import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import os
import pandas as pd
from matplotlib.widgets import Slider
import global_vars

class SignalData:
    def __init__(self, time=None, rppg=None, ecg=None, ppg_red=None, ppg_ir=None, ppg_green=None):
        self.time = np.array(time) if time is not None else np.array([])
        self.rppg = np.array(rppg) if rppg is not None else np.array([])
        self.ecg = np.array(ecg) if ecg is not None else np.array([])
        self.ppg_red = np.array(ppg_red) if ppg_red is not None else np.array([])
        self.ppg_ir = np.array(ppg_ir) if ppg_ir is not None else np.array([])
        self.ppg_green = np.array(ppg_green) if ppg_green is not None else np.array([])

    def get_signal(self, signal_name):
        return getattr(self, signal_name, np.array([]))

    def has_ppg(self):
        return len(self.ppg_red) > 0 or len(self.ppg_ir) > 0 or len(self.ppg_green) > 0

class DataLoader:
    def __init__(self, version):
        self.version = version

    def load(self, data_path):
        if self.version == '1':
            return self._load_v1(data_path)
        else:
            return self._load_v2(data_path)

    def _load_v1(self, data_path):
        time, rppg, ecg = [], [], []
        file_path = os.path.join(data_path, 'merged_log.csv')
        if not os.path.exists(file_path):
            return SignalData()
            
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    time.append(float(parts[0]))
                    rppg.append(float(parts[1]) if parts[1] != '' else 0.0)
                    ecg.append(float(parts[2]) if parts[2] != '' else 0.0)
        return SignalData(time=time, rppg=rppg, ecg=ecg)

    def _load_v2(self, data_path):
        file_path = os.path.join(data_path, 'normalized_log.csv')
        if not os.path.exists(file_path):
            return SignalData()
            
        df = pd.read_csv(file_path, dtype=float)
        return SignalData(
            time=df['timestamp'].tolist(),
            rppg=df['rppg'].tolist(),
            ecg=df['ecg'].tolist(),
            ppg_red=df['ppg_red'].tolist(),
            ppg_ir=df['ppg_ir'].tolist(),
            ppg_green=df['ppg_green'].tolist()
        )

class SignalCleaner:
    def __init__(self, fs=512):
        self.fs = fs

    def clean(self, sig, config):
        if len(sig) == 0:
            return np.array([], dtype=bool)
        
        mask = np.ones(len(sig), dtype=bool)
        if "std" in config:
            mask &= self._std_filter(sig, config["std"])
        if "diff" in config:
            mask &= self._diff_filter(sig, config["diff"])
        if "welch" in config:
            mask &= self._welch_filter(sig, config["welch"])
        return mask

    def _std_filter(self, sig, params):
        window_len = int(params["window_size"] * self.fs)
        threshold = params["threshold"]
        global_std = np.std(sig)
        mask = np.ones(len(sig), dtype=bool)
        
        for start in range(0, len(sig) - window_len, window_len):
            seg = sig[start:start + window_len]
            if np.std(seg) > global_std * threshold:
                mask[start:start + window_len] = False
        return mask

    def _diff_filter(self, sig, params):
        window_len = int(params["window_size"] * self.fs)
        threshold = params["threshold"]
        global_diff = np.max(sig) - np.min(sig)
        mask = np.ones(len(sig), dtype=bool)
        
        for start in range(0, len(sig) - window_len, window_len):
            seg = sig[start:start + window_len]
            if (np.max(seg) - np.min(seg)) > global_diff * threshold:
                mask[start:start + window_len] = False
        return mask

    def _welch_filter(self, sig, params):
        window_len = int(params["window_size"] * self.fs)
        freq_tolerance = params["bpm_tolerance"] / 60
        gf, gPxx = signal.welch(sig, fs=self.fs, nperseg=window_len)
        peak_freq = gf[np.argmax(gPxx)]
        mask = np.ones(len(sig), dtype=bool)
        
        for start in range(0, len(sig) - window_len, window_len):
            seg = sig[start:start + window_len]
            f, Pxx = signal.welch(seg, fs=self.fs, nperseg=window_len)
            if abs(f[np.argmax(Pxx)] - peak_freq) > freq_tolerance:
                mask[start:start + window_len] = False
        return mask

class DataLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        if not os.path.exists(log_path):
            os.makedirs(log_path)

    def log_cleaned_data(self, file_name, data, masks):
        signal_names = ['rppg', 'ecg']
        if data.has_ppg():
            signal_names.extend(['ppg_red', 'ppg_ir', 'ppg_green'])
        
        combined_mask = np.ones(len(data.time), dtype=bool)
        for name in signal_names:
            if name in masks:
                combined_mask &= masks[name]
        
        windows = self._find_clean_windows(combined_mask)
        for idx, (start, end) in enumerate(windows):
            self._save_window(file_name, idx, data, signal_names, start, end)

    def _find_clean_windows(self, mask):
        windows = []
        start = 0
        while start < len(mask):
            if mask[start]:
                end = start
                while end < len(mask) and mask[end]:
                    end += 1
                windows.append((start, end))
                start = end
            else:
                start += 1
        return windows

    def _save_window(self, file_name, idx, data, signal_names, start, end):
        base_name = file_name.replace('.csv', f'_{idx+1}.csv')
        output_path = os.path.join(self.log_path, base_name)
        
        df_dict = {'Time': data.time[start:end]}
        for name in signal_names:
            df_dict[name] = data.get_signal(name)[start:end]
        
        pd.DataFrame(df_dict).to_csv(output_path, index=False)

    def modify_cleaned_data(self, file_name, option):
        full_path = os.path.join(self.log_path, file_name)
        if not os.path.exists(full_path):
            return

        if option == 'reject':
            os.remove(full_path)
            return

        df = pd.read_csv(full_path)
        if option == 'reverse':
            for col in df.columns:
                if 'ecg' in col.lower():
                    df[col] = -df[col]
        
        # Normalize
        for col in df.columns:
            if col != 'Time':
                df[col] = (df[col] - df[col].mean()) / df[col].std()
        
        df.to_csv(full_path, index=False)

class WashDataUI:
    def __init__(self, mode='cleaning', version='1'):
        self.mode = mode
        self.version = version
        self.fig = None
        self.axes = {}
        self.sliders = {}
        self.callbacks = {}
        self.signal_names = self._get_signal_names()
        self._init_plot()

    def _get_signal_names(self):
        if self.version == '2':
            return ['ecg', 'rppg', 'ppg_red', 'ppg_ir', 'ppg_green']
        return ['ecg', 'rppg']

    def _init_plot(self):
        n_signals = len(self.signal_names)
        self.fig, axes_list = plt.subplots(n_signals, 1, figsize=(16, 3 * n_signals))
        if n_signals == 1: axes_list = [axes_list]
        
        for i, name in enumerate(self.signal_names):
            self.axes[name] = axes_list[i]
        
        plt.subplots_adjust(bottom=0.2 if self.mode == 'cleaning' else 0.1)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        if self.mode == 'cleaning':
            self._init_sliders()
            self.fig.suptitle("Controls: 'y' Accept, 'n' Reject, 'q' Quit", fontsize=12)
        else:
            self.fig.suptitle("Controls: 'y' Accept, 'n' Reject, 'r' Reverse, 'q' Quit", fontsize=12)

    def _init_sliders(self):
        slider_configs = []
        if self.version == '2':
            slider_configs = [
                ('ecg_std', 'ECG STD', 0.05, 0.02, 0.12, 0.03),
                ('rppg_std', 'RPPG STD', 0.20, 0.02, 0.12, 0.03),
                ('rppg_bpm', 'RPPG BPM', 0.35, 0.02, 0.12, 0.03),
                ('ppg_red_std', 'Red STD', 0.50, 0.02, 0.12, 0.03),
                ('ppg_ir_std', 'IR STD', 0.65, 0.02, 0.12, 0.03),
                ('ppg_green_std', 'Green STD', 0.80, 0.02, 0.12, 0.03)
            ]
        else:
            slider_configs = [
                ('ecg_std', 'ECG STD', 0.1, 0.05, 0.2, 0.03),
                ('rppg_std', 'RPPG STD', 0.4, 0.05, 0.2, 0.03),
                ('rppg_bpm', 'RPPG BPM', 0.7, 0.05, 0.2, 0.03)
            ]

        for name, label, left, bottom, width, height in slider_configs:
            ax = plt.axes([left, bottom, width, height])
            vmin, vmax, vinit, step = (0.5, 3.0, 1.5, 0.1) if 'std' in name else (5, 30, 15, 1)
            slider = Slider(ax, label, vmin, vmax, valinit=vinit, valstep=step)
            slider.on_changed(lambda val, n=name: self._on_slider_change(n, val))
            self.sliders[name] = slider

    def update_plot(self, data, masks=None, title_suffix=""):
        for name in self.signal_names:
            ax = self.axes[name]
            ax.clear()
            sig = data.get_signal(name)
            if len(sig) == 0: continue
            
            ax.plot(data.time, sig, label=f'{name.upper()}')
            
            if masks and name in masks:
                ax.fill_between(data.time, sig, where=~masks[name], color='red', alpha=0.3, label='Artifact')
            
            ax.set_title(f'{name.upper()} Signal {title_suffix}')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        self.fig.canvas.draw()

    def get_config(self):
        config = {
            'ecg': {'std': {'window_size': 1, 'threshold': 1.5}},
            'rppg': {'std': {'window_size': 1, 'threshold': 1.5}, 
                    'welch': {'window_size': 5, 'bpm_tolerance': 15}}
        }
        if self.version == '2':
            for sig in ['ppg_red', 'ppg_ir', 'ppg_green']:
                config[sig] = {'std': {'window_size': 1, 'threshold': 1.5}}

        for name, slider in self.sliders.items():
            val = slider.val
            if name == 'ecg_std': config['ecg']['std']['threshold'] = val
            elif name == 'rppg_std': config['rppg']['std']['threshold'] = val
            elif name == 'rppg_bpm': config['rppg']['welch']['bpm_tolerance'] = val
            elif name == 'ppg_red_std': config['ppg_red']['std']['threshold'] = val
            elif name == 'ppg_ir_std': config['ppg_ir']['std']['threshold'] = val
            elif name == 'ppg_green_std': config['ppg_green']['std']['threshold'] = val
        return config

    def on_action(self, callback):
        self.callbacks['action'] = callback

    def on_config_change(self, callback):
        self.callbacks['config'] = callback

    def _on_key(self, event):
        if event.key in ['y', 'n', 'r', 'q'] and 'action' in self.callbacks:
            self.callbacks['action'](event.key)

    def _on_slider_change(self, name, val):
        if 'config' in self.callbacks:
            self.callbacks['config']()

class WashDataController:
    def __init__(self, data_path, log_path, start=0, end=None):
        self.data_path = data_path
        self.log_path = log_path
        self.start = start
        self.end = end
        self.version = global_vars.mirror_version
        self.loader = DataLoader(self.version)
        self.cleaner = SignalCleaner()
        self.logger = DataLogger(log_path)
        self.ui = None
        self.files = []
        self.current_idx = 0
        self.current_data = None

    def run_cleaning(self):
        self.files = self._get_files(self.data_path, raw=True)
        if not self.files:
            print("No files to process.")
            return

        self.ui = WashDataUI(mode='cleaning', version=self.version)
        self.ui.on_action(self._handle_cleaning_action)
        self.ui.on_config_change(self._update_cleaning_view)
        
        self._load_next_raw()
        plt.show(block=True)

    def run_checking(self):
        self.files = self._get_files(self.log_path, raw=False)
        if not self.files:
            print("No cleaned files to check.")
            return

        self.ui = WashDataUI(mode='checking', version=self.version)
        self.ui.on_action(self._handle_checking_action)
        
        self._load_next_cleaned()
        plt.show(block=True)

    def _get_files(self, path, raw=True):
        files = []
        if not os.path.exists(path): return []
        
        for f in sorted(os.listdir(path)):
            if raw and not os.path.isdir(os.path.join(path, f)): continue
            if not raw and not f.endswith('.csv'): continue
            
            try:
                pid = int(f.split('_')[-1] if raw else f.split('_')[1])
                if pid < self.start or (self.end is not None and pid > self.end):
                    continue
                files.append(f)
            except: continue
        return files

    def _load_next_raw(self):
        if self.current_idx >= len(self.files):
            print("All files processed.")
            plt.close('all')
            return

        file_name = self.files[self.current_idx]
        print(f"Processing {file_name} ({self.current_idx + 1}/{len(self.files)})...")
        self.current_data = self.loader.load(os.path.join(self.data_path, file_name))
        self._update_cleaning_view()

    def _update_cleaning_view(self):
        if self.current_data is None: return
        config = self.ui.get_config()
        masks = {}
        for name in self.ui.signal_names:
            if name in config:
                masks[name] = self.cleaner.clean(self.current_data.get_signal(name), config[name])
        self.ui.update_plot(self.current_data, masks, title_suffix=f"- {self.files[self.current_idx]}")

    def _handle_cleaning_action(self, action):
        if action == 'q':
            plt.close('all')
            return
        
        if action == 'y':
            config = self.ui.get_config()
            masks = {}
            for name in self.ui.signal_names:
                if name in config:
                    masks[name] = self.cleaner.clean(self.current_data.get_signal(name), config[name])
            self.logger.log_cleaned_data(f'{self.files[self.current_idx]}.csv', self.current_data, masks)
        
        self.current_idx += 1
        self._load_next_raw()

    def _load_next_cleaned(self):
        if self.current_idx >= len(self.files):
            print("All files checked.")
            plt.close('all')
            return

        file_name = self.files[self.current_idx]
        print(f"Checking {file_name} ({self.current_idx + 1}/{len(self.files)})...")
        
        df = pd.read_csv(os.path.join(self.log_path, file_name))
        data = SignalData(time=df['Time'].tolist())
        
        # Normalize for display
        for col in df.columns:
            if col == 'Time': continue
            norm_sig = (df[col] - df[col].mean()) / df[col].std()
            if 'rppg' in col.lower(): data.rppg = norm_sig.tolist()
            elif 'ecg' in col.lower(): data.ecg = norm_sig.tolist()
            elif 'red' in col.lower(): data.ppg_red = norm_sig.tolist()
            elif 'ir' in col.lower(): data.ppg_ir = norm_sig.tolist()
            elif 'green' in col.lower(): data.ppg_green = norm_sig.tolist()
            
        self.current_data = data
        self.ui.update_plot(data, title_suffix=f"- {file_name}")

    def _handle_checking_action(self, action):
        if action == 'q':
            plt.close('all')
            return
            
        file_name = self.files[self.current_idx]
        if action == 'n':
            self.logger.modify_cleaned_data(file_name, 'reject')
        elif action == 'r':
            self.logger.modify_cleaned_data(file_name, 'reverse')
        
        self.current_idx += 1
        self._load_next_cleaned()

def main():
    data_path = input("Input data path:").strip()
    log_path = input("Input log path:").strip()
    start = input("Input starting point (default 0):").strip()
    end = input("Input ending point (default None):").strip()
    
    start = int(start) if start.isdigit() else 0
    end = int(end) if end.isdigit() else None
    
    app = WashDataController(data_path, log_path, start, end)
    
    print("\n--- Starting Cleaning Phase ---")
    app.run_cleaning()
    
    # Reset for checking phase
    app.current_idx = 0
    print("\n--- Starting Checking Phase ---")
    app.run_checking()

if __name__ == "__main__":
    main()
