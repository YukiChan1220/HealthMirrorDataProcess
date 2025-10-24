import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import os
import pandas as pd
import threading
from queue import Queue
from abc import ABC, abstractmethod
import global_vars

mirror_version = global_vars.mirror_version


class SignalData:
    def __init__(self, time=None, rppg=None, ecg=None, ppg_red=None, ppg_ir=None, ppg_green=None):
        self.time = time if time is not None else []
        self.rppg = rppg if rppg is not None else []
        self.ecg = ecg if ecg is not None else []
        self.ppg_red = ppg_red if ppg_red is not None else []
        self.ppg_ir = ppg_ir if ppg_ir is not None else []
        self.ppg_green = ppg_green if ppg_green is not None else []

    def get_signal(self, signal_name):
        return getattr(self, signal_name, [])

    def has_ppg(self):
        return len(self.ppg_red) > 0 or len(self.ppg_ir) > 0 or len(self.ppg_green) > 0


class DataLoader(ABC):
    @abstractmethod
    def load(self, data_path):
        pass


class MirrorV1Loader(DataLoader):
    def load(self, data_path):
        try:
            time, rppg, ecg = [], [], []
            with open(os.path.join(data_path, 'merged_log.csv'), 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        time.append(float(parts[0]))
                        rppg.append(float(parts[1]) if parts[1] != '' else 0.0)
                        ecg.append(float(parts[2]) if parts[2] != '' else 0.0)
            return SignalData(time=time, rppg=rppg, ecg=ecg)
        except Exception as e:
            print(f"Error loading {data_path}: {e}")
            return SignalData()


class MirrorV2Loader(DataLoader):
    def load(self, data_path):
        try:
            df = pd.read_csv(os.path.join(data_path, 'normalized_log.csv'), dtype=float)
            return SignalData(
                time=df['timestamp'].tolist(),
                rppg=df['rppg'].tolist(),
                ecg=df['ecg'].tolist(),
                ppg_red=df['ppg_red'].tolist(),
                ppg_ir=df['ppg_ir'].tolist(),
                ppg_green=df['ppg_green'].tolist()
            )
        except Exception as e:
            print(f"Error loading {data_path}: {e}")
            return SignalData()


class SignalCleaner:
    def __init__(self, fs=512):
        self.fs = fs

    def clean(self, sig, config):
        sig = np.array(sig)
        mask = np.ones(len(sig), dtype=bool)
        
        if len(sig) == 0:
            mask[:] = False
            return mask

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
            seg_std = np.std(seg)
            if seg_std > global_std * threshold:
                mask[start:start + window_len] = False
        return mask

    def _diff_filter(self, sig, params):
        window_len = int(params["window_size"] * self.fs)
        threshold = params["threshold"]
        global_diff = np.max(sig) - np.min(sig)
        mask = np.ones(len(sig), dtype=bool)
        
        for start in range(0, len(sig) - window_len, window_len):
            seg = sig[start:start + window_len]
            seg_diff = np.max(seg) - np.min(seg)
            if seg_diff > global_diff * threshold:
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
            seg_peak_freq = f[np.argmax(Pxx)]
            if abs(seg_peak_freq - peak_freq) > freq_tolerance:
                mask[start:start + window_len] = False
        return mask


class DataProcessor:
    def __init__(self, fs=512):
        self.fs = fs
        self.loader = MirrorV1Loader() if mirror_version == '1' else MirrorV2Loader()
        self.cleaner = SignalCleaner(fs)
        self.data = SignalData()
        self.masks = {}

    def load_data(self, data_path):
        self.data = self.loader.load(data_path)
        self.masks = {}

    def update_signal_mask(self, signal_name, config):
        sig = self.data.get_signal(signal_name)
        self.masks[signal_name] = self.cleaner.clean(sig, config)
        return self.masks[signal_name]

    def get_combined_mask(self, signal_names):
        if not self.masks:
            return np.ones(len(self.data.time), dtype=bool)
        mask = np.ones(len(self.data.time), dtype=bool)
        for name in signal_names:
            if name in self.masks:
                mask &= self.masks[name]
        return mask


class PlotterBase(ABC):
    def __init__(self, plot_event, plot_update_event):
        self.data = SignalData()
        self.masks = {}
        self.signal_queue = None
        self.config_queue = None
        self.event_queue = None
        self.configs = {}
        self.fig = None
        self.axes = {}
        self.sliders = {}
        self.buttons = {}
        self.event = plot_event
        self.plot_update_event = plot_update_event
        self._init_plot()

    @abstractmethod
    def _init_plot(self):
        pass

    @abstractmethod
    def _get_signal_names(self):
        pass

    def _plot_signals(self):
        for name, ax in self.axes.items():
            ax.clear()
            sig = self.data.get_signal(name)
            ax.plot(self.data.time, sig, label=f'{name.upper()} Signal')
            
            if name in self.masks and hasattr(self, 'show_mask') and self.show_mask:
                ax.fill_between(self.data.time, sig, where=~self.masks[name], 
                               color='red', alpha=0.5, label='Marked Artifacts')
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'{name.upper()} Signal')
            ax.legend()
            ax.grid()
        plt.draw()

    def update_once(self):
        if not self.plot_update_event.is_set():
            return
        try:
            from queue import Empty
            item = self.signal_queue.get_nowait()
            self._process_queue_item(item)
            self._plot_signals()
            self.plot_update_event.clear()
        except Empty:
            return

    @abstractmethod
    def _process_queue_item(self, item):
        pass

    def __call__(self, signal_queue, config_queue=None, event_queue=None):
        self.signal_queue = signal_queue
        self.config_queue = config_queue
        self.event_queue = event_queue


class RawDataPlotter(PlotterBase):
    def __init__(self, plot_event, plot_update_event):
        self.show_mask = True
        super().__init__(plot_event, plot_update_event)

    def _get_signal_names(self):
        if mirror_version == '2':
            return ['ecg', 'rppg', 'ppg_red', 'ppg_ir', 'ppg_green']
        return ['ecg', 'rppg']

    def _init_plot(self):
        plt.ion()
        signal_names = self._get_signal_names()
        n_signals = len(signal_names)
        
        self.fig, axes_list = plt.subplots(n_signals, 1, figsize=(20, 4 * n_signals))
        if n_signals == 1:
            axes_list = [axes_list]
        
        for i, name in enumerate(signal_names):
            self.axes[name] = axes_list[i]
        
        plt.subplots_adjust(bottom=0.12)
        
        ax_accept = plt.axes([0.81, 0.02, 0.08, 0.03])
        ax_reject = plt.axes([0.90, 0.02, 0.08, 0.03])
        self.buttons['accept'] = plt.Button(ax_accept, 'Accept')
        self.buttons['reject'] = plt.Button(ax_reject, 'Reject')
        self.buttons['accept'].on_clicked(lambda e: self._button_handler(e, 'raw_accept'))
        self.buttons['reject'].on_clicked(lambda e: self._button_handler(e, 'raw_reject'))
        
        if mirror_version == '2':
            slider_configs = [
                ('ecg_std', 'ECG STD', 0.02, 0.02, 0.12, 0.03),
                ('rppg_std', 'RPPG STD', 0.15, 0.02, 0.12, 0.03),
                ('rppg_bpm', 'RPPG BPM', 0.28, 0.02, 0.12, 0.03),
                ('ppg_red_std', 'PPG Red', 0.41, 0.02, 0.12, 0.03),
                ('ppg_ir_std', 'PPG IR', 0.54, 0.02, 0.12, 0.03),
                ('ppg_green_std', 'PPG Green', 0.67, 0.02, 0.12, 0.03)
            ]
        else:
            slider_configs = [
                ('ecg_std', 'ECG STD', 0.05, 0.02, 0.15, 0.03),
                ('rppg_std', 'RPPG STD', 0.25, 0.02, 0.15, 0.03),
                ('rppg_bpm', 'RPPG BPM', 0.45, 0.02, 0.15, 0.03)
            ]
        
        for name, label, left, bottom, width, height in slider_configs:
            label_ax = plt.axes([left, bottom + height + 0.005, width, 0.02])
            label_ax.axis('off')
            label_ax.text(0.5, 0.5, label, ha='center', va='center', 
                         fontsize=9, fontweight='bold', transform=label_ax.transAxes)
            
            ax = plt.axes([left, bottom, width, height])
            if 'std' in name:
                vmin, vmax, vinit, step = 0.5, 3.0, 1.5, 0.1
            else:
                vmin, vmax, vinit, step = 5, 30, 15, 1
            self.sliders[name] = plt.Slider(ax, '', vmin, vmax, 
                                           valinit=vinit, valstep=step)
            self.sliders[name].on_changed(lambda val, n=name: self._slider_handler(val, n))
        
        self.configs = {
            'ecg': {'std': {'window_size': 1, 'threshold': 1.5}},
            'rppg': {'std': {'window_size': 1, 'threshold': 1.5}, 
                    'welch': {'window_size': 5, 'bpm_tolerance': 15}}
        }
        
        if mirror_version == '2':
            for sig in ['ppg_red', 'ppg_ir', 'ppg_green']:
                self.configs[sig] = {'std': {'window_size': 1, 'threshold': 1.5}}

    def _button_handler(self, event, event_name):
        self.event.set()
        self.event_queue.put(event_name)

    def _slider_handler(self, val, slider_name):
        if slider_name == 'ecg_std':
            self.configs['ecg']['std']['threshold'] = val
        elif slider_name == 'rppg_std':
            self.configs['rppg']['std']['threshold'] = val
        elif slider_name == 'rppg_bpm':
            self.configs['rppg']['welch']['bpm_tolerance'] = val
        elif slider_name == 'ppg_red_std':
            self.configs['ppg_red']['std']['threshold'] = val
        elif slider_name == 'ppg_ir_std':
            self.configs['ppg_ir']['std']['threshold'] = val
        elif slider_name == 'ppg_green_std':
            self.configs['ppg_green']['std']['threshold'] = val
        
        self.config_queue.put(self.configs.copy())
        self.event.set()
        self.event_queue.put('raw_update')

    def _process_queue_item(self, item):
        self.data, self.masks = item


class CleanedDataPlotter(PlotterBase):
    def __init__(self, plot_event, plot_update_event):
        self.show_mask = False
        super().__init__(plot_event, plot_update_event)

    def _get_signal_names(self):
        if mirror_version == '2':
            return ['ecg', 'rppg', 'ppg_red', 'ppg_ir', 'ppg_green']
        return ['ecg', 'rppg']

    def _init_plot(self):
        plt.ion()
        signal_names = self._get_signal_names()
        n_signals = len(signal_names)
        
        self.fig, axes_list = plt.subplots(n_signals, 1, figsize=(20, 4 * n_signals))
        if n_signals == 1:
            axes_list = [axes_list]
        
        for i, name in enumerate(signal_names):
            self.axes[name] = axes_list[i]
        
        plt.subplots_adjust(bottom=0.12)
        
        ax_reverse = plt.axes([0.65, 0.02, 0.1, 0.03])
        ax_accept = plt.axes([0.76, 0.02, 0.1, 0.03])
        ax_reject = plt.axes([0.87, 0.02, 0.1, 0.03])
        self.buttons['reverse'] = plt.Button(ax_reverse, 'Reverse')
        self.buttons['accept'] = plt.Button(ax_accept, 'Accept')
        self.buttons['reject'] = plt.Button(ax_reject, 'Reject')
        self.buttons['reverse'].on_clicked(lambda e: self._button_handler(e, 'cleaned_reverse'))
        self.buttons['accept'].on_clicked(lambda e: self._button_handler(e, 'cleaned_accept'))
        self.buttons['reject'].on_clicked(lambda e: self._button_handler(e, 'cleaned_reject'))
    def _button_handler(self, event, event_name):
        self.event.set()
        self.event_queue.put(event_name)

    def _process_queue_item(self, item):
        self.data = item


class DataLogger:
    def __init__(self, log_path):
        self.log_path = log_path

    def log_cleaned_data(self, file_name, data, masks):
        signal_names = ['rppg', 'ecg']
        if data.has_ppg():
            signal_names.extend(['ppg_red', 'ppg_ir', 'ppg_green'])
        
        combined_mask = np.ones(len(data.time), dtype=bool)
        for name in signal_names:
            if name in masks:
                combined_mask &= masks[name]
        
        clean_windows = self._find_clean_windows(combined_mask)
        
        file_idx = 0
        for start, end in clean_windows:
            if start < end:
                self._save_window(file_name, file_idx, data, signal_names, start, end)
                file_idx += 1

    def _find_clean_windows(self, mask):
        windows = []
        window_begin = 0
        window_end = 0
        
        while window_begin < len(mask):
            if mask[window_begin]:
                while window_end < len(mask) and mask[window_end]:
                    window_end += 1
                windows.append((window_begin, window_end))
            if window_end <= window_begin:
                window_end = window_begin + 1
            window_begin = window_end + 1
        
        return windows

    def _save_window(self, file_name, idx, data, signal_names, start, end):
        base_name = file_name.replace('.csv', f'_{idx+1}.csv')
        output_path = os.path.join(self.log_path, base_name)
        
        df_dict = {'Time': data.time[start:end]}
        for name in signal_names:
            sig = data.get_signal(name)
            df_dict[name] = sig[start:end]
        
        df = pd.DataFrame(df_dict)
        df.to_csv(output_path, index=False)

    def modify_cleaned_data(self, file_path, option):
        try:
            full_path = os.path.join(self.log_path, file_path)
            
            if option == 'reject':
                os.remove(full_path)
                return
            
            df = pd.read_csv(full_path)
            
            if option == 'reverse':
                for col in df.columns:
                    if col != 'Time' and 'ecg' in col.lower():
                        df[col] = -df[col]
            
            for col in df.columns:
                if col != 'Time':
                    df[col] = (df[col] - df[col].mean()) / df[col].std()
            
            df.to_csv(full_path, index=False)
            
        except Exception as e:
            print(f"Error {option} {file_path}: {e}")



class Pipeline:
    def __init__(self):
        self.data_processor = None
        self.data_plotter = None
        self.data_logger = None
        self.signal_queue = Queue()
        self.config_queue = Queue()
        self.event_queue = Queue()
        self.plot_event = threading.Event()
        self.update_event = threading.Event()
        self.plot_event.set()

    def start_cleaning(self, data_path, log_path, starting_point=0, ending_point=None):
        self.data_processor = DataProcessor()
        self.data_plotter = RawDataPlotter(self.plot_event, self.update_event)
        self.data_plotter(self.signal_queue, self.config_queue, self.event_queue)
        self.data_logger = DataLogger(log_path)

        if not os.path.exists(log_path):
            os.makedirs(log_path)
        
        for path in sorted(os.listdir(data_path)):
            if global_vars.user_interrupt:
                break
            
            try:
                patient_num = int(path.split('_')[-1])
            except (ValueError, IndexError):
                continue
            
            if patient_num < starting_point or (ending_point is not None and patient_num > ending_point):
                continue
            
            self.data_processor.load_data(os.path.join(data_path, path))
            print(f"Processing patient {path}...")
            
            signal_names = ['rppg', 'ecg']
            if self.data_processor.data.has_ppg():
                signal_names.extend(['ppg_red', 'ppg_ir', 'ppg_green'])
            
            configs = self.data_plotter.configs.copy()
            masks = {}
            for name in signal_names:
                if name in configs:
                    masks[name] = self.data_processor.update_signal_mask(name, configs[name])
            
            self.signal_queue.put((self.data_processor.data, masks))
            self.update_event.set()
            self.data_plotter.update_once()
            plt.pause(0.01)

            while not global_vars.user_interrupt:
                plt.pause(0.01)
                self.data_plotter.update_once()
                
                if not self.event_queue.empty():
                    event = self.event_queue.get()
                    
                    if event == 'raw_reject':
                        break
                    elif event == 'raw_update':
                        if not self.config_queue.empty():
                            configs = self.config_queue.get()
                        
                        masks = {}
                        for name in signal_names:
                            if name in configs:
                                masks[name] = self.data_processor.update_signal_mask(name, configs[name])
                        
                        self.signal_queue.put((self.data_processor.data, masks))
                        self.update_event.set()
                    elif event == 'raw_accept':
                        if not self.config_queue.empty():
                            configs = self.config_queue.get()
                        
                        masks = {}
                        for name in signal_names:
                            if name in configs:
                                masks[name] = self.data_processor.update_signal_mask(name, configs[name])
                        
                        self.data_logger.log_cleaned_data(f'{path}.csv', self.data_processor.data, masks)
                        break
                    else:
                        print(f"Error: Unknown event {event}")
        
        plt.close('all')

    def start_checking_cleaning(self, log_path, starting_point=0, ending_point=None):
        self.data_plotter = CleanedDataPlotter(self.plot_event, self.update_event)
        self.data_plotter(self.signal_queue, event_queue=self.event_queue)
        self.data_logger = DataLogger(log_path)

        if not os.path.exists(log_path):
            print(f"Error: Log path {log_path} does not exist")
            return
        
        for file in sorted(os.listdir(log_path)):
            if global_vars.user_interrupt:
                break
            
            if not file.endswith('.csv'):
                continue
            
            try:
                patient_num = int(file.split('_')[1])
            except (ValueError, IndexError):
                continue
            
            if patient_num < starting_point or (ending_point is not None and patient_num > ending_point):
                continue
            
            print(f"Checking cleaned data {file}...")

            try:
                df = pd.read_csv(os.path.join(log_path, file))
                
                data = SignalData()
                data.time = df['Time'].tolist()
                
                for col in df.columns:
                    col_lower = col.lower()
                    if 'rppg' in col_lower:
                        data.rppg = ((df[col] - df[col].mean()) / df[col].std()).tolist()
                    elif 'ecg' in col_lower:
                        data.ecg = ((df[col] - df[col].mean()) / df[col].std()).tolist()
                    elif 'ppg_red' in col_lower or 'red' in col_lower:
                        data.ppg_red = ((df[col] - df[col].mean()) / df[col].std()).tolist()
                    elif 'ppg_ir' in col_lower or col_lower == 'ir':
                        data.ppg_ir = ((df[col] - df[col].mean()) / df[col].std()).tolist()
                    elif 'ppg_green' in col_lower or 'green' in col_lower:
                        data.ppg_green = ((df[col] - df[col].mean()) / df[col].std()).tolist()
                
                self.signal_queue.put(data)
                self.update_event.set()
                self.data_plotter.update_once()
                plt.pause(0.01)

                while not global_vars.user_interrupt:
                    plt.pause(0.01)
                    self.data_plotter.update_once()
                    
                    if not self.event_queue.empty():
                        event = self.event_queue.get()
                        
                        if event == 'cleaned_accept':
                            break
                        elif event == 'cleaned_reject':
                            self.data_logger.modify_cleaned_data(file, 'reject')
                            break
                        elif event == 'cleaned_reverse':
                            self.data_logger.modify_cleaned_data(file, 'reverse')
                            break
                        else:
                            print(f"Error: Unknown event {event}")
            
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        plt.close('all')


def main():
    data_path = input("Input data path:").strip()
    log_path = input("Input log path:").strip()
    starting_point = input("Input starting point (default 0):").strip()
    ending_point = input("Input ending point (default None):").strip()
    starting_point = int(starting_point) if starting_point.isdigit() else 0
    ending_point = int(ending_point) if ending_point.isdigit() else None
    
    pipeline = Pipeline()
    pipeline.start_cleaning(data_path=data_path, log_path=log_path, 
                           starting_point=starting_point, ending_point=ending_point)
    pipeline.start_checking_cleaning(log_path=log_path, 
                                     starting_point=starting_point, ending_point=ending_point)


if __name__ == "__main__":
    main()
