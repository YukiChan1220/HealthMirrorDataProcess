from data.patient_info import PatientInfo
from data.load_data import DataLoader
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import csv
import os
import pandas as pd
import neurokit2 as nk
from ecg.ecg_process import ECGProcess

lab = False

if lab:
    data_dir = "./lab_mirror_data"
    output_file = "lab_overall_patient_info.csv"
    merged_patient_file = "lab_merged_patient_info.csv"
    cleaned_dir = "./lab_test_cleaned"
else:
    data_dir = "./patient_data"
    output_file = "overall_patient_info.csv"
    merged_patient_file = "merged_patient_info.csv"
    cleaned_dir = "./test_sliced"

def load_all_patients(data_dir=data_dir, output_file=output_file):
    patient_info = PatientInfo(data_dir, save_dir=output_file, mode="file")
    patient_info_list = patient_info.extract(data_file=merged_patient_file)
    return patient_info_list

def load_patient_with_bp(data_dir=data_dir, output_file=output_file):
    patient_info = PatientInfo(data_dir, save_dir=output_file, mode="file")
    patient_info_list = patient_info.extract(data_file=merged_patient_file)
    patient_with_bp = [p for p in patient_info_list if int(p['low_blood_pressure']) != -1 and int(p['high_blood_pressure']) != -1]
    return patient_with_bp

def load_data_for_patients(patient_list, raw_dir=data_dir, cleaned_dir=cleaned_dir):
    patient_ids = [int(p['lab_patient_id']) for p in patient_list]
    data_loader = DataLoader(raw_dir=raw_dir, cleaned_dir=cleaned_dir)
    raw_data_loader = data_loader.load_raw_data(patient_id=patient_ids)
    cleaned_data_loader = data_loader.load_cleaned_data(patient_id=patient_ids)
    return raw_data_loader, cleaned_data_loader

def load_reference_waveforms(ref_dir):
    for f in os.listdir(ref_dir):
        if f.endswith(".csv"):
            try:
                file_path = os.path.join(ref_dir, f)
                df = pd.read_csv(file_path)
                timestamps = df.loc[:, 'timestamps'].astype(float).to_numpy()
                ecg_signal = df.loc[:, 'ecg'].astype(float).to_numpy()
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            yield int(f[4:7]), timestamps, ecg_signal

def filter_signal(data, fs=512, lowcut=0.5, highcut=5.0, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    data = signal.filtfilt(b, a, data)
    return data

def find_rppg_peaks(signal_data, fs=512, min_distance=None):
    if min_distance is None:
        min_distance = int(fs * 0.35)
    peaks, properties = signal.find_peaks(signal_data, distance=min_distance, height=0)
    return peaks

def calculate_ptt(time, rppg_signal, ecg_signal, ecg_processor):
    rppg_peaks = find_rppg_peaks(rppg_signal, fs=512)
    ecg_processor.process(ecg_signal)
    ecg_peaks = ecg_processor.get_peaks()
    
    if len(rppg_peaks) == 0 or len(ecg_peaks) == 0:
        return None, None, None
    
    matched_pairs = []
    ptt_values = []
    
    for ecg_idx in ecg_peaks:
        ecg_time = time[ecg_idx]
        future_rppg_peaks = rppg_peaks[rppg_peaks > ecg_idx]
        
        if len(future_rppg_peaks) > 0:
            rppg_idx = future_rppg_peaks[0]
            rppg_time = time[rppg_idx]
            ptt = rppg_time - ecg_time

            if 0.05 < ptt < 0.4:
                matched_pairs.append((ecg_idx, rppg_idx))
                ptt_values.append(ptt)
    
    if len(ptt_values) == 0:
        return None, None, None
    
    ptt_median = np.median(ptt_values)
    ptt_filtered = [p for p in ptt_values if abs(p - ptt_median) < 0.1]
    
    if len(ptt_filtered) == 0:
        return None, None, None
    
    ptt_final = np.mean(ptt_filtered)
    std = np.std(ptt_filtered)
    
    return ptt_final, None, std

class RawSignalViewer:
    def __init__(self, dataloader, reference_waveforms=None, method='nk'):
        self.dataloader = dataloader
        self.dataframe = None
        self.current_raw_idx = 0
        self.fig, self.axes = plt.subplots(3, 2, figsize=(18, 12))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.ptt_results = []
        self.fs = 512
        self.ecg_processor = ECGProcess(method=method, fs=self.fs)
        self.patient_id = None

        self.selected_signal = 0
        self.signal_list = ['ecg', 'rppg', 'ppg']
        self.clipped_segments = {s: [] for s in self.signal_list}
        self.current_segment_idx = {s: 0 for s in self.signal_list}
        
    def on_key_press(self, event):
        if event.key == 'y':
            print("Accept")
            if self.current_ptt is not None:
                self.ptt_results.append((self.current_patient_id, self.current_ptt, 
                                        None, self.current_std))
            self.current_raw_idx += 1
            self.clipped_segments = {s: [] for s in self.signal_list}
            self.patient_id, self.dataframe = next(self.dataloader, (None, None))
            if self.dataframe is None:
                plt.close(self.fig)
                return
            self.update_plot()
        elif event.key == 'n':
            print("Reject")
            self.current_raw_idx += 1
            self.clipped_segments = {s: [] for s in self.signal_list}
            self.patient_id, self.dataframe = next(self.dataloader, (None, None))
            if self.dataframe is None:
                plt.close(self.fig)
                return
            self.update_plot()
        elif event.key == 'esc':
            print("Quit")
            plt.close(self.fig)
        elif event.key == 'up':
            self.axes[self.selected_signal, 0].set_facecolor('white')
            self.axes[self.selected_signal, 1].set_facecolor('white')
            self.selected_signal = (self.selected_signal - 1) % len(self.signal_list)
            self.axes[self.selected_signal, 0].set_facecolor('lightgray')
            self.axes[self.selected_signal, 1].set_facecolor('lightgray')
            self.fig.canvas.draw()
            print(f"Selected {self.signal_list[self.selected_signal].upper()} signal")
        elif event.key == 'down':
            self.axes[self.selected_signal, 0].set_facecolor('white')
            self.axes[self.selected_signal, 1].set_facecolor('white')
            self.selected_signal = (self.selected_signal + 1) % len(self.signal_list)
            self.axes[self.selected_signal, 0].set_facecolor('lightgray')
            self.axes[self.selected_signal, 1].set_facecolor('lightgray')
            self.fig.canvas.draw()
            print(f"Selected {self.signal_list[self.selected_signal].upper()} signal")
        elif event.key == 'right':
            print("Next Segment")
            self.current_segment_idx[self.signal_list[self.selected_signal]] += 1
            self.update_subplot(signal_idx=self.selected_signal, segment=True)
        elif event.key == 'w':
            print("Save segment")
            self.save_segment(signal_idx=self.selected_signal)  
    
    def update_subplot(self, signal_idx=0, segment=False):
        signal_name = self.signal_list[signal_idx]
        if not segment:
            timestamps = self.dataframe['Timestamp'].to_numpy()
            self.axes[signal_idx, 0].clear()
            self.axes[signal_idx, 1].clear()
            if signal_name == 'rppg':
                rppg_signal = self.dataframe['RPPG'].to_numpy()
                rppg_filtered = rppg_signal
                rppg_peaks = find_rppg_peaks(rppg_filtered, fs=512)
                self.axes[signal_idx, 0].plot(timestamps, rppg_filtered, label='rPPG Signal')
                self.axes[signal_idx, 0].plot(timestamps[rppg_peaks], rppg_filtered[rppg_peaks], "x", label='rPPG Peaks')
                self.axes[signal_idx, 0].set_title(f'rPPG Signal - Patient {patient_id}')
                self.axes[signal_idx, 0].set_xlabel('Time (s)')
                self.axes[signal_idx, 0].set_ylabel('Amplitude')
                self.axes[signal_idx, 0].legend()
                self.axes[signal_idx, 0].grid(True, alpha=0.3)
                self.fig.canvas.draw()

            elif signal_name == 'ecg':
                ecg_signal = self.dataframe['ECG'].to_numpy()
                ecg_filtered = ecg_signal
                ptt, _, std = calculate_ptt(timestamps, rppg_filtered, ecg_filtered, self.ecg_processor)
                ecg_peaks = self.ecg_processor.get_peaks()
                additional_signals = self.ecg_processor.get_additional_signals()
                self.axes[signal_idx, 0].plot(timestamps, ecg_filtered, label='ECG Signal')
                self.axes[signal_idx, 0].plot(timestamps[ecg_peaks], ecg_filtered[ecg_peaks], "o", label='ECG Peaks')

                if self.ecg_processor.method == 'pt':
                    pantompkins = additional_signals['pantompkins']
                    pt_peaks = additional_signals['pt_peaks']
                    self.axes[signal_idx, 0].plot(timestamps, pantompkins, label='Pan-Tompkins', alpha=0.6)
                    self.axes[signal_idx, 0].plot(timestamps[pt_peaks], pantompkins[pt_peaks], "s", label='PT Peaks', markersize=4)

                if self.ecg_processor.method == 'nk':
                    ecg_quality = additional_signals['quality']
                    self.axes[signal_idx, 0].plot(timestamps, ecg_quality, label='ECG Quality', alpha=0.6)
                
                ptt_text = f"PTT: {ptt:.3f}s, Std: {std:.3f}" if ptt is not None else "PTT: N/A"
                if self.ecg_processor.method == 'pt':
                    self.axes[signal_idx, 0].set_title(f'ECG Signal - {ptt_text}')
                elif self.ecg_processor.method == 'nk':
                    self.axes[signal_idx, 0].set_title(f'ECG Signal - {ptt_text}, Mean quality: {np.mean(ecg_quality):.3f}')
                self.axes[signal_idx, 0].set_xlabel('Time (s)')
                self.axes[signal_idx, 0].set_ylabel('Amplitude')
                self.axes[signal_idx, 0].legend()
                self.axes[signal_idx, 0].grid(True, alpha=0.3)
                self.fig.canvas.draw()

            elif signal_name == 'ppg':
                # TODO: Implement PPG plotting
                pass

        else:
            if signal_name == 'rppg':
                # TODO: Implement clipped segment plotting for rPPG
                pass
            elif signal_name == 'ecg':
                self.axes[signal_idx, 1].clear()
                self.axes[signal_idx, 1].set_title('Clipped Segment Viewer')
                if self.current_segment_idx[signal_name] < len(self.clipped_segments[signal_name]):
                    seg_df = self.clipped_segments[signal_name][self.current_segment_idx[signal_name]]
                linear_sim = None
                cosine_sim = None
                if reference_waveforms is not None:
                    linear_sims = []
                    cosine_sims = []
                    target_length = 512
                    new_timestamps = np.linspace(seg_df['Timestamp'].iloc[0], seg_df['Timestamp'].iloc[-1], target_length)
                    resampled_ecg = interp1d(seg_df['Timestamp'], seg_df['ECG'], kind='cubic', fill_value='extrapolate')(new_timestamps)
                    for patient_id, timestamps, ref_ecg in reference_waveforms:
                        linear = np.corrcoef(resampled_ecg, ref_ecg)[0, 1]
                        linear_sims.append((patient_id, linear))
                        cosine = np.dot(resampled_ecg, ref_ecg) / (np.linalg.norm(resampled_ecg) * np.linalg.norm(ref_ecg))
                        cosine_sims.append((patient_id, cosine))
                    linear_sim = np.mean([s[1] for s in linear_sims]) if linear_sims else -1.0
                    cosine_sim = np.mean([s[1] for s in cosine_sims]) if cosine_sims else -2.0

                self.axes[signal_idx, 1].plot(seg_df['Timestamp'], seg_df['RPPG'], label='Clipped rPPG Signal')
                self.axes[signal_idx, 1].plot(seg_df['Timestamp'], seg_df['ECG'], label='Clipped ECG Signal')
                self.axes[signal_idx, 1].set_xlabel('Time (s)')
                self.axes[signal_idx, 1].set_ylabel('Amplitude')
                self.axes[signal_idx, 1].legend()
                self.axes[signal_idx, 1].grid(True, alpha=0.3)
                self.axes[signal_idx, 1].set_title(f'Clipped Segment {self.current_clipped_idx + 1}/{len(self.clipped_segments)}-Patient {self.patient_id}. Linear: {linear_sim:.3f}, Cosine: {cosine_sim:.3f}')
        
    def save_segment(self, signal_idx=0):
        signal_name = self.signal_list[signal_idx]
        if self.current_segment_idx[signal_name] < len(self.clipped_segments[signal_name]):
            seg_df = self.clipped_segments[signal_name][self.current_segment_idx[signal_name]]
            filename = f"seg_{self.current_patient_id}_{self.current_raw_idx+1}_{self.current_segment_idx[signal_name] + 1}_{signal_name}.csv"
            target_length = 512
            new_timestamps = np.linspace(seg_df['Timestamp'].iloc[0], seg_df['Timestamp'].iloc[-1], target_length)
            resampled_ecg = interp1d(seg_df['Timestamp'], seg_df[signal_name.upper()], kind='cubic', fill_value='extrapolate')(new_timestamps)
            save_df = pd.DataFrame({
                'Timestamp': new_timestamps,
                signal_name: resampled_ecg
            })
            save_df.to_csv(filename, index=False)
            print(f"Saved clipped segment to {filename}")

    def update_plot(self):
        for self.selected_signal in range(len(self.signal_list)):
            self.update_subplot(signal_idx=self.selected_signal, segment=False)

        self.fig.suptitle("Press 'y' to accept, 'n' to reject, 'esc' to quit", fontsize=12)
        self.fig.tight_layout()
        self.fig.canvas.draw()
    
    def __call__(self):
        self.patient_id, self.dataframe = next(self.dataloader, (None, None))
        if self.dataframe is None:
            plt.close(self.fig)
            return
        self.update_plot()
        plt.show()

        return self.current_raw_idx + 1


if __name__ == '__main__':
    bp_patient_list = load_patient_with_bp()
    raw_data_loader, cleaned_data_loader = load_data_for_patients(bp_patient_list)
    reference_waveforms = list(load_reference_waveforms('./reference_ecg'))
    
    viewer = RawSignalViewer(cleaned_data_loader, reference_waveforms=reference_waveforms, method='pt')
    total_processed = viewer()
    print(f"Total processed raw signals: {total_processed}")
    
