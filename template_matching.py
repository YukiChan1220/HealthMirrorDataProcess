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
    raw_data = list(data_loader.load_raw_data(patient_id=patient_ids))
    cleaned_data = list(data_loader.load_cleaned_data(patient_id=patient_ids))
    return raw_data, cleaned_data

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
    def __init__(self, data_list, reference_waveforms=None, method='nk'):
        self.data_list = data_list if data_list is not None else []
        self.current_raw_idx = 0
        self.current_clipped_idx = 0
        self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 12))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.ptt_results = []
        self.clipped_segments = []
        self.fs = 512
        self.ecg_processor = ECGProcess(method=method, fs=self.fs)
        
    def on_key_press(self, event):
        if event.key == 'y':
            print("Accept")
            if self.current_ptt is not None:
                self.ptt_results.append((self.current_patient_id, self.current_ptt, 
                                        None, self.current_std))
            self.current_raw_idx += 1
            self.current_clipped_idx = 0
            self.clipped_segments = []
            self.update_plot()
        elif event.key == 'n':
            print("Reject")
            self.current_raw_idx += 1
            self.current_clipped_idx = 0
            self.clipped_segments = []
            self.update_plot()
        elif event.key == 'q':
            print("Quit")
            plt.close(self.fig)
        elif event.key == 'c':
            print("Next Segment")
            self.current_clipped_idx += 1
            self.update_plot(segment=True)
        elif event.key == 'v':
            print("Save segment")
            self.update_plot(save=True)
            # Implement saving functionality here   
    
    def update_plot(self, segment=False, save=False):
        if self.current_raw_idx >= len(self.data_list):
            print(f"All {len(self.data_list)} segments processed.")
            plt.close(self.fig)
            return
        
        if len(self.clipped_segments) == 0:
            patient_id, timestamps, rppg_signal, ecg_signal = self.data_list[self.current_raw_idx]
            self.current_patient_id = patient_id
            
            print(f"\nPatient ID: {patient_id} [{self.current_raw_idx + 1}/{len(self.data_list)}]")
            
            #rppg_filtered = filter_signal(rppg_signal, fs=512, lowcut=0.5, highcut=5.0, order=4)
            #ecg_filtered = filter_signal(ecg_signal, fs=512, lowcut=0.1, highcut=150.0, order=2)
            rppg_filtered = rppg_signal
            ecg_filtered = ecg_signal
            
            ptt, _, std = calculate_ptt(timestamps, rppg_filtered, ecg_filtered, self.ecg_processor)
            self.current_ptt = ptt
            self.current_std = std
            
            if ptt is not None:
                print(f"PTT: {ptt:.3f} seconds, Std: {std:.3f}")
            else:
                print("PTT: Unable to calculate")
            
            rppg_peaks = find_rppg_peaks(rppg_filtered, fs=512)
            ecg_peaks = self.ecg_processor.get_peaks()
            additional_signals = self.ecg_processor.get_additional_signals()

            for ecg_peak_idx in range(len(ecg_peaks)):
                start_idx = max(0, ecg_peaks[ecg_peak_idx-1] + int(0.7 * (ecg_peaks[ecg_peak_idx] - ecg_peaks[ecg_peak_idx-1])) if ecg_peak_idx > 0 else 0)
                end_idx = min(len(timestamps), ecg_peaks[ecg_peak_idx] + int(0.7 * (ecg_peaks[ecg_peak_idx+1] - ecg_peaks[ecg_peak_idx])) if ecg_peak_idx < len(ecg_peaks) - 1 else len(timestamps))
                self.clipped_segments.append((
                    patient_id,
                    timestamps[start_idx:end_idx],
                    rppg_filtered[start_idx:end_idx],
                    ecg_filtered[start_idx:end_idx]
                ))
        
        if save:
            seg_patient_id, seg_timestamps, seg_rppg, seg_ecg = self.clipped_segments[self.current_clipped_idx]
            filename = f"seg_{self.current_patient_id}_{self.current_raw_idx+1}_{self.current_clipped_idx + 1}.csv"
            target_length = 512
            new_timestamps = np.linspace(seg_timestamps[0], seg_timestamps[-1], target_length)
            resampled_ecg = interp1d(seg_timestamps, seg_ecg, kind='cubic', fill_value='extrapolate')(new_timestamps)
            csv_data = zip(new_timestamps, resampled_ecg)
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['timestamps', 'ecg'])
                writer.writerows(csv_data)
            print(f"Saved clipped segment to {filename}")
            return
            
        if not segment:
            self.axes[0].clear()
            self.axes[1].clear()
            
            self.axes[0].plot(timestamps, rppg_filtered, label='rPPG Signal')
            self.axes[0].plot(timestamps[rppg_peaks], rppg_filtered[rppg_peaks], "x", label='rPPG Peaks')
            self.axes[0].set_title(f'rPPG Signal - Patient {patient_id}')
            self.axes[0].set_xlabel('Time (s)')
            self.axes[0].set_ylabel('Amplitude')
            self.axes[0].legend()
            self.axes[0].grid(True, alpha=0.3)
            
            self.axes[1].plot(timestamps, ecg_filtered, label='ECG Signal')
            self.axes[1].plot(timestamps[ecg_peaks], ecg_filtered[ecg_peaks], "o", label='ECG Peaks')

            if self.ecg_processor.method == 'pt':
                pantompkins = additional_signals['pantompkins']
                pt_peaks = additional_signals['pt_peaks']
                self.axes[1].plot(timestamps, pantompkins, label='Pan-Tompkins', alpha=0.6)
                self.axes[1].plot(timestamps[pt_peaks], pantompkins[pt_peaks], "s", label='PT Peaks', markersize=4)

            if self.ecg_processor.method == 'nk':
                ecg_quality = additional_signals['quality']
                self.axes[1].plot(timestamps, ecg_quality, label='ECG Quality', alpha=0.6)
            
            ptt_text = f"PTT: {ptt:.3f}s, Std: {std:.3f}" if ptt is not None else "PTT: N/A"
            if self.ecg_processor.method == 'pt':
                self.axes[1].set_title(f'ECG Signal - {ptt_text}')
            elif self.ecg_processor.method == 'nk':
                self.axes[1].set_title(f'ECG Signal - {ptt_text}, Mean quality: {np.mean(ecg_quality):.3f}')
            self.axes[1].set_xlabel('Time (s)')
            self.axes[1].set_ylabel('Amplitude')
            self.axes[1].legend()
            self.axes[1].grid(True, alpha=0.3)

        self.axes[2].clear()
        self.axes[2].set_title('Clipped Segment Viewer')
        if self.current_clipped_idx < len(self.clipped_segments):
            seg_patient_id, seg_timestamps, seg_rppg, seg_ecg = self.clipped_segments[self.current_clipped_idx]

            linear_sim = None
            cosine_sim = None
            if reference_waveforms is not None:
                linear_sims = []
                cosine_sims = []
                target_length = 512
                new_timestamps = np.linspace(seg_timestamps[0], seg_timestamps[-1], target_length)
                resampled_ecg = interp1d(seg_timestamps, seg_ecg, kind='cubic', fill_value='extrapolate')(new_timestamps)
                for patient_id, timestamps, ref_ecg in reference_waveforms:
                    linear = np.corrcoef(resampled_ecg, ref_ecg)[0, 1]
                    linear_sims.append((patient_id, linear))
                    cosine = np.dot(resampled_ecg, ref_ecg) / (np.linalg.norm(resampled_ecg) * np.linalg.norm(ref_ecg))
                    cosine_sims.append((patient_id, cosine))
                linear_sim = np.mean([s[1] for s in linear_sims]) if linear_sims else -1.0
                cosine_sim = np.mean([s[1] for s in cosine_sims]) if cosine_sims else -2.0

            self.axes[2].plot(seg_timestamps, seg_rppg, label='Clipped rPPG Signal')
            self.axes[2].plot(seg_timestamps, seg_ecg, label='Clipped ECG Signal')
            self.axes[2].set_xlabel('Time (s)')
            self.axes[2].set_ylabel('Amplitude')
            self.axes[2].legend()
            self.axes[2].grid(True, alpha=0.3)
            self.axes[2].set_title(f'Clipped Segment {self.current_clipped_idx + 1}/{len(self.clipped_segments)}-Patient {seg_patient_id}. Linear: {linear_sim:.3f}, Cosine: {cosine_sim:.3f}')
        
        self.fig.suptitle("Press 'y' to accept, 'n' to reject, 'q' to quit, 'c' to view next segment, 'v' to save current segment", fontsize=12)
        self.fig.tight_layout()
        self.fig.canvas.draw()
    
    def show(self):
        self.update_plot()
        plt.show()
        return self.ptt_results

if __name__ == '__main__':
    bp_patient_list = load_patient_with_bp()
    raw_data, cleaned_data = load_data_for_patients(bp_patient_list)
    reference_waveforms = list(load_reference_waveforms('./reference_ecg'))
    
    #viewer = RawSignalViewer(list(cleaned_data), reference_waveforms=reference_waveforms)
    viewer = RawSignalViewer(list(cleaned_data), reference_waveforms=reference_waveforms, method='pt')
    ptt_results = viewer.show()
    
    print(f"\n\nTotal accepted: {len(ptt_results)}")
    for patient_id, ptt, _, std in ptt_results:
        print(f"Patient {patient_id}: PTT={ptt:.3f}s, Std={std:.3f}")
