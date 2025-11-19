from data.patient_info import PatientInfo
from data.load_data import DataLoader
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import csv

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
    reference_waveforms = {}
    data_loader = DataLoader(cleaned_dir=ref_dir)
    for patient_id, timestamps, rppg_signal, ecg_signal in data_loader.load_cleaned_data():
        reference_waveforms[patient_id] = (timestamps, rppg_signal, ecg_signal)
    return reference_waveforms

def filter_signal(data, fs=512, lowcut=0.5, highcut=5.0, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    data = signal.filtfilt(b, a, data)
    return data

def ecg_peak_pantompkins(ecg_signal, fs=512):
    filtered_ecg = filter_signal(ecg_signal, fs=fs, lowcut=5.0, highcut=15.0, order=2)
    derivative_ecg = np.gradient(filtered_ecg)
    squared_ecg = np.square(derivative_ecg)
    
    window_size = int(0.150 * fs)
    pantompkins = np.convolve(squared_ecg, np.ones(window_size)/window_size, mode='same')
    pantompkins = 2 * pantompkins / np.max(pantompkins)
    
    signal_std = np.std(pantompkins)
    pt_peaks, _ = signal.find_peaks(
        pantompkins,
        distance=int(fs * 0.35),
        prominence=0.1 * signal_std,
    )

    peaks = []
    for peak in pt_peaks:
        left = max(0, peak - int(0.1 * fs))
        right = min(len(ecg_signal), peak + int(0.1 * fs))
        p = np.argmax(ecg_signal[left:right]) + left
        peaks.append(p)
    
    print(f"Detected {len(peaks)}/{len(pt_peaks)} ECG peaks using Pan-Tompkins algorithm.")
    return peaks, pantompkins, pt_peaks

def find_peaks_new(signal_data, signal_type='rppg', fs=512, min_distance=None):
    if min_distance is None:
        min_distance = int(fs * 0.35)
    
    if signal_type == 'ecg':
        peaks, pantompkins, pt_peaks = ecg_peak_pantompkins(signal_data, fs=fs)
        return peaks, pantompkins, pt_peaks
    else:
        peaks, properties = signal.find_peaks(signal_data, distance=min_distance, height=0)
        return peaks, properties

def calculate_ptt_new(time, rppg_signal, ecg_signal):
    rppg_peaks, _ = find_peaks_new(rppg_signal, signal_type='rppg', fs=512)
    ecg_peaks, _, _ = find_peaks_new(ecg_signal, signal_type='ecg', fs=512)
    
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

def ptt_signals(timestamps, rppg_signal, ecg_signal, peak=True, filter=True, fs=512):
    if filter:
        rppg_signal = filter_signal(rppg_signal, fs=fs)
        ecg_signal = filter_signal(ecg_signal, fs=fs)
    if peak:
        ptt, align, std = calculate_ptt_new(timestamps, rppg_signal, ecg_signal)
        if ptt is not None:
            print(f"PTT: {ptt:.3f} seconds, Std: {std:.3f}")
        return ptt, align, std
    return None, None, None

class RawSignalViewer:
    def __init__(self, data_list, reference_waveform_list=None):
        self.data_list = data_list if data_list is not None else []
        self.current_raw_idx = 0
        self.current_clipped_idx = 0
        self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 12))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.ptt_results = []
        self.clipped_segments = []
        self.fs = 512
        
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
            
            ptt, _, std = calculate_ptt_new(timestamps, rppg_filtered, ecg_filtered)
            self.current_ptt = ptt
            self.current_std = std
            
            if ptt is not None:
                print(f"PTT: {ptt:.3f} seconds, Std: {std:.3f}")
            else:
                print("PTT: Unable to calculate")
            
            rppg_peaks, _ = find_peaks_new(rppg_filtered, signal_type='rppg', fs=512)
            ecg_peaks, pantompkins, pt_peaks = find_peaks_new(ecg_filtered, signal_type='ecg', fs=512)

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
            new_timestamps = np.linspace(0, 1, target_length)
            resampled_ecg = interp1d(seg_timestamps, seg_ecg, kind='cubic', fill_value='extrapolate')(new_timestamps)
            csv_data = zip(new_timestamps, resampled_ecg)
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['timestamp', 'ecg_signal'])
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
            self.axes[1].plot(timestamps, pantompkins, label='Pan-Tompkins', alpha=0.6)
            self.axes[1].plot(timestamps[ecg_peaks], ecg_filtered[ecg_peaks], "o", label='ECG Peaks')
            self.axes[1].plot(timestamps[pt_peaks], pantompkins[pt_peaks], "s", label='PT Peaks', markersize=4)
            
            ptt_text = f"PTT: {ptt:.3f}s, Std: {std:.3f}" if ptt is not None else "PTT: N/A"
            self.axes[1].set_title(f'ECG Signal - {ptt_text}')
            self.axes[1].set_xlabel('Time (s)')
            self.axes[1].set_ylabel('Amplitude')
            self.axes[1].legend()
            self.axes[1].grid(True, alpha=0.3)

        self.axes[2].clear()
        self.axes[2].set_title('Clipped Segment Viewer')
        if self.current_clipped_idx < len(self.clipped_segments):
            seg_patient_id, seg_timestamps, seg_rppg, seg_ecg = self.clipped_segments[self.current_clipped_idx]

            self.axes[2].plot(seg_timestamps, seg_rppg, label='Clipped rPPG Signal')
            self.axes[2].plot(seg_timestamps, seg_ecg, label='Clipped ECG Signal')
            self.axes[2].set_xlabel('Time (s)')
            self.axes[2].set_ylabel('Amplitude')
            self.axes[2].legend()
            self.axes[2].grid(True, alpha=0.3)
            self.axes[2].set_title(f'Clipped Segment {self.current_clipped_idx + 1}/{len(self.clipped_segments)} for Patient {seg_patient_id}')
        
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
    
    viewer = RawSignalViewer(list(cleaned_data))
    ptt_results = viewer.show()
    
    print(f"\n\nTotal accepted: {len(ptt_results)}")
    for patient_id, ptt, _, std in ptt_results:
        print(f"Patient {patient_id}: PTT={ptt:.3f}s, Std={std:.3f}")
