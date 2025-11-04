from data.patient_info import PatientInfo

data_dir = "./patient_data"
#data_dir = "./lab_mirror_data"
output_file = "overall_patient_info.csv"
#output_file = "lab_overall_patient_info.csv"
patient_info = PatientInfo(data_dir, save_dir=output_file, mode="dir")
patient_info_list = patient_info.extract()
patient_info.save()
