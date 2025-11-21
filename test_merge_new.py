from log.merge_new import FileMerger
import matplotlib.pyplot as plt
import global_vars

global_vars.mirror_version = "2"
merge1 = FileMerger(path="./mirror4_data/patient_000004", log=True)
df1 = merge1()

fig, ax = plt.subplots(3, 1, figsize=(10, 8))
ax[0].plot(df1["Timestamp"], df1["RPPG"], label="RPPG Signal")
ax[1].plot(df1["Timestamp"], df1["ECG"], label="ECG Signal")
ax[2].plot(df1["Timestamp"], df1["PPG_IR"], label="PPG IR Signal")
ax[0].set_xlabel("Timestamp")
ax[0].set_ylabel("RPPG Signal")
ax[1].set_xlabel("Timestamp")
ax[1].set_ylabel("ECG Signal")
ax[2].set_xlabel("Timestamp")
ax[2].set_ylabel("PPG IR Signal")
fig.suptitle("RPPG, ECG, and PPG IR Signals Over Time")
ax[0].legend()
ax[1].legend()
ax[2].legend()
plt.show()