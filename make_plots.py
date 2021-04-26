import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import datetime

idx = pd.IndexSlice

DETREND_WINDOW = "100T"
DLL_WINDOW = "100T"
REFERENCE_POINT = 1
DATA_FILE = "AQ1_EXP102.hdf5"
DATES_SLICE = slice("2020-03-28", "2020-04-05")
TIME_SLICE = slice(datetime.time(7), datetime.time(19))

data = h5py.File(DATA_FILE)
currents = pd.DataFrame(
        data=np.hstack([x[:].T for x in data["Beam_currents"].values()]),
        index=pd.to_datetime(data["Index"][:], unit='s'),
        columns=pd.MultiIndex.from_product([[f"v{i}" for i in "123"], data["Beam_cells"][:]])
        )
currents = currents.loc[DATES_SLICE]
cells = currents.columns.levels[1].values
reference_point = cells[np.argmin(np.abs(cells - REFERENCE_POINT))]
currents = currents.resample("T").mean()
currents = currents - currents.rolling(DETREND_WINDOW).mean()

dll = np.subtract(
        np.transpose(currents.values.reshape((currents.shape[0], 3, -1)), axes=[1, 0, 2]),
        np.transpose(currents.loc[:, idx[:, reference_point]].values[:, :, None], axes=[1, 0, 2])
)
dll = dll**2
dll = np.mean(dll, axis=0)
dll = pd.DataFrame(dll, index=currents.index, columns=(cells-reference_point))
dll = dll.rolling(DLL_WINDOW).mean()

plotting_data = dll.loc[TIME_SLICE].resample('D').mean().T.to_dict(orient='series')
fig, axes = plt.subplots(1, 3, figsize=(10, 5))
for key, value in plotting_data.items():
    if key == pd.Timestamp('2020-04-03') or key == pd.Timestamp('2020-03-30'):
        continue
    axes[0].plot(value, label=key.strftime("%b %d"))
    axes[1].plot(value.index**(2/3), value, label=key.strftime("%b %d"))
    axes[2].loglog(value, label=key.strftime("%b %d"))
axes[0].legend()
fig.savefig("DLL.png")
plt.show()
