import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

idx = pd.IndexSlice

DETREND_WINDOW = "100T"
DLL_WINDOW = "100T"
REFERENCE_POINT = 1
DATA_FILE = "AQ1_EXP102.hdf5"
DATES_SLICE = slice("2021-03-28", "2021-04-06")

data = h5py.File(DATA_FILE)
currents = pd.DataFrame(
        data=np.hstack([x[:].T for x in data["Beam_currents"].values()]),
        index=pd.to_datetime(data["Index"][:], unit='s'),
        columns=pd.MultiIndex.from_product([[f"v{i}" for i in "123"], data["Beam_cells"][:]])
        )
# currents = currents.loc[DATES_SLICE]
cells = currents.columns.levels[1].values
reference_point = cells[np.argmin(np.abs(cells - REFERENCE_POINT))]
currents = currents.resample("T").mean()
currents = currents - currents.rolling(DETREND_WINDOW).mean()

dll = currents.values.reshape((currents.shape[0], -1, 3)) - currents.loc[:, idx[:, reference_point]].values[:, None, :]
dll = np.mean(dll, axis=2)
dll = pd.DataFrame(dll, index=currents.index, columns=(cells-reference_point))
dll = dll.rolling(DLL_WINDOW).mean()
print(dll)
