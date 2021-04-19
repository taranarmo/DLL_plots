import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

idx = pd.IndexSlice

DATA_FILE = "AQ1_EXP102.hdf5"
data = h5py.File(DATA_FILE)
currents = pd.DataFrame(
        data=np.hstack([x[:].T for x in data["Beam_currents"].values()]),
        index=pd.to_datetime(data["Index"][:], unit='s'),
        columns=pd.MultiIndex.from_product([[f"v{i}" for i in "123"], data["Beam_cells"][:]])
        )
currents = currents - currents.rolling("100T").mean()

