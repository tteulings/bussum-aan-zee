import pandas as pd

data = pd.read_csv("datasealvl.txt", header=None, sep='\s+')
data.columns = ["altimeter", "cycle", "year", "observations", "weighted_observations", "GMSLNA", "std_GMSLNA", "smoothed_GMSLNA", "GMSLA", "std_GMSLA", "smoothed_GMSLA", "smoothed_GMSLA_(semi)-an_rm"]
data.to_csv("sealevel_data.csv", index=False)

