import pandas as pd

# data = pd.read_csv("datasealvl.txt", header=None, sep='\s+')
# data.columns = ["altimeter", "cycle", "year", "observations", "weighted_observations", "GMSLNA", "std_GMSLNA", "smoothed_GMSLNA", "GMSLA", "std_GMSLA", "smoothed_GMSLA", "smoothed_GMSLA_(semi)-an_rm"]
# data.to_csv("sealevel_data.csv", index=False)


data = pd.read_csv("co2_data.txt", header=None, sep='\s+')
data.columns = ["year", "month", "day", "year_decimal", "ppm", "#days", "1_yr_ago", "10_yr_ago", "increase_since_1800"]
data.to_csv("c02_data.csv", index=False)

