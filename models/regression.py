import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snb
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.regression.linear_model import OLS
from statsmodels.graphics.regressionplots import abline_plot

sealevel_data = pd.read_csv("../sealevel_data.csv")
co2_data = pd.read_csv("../c02_data.csv")
co2_data = co2_data.drop(np.where(co2_data["ppm"]<0)[0]).reset_index()

matching_rows = np.zeros((1019, 2))
diffs = np.zeros(1019)


for i,x in sealevel_data.iterrows():
    diff = abs(co2_data["year_decimal"] - x["year"])
    index = np.where(diff == np.min(diff))[0][0]
    matching_rows[i] = co2_data.loc[index][["year_decimal","ppm"]]

    diffs[i] = np.min(diff)
    

matching_rows_df = pd.DataFrame(matching_rows, columns = ["year_decimal","ppm"])
sealevel_data = sealevel_data.join(matching_rows_df)

# plt.hist(diffs, bins=40)
# plt.show()

# plt.scatter(sealevel_data["GMSLNA"], sealevel_data["ppm"])
# plt.show()

import statsmodels.formula.api as smf
x = sealevel_data["year"]
mod = OLS(sealevel_data["GMSLNA"], sm.add_constant(x))
# print(sealevel_data["year"][0])
res = mod.fit()
p = res.params
xp = np.arange(2020, 2119)
print(x)
print(pd.DataFrame(xp))
new_y = res.predict(sm.add_constant(pd.DataFrame(xp)))

plt.plot(xp, new_y)
plt.plot(x, p.const+p.year*x)
plt.plot(x, sealevel_data["GMSLNA"])
plt.show()


# snb.regplot(sealevel_data["GMSLNA"], sealevel_data["ppm"])
# plt.show()

# new_GMSLNA = res.predict()
# coefs = np.polyfit(sealevel_data["GMSLNA"], sealevel_data["year"], 1)
# print(xp)
# new_GMSLNA = np.polyval(xp, coefs)
# print(new_GMSLNA)
# plt.plot(new_GMSLNA)
# # plt.plot(sealevel_data["year"], sealevel_data["GMSLNA"])
# plt.show()
