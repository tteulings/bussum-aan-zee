import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.arima.model import ARIMA

from statsmodels.tsa.api import acf, pacf, graphics


data = pd.read_csv("../sealevel_data.csv")


### Plot sealevel ###
plt.plot( data["GMSLNA"])
plt.show()

x = data["GMSLNA"]

### ACF and PACF ###
graphics.plot_acf(x)
graphics.plot_pacf(x)
plt.show()


### AR  ###
sel = ar_select_order(x, 13, old_names=False, seasonal=True, period=37)
sel.ar_lags
res = sel.model.fit()
ax = res.plot_predict(1000, 1100)
ax = x.plot(fig=ax)

plt.show()



### ARIMA RANDOM WALK ###
mod1 = ARIMA(x, seasonal_order = (0,1,0, 37))
res1 = mod1.fit()

predict = res1.get_forecast(100)
predictions = predict.predicted_mean[-100:]
lower = predict.conf_int(0.95)["lower GMSLNA"]
upper = predict.conf_int(0.95)["upper GMSLNA"]

ax = plt.plot(predictions)
plt.fill_between(np.arange(x.size, x.size+predictions.size), upper, lower,  alpha=0.5)
x.plot(fig=ax)

plt.show()

