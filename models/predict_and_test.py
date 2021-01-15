import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


### AR4 MODEL ###
def ar_model(data, test):
    from statsmodels.tsa.ar_model import AutoReg, ar_select_order

    sel = ar_select_order(data["GMSLNA"], 20, old_names=False, seasonal=True, period=37)
    res = sel.model.fit()

    oos_predictions = res.predict(start=test.index.values[0], end=test.index.values[-1])
    is_predictions = res.predict()

    return res, oos_predictions, is_predictions


### ARIMA RANDOM WALK ###
def random_walk(data, test):
    from statsmodels.tsa.arima.model import ARIMA

    mod = ARIMA(data, seasonal_order=(0, 1, 0, 37))
    res = mod.fit()

    oos_predictions = res.predict(start=test.index.values[0], end=test.index.values[-1])
    is_predictions = res.predict()

    return res, oos_predictions, is_predictions

def regression(data, test):
    import statsmodels.api as sm
    from statsmodels.regression.linear_model import OLS

    x = data.index.values
    mod = OLS(data, sm.add_constant(x))
    res = mod.fit()

    xp = np.arange(test.index.values[0], test.index.values[-1]+1)


    oos_predictions = res.predict(sm.add_constant(xp))
    print(oos_predictions)
    is_predictions = res.predict()

    return res, oos_predictions, is_predictions

def print_and_plot(test, train, oos_predictions, is_predictions, BIC):
    oos_MSE = mean_squared_error(oos_predictions, test)

    print(f"Out of Sample MSE: {oos_MSE}")
    print(f"Model BIC: {BIC}")

    ax = train.plot(label="Train GMSLNA")
    ax.plot(test, label="Test GMSLNA")
    ax.plot(oos_predictions, label="Out of Sample Predictions")
    ax.plot(is_predictions, label="Insample Predictions")

    plt.legend()
    plt.show()


def predict_and_test(model_name, amount_of_predictions=100):
    model_functions = {
        "AR": ar_model,
        "RW": random_walk,
        "REG": regression
    }

    print(f"Predicting and testing {model_name}")

    data = pd.read_csv("../sealevel_data.csv")
    sealevel_data = data[["GMSLNA"]]
    train, test = train_test_split(sealevel_data, shuffle=False, train_size=0.85)

    model_function = model_functions[model_name]
    model, oos_predictions, is_predictions = model_function(train, test)

    print_and_plot(test, train, oos_predictions, is_predictions, model.bic)


predict_and_test("RW")
predict_and_test("AR")
