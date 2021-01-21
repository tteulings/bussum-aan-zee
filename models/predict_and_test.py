import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def forecast(model_name):
    model_forcasts={"AR":ar_forecast, "ARM": arima_forecast}

    forecast_func = model_forcasts[model_name]

    data = pd.read_csv("../monthly_sealevel_data.csv")
    sealevel_data = data[["year", "GMSLNA"]]

    forecast_func(sealevel_data, 2020)


def ar_forecast(data, end_year, start_year=2020):
    model = ar_model(data)
    xp = np.arange(start_year, end_year, 1/37)

    y_forecast = model.predict(start = 1019, end = 1500)

    plt.plot(y_forecast)
    plt.plot(data[["GMSLNA"]])
    plt.show()
    
def arima_forecast(data, end_year, start_year=2020):
    model = arima_model(data["GMSLNA"])
    xp = np.arange(start_year, end_year, 1/37)

    y_forecast = model.predict(start = 341, end = 500)

    plt.plot(y_forecast)
    plt.plot(data[["GMSLNA"]])
    plt.show()


### AR4 MODEL ###
def ar_predict(data, test):
    res = ar_model(data)

    oos_predictions = res.predict(start=test.index.values[0], end=test.index.values[-1])

    return res, oos_predictions, res.bic


def ar_model(data):
    from statsmodels.tsa.ar_model import AutoReg, ar_select_order

    sel = ar_select_order(data["GMSLNA"], 20, old_names=False, seasonal=True, period=12)
    res = sel.model.fit()

    return res


### ARIMA RANDOM WALK ###
def random_walk(data, test):
    from statsmodels.tsa.arima.model import ARIMA

    import pmdarima as pm
 

    mod = ARIMA(data, seasonal_order=(0, 1, 0, 12))
    res = mod.fit()

    oos_predictions = res.predict(start=test.index.values[0], end=test.index.values[-1])

    return res, oos_predictions, res.bic

def predict_arima(data, test):
    
    res = arima_model(data)
    oos_predictions = res.predict(start=test.index.values[0], end=test.index.values[-1])
    oos_predictions = pd.DataFrame(oos_predictions, index=test.index.values)

    return res, oos_predictions, res.bic

def arima_model(data):
    import pmdarima as pm
    
    # model = pm.auto_arima(data, d=1, D=1,
    #                   m=12, trend='c', seasonal=True, 
    #                   start_p=0, start_q=0, max_order=10, test='adf',
    #                   stepwise=False, trace=True)
    from statsmodels.tsa.arima.model import ARIMA


    mod = ARIMA(data, order=(1,1,2),seasonal_order=(1, 1, 2, 12)).fit()

    return mod


def regression(data, test):
    import statsmodels.api as sm
    from statsmodels.regression.linear_model import OLS

    x = data.index.values
    mod = OLS(data, sm.add_constant(x))
    res = mod.fit()

    xp = np.arange(test.index.values[0], test.index.values[-1]+1)
    oos_predictions = pd.DataFrame(res.predict(sm.add_constant(xp)), index=xp)

    return res, oos_predictions, res.bic

def print_and_plot(test, train, oos_predictions, BIC):
    oos_MSE = mean_squared_error(oos_predictions, test)

    print(f"Out of Sample MSE: {oos_MSE}")
    print(f"Model BIC: {BIC}")

    ax = train.plot(label="Train GMSLNA")
    ax.plot(test, label="Test GMSLNA")
    ax.plot(oos_predictions, label="Out of Sample Predictions")

    plt.legend()
    plt.show()

model_functions = {
    "AR": ar_predict,
    "RW": random_walk,
    "REG": regression,
    "ARM": predict_arima
}

def predict_and_test(model_name):

    print(f"Predicting and testing {model_name}")

    data = pd.read_csv("../monthly_sealevel_data.csv")
    sealevel_data = data[["GMSLNA"]]
    train, test = train_test_split(sealevel_data, shuffle=False, train_size=0.8)

    model_function = model_functions[model_name]
    model, oos_predictions, bic = model_function(train, test)

    print_and_plot(test, train, oos_predictions, bic)


def predict_and_test_all():
    data = pd.read_csv("../monthly_sealevel_data.csv")
    sealevel_data = data[["year", "GMSLNA"]]
    train_w_year, test_w_year = train_test_split(sealevel_data, shuffle=False, train_size=0.9)
    train, test = train_w_year[["GMSLNA"]], test_w_year[["GMSLNA"]]

    
    fig, ax = plt.subplots(4)
    
    for i, (model_name, model) in enumerate(model_functions.items()):
        print(f"Predicting and testing {model_name}")
        model, oos_predictions, bic = model(train, test)
                
        ax[i].plot(sealevel_data["GMSLNA"], ".-")
        ax[i].plot(oos_predictions)
        ax[i].set_title(f"{model_name}, MSE: {mean_squared_error(oos_predictions, test)}, BIC: {bic}")

    plt.tight_layout()

    plt.show()

    


# predict_and_test("REG")
# # predict_and_test("RW")
# predict_and_test("AR")
# predict_and_test("RW")
# predict_and_test("ARM")

predict_and_test_all()

# data = pd.read_csv("../monthly_sealevel_data.csv")
# sealevel_data = data[["GMSLNA"]]

# arima_model(sealevel_data)
# forecast("ARM")

