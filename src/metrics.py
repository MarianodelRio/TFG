import numpy as np
from matplotlib import pyplot as plt
from preprocessing import denormalize_data

def error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted

def percentage_error(actual: np.ndarray, predicted: np.ndarray):
    """ Percentage error """
    return error(actual, predicted) / actual

def mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    return np.mean(np.square(error(actual, predicted)))

def mae(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Absolute Error """
    return np.mean(np.abs(error(actual, predicted)))

def mape(actual: np.ndarray, predicted: np.ndarray):
    """Mean Absolute Percentage Error"""
    return 100*np.mean(np.abs(percentage_error(actual, predicted)))

METRICS = {
    "mse": mse,
    "mae": mae,
    "mape": mape,
}

def evaluate(
    x_test_denorm: np.ndarray,    
    actual: np.ndarray,
    predicted: np.ndarray,
    metrics
):
    results = {}
    cont = 0
    for name in metrics:
        try:
            res = []

            for i, (y, o) in enumerate(zip(actual, predicted)):
                res.append(METRICS[name](y, o))

                if (cont <0):  
                    precio = x_test_denorm[i, :, -1]
                    oo = np.concatenate((precio, o))
                    yy = np.concatenate((precio, y))
                    plt.plot(yy, label="real")
                    plt.plot(oo, label="predicted")
                    plt.title("{}: {}".format(name, METRICS[name](y, o)))
                    plt.legend()
                    plt.show()
                    cont += 1

            results[name] = np.mean(res)

        except Exception as err:
            results[name] = np.nan
            print("Unable to compute metric {0}: {1}".format(name, err))

    return results