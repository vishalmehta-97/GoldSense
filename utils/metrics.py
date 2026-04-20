# utils/metrics.py

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    directional = np.mean(
        (np.sign(y_true[1:] - y_true[:-1]) ==
         np.sign(y_pred[1:] - y_pred[:-1]))
    )

    return rmse, mae, r2, directional