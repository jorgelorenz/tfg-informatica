import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from arch import arch_model
import os


plt.rcParams.update({
    'axes.titlesize': 16,       
    'axes.labelsize': 14,       
    'xtick.labelsize': 12,      
    'ytick.labelsize': 12,      
    'legend.fontsize': 12,      
    'figure.titlesize': 18
})

seq_length=8

folder_datasets = 'datasets/'
output_folder = './output/'
files_2018_2021 = ["^GSPC_2018_2021.csv", "^IBEX_2018_2021.csv"]
files_2021_2024 = ["^GSPC_2021_2024.csv", "^IBEX_2021_2024.csv"]

def load_data_for_ml(file, test_sample=0.8):
    path = os.path.join(folder_datasets, file)
    df = pd.read_csv(path, index_col=0)
    df.dropna(inplace=True)

    returns = df[['Daily log return']].values
    variance = returns ** 2
    vol = np.sqrt(variance)

    fechas = pd.to_datetime(df.index)

    split_idx = int(len(vol) * test_sample)
    vol_train_raw, vol_test_raw = vol[:split_idx], vol[split_idx:]
    fechas_train, fechas_test = fechas[:split_idx], fechas[split_idx:]

    scaler = MinMaxScaler()
    scaler.fit(vol_train_raw)

    vol_train = scaler.transform(vol_train_raw)
    vol_test = scaler.transform(vol_test_raw)

    X_train, y_train = [], []
    for i in range(len(vol_train) - seq_length):
        X_train.append(vol_train[i:i + seq_length])
        y_train.append(vol_train[i + seq_length])

    X_test, y_test = [], []
    for i in range(len(vol_test) - seq_length):
        X_test.append(vol_test[i:i + seq_length])
        y_test.append(vol_test[i + seq_length])

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), scaler,fechas_test[seq_length:]

def build_lstm(units=50, activation='tanh', dropout=0.0, recurrent_dropout=0.0, learning_rate=0.001, **kwargs):
    model = Sequential()
    model.add(LSTM(units=units, activation=activation, dropout=dropout, recurrent_dropout=recurrent_dropout, input_shape=(seq_length, 1)))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def create_train_test_model_k_fold(X_train, y_train, param_grid, name, scaler, X_test, y_test, fechas_test):
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = KerasRegressor(model=build_lstm, verbose=0)

    tcsv = TimeSeriesSplit(n_splits=3)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=tcsv)
    grid_result = grid.fit(X_train, y_train)

    y_test = scaler.inverse_transform(y_test)
    y_pred = scaler.inverse_transform(grid_result.best_estimator_.predict(X_test))

    mse = mean_squared_error(y_test, y_pred)

    #--- Graficar y guardar imagen ---
    plt.figure(figsize=(10, 5))
    plt.plot(fechas_test, y_test, label="Volatilidad Real")
    plt.plot(fechas_test, y_pred, label="Predicción LSTM")
    plt.title("Volatilidad real vs predicha")
    plt.xlabel("Fecha")
    plt.ylabel("Volatilidad")
    plt.legend()
    plt.grid()
    plot_path = output_folder +f"{name}_vol_pred.png"
    plt.savefig(plot_path)
    plt.close()

    #--- Guardar resultados en Excel ---
    resultados = pd.DataFrame({
        'Fecha': fechas_test,
        'y_real': y_test.flatten(),
        'y_predicho': y_pred.flatten()
    })

    excel_path = output_folder + f"{name}_resultados.xlsx"
    resultados.to_excel(excel_path, index=False)

    resultados_grid = pd.DataFrame(grid_result.cv_results_)
    excel_grid_path = output_folder + f"{name}_gridsearch_results.xlsx"
    resultados_grid.to_excel(excel_grid_path, index=False)

    #--- Guardar resumen con el mejor modelo y métricas en un Excel aparte ---
    resumen = pd.DataFrame([{
        "archivo": name,
        "mejores_parametros": str(grid_result.best_params_),
        "MSE_test": mse
    }])

    resumen_path = output_folder + f"{name}_resumen.xlsx"
    resumen.to_excel(resumen_path, index=False)

def rolling_garch_forecast(df, name, test_sample=0.8):
    df = df.dropna()
    returns = df['Daily log return'].values
    dates = pd.to_datetime(df.index)

    split_idx = int(len(returns) * test_sample)
    returns_train = returns[:split_idx]
    returns_test = returns[split_idx:]
    dates_test = dates[split_idx:]

    garch_preds = []
    egarch_preds = []

    full_returns = list(returns_train)

    for t in range(len(returns_test)):
        window = np.array(full_returns)

        # GARCH
        try:
            garch_model = arch_model(window, vol='GARCH', p=1, q=1)
            garch_fit = garch_model.fit(disp='off')
            garch_forecast = garch_fit.forecast(horizon=1)
            garch_var = garch_forecast.variance.values[-1, 0]
        except Exception as e:
            print(f"[GARCH ERROR] t={t}: {e}")
            garch_var = np.nan

        # EGARCH
        try:
            egarch_model = arch_model(window, vol='EGARCH', p=1, q=1)
            egarch_fit = egarch_model.fit(disp='off')
            egarch_forecast = egarch_fit.forecast(horizon=1)
            egarch_var = egarch_forecast.variance.values[-1, 0]
        except Exception as e:
            print(f"[EGARCH ERROR] t={t}: {e}")
            egarch_var = np.nan

        garch_preds.append(garch_var)
        egarch_preds.append(egarch_var)

        full_returns.append(returns_test[t])

    var_real = returns_test ** 2

    garch_preds = np.sqrt(np.array(garch_preds)[seq_length:])
    egarch_preds = np.sqrt(np.array(egarch_preds)[seq_length:])
    vol_real = np.sqrt(var_real[seq_length:])
    dates_test = dates_test[seq_length:]

    mse_garch = np.mean((garch_preds - vol_real) ** 2)
    mse_egarch = np.mean((egarch_preds - vol_real) ** 2)

    # Graficar
    plt.figure(figsize=(10, 5))
    plt.plot(dates_test, vol_real, label="Volatilidad real")
    plt.plot(dates_test, garch_preds, label="GARCH rolling")
    plt.plot(dates_test, egarch_preds, label="EGARCH rolling")
    plt.title(f"Rolling Forecast — {name}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plot_path = f"./output/{name}_rolling_garch.png"
    plt.savefig(plot_path)
    plt.close()

    # Guardar Excel
    df_out = pd.DataFrame({
        'Fecha': dates_test,
        'Volatilidad real': vol_real,
        'GARCH_rolling': garch_preds,
        'EGARCH_rolling': egarch_preds
    })
    df_out.to_excel(f"./output/{name}_rolling_garch.xlsx", index=False)

    resumen = pd.DataFrame([{
        'archivo': name,
        'MSE_GARCH_rolling': mse_garch,
        'MSE_EGARCH_rolling': mse_egarch
    }])
    resumen.to_excel(f"./output/{name}_rolling_garch_resumen.xlsx", index=False)

    return resumen

if __name__ == "__main__":
    param_grid = {
    'model__units': [32, 50, 100],
    'model__activation': ['tanh'],
    'model__recurrent_dropout': [0.0, 0.2],
    'model__learning_rate': [0.001],
    'fit__epochs': [5, 10],
    }

    for file in os.listdir(folder_datasets):
        if file in files_2018_2021 or file in files_2021_2024:
            print(f"Procesando archivo: {file}")
            X, y, x_test, y_test, scaler, fechas = load_data_for_ml(file)
            create_train_test_model_k_fold(X, y, param_grid, file.split('.')[0],  scaler, x_test, y_test, fechas)

    for file in os.listdir(folder_datasets):
        if file in files_2018_2021 or file in files_2021_2024:
            df = pd.read_csv(os.path.join(folder_datasets, file), index_col=0)
            rolling_garch_forecast(df, name=file.split(".")[0])
