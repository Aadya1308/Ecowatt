"""Load trained seq2seq models and show prediction results."""

import os
import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

INPUT_SEQUENCE_LENGTH = 72
TARGET_SEQUENCE_LENGTH = 72
BATCH_SIZE = 80
NUM_STEPS_TO_PREDICT = 24
NUM_PLOTS = 5


def preprocess_df(df, input_sequence_length, target_sequence_length, shuffle=False):
    seq_len = input_sequence_length + target_sequence_length
    sequential_data = []
    prev_days = deque(maxlen=seq_len)
    for row in df:
        prev_days.append(row)
        if len(prev_days) == seq_len:
            sequential_data.append(np.array(prev_days))

    if shuffle:
        random.shuffle(sequential_data)

    x = np.array(sequential_data)
    encoder_input = x[:, :input_sequence_length, :]
    decoder_output = x[:, input_sequence_length:, :]
    decoder_input = np.zeros((decoder_output.shape[0], decoder_output.shape[1], 1))
    return encoder_input, decoder_input, decoder_output


def load_and_prepare_data():
    csv_path = os.path.join(SCRIPT_DIR, "kaggle_data_1h.csv")
    df = pd.read_csv(csv_path, sep=",", low_memory=False, index_col="time", encoding="utf-8")
    df = df.drop(
        [
            "Voltage",
            "Global_intensity",
            "Global_reactive_power",
            "Sub_metering_1",
            "Sub_metering_2",
            "Sub_metering_3",
        ],
        axis=1,
    )
    df = df.round(5)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    df = df.ewm(alpha=0.15).mean()
    df.dropna(inplace=True)

    factor = 3
    upper_lim = df["Global_active_power"].mean() + df["Global_active_power"].std() * factor
    lower_lim = df["Global_active_power"].mean() - df["Global_active_power"].std() * factor
    df = df[(df["Global_active_power"] < upper_lim) & (df["Global_active_power"] > lower_lim)]

    times = pd.to_datetime(df.index).strftime("%Y-%m-%d").tolist()
    temp = []
    for index, value in enumerate(times):
        if value > "2007-08-08" and value < "2007-09-01":
            temp.append(float(df["Global_active_power"].iloc[index]))
    temp = pd.Series(temp).ewm(alpha=0.15).mean().values

    k = 0
    for index, value in enumerate(times):
        if value > "2008-08-08" and value < "2008-09-01":
            df.iloc[index, df.columns.get_loc("Global_active_power")] = temp[k]
            k += 1

    train_len = int(len(df) * 0.8)
    test_df = df[train_len:]
    train_df = df[:train_len]
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_values = scaler.fit_transform(np.float64(train_df.values))
    test_values = scaler.transform(np.float64(test_df.values))
    return scaler, train_values, test_values


def predict(x, encoder_model, decoder_model, num_steps):
    states = encoder_model.predict(x, verbose=0)
    if not isinstance(states, list):
        states = [states]

    decoder_input = np.zeros((x.shape[0], 1, 1))
    y_predicted = []
    for _ in range(num_steps):
        outputs_and_states = decoder_model.predict([decoder_input, *states], batch_size=BATCH_SIZE, verbose=0)
        output = outputs_and_states[0]
        states = outputs_and_states[1:]
        y_predicted.append(output)

    return np.concatenate(y_predicted, axis=1)


def save_prediction_plot(x, y_true, y_pred, output_path):
    plt.figure(figsize=(15, 3))
    past = x[:, 0]
    true = y_true[:, 0]
    pred = y_pred[:, 0]

    plt.plot(range(len(past)), past, "o--b", label="Seen (past) values")
    plt.plot(range(len(past), len(true) + len(past)), true, "x--b", label="True future values")
    plt.plot(range(len(past), len(pred) + len(past)), pred, "o--y", label="Predictions")
    plt.legend(loc="best")
    plt.title("Predictions vs. true values")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def inverse_scale(values, scaler):
    shape = values.shape
    flat = values.reshape(-1)
    restored = scaler.inverse_transform(flat.reshape(-1, 1)).reshape(shape)
    return restored


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def print_metrics(label, y_true, y_pred):
    print(f"\n{label}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.5f}")
    print(f"  R^2:  {r2_score(y_true, y_pred):.4f}")
    print(f"  MAE:  {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"  MAPE: {mean_absolute_percentage_error(y_true, y_pred):.3f} %")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    encoder_path = os.path.join(SCRIPT_DIR, "encoder.h5")
    decoder_path = os.path.join(SCRIPT_DIR, "decoder.h5")
    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
        raise FileNotFoundError("Run seq2seq.py first to create encoder.h5 and decoder.h5")

    print("Loading models...")
    encoder_model = keras.models.load_model(encoder_path)
    decoder_model = keras.models.load_model(decoder_path)

    print("Preparing test data...")
    scaler, _, test_values = load_and_prepare_data()
    test_x, _, test_y = preprocess_df(
        test_values, INPUT_SEQUENCE_LENGTH, TARGET_SEQUENCE_LENGTH, shuffle=False
    )

    print(f"Running predictions on {test_x.shape[0]} test sequences...")
    test_y_predicted = predict(test_x, encoder_model, decoder_model, NUM_STEPS_TO_PREDICT)

    inv_y = inverse_scale(test_y[:, :NUM_STEPS_TO_PREDICT, :], scaler)
    inv_yhat = inverse_scale(test_y_predicted, scaler)

    print_metrics("24-hour forecast metrics", inv_y[:, :, 0], inv_yhat[:, :, 0])
    print_metrics("6-hour forecast metrics", inv_y[:, :6, 0], inv_yhat[:, :6, 0])

    indices = np.random.default_rng(42).choice(test_x.shape[0], size=NUM_PLOTS, replace=False)
    plot_paths = []
    for i, index in enumerate(indices, start=1):
        inv_x = inverse_scale(test_x[index : index + 1], scaler)[0]
        inv_true = inverse_scale(test_y[index : index + 1, :NUM_STEPS_TO_PREDICT], scaler)[0]
        inv_pred = inverse_scale(test_y_predicted[index : index + 1], scaler)[0]
        plot_path = os.path.join(RESULTS_DIR, f"prediction_{i}.png")
        save_prediction_plot(inv_x, inv_true, inv_pred, plot_path)
        plot_paths.append(plot_path)
        print(f"Saved {plot_path}")

    print(f"\nDone. Open the results folder to review plots:\n{RESULTS_DIR}")
    return plot_paths, RESULTS_DIR


if __name__ == "__main__":
    main()
