# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 15:15:04 2022

@author: Vlada
"""

import os

from datetime import date, timedelta, datetime
from dateutil.easter import easter, EASTER_ORTHODOX
# import sys

# import IPython
# import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

mpl.rcParams["figure.figsize"] = (8, 6)
mpl.rcParams["axes.grid"] = False

# Ucitavanje fajlova u dataframe-ove

path = os.path.dirname(os.path.realpath(__file__))
files = os.listdir(path + "\\EMS")
ems_load = pd.read_csv(path + "\\EMS\\" + files[0])
ems_weather_hourly = pd.read_csv(path + "\\EMS\\" + files[2])
ems_weather_daily = pd.read_csv(path + "\\EMS\\" + files[1])

# Razdvajanje kolona

ems_load[["Timestamp", "Load"]] = ems_load["Timestamp;Load"].str.split(
    ";", expand=True
    )
ems_load.pop("Timestamp;Load")

ems_load["Timestamp"] = pd.to_datetime(ems_load["Timestamp"])

ems_weather_hourly[["Timestamp", "WeatherType", "WeatherValue"]] =\
    ems_weather_hourly["Timestamp;WeatherType;WeatherValue"
                       ].str.split(";", expand=True)

ems_weather_daily[["Timestamp", "WeatherType", "WeatherValue"]] =\
    ems_weather_daily["Timestamp;WeatherType;WeatherValue"
                      ].str.split(";", expand=True)

ems_weather_hourly["Timestamp"] = pd.to_datetime(
    ems_weather_hourly["Timestamp"])
ems_weather_daily["Timestamp"] = pd.to_datetime(
    ems_weather_daily["Timestamp"])


ems_weather_hourly.pop("Timestamp;WeatherType;WeatherValue")
ems_weather_daily.pop("Timestamp;WeatherType;WeatherValue")

# Preuredjivanje df

ems_weather_hourly = ems_weather_hourly.pivot(
    "Timestamp", columns="WeatherType", values="WeatherValue")
ems_weather_hourly = ems_weather_hourly.reset_index()

ems_weather_daily = ems_weather_daily.pivot(
    "Timestamp", columns="WeatherType", values="WeatherValue")
ems_weather_daily = ems_weather_daily.reset_index()

ems_load["Load"] = ems_load["Load"].astype(float, errors="ignore")
ems_weather_hourly[["Cloud", "Temperature", "Wind"]] = ems_weather_hourly[
    ["Cloud", "Temperature", "Wind"]].astype(float, errors="ignore")

ems_weather_daily[ems_weather_daily.columns[1:]] = ems_weather_daily[
    ems_weather_daily.columns[1:]].astype(float, errors="ignore")

# Uklanjanje podataka s obzirom da podaci o opterecenju postoje od 15.4.
# a zatim i popunjavanje nedostajucih datuma, za pocetak nan vrednostima

datelist = pd.date_range(
    ems_load["Timestamp"].iloc[0], ems_load["Timestamp"].iloc[-1], freq="1H")

ems_weather_hourly = ems_weather_hourly[
    ems_weather_hourly.Timestamp.dt.date >= datetime.date(ems_load[
        "Timestamp"].iloc[0])]
ems_weather_daily = ems_weather_daily[
    ems_weather_daily.Timestamp.dt.date >= datetime.date(ems_load[
        "Timestamp"].iloc[0])]

# Umetanje nedostajucih vremenskih trenutaka

ems_load = ems_load.set_index("Timestamp").asfreq("1H")
ems_weather_hourly = ems_weather_hourly.set_index("Timestamp").asfreq("1H")
ems_weather_daily = ems_weather_daily.set_index("Timestamp").asfreq("1D")

#####
ems_load.describe()

# 3 nule se pojavljuju u potrosnji, verovatno usled planskih iskljucenja mreze
# Resiti nedostajuce podatke, ima ih 24, u pitanju je citav jedan dan.
# print(ems_load["Load"].isna().sum())

ems_weather_hourly.describe()
# Podaci o oblacnosti imaju veliki broj nedostajucih podataka, cak 30317 (65%.)
# Podaci o brzini vetra imaju 14334 nedostajucih podataka.
# Mogu se popuniti koriscenjem prosecne brzine vetra.
# Temperatura ima jedan nedostajuci podatak i to ce se resiti interpolacijom.
# Maksimalne i minimalne izmerene casovne vrednosti temperature treba zameniti
# najvisim i najnizim izmerenim dnevnim vrednostima.

# print(ems_weather_hourly["Temperature"].isna().sum())
ems_weather_daily.describe()
# Nedostaju 3 podatka o dnevnim vremenskim podacima za sve kolone.
# print(ems_weather_daily[ems_weather_daily["Min temperature"].isna()])

###############################

# Sredjivanje nan vrednosti

ems_load_nan = ems_load[ems_load["Load"].isna()]

# Postavljamo za svaki sat prosecnu vrednost potrosnje u tom satu tog meseca

for i in range(0, len(ems_load_nan)):
    privremeno = ems_load.loc[
        (ems_load.index.month == ems_load_nan.index[i].month)
        & (ems_load.index.year == ems_load_nan.index[i].year)
        & (ems_load.index.hour == ems_load_nan.index[i].hour)
    ]
    ems_load["Load"][privremeno.index[8] == ems_load.index] = round(
        float(privremeno.mean())
    )

# Zamena nula koje se pojavljuju u opterecenju srednjom vrednoscu
# prethodnog i narednog opterecenja

ems_load["Load"] = ems_load["Load"].replace(0, np.nan)
ems_load["Load"] = round((ems_load["Load"].ffill() + ems_load["Load"].bfill())
                         / 2)

# U nedostajucu vrednost ubacujemo srednju vrednost
# prethodne i naredne temperaturne vrednosti

ems_temp_nan = ems_weather_hourly[ems_weather_hourly["Temperature"].isna()]
ems_weather_hourly["Temperature"] = (
    ems_weather_hourly["Temperature"].ffill()
    + ems_weather_hourly["Temperature"].bfill()
) / 2

# Posmatramo weather daily podatke

ewd_nan = ems_weather_daily[ems_weather_daily.isnull().any(axis=1)]
ems_weather_daily = (ems_weather_daily.ffill() + ems_weather_daily.bfill()) / 2
# print(ewd_nan)

# Podaci o oblacnosti nece biti uzeti u obzir prilikom kreiranja modela
# nan vrednosti brzine vetra ce biti zamenjene prosecnom brzinom vetra
# za svaki dan, a taj podatak je dat u dnevnim vremenskim podacima

ems_wind_nan = ems_weather_hourly["Wind"][ems_weather_hourly["Wind"].isna()]
ems_wind_nan = ems_wind_nan.fillna(ems_weather_daily["Avg Wind"])

s = ems_wind_nan[ems_wind_nan.index.strftime("%H:%M:%S") == "00:00:00"]
s.index = s.index.date
ems_wind_nan[:] = s.reindex(ems_wind_nan.index.date).values

ems_weather_hourly["Wind"][ems_wind_nan.index] = ems_wind_nan
ems_weather_hourly["Wind"] = (
    ems_weather_hourly["Wind"].ffill() + ems_weather_hourly["Wind"].bfill()
) / 2

# print(ems_weather_hourly[ems_weather_hourly["Wind"].isna()])

# Zamena maksimalnih i minimalnih satnih vrednosti najvisim i najnizim
# izmerenim vrednostima temperature u toku dana iz fajla ems_weather_daily

temp_min_index = ems_weather_hourly.loc[
    ems_weather_hourly.groupby(ems_weather_hourly.index.date)["Temperature"]
    .agg(["idxmin"])
    .stack()]
temp_max_index = ems_weather_hourly.loc[
    ems_weather_hourly.groupby(ems_weather_hourly.index.date)["Temperature"]
    .agg(["idxmax"])
    .stack()]

temp_min = ems_weather_daily["Min temperature"]
temp_max = ems_weather_daily["Max temperature"]

temp_min.index = temp_min_index.index
temp_max.index = temp_max_index.index

ems_weather_hourly["Temperature"][temp_min_index.index] = temp_min
ems_weather_hourly["Temperature"][temp_max_index.index] = temp_max
###############
# Kreiranje df sa svim promenljivim koji ce biti korisceni u modelu

df = pd.concat([ems_load, ems_weather_hourly[["Temperature", "Wind"]]], axis=1)

df.describe()

# sredjivanje periodicnosti

timestamp_s = datelist.map(pd.Timestamp.timestamp)
day = 24 * 60 * 60
year = (365.2425) * day

df["Day sin"] = np.sin(timestamp_s * (2 * np.pi / day))
df["Day cos"] = np.cos(timestamp_s * (2 * np.pi / day))
df["Year sin"] = np.sin(timestamp_s * (2 * np.pi / year))
df["Year cos"] = np.cos(timestamp_s * (2 * np.pi / year))

# Obelezavanje praznika


def Holidays(year):
    """

    Za izabranu godinu, funkcija vraca listu datuma praznika u Srbiji te godine

    """
    JAN = 1
    FEB = 2
    MAY = 5
    NOV = 11
    WEEKEND = [5, 6]
    SUN = 7
    praznici_datumi = []
    # New Year's Day
    praznici_datumi.append(date(year, JAN, 1))
    praznici_datumi.append(date(year, JAN, 2))
    if date(year, JAN, 1).weekday() in WEEKEND:
        praznici_datumi.append(date(year, JAN, 3))
    # Orthodox Christmas
    praznici_datumi.append(date(year, JAN, 7))
    # International Workers' Day
    praznici_datumi.append(date(year, MAY, 1))
    praznici_datumi.append(date(year, MAY, 2))
    if date(year, MAY, 1).weekday() in WEEKEND:
        if date(year, MAY, 2) == easter(year, method=EASTER_ORTHODOX):
            praznici_datumi.append(date(year, MAY, 4))
        else:
            praznici_datumi.append(date(year, MAY, 3))
    # Armistice day
    praznici_datumi.append(date(year, NOV, 11))
    if date(year, NOV, 11).weekday() == SUN:
        praznici_datumi.append(date(year, NOV, 12))
    # Easter
    praznici_datumi.append(easter(year, method=EASTER_ORTHODOX)
                           - timedelta(days=2))
    praznici_datumi.append(easter(year, method=EASTER_ORTHODOX)
                           - timedelta(days=1))
    praznici_datumi.append(easter(year, method=EASTER_ORTHODOX))
    praznici_datumi.append(easter(year, method=EASTER_ORTHODOX)
                           + timedelta(days=1))
    # Statehood day
    praznici_datumi.append(date(year, FEB, 15))
    praznici_datumi.append(date(year, FEB, 16))
    if date(year, FEB, 15).weekday() in WEEKEND:
        praznici_datumi.append(date(year, FEB, 17))
    return praznici_datumi


years = [2013, 2014, 2015, 2016, 2017, 2018]
rs_holidays = []
for year in years:
    rs_holidays.extend(Holidays(year))

df["Holidays"] = 0
df.loc[df.index.isin(rs_holidays), "Holidays"] = 1

s = df["Holidays"][df["Holidays"].index.strftime("%H:%M:%S") == "00:00:00"]
s.index = s.index.date
df.loc[:, ["Holidays"]] = s.reindex(df["Holidays"].index.date).values

# Obelezavanje vikenda i radnih dana

df["Weekdays"] = df.index.to_series().dt.dayofweek

df.loc[df["Weekdays"] <= 4, "Weekdays"] = 0
df.loc[df["Weekdays"] > 4, "Weekdays"] = 1

df.head()

# sys.exit()

################
# Podela seta podataka

column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n * 0.7)]
val_df = df[int(n * 0.7):int(n * 0.9)]
test_df = df[int(n * 0.9):]

num_features = df.shape[1]

# Normalizacija

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name="Column", value_name="Normalized")
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x="Column", y="Normalized", data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)

# Klasa za kreiranje prozora podataka


class WindowGenerator:
    def __init__(
        self,
        input_width,
        label_width,
        shift,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_columns=None
    ):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)
            }
        self.column_indices = {name: i for i, name
                               in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(
            self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(
            self.total_window_size)[self.labels_slice]

    # Podela prozora

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in
                 self.label_columns], axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    # Plot

    def plot(self, model=None, plot_col="Load", max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 10))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f"{plot_col} [normed]")
            plt.plot(
                self.input_indices,
                inputs[n, :, plot_col_index],
                label="Inputs",
                marker=".",
                zorder=-10
            )

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col,
                                                                 None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(
                self.label_indices,
                labels[n, :, label_col_index],
                edgecolors="k",
                label="Labels",
                c="#2ca02c",
                s=64
            )
            if model is not None:
                predictions = model(inputs)
                plt.scatter(
                    self.label_indices,
                    predictions[n, :, label_col_index],
                    marker="X",
                    edgecolors="k",
                    label="Predictions",
                    c="#ff7f0e",
                    s=64
                )

            if n == 0:
                plt.legend()

        plt.xlabel("Time [h]")

    # Kreiranje dataseta

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32
        )

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, "_example", None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    def __repr__(self):
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.input_indices}",
                f"Label indices: {self.label_indices}",
                f"Label column name(s): {self.label_columns}"
            ]
        )


# Primer prozora

w1 = WindowGenerator(input_width=24, label_width=1, shift=1,
                     label_columns=["Load"])

w1.train.element_spec

for example_inputs, example_labels in w1.train.take(1):
    print(f"Inputs shape (batch, time, features): {example_inputs.shape}")
    print(f"Labels shape (batch, time, features): {example_labels.shape}")

#####################

# Trening procedura

MAX_EPOCHS = 20


def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, mode="min"
    )

    model.compile(
        loss=tf.losses.MeanSquaredError(),
        optimizer=tf.optimizers.Adam(),
        metrics=[tf.metrics.MeanAbsoluteError()],
    )

    history = model.fit(
        window.train,
        epochs=MAX_EPOCHS,
        validation_data=window.val,
        callbacks=[early_stopping],
    )
    return history


# Multi-steps

OUT_STEPS = 24
multi_window = WindowGenerator(input_width=24, label_width=OUT_STEPS,
                               shift=OUT_STEPS)

multi_window.plot()
multi_window


class MultiStepLastBaseline(tf.keras.Model):
    def call(self, inputs):
        return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])


last_baseline = MultiStepLastBaseline()
last_baseline.compile(
    loss=tf.losses.MeanSquaredError(), metrics=[tf.metrics.MeanAbsoluteError()]
)

multi_val_performance = {}
multi_performance = {}

multi_val_performance["Last"] = last_baseline.evaluate(multi_window.val)
multi_performance["Last"] = last_baseline.evaluate(multi_window.test,
                                                   verbose=0)
multi_window.plot(last_baseline)


class RepeatBaseline(tf.keras.Model):
    def call(self, inputs):
        return inputs


repeat_baseline = RepeatBaseline()
repeat_baseline.compile(
    loss=tf.losses.MeanSquaredError(), metrics=[tf.metrics.MeanAbsoluteError()]
)

multi_val_performance["Repeat"] = repeat_baseline.evaluate(multi_window.val)
multi_performance["Repeat"] = repeat_baseline.evaluate(multi_window.test,
                                                       verbose=0)
multi_window.plot(repeat_baseline)

# linear

multi_linear_model = tf.keras.Sequential(
    [
        # Take the last time-step.
        # Shape [batch, time, features] => [batch, 1, features]
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        # Shape => [batch, 1, out_steps*features]
        tf.keras.layers.Dense(
            OUT_STEPS * num_features,
            kernel_initializer=tf.initializers.zeros()
        ),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_features]),
    ]
)

history = compile_and_fit(multi_linear_model, multi_window)

# IPython.display.clear_output()
multi_val_performance["Linear"] = multi_linear_model.evaluate(multi_window.val)
multi_performance["Linear"] = multi_linear_model.evaluate(multi_window.test,
                                                          verbose=0)
multi_window.plot(multi_linear_model)

# dense
multi_dense_model = tf.keras.Sequential(
    [
        # Take the last time step.
        # Shape [batch, time, features] => [batch, 1, features]
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        # Shape => [batch, 1, dense_units]
        tf.keras.layers.Dense(512, activation="relu"),
        # Shape => [batch, out_steps*features]
        tf.keras.layers.Dense(
            OUT_STEPS * num_features,
            kernel_initializer=tf.initializers.zeros()
        ),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_features]),
    ]
)

history = compile_and_fit(multi_dense_model, multi_window)

# IPython.display.clear_output()
multi_val_performance["Dense"] = multi_dense_model.evaluate(multi_window.val)
multi_performance["Dense"] = multi_dense_model.evaluate(multi_window.test,
                                                        verbose=0)
multi_window.plot(multi_dense_model)

# cnn
CONV_WIDTH = 3
multi_conv_model = tf.keras.Sequential(
    [
        # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
        tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
        # Shape => [batch, 1, conv_units]
        tf.keras.layers.Conv1D(256, activation="relu",
                               kernel_size=(CONV_WIDTH)),
        # Shape => [batch, 1,  out_steps*features]
        tf.keras.layers.Dense(
            OUT_STEPS * num_features,
            kernel_initializer=tf.initializers.zeros()
        ),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_features]),
    ]
)

history = compile_and_fit(multi_conv_model, multi_window)

# IPython.display.clear_output()

multi_val_performance["Conv"] = multi_conv_model.evaluate(multi_window.val)
multi_performance["Conv"] = multi_conv_model.evaluate(multi_window.test,
                                                      verbose=0)
multi_window.plot(multi_conv_model)

# rnn

multi_lstm_model = tf.keras.Sequential(
    [
        # Shape [batch, time, features] => [batch, lstm_units].
        # Adding more `lstm_units` just overfits more quickly.
        tf.keras.layers.LSTM(32, return_sequences=False),
        # Shape => [batch, out_steps*features].
        tf.keras.layers.Dense(
            OUT_STEPS * num_features,
            kernel_initializer=tf.initializers.zeros()
        ),
        # Shape => [batch, out_steps, features].
        tf.keras.layers.Reshape([OUT_STEPS, num_features]),
    ]
)

history = compile_and_fit(multi_lstm_model, multi_window)

# IPython.display.clear_output()

multi_val_performance["LSTM"] = multi_lstm_model.evaluate(multi_window.val)
multi_performance["LSTM"] = multi_lstm_model.evaluate(multi_window.test,
                                                      verbose=0)
multi_window.plot(multi_lstm_model)

# Autoregressive model


class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the LSTM state.
        prediction, state = self.warmup(inputs)

        # Insert the first prediction.
        predictions.append(prediction)

        # Run the rest of the prediction steps.
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state, training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions


feedback_model = FeedBack(units=32, out_steps=OUT_STEPS)

prediction, state = feedback_model.warmup(multi_window.example[0])

print(
    "Output shape (batch, time, features): ",
    feedback_model(multi_window.example[0]).shape
)

history = compile_and_fit(feedback_model, multi_window)

# IPython.display.clear_output()

multi_val_performance["AR LSTM"] = feedback_model.evaluate(multi_window.val)
multi_performance["AR LSTM"] = feedback_model.evaluate(multi_window.test,
                                                       verbose=0)
multi_window.plot(feedback_model)
plt.show()

# Performance

x = np.arange(len(multi_performance))
width = 0.3

metric_name = "mean_absolute_error"
metric_index = multi_lstm_model.metrics_names.index("mean_absolute_error")
val_mae = [v[metric_index] for v in multi_val_performance.values()]
test_mae = [v[metric_index] for v in multi_performance.values()]

plt.ylabel("MAE (average over all times and outputs)")
_ = plt.legend()
plt.bar(x - 0.17, val_mae, width, label="Validation")
plt.bar(x + 0.17, test_mae, width, label="Test")
plt.xticks(ticks=x, labels=multi_performance.keys(), rotation=45)

for name, value in multi_performance.items():
    print(f"{name:8s}: {value[1]:0.4f}")
