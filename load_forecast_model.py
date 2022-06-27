# This Python file uses the following encoding: utf-8
"""
Created on Sun Jun  5 15:15:04 2022

@author: Vlada
"""

import os

from datetime import datetime
from holidays import Holidays
from wg_class import WindowGenerator
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

path = os.path.dirname(os.path.realpath(__file__))
files = os.listdir(path + "\\EMS")


def main():

    # Loading files to dataframes
    ems_load, ems_weather_daily, ems_weather_hourly = loading_file()

    # Rearrangement of dfs
    ems_load, ems_weather_daily, ems_weather_hourly = df_rearrangement(
        ems_load, ems_weather_daily, ems_weather_hourly)

    # Handling missing and corrupt data
    ems_load, ems_weather_daily, ems_weather_hourly = filling_missing_data(
        ems_load, ems_weather_daily, ems_weather_hourly)

    # List of dates that will be taken into consideration
    datelist = pd.date_range(ems_load.index[0], ems_load.index[-1], freq='1H')

    # Completing the df that will be used as an input to the model
    df = pd.concat([ems_load, ems_weather_hourly[["Temperature", "Wind"]]],
                   axis=1)

    # Handling periodicity
    df = periodicity(df, datelist)

    # Holidays
    df = adding_holidays(df)

    # Weekends and workdays
    df = adding_weekdays(df)

    # Spliting the data
    train_df, val_df, test_df = spliting_the_data(df)

    train_mean = train_df.mean()
    train_std = train_df.std()

    # Normalization
    train_df = Normalization(train_df)
    val_df = Normalization(val_df)
    test_df = Normalization(test_df)

    df_std = (df - train_mean) / train_std
    df_std = df_std.melt(var_name="Column", value_name="Normalized")
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x="Column", y="Normalized", data=df_std)
    _ = ax.set_xticklabels(df.keys(), rotation=90)

    num_features = df.shape[1]

    # Building a model

    OUT_STEPS = 24
    multi_window = WindowGenerator(input_width=24, label_width=OUT_STEPS,
                                   shift=OUT_STEPS,
                                   train_df=train_df,
                                   val_df=val_df,
                                   test_df=test_df)

    multi_window.plot()

    # RNN

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

    compile_and_fit(multi_lstm_model, multi_window)

    # IPython.display.clear_output()

    val_mae = multi_lstm_model.evaluate(multi_window.val)
    test_mae = multi_lstm_model.evaluate(multi_window.test, verbose=0)
    multi_window.plot(multi_lstm_model)
    plt.show()
    # Performance

    x = np.arange(1)
    width = 0.2

    plt.ylabel("MAE (average over all times and outputs)")

    plt.bar(x - 0.1, val_mae[0], width, label="Validation")
    plt.bar(x + 0.1, test_mae[0], width, label="Test")
    plt.xticks(ticks=x, labels=['Multi LSTM model'], rotation=45)

    _ = plt.legend()

    return (ems_load, ems_weather_daily, ems_weather_hourly, df)


##################################
# Functions


def loading_file():
    '''
    Function loads ems__load, ems_weather_daily, and ems_weather_hourly from
    folder
    '''

    ems_load = pd.read_csv(path + '\\EMS\\' + files[0], delimiter=';')
    ems_weather_daily = pd.read_csv(path + '\\EMS\\' + files[1], delimiter=';')
    ems_weather_hourly = pd.read_csv(path + '\\EMS\\' + files[2],
                                     delimiter=';')

    ems_load['Timestamp'] = pd.to_datetime(ems_load['Timestamp'])
    ems_weather_daily['Timestamp'] = pd.to_datetime(
        ems_weather_daily['Timestamp'])
    ems_weather_hourly['Timestamp'] = pd.to_datetime(
        ems_weather_hourly['Timestamp'])

    return (ems_load, ems_weather_daily, ems_weather_hourly)


##########################


def df_rearrangement(df1, df2, df3):
    '''
    Function rearranges ems_load, ems_weather_daily and ems_weather_hourly.
    '''
    ems_load = df1
    ems_weather_daily = df2
    ems_weather_hourly = df3
    try:
        ems_weather_hourly = ems_weather_hourly.pivot(
            "Timestamp", columns="WeatherType", values="WeatherValue")
        ems_weather_hourly = ems_weather_hourly.reset_index()

        ems_weather_daily = ems_weather_daily.pivot(
            "Timestamp", columns="WeatherType", values="WeatherValue")
        ems_weather_daily = ems_weather_daily.reset_index()

        # Numbers read as strings, should be float instead

        ems_load["Load"] = ems_load["Load"].astype(float, errors="ignore")

        ems_weather_hourly[["Cloud", "Temperature", "Wind"]] =\
            ems_weather_hourly[
            ["Cloud", "Temperature", "Wind"]].astype(float, errors="ignore")

        ems_weather_daily[ems_weather_daily.columns[1:]] = ems_weather_daily[
            ems_weather_daily.columns[1:]].astype(float, errors="ignore")

        # Removal of data up to the first recorded time in ems_load.

        ems_weather_hourly = ems_weather_hourly[
            ems_weather_hourly.Timestamp.dt.date >= datetime.date(ems_load[
                "Timestamp"].iloc[0])]
        ems_weather_daily = ems_weather_daily[
            ems_weather_daily.Timestamp.dt.date >= datetime.date(ems_load[
                "Timestamp"].iloc[0])]

        # Insertion of missing time segments

        ems_load = ems_load.set_index("Timestamp").asfreq("1H")
        ems_weather_hourly = ems_weather_hourly.set_index("Timestamp").asfreq(
            "1H")
        ems_weather_daily = ems_weather_daily.set_index("Timestamp").asfreq(
            "1D")
    except KeyError:
        print('DFs not properly formatted')

    return (ems_load, ems_weather_daily, ems_weather_hourly)


###############################


def filling_missing_data(df1, df2, df3):
    '''
    Inside of the function, a missing data is handled
    '''
    ems_load = df1
    ems_weather_daily = df2
    ems_weather_hourly = df3
    try:

        ems_load_nan = ems_load[ems_load["Load"].isna()]

        # For each missing hour, a monthly average for that hour replaces
        # a missing value.

        for i in range(0, len(ems_load_nan)):
            privremeno = ems_load.loc[
                (ems_load.index.month == ems_load_nan.index[i].month)
                & (ems_load.index.year == ems_load_nan.index[i].year)
                & (ems_load.index.hour == ems_load_nan.index[i].hour)]

        ems_load["Load"][privremeno.index[8] == ems_load.index] = round(
            float(privremeno.mean()))

        # Load zeros are replaced with the average between prior and subsequent
        # value

        ems_load["Load"] = ems_load["Load"].replace(0, np.nan)
        ems_load["Load"] = round((ems_load["Load"].ffill()
                                  + ems_load["Load"].bfill())
                                 / 2)

        # Missing data here is replaced with the average between prior and
        # subsequent value, too.

        ems_weather_hourly["Temperature"] = (
            ems_weather_hourly["Temperature"].ffill()
            + ems_weather_hourly["Temperature"].bfill()
            ) / 2

        ems_weather_daily = (ems_weather_daily.ffill()
                             + ems_weather_daily.bfill()
                             ) / 2

        # Cloudiness data are not to be taken into consideration while building
        # model, and Wind missing data will be replaced by daily average

        ems_wind_nan = ems_weather_hourly["Wind"][
            ems_weather_hourly["Wind"].isna()]
        ems_wind_nan = ems_wind_nan.fillna(ems_weather_daily["Avg Wind"])

        s = ems_wind_nan[ems_wind_nan.index.strftime("%H:%M:%S") == "00:00:00"]
        s.index = s.index.date
        ems_wind_nan[:] = s.reindex(ems_wind_nan.index.date).values

        ems_weather_hourly["Wind"][ems_wind_nan.index] = ems_wind_nan
        ems_weather_hourly["Wind"] = (
            ems_weather_hourly["Wind"].ffill() + ems_weather_hourly[
                "Wind"].bfill()) / 2

        # Min and max hourly values for temperature in ems_weather_hourly will
        # be replaced with min and max value from ems_weather_daily

        temp_min_index = ems_weather_hourly.loc[
            ems_weather_hourly.groupby(ems_weather_hourly.index.date)[
                "Temperature"].agg(["idxmin"]).stack()]
        temp_max_index = ems_weather_hourly.loc[
            ems_weather_hourly.groupby(ems_weather_hourly.index.date)[
                "Temperature"].agg(["idxmax"]).stack()]

        temp_min = ems_weather_daily["Min temperature"]
        temp_max = ems_weather_daily["Max temperature"]

        temp_min.index = temp_min_index.index
        temp_max.index = temp_max_index.index

        ems_weather_hourly["Temperature"][temp_min_index.index] = temp_min
        ems_weather_hourly["Temperature"][temp_max_index.index] = temp_max

    except KeyError:
        print('Dfs not properly formatted')

    return (ems_load, ems_weather_daily, ems_weather_hourly)


###################################################################


def periodicity(df, datelist):
    '''
    Function takes dataframe as an input and returns it after adding columns
    Day sin, Day cos, Year sin, and Year cos.
    '''
    datelist = datelist
    timestamp_s = datelist.map(pd.Timestamp.timestamp)
    day = 24 * 60 * 60
    year = (365.2425) * day

    df["Day sin"] = np.sin(timestamp_s * (2 * np.pi / day))
    df["Day cos"] = np.cos(timestamp_s * (2 * np.pi / day))
    df["Year sin"] = np.sin(timestamp_s * (2 * np.pi / year))
    df["Year cos"] = np.cos(timestamp_s * (2 * np.pi / year))

    return df


###################################################################


def adding_holidays(df):
    '''
    Function takes df and adds holidays column to it
    '''
    years = [2013, 2014, 2015, 2016, 2017, 2018]
    rs_holidays = []
    for year in years:
        rs_holidays.extend(Holidays(year))
    df["Holidays"] = 0
    df.loc[df.index.isin(rs_holidays), "Holidays"] = 1
    s = df["Holidays"][df["Holidays"].index.strftime("%H:%M:%S") == "00:00:00"]
    s.index = s.index.date
    df.loc[:, ["Holidays"]] = s.reindex(df["Holidays"].index.date).values

    return df


###################################################################


def adding_weekdays(df):
    '''
    Function takes df and adds weekdays column to it where weekends are marked
    '''
    df["Weekdays"] = df.index.to_series().dt.dayofweek
    df.loc[df["Weekdays"] <= 4, "Weekdays"] = 0
    df.loc[df["Weekdays"] > 4, "Weekdays"] = 1

    return df


###################################################################


def spliting_the_data(df):
    '''
    Function takes dataframe as an input and splits it into dfs used for
    training, validation and testing
    '''
    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]

    return (train_df, val_df, test_df)


###################################################################


def Normalization(df):
    '''
    Function takes df and returns it after normalizing its values
    '''
    df = (df-df.mean()) / df.std()

    return df


###################################################################


def compile_and_fit(model, window, patience=2):
    MAX_EPOCHS = 20
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


###################################################################

if __name__ == '__main__':
    ems_load, ems_weather_daily, ems_weather_hourly, df = main()
