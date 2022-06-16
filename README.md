**Electric Power Load Forecasting**

Table of Contents

- Introduction;
- Technology;
- Input Data;
- Observations Regarding Input Data;
- The model;

**Introduction**

This project deals with the forecast of electricity consumption 24 hours in
advance for a single consumer, using data on previous consumption and weather
data for the same location. It is created as a part of the evaluation process.
The code is separated into three files. Apart from the main script, two additional
files are created to hold the Window Generator class and the Holidays function.

**Technology**

The code is fully written in Python. Packages versions that were used are:
Pandas 1.3.5, TensorFlow 2.9.1, Keras 2.9.0, Numpy 1.21.2.

**Input Data**

The input consists of three different files: "EMS_load.csv",
"EMS_Weather_Daily.csv", and "EMS_Weather_Hourly.csv". The data is collected in the period from 2013 to 2018.  

**Observations Regarding Input Data**

Load data are available from April 15, 2013, while weather data are available
from January 1, 2013, so weather data from January 1 to April 15 need to be
ignored when building a dataset.

In addition, there are time moments for which individual variables do not have a defined value, so this data was filled in. There are also three outliers in consumption, three zero consumption at the end of March 2014, 2015, and 2016, all at 2:00 in the morning. These are probably planned grid outages by the competent electricity distributor, but, although these are potentially real-world data, these 3 zero data have been interpolated to make better use of the model.

Load data also contained one full day of missing data, so these were filled with the mean value for each hour individually during that month.

Due to a large number of missing data (over 65% of total data, or more than 30,000), as well as questionable correlations with electricity consumption,
Cloudiness data are completely excluded from the model. In addition, variables that could be used to predict cloudiness are missing (eg solar irradiation data for a given location).

Wind data were taken into account, since, although they have a large number of missing data, there is data on the average wind speed for each day, and this data was inserted where there was no value in the file "EMS_Weather_Hourly.csv".

In addition to these variables, holidays and weekends are also marked, ie non-working days.

**The Model**

After preprocessing, the created dataset is passed to models for training.
Using Sequential constructor, several different models were built and trained: Linear, Dense, Convolution Neural Network, and Recurrent Neural Network. Also, the Autoregressive Model was built on the same dataset.

The performance of each model is shown on the diagram Performance.png.
Feature_importance.png shows the importance that the linear model, built for predicting the load one hour into the future, assigns to each feature. By far the most weight is placed on the previous consumption itself. Temperature and time data also carried some weight, while wind speed, holidays, and weekends did not have a significant impact on the model.
Images Multi-LSTM-model and Autoregressive Model show the output of those models.
