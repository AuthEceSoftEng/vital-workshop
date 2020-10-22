# %%
PROJECT   = 'pide23f7ea9d71'
READ_KEY  = 'd5c81802-a1f6-46a2-ad31-8be10d7172f4'

URL = "https://cenote.sidero.services/api/projects/" + PROJECT

import requests

def cenote_get(event_name, query_type, query_dict={}):
    get_url = "{0}/queries/{1}".format(URL, query_type)
    query_dict["readKey"] = READ_KEY
    query_dict["event_collection"] = event_name
    r = requests.get(get_url, query_dict)
    r = dict(r.json())
    if r["ok"]:
        return r["results"]
    else:
        raise RuntimeError("Query unsuccessful due to {0}. Message is: {1}".format(
            r["results"], r["message"]))
        
import pandas

# Custom filter on extraction to get data from a given sensor only
cus_filter = '[{"property_name":"un","operator":"eq","property_value":"ADU-700HP_V2"}]'
res = cenote_get("measurements", "extraction", {"filters": cus_filter, "latest": 50000})

# Transform query result to a pandas DataFrame
df = pandas.DataFrame(res)
df = df.sort_values(by="epoch").drop_duplicates(subset="epoch").reset_index(drop=True)

# Convert keys
df = df.convert_dtypes()
keys = ['solar', 'precipitation', 'windspeed', 'winddirection', 'temperature', 'humidity', 
        'cntv2', 'vwc1', 'temp1', 'ec1', 'vwc2', 'temp2', 'ec2', 'analogv1', 'cntv1',
        'analogv3', 'cntv3', 'cntvm']

for key in keys:
    df[key] = pandas.to_numeric(df[key], errors="coerce")

df.epoch = pandas.to_datetime(df.epoch, unit='ms')

import numpy as np

df.loc[df.temperature < -100, ('temperature')] = np.nan
for key in ["windspeed", "winddirection", "humidity"]:
    df.loc[df[key] < 0, (key)] = np.nan

# %%
df

# %%
df.columns

# %%
"""
# Weather forecast
"""

# %%
"""
## Build a meteo DataFrame
"""

# %%
"""
### Select only the relevant features (columns)
"""

# %%
meteo_df = df[['epoch', 'solar', 'precipitation', 'windspeed', 'winddirection', 'humidity', 'temperature']]
meteo_df

# %%
"""
### Resample to 3 hours
"""

# %%
meteo_df = meteo_df.set_index("epoch").resample("3H").mean()
meteo_df

# %%
"""
## Build a dataset --- predict given the current settings the temperature in 3 hours!
"""

# %%
dataset_df = meteo_df.copy(deep=True)

# %%
"""
### Extra feature: hour in 24 hour format
"""

# %%
dataset_df.loc[:, ("hour")] = [obj.hour for obj in meteo_df.index.time]
dataset_df

# %%
"""
### Add a column with the future temperature (the target variable)
"""

# %%
future_temp = dataset_df.temperature.tolist()
# Delete the first element to offset all by one ...
future_temp = future_temp[1:]
# Delete the last row of dataset_df as we have no target for it ...
dataset_df.drop(dataset_df.tail(1).index,inplace=True)
# And now add the target!
dataset_df.loc[:, ('future_temperature')] = future_temp
dataset_df

# %%
dataset_df = dataset_df.dropna()
dataset_df

# %%
import matplotlib.pyplot as plt

f = plt.figure()
plt.matshow(dataset_df.corr(), fignum=f.number)
plt.xticks(range(dataset_df.shape[1]), dataset_df.columns, fontsize=10, rotation=30, horizontalalignment='left')
plt.yticks(range(dataset_df.shape[1]), dataset_df.columns, fontsize=10)
cb = plt.colorbar()
plt.show()

# %%
temp_diff = dataset_df.temperature - dataset_df.future_temperature
temp_diff.describe()

# %%
temp_diff.hist()
plt.show()

# %%
"""
## Use scikit-learn to build a model


[sklearn](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning) is an open source machine learning library that supports supervised and unsupervised learning. It also provides various tools for model fitting, data preprocessing, model selection and evaluation, and many other utilities.
"""

# %%
"""
### Create numpy arrays 
"""

# %%
X = dataset_df.loc[:, dataset_df.columns != 'future_temperature'].to_numpy()
y = dataset_df.loc[:, dataset_df.columns == 'future_temperature'].to_numpy().flatten()
y

# %%
"""
### Split the data to training and testing
"""

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=2020)

# %%
print("Training size: {0}\n Testing size: {1}".format(
    len(y_train), len(y_test)))

# %%
"""
### Build a model
"""

# %%
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train, y_train)

y_hat = reg.predict(X_test)

# %%
"""
### Validate --- Model assessment
"""

# %%
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt

print("RMSE: {0:.2f}, R2: {1:.2f}".format( 
      sqrt(mean_squared_error(y_true=y_test, y_pred=y_hat)),
      r2_score(y_true=y_test, y_pred=y_hat)
     ))

# %%
import matplotlib.pyplot as plt

plt.plot(y_test, y_hat, 'o')

axis_min = min(min(y_test), min(y_hat))
axis_max = max(max(y_test), max(y_hat))

plt.scatter(y_test, y_hat, linestyle="None", marker="o")
plt.xlim(axis_min, axis_max)
plt.ylim(axis_min, axis_max)
plt.xlabel("Observed")
plt.ylabel("Predicted")
plt.plot([axis_min, axis_max], [axis_min, axis_max], ls="--", c=".3")
m, b = np.polyfit(y_test, y_hat, 1)
step = (axis_max - axis_min ) / 100
x_lsq = np.arange(axis_min, axis_max+step, step)
plt.plot(x_lsq, [i * m + b for i in x_lsq], '-')
plt.show()
