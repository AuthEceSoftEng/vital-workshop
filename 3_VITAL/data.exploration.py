# %%
"""
# An introduction in processing and visualization of IoT data
"""

# %%
"""
## 1. Getting the data via HTTP GET request
"""

# %%
"""
### 1.1 Definition of keys and API access point

cenote Documentation:

http://assist.ee.auth.gr/docs/
"""

# %%
PROJECT   = 'pide23f7ea9d71'
READ_KEY  = 'c55787bd-f262-413a-abb5-dfd07896ef75'

URL = "https://cenote.sidero.services/api/projects/" + PROJECT

# %%
"""
### 1.2 Define and test the functions
"""

# %%
"""
We begin by defining an abstract function that performs a GET request.

To account for the different type of queries, this function is query type agnostic.
It thus passes the type of query and the query arguments to the GET request.
Should the query by unsuccessful, a RuntimeError is raised.
"""

# %%
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

# %%
"""
We test the efficacy of the function on known data and known unknowns (:
"""

# %%
cenote_get("measurements", "extraction", {"latest": 1})

# %%
"""
**Cenote also supports some higher level quering ...**
"""

# %%
cenote_get("measurements", "minimum", {"target_property": "temperature"})

# %%
"""
### 1.3 Get and store the data in a dataframe
"""

# %%
"""
pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool.

It provides:

* a fast and efficient **DataFrame** object for data manipulation with integrated indexing;
* tools for reading and writing data between in-memory data structures and different formats: CSV and text files, Microsoft Excel, SQL databases, and the fast HDF5 format;
* flexible reshaping and pivoting of data sets;
* intelligent label-based slicing, fancy indexing, and subsetting of large data sets


A quick [guide may be found here](https://pandas.pydata.org/docs/user_guide/index.html#user-guide), with the complete [API reference being provided here](https://pandas.pydata.org/docs/reference/index.html)
"""

# %%
import pandas

# Custom filter on extraction to get data from a given sensor only
cus_filter = '[{"property_name":"un","operator":"eq","property_value":"ADU-700HP_V2"}]'
res = cenote_get("measurements", "extraction", {"filters": cus_filter, "latest": 50000})

# Transform query result to a pandas DataFrame
df = pandas.DataFrame(res)
df = df.sort_values(by="epoch").drop_duplicates(subset="epoch").reset_index(drop=True)
df

# %%
"""
## 2. Initial data exploration
"""

# %%
"""
Now is a good idea to gauge what kind of data are stored in the DataFrame.

To this end, we explore the columns and the types of data stored.
"""

# %%
df.columns

# %%
df.dtypes

# %%
"""
**Some of these data types may be auto-magically corrected ...**
"""

# %%
df = df.convert_dtypes()
df.dtypes

# %%
"""
**While for others, we need to manually fix them ...**
"""

# %%
keys = ['solar', 'precipitation', 'windspeed', 'winddirection', 'temperature', 'humidity', 
        'cntv2', 'vwc1', 'temp1', 'ec1', 'vwc2', 'temp2', 'ec2', 'analogv1', 'cntv1',
        'analogv3', 'cntv3', 'cntvm']

for key in keys:
    df[key] = pandas.to_numeric(df[key], errors="coerce")

df.epoch = pandas.to_datetime(df.epoch, unit='ms')

df.dtypes

# %%
"""
**We can also very easily see some statistical moments that describe our data**
"""

# %%
df.describe()

# %%
"""
Some outliers may then be easily corrected.

Here, we use the .loc function which provides quick indexing of the data, in order to select the subsets fulfilling a given condition.
"""

# %%
import numpy as np

df.loc[df.temperature < -100, ('temperature')] = np.nan
for key in ["windspeed", "winddirection", "temperature", "humidity"]:
  df.loc[df[key] < 0, (key)] = np.nan

# %%
"""
## 3. Visualization of data
"""

# %%
"""
An important aspect of data processing is visualization.

Most of the work here utilizes the [matplotlib](https://matplotlib.org/) package beneath the hoods. We will use it on a higher level through pandas and through another library called [seaborn](https://seaborn.pydata.org/).
"""

# %%
"""
### 3.1 Line plot and some initializations
"""

# %%
import matplotlib.pyplot as plt

df.plot(x="epoch", y="temperature") # kind="line"
plt.show()

# %%
"""
### 3.2 Boxplots
"""

# %%
# Groups the data per week
df['per'] = df.set_index("epoch").index.to_period('W')
df.per


df.boxplot(column="temperature", by="per", rot=90)
plt.show()

# %%
df['per'] = df.set_index("epoch").index.to_period('M')
df.boxplot(column="temperature", by="per")
plt.show()

# %%
"""
### 3.3 Barplots
"""

# %%
"""
It is quite easy to aggregate data per week ...
"""

# %%
df.set_index("epoch").precipitation.resample("W").sum().plot(kind="bar")
plt.tight_layout()
plt.show()

# %%
"""
### 3.3 Using seaborn
"""

# %%
import seaborn

# %%
"""
The DataFrame object is based on the tools used in the R programming language.

There, lots of functions expect the data to be in the "long format". The seaborn library follows this approach, too. To this end we are going to manipulate our data through pandas before feeding it into seaborn. 
"""

# %%
sdf = pandas.wide_to_long(
    df=df,
    stubnames=["vwc", "temp", "ec"],
    i="epoch",
    j="sensor"
)
sdf

# %%
sdf = sdf.reset_index().melt(
    id_vars=["epoch", "sensor"],
    value_vars=["vwc", "temp", "ec"]
)
sdf

# %%
sdf['per'] = sdf.set_index("epoch").index.to_period('M')

# %%
g = seaborn.catplot(
    x="per", y="value",
    hue="sensor", col="variable",
    data=sdf, kind="box",
    height=2, aspect=1.75, col_wrap=2, sharey=False)
g.set_xticklabels(rotation=30)
plt.tight_layout()
plt.show()

