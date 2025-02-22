---
layout: post
title: Climate Visualization
subtitle : Visualizing NOAA climate data using Plotly
tags: [python, SQL, data viz]
author: Asmita Majumder
comments : False

---

In this blog post, we'll explore the NOAA climate data using several interesting and interactive data visualizations. 

We begin by importing the packages that we will use. 


```python
import sqlite3
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from plotly import express as px
```

## 1. Create a Database

Before we can create our database, we must create a function that will allow us to clean the data we wish to put into it.


```python
def prepare_df(df):
    df = df.set_index(keys=["ID", "Year"])
    df = df.stack()
    df = df.reset_index()
    df = df.rename(columns = {"level_2"  : "Month" , 0 : "Temp"})
    df["Month"] = df["Month"].str[5:].astype(int)
    df["Temp"]  = df["Temp"] / 100
    df["FIPS 10-4"] = df["ID"].str[0:2]
    return(df)
```

We can now create our database. 


```python
conn = sqlite3.connect("bp1.db")
```

Our first step will be to read the file containing the temperature data in chunks and copy the information over to the `temperatures` table in our database.


```python
df_iter = pd.read_csv("temps.csv", chunksize = 100000)
for df in df_iter:
    df = prepare_df(df)
    df.to_sql("temperatures", conn, if_exists = "append", index = False)
```  

We add the `stations` table to our database.


```python
url = "https://raw.githubusercontent.com/PhilChodrow/PIC16B/master/datasets/noaa-ghcn/station-metadata.csv"
stations = pd.read_csv(url)
stations.to_sql("stations", conn, if_exists = "replace", index = False)
```

We add the `countries` table to our database.


```python
countries_url = "https://raw.githubusercontent.com/mysociety/gaze/master/data/fips-10-4-to-iso-country-codes.csv"
countries = pd.read_csv(countries_url)
countries.to_sql("countries", conn, if_exists = "replace", index = False)
```

We have now finished populating our database. Let us check to make sure we have done so correctly.


```python
cursor = conn.cursor()

cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")

for result in cursor.fetchall():
    print(result[0])
```

    CREATE TABLE "temperatures" (
    "ID" TEXT,
      "Year" INTEGER,
      "Month" INTEGER,
      "Temp" REAL,
      "FIPS 10-4" TEXT
    )
    CREATE TABLE "stations" (
    "ID" TEXT,
      "LATITUDE" REAL,
      "LONGITUDE" REAL,
      "STNELEV" REAL,
      "NAME" TEXT
    )
    CREATE TABLE "countries" (
    "FIPS 10-4" TEXT,
      "ISO 3166" TEXT,
      "Name" TEXT
    )
    

Looks good! We finish off by closing the connection to the database. 


```python
conn.close()
```

## 2. Write a Query Function

We now want to write a query function that allows us to retrieve data from our database and returns it in the form of a dataframe. This data should be of a given month, country, and time frame.


```python
def query_climate_database(country, year_begin, year_end, month):
    conn = sqlite3.connect("bp1.db")
    cmd = \
    f"""
    SELECT S.name, S.latitude, S.longitude, C.name, T.year, T.month, T.temp 
    FROM temperatures T 
    LEFT JOIN stations S on T.id = S.id
    LEFT JOIN countries C on T.[FIPS 10-4] = C.[FIPS 10-4]
    WHERE T.month = {month} AND C.name = "{country}" AND (T.year >= {year_begin} AND T.year <= {year_end}) 
    """
    
    df = pd.read_sql_query(cmd, conn)
    conn.close()
    return df
```

Let us test the function to make sure it works: 


```python
query_climate_database(country = "India", 
                       year_begin = 1980, 
                       year_end = 2020,
                       month = 1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NAME</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>Name</th>
      <th>Year</th>
      <th>Month</th>
      <th>Temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1980</td>
      <td>1</td>
      <td>23.48</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1981</td>
      <td>1</td>
      <td>24.57</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1982</td>
      <td>1</td>
      <td>24.19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1983</td>
      <td>1</td>
      <td>23.51</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1984</td>
      <td>1</td>
      <td>24.81</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3147</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1983</td>
      <td>1</td>
      <td>5.10</td>
    </tr>
    <tr>
      <th>3148</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1986</td>
      <td>1</td>
      <td>6.90</td>
    </tr>
    <tr>
      <th>3149</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1994</td>
      <td>1</td>
      <td>8.10</td>
    </tr>
    <tr>
      <th>3150</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1995</td>
      <td>1</td>
      <td>5.60</td>
    </tr>
    <tr>
      <th>3151</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1997</td>
      <td>1</td>
      <td>5.70</td>
    </tr>
  </tbody>
</table>
<p>3152 rows × 7 columns</p>
</div>



Our output looks as we expected it to! Let's move on. 

## 3. Write a Geographic Scatter Function for Yearly Temperature Increases

Time to plot the data! Our goal in this section is to answer the question: *"How does the average yearly change in temperature vary within a given country?"*

First, we will write a function that will allow use to fit a linear regression model onto a piece of data from the set we're working with and return the coefficient of the resulting model. 


```python
from sklearn.linear_model import LinearRegression

def coef(data_group):
    x = data_group[["Year"]] # predictor data
    y = data_group["Temp"]   # target data
    LR = LinearRegression()
    LR.fit(x, y)
    return LR.coef_[0]
```

We will now define our plotting function. This function will be given a country, a month, a range of years, a minimum number of observation years, and various arguments appropriate for a `px.scatter_mapbox`. Within the function, we will use the given inputs to pull from our database and create a dataframe using the query function we defined above. We will then calculate the linear regression coefficients for each of the stations in our dataframe, and plot this on a geographic scatterplot. 


```python
def temperature_coefficient_plot(country, year_begin, year_end, month, min_obs, **kwargs):

    df = query_climate_database(country, year_begin, year_end, month)

    df["Years of Data"] = df.groupby(["NAME"])["Temp"].transform(len)
    df = df[df["Years of Data"] >= min_obs]
    
    coef_df = df.groupby(["NAME"]).apply(coef)
    coef_df = coef_df.reset_index()
    coef_df.rename(columns={0:"Estimated Yearly Increase (°C)"}, inplace=True )
    
    
    df = pd.merge(df, coef_df, on = ["NAME"])
    df["Temp"].round(decimals = 4)
    
    fig = px.scatter_mapbox(df, 
                            lat = "LATITUDE", 
                            lon = "LONGITUDE", 
                            hover_name = "NAME", 
                            color = "Estimated Yearly Increase (°C)",
                            color_continuous_midpoint = 0,
                            title = f"Estimates of yearly increase in temperature in Month {month} for stations in {country}, years {year_begin} - {year_end}",
                            **kwargs)
    
    fig.update_layout(margin={"r":0, "t":50, "l":0, "b":0})
    
    return fig
    
```

We test our function on Januaries in India from 1980 to 2020. 


```python
# assumes you have imported necessary packages
color_map = px.colors.diverging.RdGy_r # choose a colormap

fig = temperature_coefficient_plot("India", 1980, 2020, 1, 
                                   min_obs = 10,
                                   zoom = 2,
                                   mapbox_style="carto-positron",
                                   color_continuous_scale=color_map)

fig.show()
```

{% include geo_scatter.html %}


## 4. Create Two More Interesting Figures

Let us start by posing some questions and considering how we might answer them. 

Our first question: *How can we observe the effects of global warming in a given country?*

One way to approach this is to take a given country and a given range of years, and calculate how the z-score for each station in that country is changing over time. 

Let us start by defining a new query function, which is similar to our previous one but will no longer depend on a given month, as we want to track data for all months over all years in our range. 


```python
def query_climate_database_2(country, year_begin, year_end):
    conn = sqlite3.connect("bp1.db")

    cmd = \
    f"""
    SELECT S.name, S.latitude, S.longitude, C.name, T.year, T.month, T.temp 
    FROM temperatures T 
    LEFT JOIN stations S on T.id = S.id
    LEFT JOIN countries C on T.[FIPS 10-4] = C.[FIPS 10-4]
    WHERE C.name = "{country}" AND (T.year >= {year_begin} AND T.year <= {year_end}) 
    """
    
    df = pd.read_sql_query(cmd, conn)
    conn.close()
    return df
```

We also know that we will need to calculate a z-score, so we define a function for calculating that.


```python
def z_score(x): 
    m = np.mean(x)
    s = np.std(x)
    return (x - m)/s
```

Now we can plot our data. We want to extract data from the database using our new query function, and then calculate the z-score for each station and month. In other words, we will calculate how far the temperature at a given station and month deviates from the average temperature of that station and month. We will then make a line plot that compares the date to the z-score of a station; each line will represent one station in the country.


```python
def z_score_plot(country, year_begin, year_end, **kwargs):

    df = query_climate_database_2(country, year_begin, year_end)
    
    df["z"] = df.groupby(["NAME", "Month"])["Temp"].transform(z_score)
    df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"].astype(str))

    fig = px.line(df, 
                  x = "Date", 
                  y = "z", 
                  hover_name = "NAME", 
                  color = "NAME",
                  title = f"Z-score vs. time in {country}, from {year_begin} - {year_end}",
                  **kwargs)
    

    
    fig.update_layout(margin={"r":0, "t":50, "l":0, "b":0})
    
    
    return fig
```

We test our function on the coutnry of Suriname, which is the smallest independent country in South America. It is important to note that this plotting function would likely not work well with a large country that has many stations. 


```python
fig = z_score_plot("Suriname", 1980, 2010)
fig.show()
```

{% include z_score_line.html %}


It appears that all but one station ceased data collection by approximately 1990. From the plot, we can see that, since roughly 2000, there has been a trend of increasing z-scores. This means that as time has gone on, the temperature has deviated more and more from the overall average, which can be taken as a sign of global warming. 

We now move on to a different question: *What is the relationship between longitude (i.e. distance from the equator) and temperature?* 

Again, we define a query function that will allow us to extract relevant data from our database. In this case, we will compare the temperatures of 3 countries at various longitudes at two given months in a given year. By default, these two months will be June and December. 


```python
def query_climate_database_3(year, country_1, country_2, country_3, sum_month = 6, win_month = 12):
    conn = sqlite3.connect("bp1.db")

    cmd = \
    f"""
    SELECT S.name, S.latitude, S.longitude, C.name, T.year, T.month, T.temp 
    FROM temperatures T 
    LEFT JOIN stations S on T.id = S.id
    LEFT JOIN countries C on T.[FIPS 10-4] = C.[FIPS 10-4]
    WHERE (T.year = {year}) AND (T.month = {sum_month} OR T.month = {win_month}) AND (C.name = "{country_1}" OR C.name = "{country_2}" OR C.name = "{country_3}")
    """
    
    df = pd.read_sql_query(cmd, conn)
    conn.close()
    return df
```

We test our query function to make sure it works as desired:


```python
query_climate_database_3(2010, "India", "China", "Mexico")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NAME</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>Name</th>
      <th>Year</th>
      <th>Month</th>
      <th>Temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MOHE</td>
      <td>52.1330</td>
      <td>122.5170</td>
      <td>China</td>
      <td>2010</td>
      <td>6</td>
      <td>18.73</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MOHE</td>
      <td>52.1330</td>
      <td>122.5170</td>
      <td>China</td>
      <td>2010</td>
      <td>12</td>
      <td>-28.72</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HUMA</td>
      <td>51.7170</td>
      <td>126.6500</td>
      <td>China</td>
      <td>2010</td>
      <td>6</td>
      <td>21.71</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HUMA</td>
      <td>51.7170</td>
      <td>126.6500</td>
      <td>China</td>
      <td>2010</td>
      <td>12</td>
      <td>-23.55</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TULIHE</td>
      <td>50.4500</td>
      <td>121.7000</td>
      <td>China</td>
      <td>2010</td>
      <td>6</td>
      <td>17.50</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1045</th>
      <td>GENERAL_JUAN_N_ALVA</td>
      <td>16.7500</td>
      <td>-99.7500</td>
      <td>Mexico</td>
      <td>2010</td>
      <td>12</td>
      <td>24.85</td>
    </tr>
    <tr>
      <th>1046</th>
      <td>MAZATLAN</td>
      <td>23.1881</td>
      <td>-105.0000</td>
      <td>Mexico</td>
      <td>2010</td>
      <td>6</td>
      <td>27.06</td>
    </tr>
    <tr>
      <th>1047</th>
      <td>MAZATLAN</td>
      <td>23.1881</td>
      <td>-105.0000</td>
      <td>Mexico</td>
      <td>2010</td>
      <td>12</td>
      <td>19.90</td>
    </tr>
    <tr>
      <th>1048</th>
      <td>TORREONCOAH</td>
      <td>21.5167</td>
      <td>-103.4333</td>
      <td>Mexico</td>
      <td>2010</td>
      <td>6</td>
      <td>30.50</td>
    </tr>
    <tr>
      <th>1049</th>
      <td>TORREONCOAH</td>
      <td>21.5167</td>
      <td>-103.4333</td>
      <td>Mexico</td>
      <td>2010</td>
      <td>12</td>
      <td>16.35</td>
    </tr>
  </tbody>
</table>
<p>1050 rows × 7 columns</p>
</div>



Now we're ready to plot! This time, we will create a scatter plot that shows the relationship between a station's distance from the equator and its temperature in a given year and two given months. Since we are dealing with two months--one in summer, one in winter--we will created a faceted plot in which each plot represents one of these two months. We will also assign different colors based on the country in which the station is located in. 


```python
def longitude_vs_temperature_plot(year, country_1, country_2, country_3, sum_month = 6, win_month = 12, **kwargs):
    df = query_climate_database_3(year, country_1, country_2, country_3, sum_month, win_month)

    df["Distance From Equator"] = np.abs(df["LONGITUDE"])
     
    fig = px.scatter(df, 
                     x = "Distance From Equator", 
                     y = "Temp", 
                     hover_name = "NAME", 
                     color = "Name", 
                     title = f"Longitude vs. Temperature in various countries, in months {sum_month} and {win_month} of year {year}",
                     facet_col = "Month", 
                     **kwargs)
    
    fig.update_layout(margin={"r":0, "t":50, "l":0, "b":0})
    
    return fig
```

As always, we test our function to make sure it works properly. This time, we will test it on India, China, and Mexico in 2010. We will leave the default months of June and December.


```python
fig = longitude_vs_temperature_plot(2010, "India", "China", "Mexico", 
                                    width = 600, 
                                    height = 300, 
                                    opacity = 0.5)

fig.show()
```

{% include longitude_scatter.html %}


Surprisingly, distance from the equator does not seem to have much of an impact on temperatures in the summer. In winter, however, we see that the stations that are farther from the equator tend to be colder, as expected. 
