---
layout: post
title: Movie Recommender
subtitle : Using a webscraper to create a program that recommends movies or TV shows to watch
tags: [python, webscraping]
author: Asmita Majumder
comments : True
---

In Blog Post 3, we will use the question of *"What movie or TV shows share actors with your favorite movie or show?"* to generate a list of recommendations of TV shows and movies we should watch next. 

**Here is a link to the Github repository containing the program we will outline in this blog post: https://github.com/asmit-a/blog-post-3**

## 1. Setup 

First, let's locate some important URLs. For this blog post, we will use the 2005 revival of "Doctor Who" as our favorite TV show. We find its IMDB page at

```
https://www.imdb.com/title/tt0436992/
```

We also note that if we click on the *Cast & Crew* link on the IMDB page, it takes us to the original url with "fullcredits/" appended to the end. 


We then create a new GitHub repository, which we will call "blog-post-3" and use to house our scraper. Upon opening a terminal and changing its directory to the location of our repository, we must enter type the following commands: 

```
conda activate PIC16B
scrapy startproject IMDB_scraper
cd IMDB_scraper
```

This will set up our scraper, and allow us to begin writing its script. 

## 2. Write Your Scraper

We begin by setting up the basics. As always, we want to import the relevant packages: 


```python
import scrapy 
from scrapy.http import Request
```

Recall that a scraper must always be written in a class that extended `scrapy.Spider`. We set up a class named `ImdbSpider`, give it a name of `imdb_spider`, and give it a list of `start_urls` that consists of the link to the IMDB page of our favorite show. 


```python
class ImdbSpider(scrapy.Spider):
    name = 'imdb_spider'
    
    # Doctor Who (2005)'s IMDB page
    start_urls = ['https://www.imdb.com/title/tt0436992/']
```

Now it's time to get to the main task: setting up our `parse` methods. These methods will instruct the spider on what to do whenever it reaches a given page. We want to set up three such methods: 
- `parse` will assume we start on the main page of a movie, and will direct the spider to the *Cast & Cew* page. 
- `parse_full_credits` will assume we start on the *Cast & Crew* page, and will call `parse_actor_page` on each of the actors listed under the "Cast" section. 
- `parse_actor_page` will assume we start on the main page of an actor, and will create a dictionary containing their name and the works in which they have participated as key-value pairs. 

First, let us examine the `parse` method. 


```python
    def parse(self, response):
        """
        Assumes we start on a movie's main page and navigates to 
        movie's Cast & Crew page. 

        @param self: an instance of this class.  
        @param response: the result of scrapy's HTTP request. 
        """
        
        movie_url = "https://www.imdb.com/title/tt0436992/"
        credits_url = movie_url + "fullcredits/"
        yield Request(credits_url, callback = self.parse_full_credits)
```

This `parse` method must take in `self` and `response` as arguments, as is typical with parse methods. Within the method, we concatenate the url of the show's main page with "fullcredits/" in order to retrieve the url of the *Cast & Crew* page to which we want to navigate, which we store in the variable `credits_url`. We then yield a `scrapy.Request` which calls `parse_full_credits` (which we will define next) upon the page accessed by `credits_url` (i.e., the link that leads to the full *Cast & Crew* page).

Let us now define the `parse_full_credits` method, which will define the spider's behavior upon reaching the *Cast & Crew* page. 


```python
    def parse_full_credits(self, response):
        """
        Assumes we start on a movie's Cast & Crew page and navigates
        to the pages of each of the cast members listed on it. 

        @param self: an instance of this class.  
        @param response: the result of scrapy's HTTP request. 
        """
        
        # retrieves all the relative paths to the actors
        # listed on the Cast & Crew page
        actor_urls = [a.attrib["href"] for a in response.css("td.primary_photo a")]
        base_url = "https://www.imdb.com"
        
        # loops through each of the relative actor urls
        for url in actor_urls: 
            full_actor_url = base_url + url
            yield Request(full_actor_url, callback = self.parse_actor_page)
```

This `parse_full_credits` method again takes in `self` and `response` as arguments. We begin by retrieving the relative urls of the IMDB pages of each of the actors listed on the *Cast & Crew* page. We then loop through each of these urls, append them to the end of the base url "https://www.imdb.com" in order to retrieve the full link to each actor's IMDB page, and finally yield a `scrapy.Request` that calls `parse_actor_page` upon the link to the actor's main page. 

Of course, we have to define `parse_actor_page`. Let's dive into this method, which will define what we want the spider to do upon reaching an actor's IMDB page. 


```python
    def parse_actor_page(self, response):
        """
        Assumes we start on an actor's IMDB page and generates
        a dictionary containing the actor name and the works
        in which they've participated as key-value pairs. 

        @param self: an instance of this class.  
        @param response: the result of scrapy's HTTP request. 
        """
        
        # retrieves the actor's name
        name_string = response.css("td.name-overview-widget__section h1.header span::text").get()
       
        # retreieves a list of the works in which actor has participated
        filmography_rows = response.css("div.filmo-row")
        filmography = [row.css("a::text").get() for row in filmography_rows]

        # loops through the title of each work 
        # in which actor has participated
        for film in filmography: 
            yield {
                "actor": name_string, 
                "movie_or_TV_name": film
            }
```

We first use CSS selectors to retrieve the actor's name from the top of the page as a string, as well as to retrieve a list of all of the works in which the actor has participated. We must then loop through each of these works one by one, adding them to a dictionary in which actors and the works in which they have appeared are stored as key-value pairs. We `yield` this dictionary. 

Now we must run our script. We can do so using the command 

```
scrapy crawl imdb_spider -o movies.csv
```

in the terminal. This will generate a file called `movies.csv`, which is a csv file that contains all the actors from our original favorite show with each of the works in which they have participated as associated values. 

## 3. Make Your Recommendations

Let us now move to Jupyter Notebook, where we will organize this data and use it to find TV show or movie recommendations. 

As expected, we begin by importing the packages that we anticipate using in this analysis. 


```python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
```

We must copy the csv file which our `imdb_spider.py` program generated to the folder in which our Notebook is located, so that we can read it in as a `pandas` dataframe. 


```python
df = pd.read_csv("results.csv")
```

We want to count how many times each movie or TV show appears in the `movie_or_TV_name` column, which we can accomplish by grouping our dataframe by the aforementioned column and then calling `.transform(len)` on it. 


```python
df["shared_actors"] = df.groupby(["movie_or_TV_name"]).transform(len)
```

Since we are focusing only on the recurrence of movies/shows and not on the actors themselves, we can remove the `actor` column from our data frame.


```python
df = df[["movie_or_TV_name", "shared_actors"]]
```

Finally, we sort the movies/shows by the `movie_count` column in descending order and remove duplicates. We also reset the index and remove the resulting index column to create a cleaner chart. 


```python
df = df.sort_values(by = "shared_actors", ascending=False)
df = df.drop_duplicates()
df = df.reset_index()
df = df[["movie_or_TV_name", "shared_actors"]]
```


```python
df.head(20)
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
      <th>movie_or_TV_name</th>
      <th>shared_actors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Doctor Who</td>
      <td>1674</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Casualty</td>
      <td>501</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Doctors</td>
      <td>500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Bill</td>
      <td>460</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Holby City</td>
      <td>360</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Doctor Who Confidential</td>
      <td>273</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Midsomer Murders</td>
      <td>249</td>
    </tr>
    <tr>
      <th>7</th>
      <td>EastEnders</td>
      <td>245</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Silent Witness</td>
      <td>231</td>
    </tr>
    <tr>
      <th>9</th>
      <td>This Morning</td>
      <td>196</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Breakfast</td>
      <td>190</td>
    </tr>
    <tr>
      <th>11</th>
      <td>New Tricks</td>
      <td>173</td>
    </tr>
    <tr>
      <th>12</th>
      <td>The One Show</td>
      <td>172</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Coronation Street</td>
      <td>169</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Loose Women</td>
      <td>143</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Lorraine</td>
      <td>130</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Good Morning Britain</td>
      <td>119</td>
    </tr>
    <tr>
      <th>17</th>
      <td>The Graham Norton Show</td>
      <td>116</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Sunday Brunch</td>
      <td>114</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Death in Paradise</td>
      <td>112</td>
    </tr>
  </tbody>
</table>
</div>



This leaves us with a list of which movies/shows share the most common actors with our original favorite TV show, "Doctor Who". The top result (other than "Doctor Who", which is expected) is "Casualty", followed by "Doctors" and "The Bill". Strangely enough, the "Doctor Who Confidential", which is a behind-the-scenes look at the making of "Doctor Who", appears only in 5th place. 

And there we have it! A neat list full of new recommendations for us to peruse. 


```python

```
