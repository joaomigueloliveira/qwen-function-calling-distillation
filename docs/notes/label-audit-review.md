# Label audit: blind review sheet

Relabel each row against [the rubric](../label-audit.md) and write the verdict (`label_sound`, `label_ambiguous` or `label_wrong`) into `data/audit/review.csv`. Then run `uv run fc-audit build`, which reports the disagreements.

## 4334

**Query:** Search for COVID-19 vaccine updates in the UK in English, with 30 results per page.

**Tools:**

- `search`: {"pageno": {"description": "The page number of the search results.", "type": "int", "default": "1"}, "country": {"description": "The country code for the search localization.", "type": "str", "default": "US"}, "lang": {"description": "The language code for the search localization.", "type": "str", "default": "en"}, "search": {"description": "The search string to query.", "type": "str", "default": "Hello"}, "perpage": {"description": "The number of results per page. Defaults to 10.", "type": "int, optional", "default": "10"}}

**Gold:** `[{"name": "search", "arguments": {"pageno": 1, "country": "UK", "lang": "en", "search": "COVID-19 vaccine updates", "perpage": 30}}]`

**Wrong predictions:**

~~~
[{"name": "search", "arguments": {"country": "GB", "lang": "en", "search": "COVID-19 vaccine updates", "perpage": 30}}]
~~~
~~~
```json
[
  {
    "name": "search",
    "arguments": {
      "search": "COVID-19 vaccine updates",
      "country": "GB",
      "lang": "en",
      "perpage": 30,
      "pageno": 1
    }
  }
]
```
~~~
~~~
[
  {
    "name": "search",
    "arguments": {
      "country": "GB",
      "lang": "en",
      "search": "COVID-19 vaccine updates",
      "perpage": 30
    }
  }
]
~~~

## 7396

**Query:** Calculate the total alimony a spouse earning $4000 per month would have to pay to a spouse earning $2000 per month over a 5-year period in California.

**Tools:**

- `california_alimony`: {"payor_monthly_income": {"description": "The monthly gross income of the payor spouse.", "type": "int"}, "recipient_monthly_income": {"description": "The monthly gross income of the recipient spouse.", "type": "int"}, "duration_years": {"description": "The duration of the alimony in years.", "type": "int"}}

**Gold:** `[{"name": "california_alimony", "arguments": {"payor_monthly_income": 4000, "recipient_monthly_income": 2000, "duration_years": 5}}]`

**Wrong predictions:**

~~~
[{"name": "california_alimony", "arguments": {"payor_monthly_income": 4000, "recipient_monthly_income": 2000, "duration_years": 5}}
~~~

## 9238

**Query:** Retrieve box office information for the movie titled 'Inception'.

**Tools:**

- `title_v2_get_business`: {"tconst": {"description": "The identifier for the movie title, starting with 'tt'. This value can be obtained from the `/title/auto-complete` or `/title/find` endpoints.", "type": "str", "default": "tt0796366"}}

**Gold:** `[{"name": "title_v2_get_business", "arguments": {"tconst": "tt1375666"}}]`

**Wrong predictions:**

~~~
[
  {
    "name": "title_v2_get_business",
    "arguments": {
      "tconst": "tt0796366"
    }
  }
]
~~~

## 9685

**Query:** Compute the standard deviation for the dataset [7.1, 14.2, 21.3, 28.4, 35.5] and determine the number of permutations when selecting 5 items from a set of 11.

**Tools:**

- `std_deviation`: {"numbers": {"description": "The list of numbers.", "type": "List[float]"}}
- `permutation_count`: {"n": {"description": "The total number of elements in the set.", "type": "int"}, "k": {"description": "The number of elements to choose for the permutation.", "type": "int"}}

**Gold:** `[{"name": "std_deviation", "arguments": {"numbers": [7.1, 14.2, 21.3, 28.4, 35.5]}}, {"name": "permutation_count", "arguments": {"n": 11, "k": 5}}]`

**Wrong predictions:**

~~~
[{"numbers": [7.1, 14.2, 21.3, 28.4, 35.5]}, {"n": 11, "k": 5}]
~~~

## 9793

**Query:** Retrieve the latest earnings information for Google's parent company Alphabet Inc. and the top 3 news articles from CoinDesk.

**Tools:**

- `earnings`: {"symb": {"description": "The stock symbol representing the company whose earnings data is to be fetched.", "type": "str", "default": "tsla"}}
- `top_news`: {"num": {"description": "The number of news articles to fetch. By default, it is set to 5.", "type": "int, optional", "default": "5"}}

**Gold:** `[{"name": "earnings", "arguments": {"symb": "GOOGL"}}, {"name": "top_news", "arguments": {"num": 3}}]`

**Wrong predictions:**

~~~
[
    {"name": "earnings", "arguments": {"symb": "GOOGL"}},
    {"name": "top_news", "arguments": {"num": "3"}}
]
~~~

## 14014

**Query:** Count the occurrences of each element in the list [7, 'apple', 'apple', 'banana', 'banana', 'banana']. Also, check if '(({[]}))' has valid parentheses.

**Tools:**

- `count_occurrences`: {"lst": {"description": "The input list.", "type": "List"}}
- `is_valid_parentheses`: {"s": {"description": "The input string.", "type": "str"}}

**Gold:** `[{"name": "count_occurrences", "arguments": {"lst": [7, "apple", "apple", "banana", "banana", "banana"]}}, {"name": "is_valid_parentheses", "arguments": {"s": "(({[]}))"}}]`

**Wrong predictions:**

~~~
[{"name": "count_occurrences", "arguments": {"lst": [7, "apple", "apple", "banana", "banana", "banana"]}, "name": "is_valid_parentheses", "arguments": {"s": "(({[]}))"}}]
~~~
~~~
[{"name": "count_occurrences", "arguments": {"lst": [7, 'apple', 'apple', 'banana', 'banana', 'banana']}}, {"name": "is_valid_parentheses", "arguments": {"s": "(({[]}))"}}]
~~~

## 14047

**Query:** Can you find info on book ID 12345 and suggest properties near 'Dubai Marina'? Also, what's the price of book ID 67890?

**Tools:**

- `extract_info`: {"is_id": {"description": "The ID of the book to retrieve information for.", "type": "str", "default": ""}}
- `auto_complete`: {"query": {"description": "The term or phrase to search for location suggestions.", "type": "str", "default": "abu dhabi"}, "hitsperpage": {"description": "The number of results to return per page. Default is 25.", "type": "int, optional", "default": "25"}, "lang": {"description": "The language of the results. Options are 'en' or 'ar'. Default is 'en'.", "type": "str, optional", "default": "en"}, "page": {"description": "The number of results to return per page. Default is 25.", "type": "int, optional", "default": "0"}}

**Gold:** `[{"name": "extract_info", "arguments": {"is_id": "12345"}}, {"name": "auto_complete", "arguments": {"query": "Dubai Marina"}}, {"name": "extract_info", "arguments": {"is_id": "67890"}}]`

**Wrong predictions:**

~~~
[{"name": "extract_info", "arguments": {"is_id": "12345"}}, {"name": "auto_complete", "arguments": {"query": "Dubai Marina", "hitsperpage": 1, "lang": "en", "page": 0}}, {"name": "auto_complete", "arguments": {"query": "Dubai Marina", "hitsperpage": 1, "lang": "en", "page": 0}}]
~~~
~~~
[{"name": "extract_info", "arguments": {"is_id": "12345"}}, {"name": "extract_info", "arguments": {"is_id": "67890"}}, {"name": "auto_complete", "arguments": {"query": "Dubai Marina", "hitsperpage": 25, "lang": "en", "page": 0}}]
~~~

## 14503

**Query:** Please reverse the sentence 'Data is beautiful' and count the frequency of each color in ['red', 'blue', 'red', 'green', 'blue', 'blue'].

**Tools:**

- `reverse_string`: {"text": {"description": "The input string.", "type": "str"}}
- `count_occurrences`: {"lst": {"description": "The input list.", "type": "List"}}

**Gold:** `[{"name": "reverse_string", "arguments": {"text": "Data is beautiful"}}, {"name": "count_occurrences", "arguments": {"lst": ["red", "blue", "red", "green", "blue", "blue"]}}]`

**Wrong predictions:**

~~~
[{"name": "reverse_string", "arguments": {"text": "Data is beautiful"}}, {"name": "count_occurrences", "arguments": {"lst": ["red", "blue", "red", "green", "blue", "blue"]}}}
~~~
~~~
[{"name": "reverse_string", "arguments": {"text": "Data is beautiful"}]}, {"name": "count_occurrences", "arguments": {"lst": ["red", "blue", "red", "green", "blue", "blue"]}}]
~~~

## 22044

**Query:** I need to get the download links and other info for a video with ID 'M7SOh0fXNiA' but I'm unsure about the geo parameter. Can you provide the info without specifying the geo?

**Tools:**

- `get_video_info`: {"is_id": {"description": "The YouTube video ID for which to fetch information.", "type": "str", "default": "UxxajLWwzqY"}, "geo": {"description": "The country code in ISO 3166 format of the end user. Defaults to None.", "type": "str, optional", "default": ""}}

**Gold:** `[{"name": "get_video_info", "arguments": {"is_id": "M7SOh0fXNiA"}}]`

**Wrong predictions:**

~~~
```json
[
    {
        "name": "get_video_info",
        "arguments": {
            "is_id": "M7SOh0fXNiA",
            "geo": ""
        }
    }
]
```
~~~
~~~
{
  "name": "get_video_info",
  "arguments": {
    "is_id": "M7SOh0fXNiA",
    "geo": null
  }
}
~~~

## 25106

**Query:** How many times does the word 'apple' appear in the sentence: 'I love apples, especially apple pie.'

**Tools:**

- `word_frequency`: {"text": {"description": "The input text.", "type": "str"}}

**Gold:** `[{"name": "word_frequency", "arguments": {"text": "I love apples, especially apple pie."}}]`

**Wrong predictions:**

~~~
[]
~~~

## 25855

**Query:** Fetch salad recipes with ingredients: kale and bell peppers.

**Tools:**

- `salad_ingredient`: {"ingredient": {"description": "The ingredient to be included in the salad recipe search.", "type": "str", "default": "cucumbers"}}

**Gold:** `[{"name": "salad_ingredient", "arguments": {"ingredient": "kale"}}, {"name": "salad_ingredient", "arguments": {"ingredient": "bell peppers"}}]`

**Wrong predictions:**

~~~
[{"name": "salad_ingredient", "arguments": {"ingredient": "kale"}}]
~~~
~~~
[{"name": "salad_ingredient", "arguments": {"ingredient": "kale"}}]
~~~

## 28567

**Query:** Can you tell me the top songs in Bolivia on January 1, 2023, and also the latest music releases in the UK?

**Tools:**

- `bolivia_songs`: {"range": {"description": "The range of the chart to fetch.", "type": "str", "default": "1-10"}, "date": {"description": "The date for which the chart information is required, in the format YYYY-MM-DD.", "type": "str", "default": "2022-05-07"}}
- `new_releases`: {"country": {"description": "The country code to fetch new releases for. Example values include 'US', 'CA', 'SE', 'IN', 'UK'.", "type": "str", "default": "us"}}

**Gold:** `[{"name": "bolivia_songs", "arguments": {"range": "daily", "date": "2023-01-01"}}, {"name": "new_releases", "arguments": {"country": "UK"}}]`

**Wrong predictions:**

~~~
```json
[
  {"name": "bolivia_songs", "arguments": {"range": "1-10", "date": "2023-01-01"}},
  {"name": "new_releases", "arguments": {"country": "uk"}}
]
```
~~~
~~~
[{"name": "bolivia_songs", "arguments": {"date": "2023-01-01", "range": "1-10"}}, {"name": "new_releases", "arguments": {"country": "uk"}}]
~~~
~~~
[{"name": "bolivia_songs", "arguments": {"range": "1-10", "date": "2023-01-01"}}, {"name": "new_releases", "arguments": {"country": "uk"}}]
~~~

## 31012

**Query:** What are the tournaments for WTA in 2022 and the results of the WTA tournament with ID '1002' from 2018?

**Tools:**

- `tournaments`: {"tour_code": {"description": "The code representing the tennis tour (e.g., 'atp', 'wta').", "type": "str", "default": "ATP"}, "season_id": {"description": "The year representing the season (e.g., 2023).", "type": "int", "default": "2020"}}
- `tournament_results_by_id_and_year_wta`: {"year": {"description": "The year of the tournament. Please note that not all tournaments are held every year.", "type": "int", "default": "2022"}, "tournament_id": {"description": "The ID of the tournament. You can retrieve this list from the WTA endpoint.", "type": "str", "default": "709"}}

**Gold:** `[{"name": "tournaments", "arguments": {"tour_code": "wta", "season_id": 2022}}, {"name": "tournament_results_by_id_and_year_wta", "arguments": {"year": 2018, "tournament_id": "1002"}}]`

**Wrong predictions:**

~~~
```json
[
  {"name": "tournaments", "arguments": {"tour_code": "WTA", "season_id": 2022}},
  {"name": "tournament_results_by_id_and_year_wta", "arguments": {"year": 2018, "tournament_id": "1002"}}
]
```
~~~
~~~
[{"name": "tournaments", "arguments": {"tour_code": "WTA", "season_id": "2022"}}, {"name": "tournament_results_by_id_and_year_wta", "arguments": {"year": "2018", "tournament_id": "1002"}}]
~~~

## 31706

**Query:** Retrieve showtimes for cinema with ID '1234' in Argentina.

**Tools:**

- `cinemas_id_showtimes`: {"is_id": {"description": "The cinema identifier.", "type": "str", "default": ""}, "filmid": {"description": "The film identifier to filter showtimes. Defaults to None.", "type": "str, optional", "default": ""}}

**Gold:** `[{"name": "cinemas_id_showtimes", "arguments": {"is_id": "1234"}}]`

**Wrong predictions:**

~~~
```json
[
  {
    "name": "cinemas_id_showtimes",
    "arguments": {
      "is_id": "1234",
      "filmid": ""
    }
  }
]
```
~~~
~~~
[{"name": "cinemas_id_showtimes", "arguments": {"is_id": "1234", "filmid": ""}}]
~~~

## 33192

**Query:** Check the diabetes risk for a lightly active individual who is 60 inches tall and weighs 120 lbs, and assess the social dynamics of a group of 8 where 3 are extroverts and 5 are introverts.

**Tools:**

- `assess_diabetes_risk`: {"weight_lbs": {"description": "Body weight in pounds.", "type": "int"}, "height_inches": {"description": "Height in inches.", "type": "int"}, "activity": {"description": "Physical activity level. Allowed values: \"sedentary\", \"lightly active\", \"moderately active\", or \"very active\".", "type": "str"}}
- `analyze_social_group_dynamics`: {"total": {"description": "The total group size.", "type": "int"}, "extroverts": {"description": "The number of extroverted members in the group.", "type": "int"}, "introverts": {"description": "The number of introverted members in the group.", "type": "int"}}

**Gold:** `[{"name": "assess_diabetes_risk", "arguments": {"weight_lbs": 120, "height_inches": 60, "activity": "lightly active"}}, {"name": "analyze_social_group_dynamics", "arguments": {"total": 8, "extroverts": 3, "introverts": 5}}]`

**Wrong predictions:**

~~~
[{"name": "assess_diabetes_risk", "arguments": {"weight_lbs": 120, "height_inches": 60, "activity": "lightly active"}}, {"name": "analyze_social_group_dynamics", "arguments": {"total": 8, "extroverts": 3, "introverts": 5}}
~~~

## 35616

**Query:** I need the current stock price for TSLA, can u fetch it plz?

**Tools:**

- `stockprice`: {"symbol": {"description": "The stock symbol/ticker (e.g., TSLA, AAPL) for which to retrieve the live price.", "type": "str", "default": "TSLA"}}

**Gold:** `[{"name": "stockprice", "arguments": {"symbol": "TSLA"}}]`

**Wrong predictions:**

~~~
[{"symbol": "TSLA"}]
~~~

## 36037

**Query:** Find related artists for the album with the ID '5gJwO7pRZAJ1rP34IqH3wP' on Shazam.

**Tools:**

- `albums_get_related_artist`: {"is_id": {"description": "The value of the 'albums->id', 'id', or 'albumadamid' field returned from various Shazam API endpoints.", "type": "str", "default": "850569437"}, "l": {"description": "The language code to return results in. Defaults to 'en-US'.", "type": "str, optional", "default": "en-US"}}

**Gold:** `[{"name": "albums_get_related_artist", "arguments": {"is_id": "5gJwO7pRZAJ1rP34IqH3wP"}}]`

**Wrong predictions:**

~~~
[
  {
    "name": "albums_get_related_artist",
    "arguments": {
      "is_id": "5gJwO7pRZAJ1rP34IqH3wP",
      "l": "en-US"
    }
  }
]
~~~
~~~
[{"name": "albums_get_related_artist", "arguments": {"is_id": "5gJwO7pRZAJ1rP34IqH3wP", "l": "en-US"}}]
~~~

## 38510

**Query:** What is the current air quality in Beijing, China, and the moon illumination in Moscow, Russia?

**Tools:**

- `current_air_quality`: {"lon": {"description": "Longitude of the location for which to retrieve air quality data.", "type": "str", "default": "-73.00597"}, "lat": {"description": "Latitude of the location for which to retrieve air quality data.", "type": "str", "default": "40.71427"}}
- `astronomy_api`: {"q": {"description": "The query parameter for retrieving data. It supports various formats such as:", "type": "str", "default": "London"}, "dt": {"description": "The date for which to retrieve the astronomical data. Defaults to None.", "type": "str, optional", "default": ""}}

**Gold:** `[{"name": "current_air_quality", "arguments": {"lon": "116.4074", "lat": "39.9042"}}, {"name": "astronomy_api", "arguments": {"q": "Moscow, Russia"}}]`

**Wrong predictions:**

~~~
[{"name": "current_air_quality", "arguments": {"lat": "40.0084", "lon": "116.4074"}}, {"name": "astronomy_api", "arguments": {"q": "Moscow"}}]
~~~
~~~
[{"name": "current_air_quality", "arguments": {"lon": "-116.407944", "lat": "39.904201"}}, {"name": "astronomy_api", "arguments": {"q": "Moscow"}}
~~~
~~~
```json
[
  {
   "name": "current_air_quality",
   "arguments": {
    "lon": "116.4075",
    "lat": "39.9042"
   }
  },
  {
   "name": "astronomy_api",
   "arguments": {
    "q": "Moscow",
    "dt": ""
   }
  }
]
```
~~~

## 40311

**Query:** A proton has a charge of approximately 1.602 x 10^-19 Coulombs. If a proton is placed in an electric field with a voltage of 1000 Volts, what is its potential energy? After that, deal me 5 random cards from a shuffled deck.

**Tools:**

- `potential_energy`: {"charge": {"description": "The electric charge.", "type": "float"}, "voltage": {"description": "The electric potential difference (voltage).", "type": "float"}}
- `draw_cards`: {"num_draw": {"description": "The number of cards to be drawn. Defaults to 1.", "type": "int, optional", "default": 1}}

**Gold:** `[{"name": "potential_energy", "arguments": {"charge": 1.602e-19, "voltage": 1000}}, {"name": "draw_cards", "arguments": {"num_draw": 5}}]`

**Wrong predictions:**

~~~
[{"name": "potential_energy", "arguments": {"charge": 1.602e-19, "voltage": 1000}}, {"name": "draw_cards", "arguments": {"num_draw": 5}}
~~~

## 41794

**Query:** How many ounces are in 2 pounds of butter, and how many grams are in 1 cup of sugar?

**Tools:**

- `convert_cooking_measurements`: {"quantity": {"description": "The quantity of the item to convert.", "type": "float"}, "from_unit": {"description": "The unit of the input quantity (e.g., 'cup', 'gram').", "type": "str"}, "to_unit": {"description": "The unit to which the quantity should be converted (e.g., 'ounce', 'kilogram').", "type": "str"}, "item": {"description": "The cooking ingredient being converted (e.g., 'flour', 'sugar').", "type": "str"}}

**Gold:** `[{"name": "convert_cooking_measurements", "arguments": {"quantity": 2, "from_unit": "pound", "to_unit": "ounce", "item": "butter"}}, {"name": "convert_cooking_measurements", "arguments": {"quantity": 1, "from_unit": "cup", "to_unit": "gram", "item": "sugar"}}]`

**Wrong predictions:**

~~~
[{"from_unit": "pound", "to_unit": "ounce", "item": "butter"}, {"from_unit": "cup", "to_unit": "gram", "item": "sugar"}]
~~~

## 43032

**Query:** What are the top 5 popular fantasy books from the past week? Also, I'm in the mood for some dad jokes about cats.

**Tools:**

- `get_weekly_popular_books_by_genre`: {"genre": {"description": "The genre of books to retrieve.", "type": "str", "default": "horror"}, "number_results": {"description": "The number of results to return. Defaults to 10.", "type": "int, optional", "default": "10"}}
- `dad_jokes_search`: {"text": {"description": "The keyword to search for in the jokes. Default is 'dad'.", "type": "str", "default": "dad"}}

**Gold:** `[{"name": "get_weekly_popular_books_by_genre", "arguments": {"genre": "fantasy", "number_results": 5}}, {"name": "dad_jokes_search", "arguments": {"text": "cat"}}]`

**Wrong predictions:**

~~~
[{"name": "get_weekly_popular_books_by_genre", "arguments": {"genre": "fantasy", "number_results": "5"}}, {"name": "dad_jokes_search", "arguments": {"text": "cats"}}]
~~~
~~~
[{"name": "get_weekly_popular_books_by_genre", "arguments": {"genre": "fantasy", "number_results": 5}}, {"name": "dad_jokes_search", "arguments": {"text": "cats"}}]
~~~
~~~
[{"name": "get_weekly_popular_books_by_genre", "arguments": {"genre": "fantasy", "number_results": 5}}, {"name": "dad_jokes_search", "arguments": {"text": "cats"}}]
~~~

## 47337

**Query:** Retrieve the trending movies in Japan, along with the recent tweets from Elon Musk and Barack Obama.

**Tools:**

- `user_tweets`: {"twttr_proxy": {"description": "Optional proxy to use for the Twitter API request.", "type": "str, optional", "default": ""}, "twttr_session": {"description": "Optional session identifier for the Twitter API request.", "type": "str, optional", "default": ""}, "cursor": {"description": "Optional cursor for paginated results.", "type": "str, optional", "default": ""}, "user_id": {"description": "The Twitter user ID for which to fetch tweets.", "type": "str, optional", "default": ""}, "username": {"description": "The Twitter username for which to fetch tweets (default is 'elonmusk').", "type": "str, default='elonmusk'", "default": "elonmusk"}}
- `trending_videos`: {"country": {"description": "The country code for which to retrieve trending videos. Defaults to 'us'.", "type": "str, optional", "default": "us"}, "lang": {"description": "The language code for the video titles and descriptions. Defaults to 'en'.", "type": "str, optional", "default": "en"}, "section": {"description": "The section of trending videos to retrieve. Possible values are \"Now\", \"Music\", \"Movies\", \"Gaming\". Defaults to 'Now'.", "type": "str, optional", "default": "Now"}}

**Gold:** `[{"name": "trending_videos", "arguments": {"country": "jp", "section": "Movies"}}, {"name": "user_tweets", "arguments": {"username": "elonmusk"}}, {"name": "user_tweets", "arguments": {"username": "BarackObama"}}]`

**Wrong predictions:**

~~~
[{"name": "trending_videos", "arguments": {"country": "jp", "lang": "ja", "section": "Movies"}}, {"name": "user_tweets", "arguments": {"user_id": "4419639", "username": "elonmusk"}}, {"name": "user_tweets", "arguments": {"user_id": "1439232", "username": "barackobama"}}]
~~~
~~~
[
  {"name": "trending_videos", "arguments": {"country": "jp", "lang": "en", "section": "Movies"}},
  {"name": "user_tweets", "arguments": {"username": "elonmusk", "cursor": "1", "twttr_session": "1", "twttr_proxy": "1"}}
]
~~~
~~~
[{"name": "trending_videos", "arguments": {"country": "jp", "lang": "ja", "section": "Movies"}}, {"name": "user_tweets", "arguments": {"username": "elonmusk"}}, {"name": "user_tweets", "arguments": {"username": "barackobama"}}]
~~~

## 47684

**Query:** Can you create a QR code for my email address, myemail@example.com? Also, I need a QR code that dials the number 123-456-7890.

**Tools:**

- `getqrcode`: {"type": {"description": "The format of the QR code to generate. Valid options are 'url', 'text', 'telno', 'mailto', or 'smsto'.", "type": "str", "default": "url"}, "value": {"description": "The content to encode within the QR code. Example values are 'https://www.google.com' for 'url', 'Some text' for 'text', '9142545474' for 'telno', and '8542487542' for 'smsto'.", "type": "str", "default": "https://www.google.com"}}

**Gold:** `[{"name": "getqrcode", "arguments": {"type": "mailto", "value": "myemail@example.com"}}, {"name": "getqrcode", "arguments": {"type": "telno", "value": "123-456-7890"}}]`

**Wrong predictions:**

~~~
```json
[
  {
    "name": "getqrcode",
    "arguments": {
      "type": "mailto",
      "value": "myemail@example.com"
    }
  },
  {
    "name": "getqrcode",
    "arguments": {
      "type": "telno",
      "value": "1234567890"
    }
  }
]
```
~~~
~~~
[{"name": "getqrcode", "arguments": {"value": "123-456-7890"}}]
~~~

## 48573

**Query:** I need to find out the exchange rates for GBP to USD, EUR, and JPY on March 15, 2023. I also need to know the exchange rates for USD to CAD, CHF, and MXN on May 31, 2023. Can you help me with that?

**Tools:**

- `get_4_date_endpoint`: {"base": {"description": "The base currency for which to get the exchange rates.", "type": "str", "default": "eur"}, "symbols": {"description": "A comma-separated string of target currencies to get exchange rates for.", "type": "str", "default": "usd"}, "date": {"description": "The date for which to fetch the exchange rates in 'YYYY-MM-DD' format. Defaults to '2023-02-01'.", "type": "str, optional", "default": "2023-02-01"}}

**Gold:** `[{"name": "get_4_date_endpoint", "arguments": {"base": "GBP", "symbols": "USD,EUR,JPY", "date": "2023-03-15"}}, {"name": "get_4_date_endpoint", "arguments": {"base": "USD", "symbols": "CAD,CHF,MXN", "date": "2023-05-31"}}]`

**Wrong predictions:**

~~~
[{"name": "get_4_date_endpoint", "arguments": {"base": "eur", "symbols": "usd,gbp,gbp", "date": "2023-03-15"}}]
~~~
~~~
[{"name": "get_4_date_endpoint", "arguments": {"base": "gbp", "symbols": "usd,eur,jpy", "date": "2023-03-15"}}, {"name": "get_4_date_endpoint", "arguments": {"base": "usd", "symbols": "cad,chf,mxn", "date": "2023-05-31"}}]
~~~
~~~
[
  {
    "name": "get_4_date_endpoint",
    "arguments": {
      "base": "gbp",
      "symbols": "usd,eur,jp",
      "date": "2023-03-15"
    }
  },
  {
    "name": "get_4_date_endpoint",
    "arguments": {
      "base": "usd",
      "symbols": "cad,CHF,mx",
      "date": "2023-05-31"
    }
  }
]
~~~

## 50566

**Query:** Who are the users on Pinterest related to 'fashion design'?

**Tools:**

- `search_user`: {"keyword": {"description": "The search term used to find users on Pinterest.", "type": "str", "default": "Trang Bui"}}

**Gold:** `[{"name": "search_user", "arguments": {"keyword": "fashion design"}}]`

**Wrong predictions:**

~~~
{"name": "search_user", "arguments": {"keyword": "fashion design"}}
</arguments>
}
~~~

## 51157

**Query:** Can you help me find the definition of the word 'obfuscate' and search for a book called 'Moby Dick'?

**Tools:**

- `getdefinition`: {"word": {"description": "The word for which the definition, pronunciation, and example usages are retrieved.", "type": "str", "default": "dance"}}
- `search`: {"s": {"description": "The name of the book to search for.", "type": "str", "default": ""}}

**Gold:** `[{"name": "getdefinition", "arguments": {"word": "obfuscate"}}, {"name": "search", "arguments": {"s": "Moby Dick"}}]`

**Wrong predictions:**

~~~
[{"name": "getdefinition", "arguments": {"word": "obfuscate"}]}
~~~

## 53165

**Query:** Fetch the active listings for agent D on page 3. Then, check if 'microsoft.com' and 'linkedin.com' are similar domains. Finally, perform a video search for 'top technology trends'.

**Tools:**

- `agent_s_active_listings`: {"zuid": {"description": "The Zillow unique identifier for the agent.", "type": "str", "default": "X1-ZU10c5uqeoxak21_5upaj"}, "page": {"description": "The page number of the results to retrieve. Defaults to None.", "type": "int, optional", "default": ""}}
- `check_similarity`: {"domain1": {"description": "The first domain to compare.", "type": "str", "default": "google.com"}, "domain2": {"description": "The second domain to compare.", "type": "str", "default": "shopify.com"}}
- `video`: {"query": {"description": "URL encoded query string for the video search.", "type": "str", "default": ""}}

**Gold:** `[{"name": "agent_s_active_listings", "arguments": {"zuid": "jkl012", "page": 3}}, {"name": "check_similarity", "arguments": {"domain1": "microsoft.com", "domain2": "linkedin.com"}}, {"name": "video", "arguments": {"query": "top technology trends"}}]`

**Wrong predictions:**

~~~
[{"name": "agent_s_active_listings", "arguments": {"zuid": "D", "page": 3}}, {"name": "check_similarity", "arguments": {"domain1": "microsoft.com", "domain2": "linkedin.com"}}, {"name": "video", "arguments": {"query": "top%20technology%20trends"}}]
~~~
~~~
[
  {"name": "agent_s_active_listings", "arguments": {"zuid": "X1-ZU10c5uqeoxak21_5upaj", "page": 3}},
  {"name": "check_similarity", "arguments": {"domain1": "microsoft.com", "domain2": "linkedin.com"}},
  {"name": "video", "arguments": {"query": "top technology trends"}}
]
~~~
~~~
[{"name": "agent_s_active_listings", "arguments": {"zuid": "X1-ZU10c5uqeoxak21_5upaj", "page": 3}}, {"name": "check_similarity", "arguments": {"domain1": "microsoft.com", "domain2": "linkedin.com"}}, {"name": "video", "arguments": {"query": "top%20technology%20trends"}}]
~~~

## 55363

**Query:** What would be the final amount if I invest $12000 at an annual interest rate of 4% for 6 years, with the interest compounded quarterly? Also, can you create a histogram for the data set [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] with 5 bins?

**Tools:**

- `compound_interest`: {"principal": {"description": "The initial investment amount.", "type": "float", "default": 1}, "rate": {"description": "The annual interest rate expressed as a decimal.", "type": "float", "default": 1}, "time": {"description": "The time in years for which the investment is held.", "type": "float"}, "n": {"description": "The number of times interest is compounded per year. Defaults to 1.", "type": "int, optional"}}
- `create_histogram`: {"data": {"description": "The data for which histogram needs to be plotted.", "type": "list", "default": 10}, "bins": {"description": "The number of equal-width bins in the range. Default is 10.", "type": "int", "default": 10}}

**Gold:** `[{"name": "compound_interest", "arguments": {"principal": 12000, "rate": 0.04, "time": 6, "n": 4}}, {"name": "create_histogram", "arguments": {"data": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], "bins": 5}}]`

**Wrong predictions:**

~~~
[{"name": "compound_interest", "arguments": {"principal": 12000, "rate": 0.04, "time": 6, "n": 4}}, {"name": "create_histogram", "arguments": {"data": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], "bins": 5}}
~~~

## 57548

**Query:** Are the words 'listen' and 'silent' anagrams of each other? Also, convert the date '2022-01-01' from 'YYYY-MM-DD' to 'DD/MM/YYYY' format.

**Tools:**

- `is_anagram`: {"word1": {"description": "The first word.", "type": "str"}, "word2": {"description": "The second word.", "type": "str"}}
- `format_date`: {"date": {"description": "The date string to convert.", "type": "str"}, "input_format": {"description": "The format of the input date string.", "type": "str"}, "output_format": {"description": "The desired format of the output date string.", "type": "str"}}

**Gold:** `[{"name": "is_anagram", "arguments": {"word1": "listen", "word2": "silent"}}, {"name": "format_date", "arguments": {"date": "2022-01-01", "input_format": "%Y-%m-%d", "output_format": "%d/%m/%Y"}}]`

**Wrong predictions:**

~~~
[{"name": "is_anagram", "arguments": {"word1": "listen", "word2": "silent"}}, {"name": "format_date", "arguments": {"date": "2022-01-01", "input_format": "YYYY-MM-DD", "output_format": "DD/MM/YYYY"}}]
~~~
~~~
[{"name": "is_anagram", "arguments": {"word1": "listen", "word2": "silent"}}]
~~~
~~~
[{"name": "is_anagram", "arguments": {"word1": "listen", "word2": "silent"}}, {"name": "format_date", "arguments": {"date": "2022-01-01", "input_format": "YYYY-MM-DD", "output_format": "DD/MM/YYYY"}}]
~~~

## 59212

**Query:** Hey there, I want to see the next batch of trending videos on TikTok after this cursor value 1234567890123456789. Can you also limit the number of videos to 10?

**Tools:**

- `trending_feed_video_posts_v2`: {"limit": {"description": "The maximum number of records to return. Defaults to 12. Must not exceed 12.", "type": "int, optional", "default": ""}, "max_cursor": {"description": "Pagination cursor. Use this to fetch the next batch of videos by providing the value received in the previous request's response.", "type": "int, optional", "default": ""}}

**Gold:** `[{"name": "trending_feed_video_posts_v2", "arguments": {"limit": 10, "max_cursor": 1234567890123456789}}]`

**Wrong predictions:**

~~~
[{"name": "trending_feed_video_posts_v2", "arguments": {"limit": 10, "max_cursor": "1234567890123456789"}}]
~~~
~~~
[
  {
    "name": "trending_feed_video_posts_v2",
    "arguments": {
      "limit": 10,
      "max_cursor": "1234567890123456789"
    }
  }
]
~~~
