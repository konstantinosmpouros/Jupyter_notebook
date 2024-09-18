# Pacific Hurricanes 1975 Season

<img src="Images/gpt_pacific.png" alt="1975 Pacific Hurricane Season" width="500" height="500">

## Overview

This project involves scraping and processing data from the Wikipedia page dedicated to the 1975 Pacific hurricane season. The primary goal is to extract detailed information about hurricanes that occurred during this season and structure this information in a CSV format. This task will involve web scraping techniques to gather raw data and natural language processing (NLP) methods to clean and structure the extracted data effectively.

## Methodology

The project's implementation is divided in two primary parts. The Data collection, and data cleaning with the help of LLMs. 


- **Data Collection / Web Scraping**: With the communication with the web page, I used the requests and BeautifulSoup libraries. The primary task was to isolate the "Systems" section, which contains hurricane names and descriptions. Once I located the section, I iterated through sibling tags to extract the relevant data. The key tags containing the data were `<div class='mw-heading mw-heading3'>` for hurricane names and `<p>` for descriptions. I continued reading these tags until encountering another `<div class='mw-heading mw-heading3'>`, indicating a new hurricane entry. At this point, I appended the collected data as a new row and reset the process for the next hurricane.

- **LLM cleaning**: In the extracted data, each hurricane has an accompanying paragraph that describes key details. This paragraph is then processed using `gpt-4o-mini` to extract structured information such as the `Number of deaths`, `Start date`, `End date`, and `Affected areas`. The `gpt-4o-mini` has proven to be highly effective at accurately identifying and extracting these details, ensuring the quality and consistency of the structured data.


## Oucome

The outcome of the project was to create a csv file containing the hurricanes that occured in the Pacific ocean in 1975 containig the following information, columns, for each hurricane:

   - `hurricane_storm_name`: The name of the hurricane.
   - `paragraph`: The paragraph that we used as input the GPT-4o Mini to extract the data.
   - `date_start`: The start date of the hurricane.
   - `date_end`: The end date of the hurricane.
   - `number_of_deaths`: The number of fatalities attributed to the hurricane.
   - `list_of_areas_affected`: A list of regions or areas affected by the hurricane.


