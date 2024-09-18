# Data Manipulation
import pandas as pd

# Regex
import re

# Web Scraping
from bs4 import BeautifulSoup
import requests

# OpenAI connection
from openai import OpenAI


url = 'https://en.wikipedia.org/wiki/1975_Pacific_hurricane_season'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')


# Initialize variables to store the hurricane names, paragraphs and a list for the whole data
name, paragraph = '', ''
data = []

# Enumerate the "mw-heading mw-heading2" and stand only in the second one, which is the "Systems" section
for i, tag in enumerate(soup.find_all('div', class_='mw-heading mw-heading2')):
    if i == 1:

        # While the 'Summary' section hasn't ended jump from sibling to sibling and extract the data
        sibling = tag.find_next_sibling()
        while sibling.get('class', 'No class') != ['mw-heading', 'mw-heading2']:

            # Loop in every sibling and extract the data from the specific div and p tags
            while True:
                # Extract the data from the specific tags
                if sibling.name == 'div' and sibling.get('class', 'No class') == ['mw-heading', 'mw-heading3']:
                    name = sibling.text
                    name = re.sub(r'\[.*?\]', '', name)
                elif sibling.name == 'p':
                    paragraph += ' ' + sibling.text
                    paragraph = re.sub(r'[\n\xa0]|\[.*?\]', '', paragraph).strip()

                # Move to the next sibling
                sibling = sibling.find_next_sibling()

                # If the next sibling is the end of section 'Summary' break, else if we move to another hurricane append a new row
                if sibling.name == 'div' and sibling.get('class', 'No class') == ['mw-heading', 'mw-heading2']: 
                    break
                elif sibling.name == 'div' and sibling.get('class', 'No class') == ['mw-heading', 'mw-heading3']:
                    data.append([name, paragraph])
                    name, paragraph = '', ''

data = pd.DataFrame(data, columns=['hurricane_storm_name', 'paragraph'])
data['date_start'], data['date_end'], data['number_of_deaths'], data['list_of_areas_affected'] = '', '', '', ''

# Initialize OpenAI's parameters
gpt_keys = pd.read_csv('../ChatGPT API Keys.txt').columns
api_key = gpt_keys[0]
org_key = gpt_keys[1]
prj_key = gpt_keys[2]

# Open connection
client = OpenAI(
    api_key=api_key,
    organization=org_key,
    project=prj_key
)


def create_prompt(paragraph):
    prompt = f"""
    Extract the following information from the given paragraph:
    1. Number of deaths
    2. Start date
    3. End date
    4. Affected areas

    Paragraph:
    {paragraph}

    Please provide the information in the following format, if no information provided reply None or 0:
    - Number of deaths: <number_of_deaths>
    - Start date: <start_date>
    - End date: <end_date>
    - Affected areas: <affected_areas>
    """
    return prompt


for i in range(len(data)):
    # Create the prompt
    prompt = create_prompt(data.iloc[i, 1])

    # Request to the GPT-4o-mini model
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.5 
    )

    # Extract response text
    response_text = response.choices[0].message.content
    
    # Extract information using regular expressions
    number_of_deaths = re.search(r'- Number of deaths: (.*?)\n', response_text).group(1)
    start_date = re.search(r'- Start date: (.*?)\n', response_text).group(1)
    end_date = re.search(r'- End date: (.*?)\n', response_text).group(1)
    affected_areas = re.search(r'- Affected areas: (.*)', response_text).group(1)

    # Assign the values to the right col
    data.iloc[i, 2], data.iloc[i, 3], data.iloc[i, 4], data.iloc[i, 5] = start_date, end_date, number_of_deaths, affected_areas

# Save the data
data.to_csv('hurricanes_1975.csv')
