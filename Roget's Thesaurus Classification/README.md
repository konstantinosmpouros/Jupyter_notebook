# Analysis of Roget's Thesaurus with Machine Learning

## Overview

This project leverages the rich lexical resource of Roget's Thesaurus to perform an in-depth analysis using machine learning techniques. The primary goal is to categorize words into distinct classes and sections, providing insights into the semantic relationships and clustering of words within the thesaurus. 

## Methodology

The project's implementation is divided in three primary parts. The Data collection and creation of Embeddings, the clustering and finally in the prediction of class and section. In each of these parts a separate notebook was created in order to be able to rerun part of the whole project independently and faster. The notebooks that were created are 4 and named as following with the exact order:

  1. **Roget's Data and Embeddings**
  2. **Roget's Clustering**
  3. **Roget's Embending's Class Prediction**
  4. **Roget's Embending's Section Prediction**

### Data Collection and Embeddings creation

- **Web Scraping**: The initial step involved scraping Roget's Thesaurus (Available on project Gutenberg) to extract all the words. This process provided the raw data needed for further analysis.

- **Embedding Generation**: For each word obtained from the thesaurus, embeddings were created. These embeddings serve as numerical representations of the words, capturing their semantic meanings in a multidimensional space. In the creation of the embeddings phase we used 2 different model to process the words. These models are the **sentece transformer** and the **universal sentence encoder**.

### Clustering

- **Two-Level Clustering**: The words were clustered on two different levels to explore their semantic relationships:
  1. **First Level (Class Clustering)**: Words were clustered into 6 broad classes. This high-level categorization provides a general grouping of words based on their overarching meanings.
  2. **Second Level (Section Clustering**: A more granular clustering was performed to divide the words into 24 sections. This detailed segmentation allows for a finer understanding of the semantic nuances among the words.

### Predictions with Machine Learning Models

- **Prediction**: With the embeddings as input, machine learning models were trained to predict the class and section for each word. This predictive analysis helps in understanding how well the embeddings capture the semantic relationships and can be used to categorize words based on their meanings.

## Reproducibility Instructions

If you wish to rerun the Jupyter notebooks included in this project to reproduce the results or explore further, please follow these guidelines:

- **Environment Setup**: This project uses Poetry for dependency management to ensure a consistent environment. Make sure you have Poetry installed on your system. Once installed, navigate to the project directory and run `poetry install` to install all necessary dependencies in a virtual environment.

- **Before run anything**: Make sure you have unziped the files of the embeddings if you to run first notebook like Roget's Clustering that need the embeddings file to be in json format and not zipped.
  
- **Running the Notebooks**: After setting up the environment, activate the virtual environment created by Poetry by running `poetry shell`. Then, launch Jupyter Notebook or JupyterLab by running `jupyter notebook` or `jupyter lab` in the terminal. You can then navigate to the notebook files and run them.

- **Expected Duration**: The total runtime can vary significantly based on your hardware and the specific models being used. As a rough estimate, expect the data collection and embeddings creation phase to take some minutes, depending on your internet connection and processing power. The clustering and machine learning model training phases are computationally intensive, especially for large datasets like this case, and takes multiple hours to complete with a min time consumption of 30min and max about 1.5 hours. It's recommended to run these processes on a machine with sufficient CPU and memory resources cause the models are designed to drain all the computational power of the cpu in order to run as fast as possible.

