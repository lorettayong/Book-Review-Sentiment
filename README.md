# Kindle Book Review Sentiment Analysis Project

This repository contains a machine learning project focuses on sentiment analysis of Kindle book reviews. The goal is to build a classification model that can accurately determine whether a review has a positive or negative sentiment.

# Project Goal

The primary objective is to develop a robust sentiment analysis model that can classify book reviews as either positive or negative, using a dataset of Kindle reviews from Amazon.

# Dataset

The dataset used for this project is the **Kindle Book Review dataset**, which is publicly available and contains a large number of reviews, each with a star rating and the full text of the review.

* **Source: [Kaggle - Amazon Kindly Book Review for Sentiment Analysis](https://www.kaggle.com/datasets/meetnagadia/amazon-kindle-book-review-for-sentiment-analysis)

# Project Structure

* Book_Review_Sentiment
  * data/
    * raw/
      * all_kindle_reviews.csv
    * processed/
      * all_kindle_reviews_processed.csv
    * images/
    * book_review_sentiment.ipynb
    * README.md
    * requirements.txt
    * .gitignore

# Initial Data Overview

The dataset consists of 12,000 Kindle book reviews. A quick exploration of the data has revealed the following key characteristics:

- Columns: The dataset has 11 columns in total, of which are of most interest to this project are `rating` and `reviewText`.
- Data Integrity: The `reviewText` column has no missing values, and there are no duplicate entries in the dataset.
- Rating Distribution: The ratings are perfectly balanced, with 2,000 reviews for each star rating of 1 to 5. This is an ideal scenario for a classification task as it avoids issues related to class imbalance.

# How to Run This Project

1. **Clone the repository:**
   ```bash
   git clone https://github.com/lorettayong/Book-Review-Sentiment.git
   cd book-review-sentiment
   ```
2. **Set up the virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   # On Windows, use '.venv\Scripts\activate'
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data:**
   * In a Python interpreter or script, run the following commands to download the necessary NLTK data:
     ```bash
     import nltk
     nltk.download('punkt')
     nltk.download('stopwords')
     nltk.download('wordnet')
     ```

5. **Run the Jupyter notebook:**
   ```bash
   jupyter notebook
   ```
   * Open `book_review_sentiment.ipynb` in your browser to follow the project's analysis steps.

# Project Phases

## Phase 1: Data Acquisition and Exploration
* **Objective:** Understand the raw data's characteristics, quality, and distribution before carrying out data transformation.
* **Key Activities:**
  * Loaded 'all_kindle_reviews.csv' dataset into a pandas DataFrame.
  * Performed an initial inspection to understand the data's structure, identify any missing values, and check for duplicates.
  * Conducted a simple exploratory data analysis to visualise the distribution of star ratings, confirming that the dataset is well-balanced.

# Next Steps (Future Work)

* **Data Preprocessing and Feature Engineering:** Transform the raw text data into a clean, more structured format that is suitable for machine learning models using sentiment mapping, text cleaning, text normalisation, vectorisation, and data splitting.
* **Model Building and Evaluation:** Train a machine learning model, such as Logistic Regression or Support Vector Machine (SVM), on the prepared training data, and assess its performance on unseen data using key metrics like accuracy, precision, recall, and F1-score.
* **Model Optimisation and Deployment:** Improve the model's performance by using techniques such as hyperparameter tuning and cross-validation, and prepare it for real-world use by creating a new branch on Github to work on developing a simple web application.
