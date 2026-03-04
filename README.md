**YouTube Comment Sentiment Analysis**

This project implements a machine learning pipeline for analyzing the sentiment of YouTube comments. The system collects comments from a video using the YouTube Data API, preprocesses the text data, and classifies each comment as positive, neutral, or negative.

The project includes data preprocessing, feature engineering, model training, hyperparameter optimization, and a simple web interface that allows users to analyze comments from any YouTube video.

**Project Overview**

The goal of this project is to automatically analyze the sentiment of YouTube comments and provide a quick overview of how users react to a video.

The system:

Collects comments using the YouTube Data API

Cleans and preprocesses the text

Converts text into numerical features using TF-IDF

Applies a trained LightGBM classification model

Displays the results through a Streamlit interface

The final output shows:

Number of positive, neutral, and negative comments

Percentage distribution

Example comments for each sentiment class

**Dataset**

The model was trained on a dataset of approximately 1 million YouTube comments.

During preprocessing:

rows containing NaN values were removed

extremely short comments (<3 characters) were removed

extremely long comments (>1000 characters) were removed

These steps helped reduce noise and improve model performance.

**Text Preprocessing**

converting text to lowercase

removing URLs

removing user mentions (@username)

removing hashtags

removing unnecessary special characters

normalizing whitespace

**Feature Engineering**

In addition to text features, several numerical features were extracted:

Comment_Length

Likes

Month

DayOfWeek

Hour

IsWeekend

These features provide additional contextual information that can improve classification performance.

**Feature Extraction**

Text was converted into numerical vectors using TF-IDF.

Several approaches were tested:

word-level TF-IDF

stopword removal

lemmatization

character n-gram TF-IDF

The best results were obtained using:

TF-IDF with character n-grams 


**Models Tested**

Several machine learning models were evaluated:

Logistic Regression

Linear SVC

XGBoost

LightGBM

Simple Neural Network (Keras)

The final model selected was LightGBM, which achieved the best balance between performance and training efficiency.

Evaluation metric used:
Macro F1 Score

Final performance:

Macro F1 = 0.0.73-0.75

**Model Optimization**

Hyperparameters were optimized using RandomizedSearchCV.

This improved the model performance compared to the baseline configuration.

**Web Interface**

A simple web interface was built using Streamlit.

The application allows users to:

Enter a YouTube video ID

Download comments using the YouTube API

Apply the trained sentiment model

Display sentiment statistics and example comments

Example output:

number of positive / neutral / negative comments

percentage distribution

sample comments for each category
