# Machine Learning Project

## Project Overview : Predicting Song Popularity Using Machine Learning

The goal of this project is to develop a machine learning model capable of predicting the popularity of a song based on various features, using data extracted from Spotify. We will leverage the Ultimate Spotify Tracks DB dataset sourced from Kaggle, which contains a wide range of song-related attributes such as tempo, duration, genre, and other musical characteristics. By using this data, the project aims to identify key factors that influence a song’s popularity on Spotify and create a model that can predict a song's popularity upon its release.


**Insights from our study**
TBD

- **Objective**: 
- **Focus Areas**: EDA & Machine Learning models





## Functionality ⚙️

- 🧹 **Data Structure**: 
- 📈 **Data Visualization**: 

## Tools Used 🛠️

- 🐍 **Python**: Main programming language used for data processing and analysis.
- 🐼 **Pandas**: Library for data manipulation and analysis.
- 📊 **Matplotlib & Seaborn**: Libraries for data visualization.
- 📈 **Power BI**: Tool for creating interactive dashboards and visualizing data insights.
- 📓 **Jupyter Notebooks**: Interactive environment for data cleaning and visualization.
- 🤖 **Scikit-learn**: Machine learning library used for building and evaluating models.
- 🌐 **Git**: Version control system for tracking changes and collaboration.

## Development Process 🚀

1. 🧹 **Data Cleaning**:  
   The first step was to clean the dataset by handling missing values and removing duplicates. Outliers were **evaluated** but were **not removed** from the dataset, as they were not deemed problematic for the models in this context, ensuring that the data was in the best shape for modeling.

2. 🔍 **Data Visualization**:  
   We visualized the relationships between the variables to identify which ones had the most impact on the **popularity** column and to understand how different features influenced the target.

3. 💻 **Machine Learning Models**:  
   - **Model Selection**: We tested different machine learning models to choose the best one for both the **classification** and **regression** tasks.
   - **Feature Selection**:
     - Chose relevant variables to train the models.
     - Dropped non-essential columns such as `['genre', 'artist_name', 'track_name', 'track_id', 'popularity']`.
   - **Popularity Classification**:
     - For the classification task, the **popularity** column was transformed into a categorical feature with five classes:
       - 'Very Low' (0-25)
       - 'Low' (25-50)
       - 'Medium' (50-75)
       - 'High' (75-90)
       - 'Top' (90-100)
   - **For Both Regression and Classification**:
     - **Train-Test Split**: Split the data into training and testing sets to ensure proper model evaluation.
     - **Data Normalization**: Applied normalization to scale the features and ensure the models were trained on properly scaled data.
     - **Principal Component Analysis (PCA)**: Used PCA for dimensionality reduction to improve model efficiency by keeping only the most important features.


## Model Performance: F1-Scores for Predicting Popular Songs 🎶 

Below are the results for our models ranked by their F1-scores in correctly predicting popular songs:

1. **Classification Problem**
   
| **Model Type**            | **F1-Score**  |
|---------------------------|---------------|
| **k-NN(k=3)**                  | 0.588         |
| **Logistic Regression**   | 0.509         |
| **Random Forest**         | 0.702         |
| **XGBoost**               | 0.51          |
| **MPL Classification**    | 0.54          |

2. **Regression Problem**
   
| **Model Type**            | **R2-Score**  |
|---------------------------|---------------|
| **k-NN(k=49)**            | 0.296         |
| **Bagging Regressor**     | 0.299         |
| **Random Forest**         | 0.447         |
| **Gradient Boosting**     | 0.515         |
| **Adaptative Boosting**   | 0.446         |
| **XGBoost Regressor**     | 0.399         |
| **MPL Regressor**         | 0.33          |


2. **Classification Problem with only two classes: *Popular* and *No Popular***
| **Model Type**            | **F1-Score**  |
|---------------------------|---------------|
| **Random Forest**         | 0.834         |


## Conclusion 📊
Through comprehensive analysis, the project reveals critical insights:
 - The best model was: *Random Forest*.
 - We need balance model to achieve the best predictions.




## Project Structure 📁

- `data/`: csv's generated for our different analysis
    - `cleaning_data.ipynb`: Jupyther Notebook to clean our data.
    - `cleaning_functions.py`: python folder with fuctions
    - `spotify_clean.csv`: data cleaned and 'popularity_class' column creation.
    - `spotify_clean_reg.csv`: data cleaned
    - `spotify_clean_reg_without_outliers`: data cleaned and with no outliers
    - `SpotifyFeatures`: original data

- `EDA/`: Exploratory Data Analisis
- `ML/`: Exploration of different ML model types
- `presentation/`:  Folder to store PDF presentations.
- `README.md`: File to describe the project and how to set it up.


## Project Presentation 🎤

The project findings are summarized in a detailed presentation, covering:

- 📋 Research goals and methodology.
- 🔍 Key insights and trends.
- 📊 Data visualizations and treatment analysis.
- 🤖 Best predictive model



## Project Members 👥

| Name       | GitHub Profile                           |
|------------|------------------------------------------|
| **Celia Manzano** | [GitHub Profile](https://github.com/cemanzanoc) |
| **Carlota Gordillo** | [GitHub Profile](https://github.com/carlotagordillo2) |
| **Laura Sánchez** | [GitHub Profile](https://github.com/laurasanchez20) |
----

## DataSet Source
The dataset used in this project is sourced from Kaggle's Ultimate Spotify Tracks DB, available at:
https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db/data

---
Feel free to reach out for any questions or suggestions!
