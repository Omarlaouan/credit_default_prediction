# Credit Default Prediction Project

Table of Contents
-----------------

1.  [Introduction](#introduction)
2.  [Project Structure](#project-structure)
3.  [Project Workflow](#project-workflow)
4.  [Data](#data)
6.  [Results](#results)
7.  [Streamlit app](#streamlit-app)
9.  [Future Work](#future-work)


## Introduction

This is a machine learning project that aims to predict whether a customer will default on a loan within 60 days of disbursement. The project uses historical customer financial data to train and evaluate several machine learning models and select the best-performing one for deployment.

## Project Structure

The project has the following structure:
credit_default_prediction/
```bash
.
├── README.md
├── data
│   ├── processed
│   │   └── test_with_pred.csv
│   └── raw
│       ├── simulations_history.csv
│       ├── test_topred.csv
│       └── train_topred.csv
├── models
│   ├── xgb_tuned.joblib
│   └── xgb_tuned_smote_model.joblib
├── notebooks
│   ├── EDA.ipynb
│   └── Modeling.ipynb
├── requirements.txt
├── src
│   └── data_processing.py
└── streamlit_app
    ├── app.py
    └── streamlit_data_input_example.csv
```


The `data` directory contains two subdirectories: `raw` and `processed`. The raw data is stored in the `data/raw` directory. In the `data/processed` directory you will find the submission file `test_with_pred.csv`. 

The `notebooks` directory contains Jupyter notebooks that document the different stages of the data analysis and modeling process.

The `src` directory contains Python script for data processing.

The `models` directory is where the trained machine learning models will be saved.

The `streamlit_app` directory contains the files for the app.

The `requirements.txt` file contains a list of Python dependencies needed to run the project.

The `README.md` file is the project documentation that you are currently reading.

## Usage 
To use this project, follow the steps below:

1. Clone the repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Navigate to the `notebooks` directory and run the `EDA.ipynb` notebook to explore the data and gain insights into the distribution of the features and the target variable.
4. Run the `Modeling.ipynb` notebook to train and evaluate machine learning models on the processed data. The notebook will also save the best-performing model to the `models` directory.
5. Navigate to the `streamlit_app` directory and run the Streamlit app using the command `streamlit run app.py`.
6. Upload a CSV file containing customer data and click the "Predict" button to view predictions and confidence levels. An example file is provided in the folder `streamlit_app`.
7. Save the results in csv format.

In addition to the above steps, you can also modify the code to experiment with different machine learning models, hyperparameters, and feature engineering techniques to improve the performance of the model. The `src` directory contains Python scripts for data processing that can be modified to create new features or change the way the data is preprocessed.

## Project Workflow
The workflow for this project is as follows:

1. **Data Exploration**: Explore the raw data to gain insights into the distribution of the features and the target variable. This will be done in the `EDA.ipynb` notebook.

2. **Modeling**: 
- **Engineer new features** from the original data to improve the performance of the machine learning models.
- **Train** machine learning models on the processed data to predict credit defaults. 
- **Evaluate** the performance of the trained models using various metrics and choose the best model for deployment.
- This will be done in the `modeling.ipynb` notebook.

## Data

Our training data set have the following structure : 
*   **Source**: private
*   **Number of features**: 15
*   **Number of observations**: 121431
*   More details in `EDA.ipynb` notebook 


## Results

We have trained multiples models (logistic regression, random forest and xgboost). Xgboost was the best performing model with the following results 

**Model Performance:**
```bash
Classification Report:
              precision    recall  f1-score   support

         0.0       0.73      0.73      0.73      7357
         1.0       0.58      0.58      0.58      4787

    accuracy                           0.67     12144
   macro avg       0.65      0.65      0.65     12144
weighted avg       0.67      0.67      0.67     12144

AUC Score:
0.7168164543550068
```

**Insights:**
- The confusion matrix shows that the model is able to correctly identify 73% of non-defaulters and 58% of defaulters. 
- The classification report also shows that the model has a better precision and recall for non-defaulters as compared to defaulters. 
- The AUC score of 0.716 indicates that the model is performing better than a random classifier. However, further improvements can be made by using more sophisticated techniques such as neural networks or ensemble models. 
- Challenges faced:
    - Some challenges we faced include dealing with the imbalanced dataset to improve model performance. We utilized SMOTE to address this issue. 
    - Additionally, hyperparameter tuning of XGBoost took approximately one hour, which limited our experimentation cycle.

## Streamlit app
We have developed a user-friendly Streamlit application to help the business understand the predictive capabilities of the model.
1.  Clone the repository to your local machine.
2.  Install the required dependencies using `pip install -r requirements.txt`.
3.  Navigate to the `streamlit_app` directory.
4.  Run the Streamlit app using the command `streamlit run app.py`.
5.  Open the displayed URL in your web browser to access the application.
6.  Upload a CSV file containing customer data and click the "Predict" button to view predictions and confidence levels. An example file is provided in the folder `streamlit_app`.
7.  Save the results in csv format

## Future Work

Here are some ideas to further improve the project:

- Retrain the XGBoost model using only the top 10 most important features identified from feature importance analysis. This may help to simplify the model and reduce overfitting. 
- Explore the use of SMOTE (Synthetic Minority Over-sampling Technique) as a hyperparameter in the XGBoost tuning. This technique may help to balance the class distribution and improve the model's ability to predict defaults.
- Gather additional information about the loan amount and use it as a feature in the model. This could potentially improve the model's performance by capturing the relationship between loan amount and default risk.
