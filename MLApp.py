#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Define the Streamlit app
def main():
    st.title('Loan Pursue Prediction App')

    # Tabs for training and prediction
    tab1, tab2 = st.tabs(["Train Model", "Predict"])

    # Train Model Tab
    with tab1:
        st.header("Train Model")
        uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
        if uploaded_file is not None:
            data = pd.read_excel(uploaded_file)
            st.write("Data Preview:")
            st.write(data.head())
            
            st.write("Columns in uploaded file:", data.columns.tolist())

            if st.button('Train Model'):
                try:
                    models, label_encoders = train_model(data)
                    st.success('Models trained successfully and saved!')
                except KeyError as e:
                    st.error(f"Column not found: {e}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    # Predict Tab
    with tab2:
        st.header("Predict")
        # Load the models and encoders
        models, label_encoders = load_model()
        
        if models is not None:
            # Input fields for new data
            loan_type = st.selectbox('Loan Type', label_encoders['Loan Type'].classes_)
            pd_rating = st.slider('PD Rating', 5, 11, 7)
            lgd = st.selectbox('LGD', label_encoders['LGD'].classes_)
            eligibility = st.selectbox('Eligibility', label_encoders['Eligibility'].classes_)
            association_spread = st.text_input('Association Spread (%)', value="1.25")
            purchased_from = st.selectbox('Purchased From', label_encoders['Purchased From'].classes_)
            industry = st.selectbox('Industry', label_encoders['Industry'].classes_)
            tenor = st.slider('Tenor', 3, 10, 5)
            construction = st.selectbox('Construction', label_encoders['Construction'].classes_)
            upfront_fee = st.text_input('Upfront Fee (%)', value="0.25")

            # Process input data
            try:
                association_spread = float(association_spread.rstrip('%')) / 100
                upfront_fee = float(upfront_fee.rstrip('%')) / 100
            except ValueError:
                st.error("Please enter valid percentages for Association Spread and Upfront Fee.")
                return

            # Encode input data
            input_data = pd.DataFrame({
                'Loan Type': [loan_type],
                'PD Rating': [pd_rating],
                'LGD': [lgd],
                'Eligibility': [eligibility],
                'Association Spread': [association_spread],
                'Purchased From': [purchased_from],
                'Industry': [industry],
                'Tenor': [tenor],
                'Construction': [construction],
                'Upfront Fee': [upfront_fee]
            })

            for column in input_data.columns:
                if column in label_encoders:
                    input_data[column] = label_encoders[column].transform(input_data[column])
                elif column in ['Association Spread', 'Upfront Fee']:
                    input_data[column] = input_data[column].astype('float')

            # Predict the outcome using multiple models
            if st.button('Predict'):
                predictions, probabilities = predict(input_data, models)
                majority_vote = 'Yes' if predictions.count('Yes') > predictions.count('No') else 'No'

                # Display the results
                result_df = pd.DataFrame({
                    'Model Type': ['Logistic Regression', 'Random Forest', 'Support Vector Machine', 'Naive Bayes', 'K-Nearest Neighbors'],
                    'Prediction': predictions,
                    'Probability': probabilities
                })

                st.write(result_df)
                st.write(f'ELC Prediction: {majority_vote}')
        else:
            st.warning('Please train the model first.')

def train_model(data):
    required_columns = ['Loan Type', 'LGD', 'Eligibility', 'Purchased From', 'Industry', 'Construction', 'Association Spread', 'Upfront Fee', 'PD Rating', 'Pursue', 'Tenor']
    
    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise KeyError(f"Missing columns: {missing_columns}")

    # Handle missing values
    data = data.fillna('')

    # Encode categorical variables
    label_encoders = {}
    for column in ['Loan Type', 'LGD', 'Eligibility', 'Purchased From', 'Industry', 'Construction']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].astype(str))
        label_encoders[column] = le

    # Convert percentage strings to float if necessary
    if data['Association Spread'].dtype == 'object':
        data['Association Spread'] = data['Association Spread'].str.rstrip('%').astype('float') / 100.0
    if data['Upfront Fee'].dtype == 'object':
        data['Upfront Fee'] = data['Upfront Fee'].str.rstrip('%').astype('float') / 100.0

    # Separate features and target
    X = data.drop('Pursue', axis=1)
    y = data['Pursue'].apply(lambda x: 1 if x == 'Yes' else 0)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train multiple models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'Naive Bayes': GaussianNB(),
        'K-Nearest Neighbors': KNeighborsClassifier()
    }

    for model in models.values():
        model.fit(X_train, y_train)

    # Save the models and encoders
    joblib.dump(models, 'models.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')

    return models, label_encoders

def load_model():
    try:
        models = joblib.load('models.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        return models, label_encoders
    except:
        return None, None

def predict(input_data, models):
    predictions = []
    probabilities = []

    for model_name, model in models.items():
        prediction = model.predict(input_data)[0]
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(input_data)[0][1]  # Probability of 'Yes'
            probabilities.append(probability)
        else:
            probabilities.append('N/A')

        prediction_text = 'Yes' if prediction == 1 else 'No'
        predictions.append(prediction_text)

    return predictions, probabilities

if __name__ == '__main__':
    main()

