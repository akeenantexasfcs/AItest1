#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
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
            
            if st.button('Train Model'):
                model, label_encoders = train_model(data)
                st.success('Model trained successfully and saved!')

    # Predict Tab
    with tab2:
        st.header("Predict")
        # Load the model and encoders
        model, label_encoders = load_model()
        
        if model is not None:
            # Input fields for new data
            loan_type = st.selectbox('Loan Type', label_encoders['Loan Type'].classes_)
            pd_rating = st.slider('PD Rating', 5, 11, 7)
            lgd = st.selectbox('LGD', label_encoders['LGD'].classes_)
            eligibility = st.selectbox('Eligibility', label_encoders['Eligibility'].classes_)
            association_spread = st.slider('Association Spread', 0.0050, 0.04, 0.015)
            purchased_from = st.selectbox('Purchased From', label_encoders['Purchased From'].classes_)
            industry = st.selectbox('Industry', label_encoders['Industry'].classes_)
            tenor = st.slider('Tenor', 3, 10, 5)
            construction = st.selectbox('Construction', label_encoders['Construction'].classes_)
            upfront_fee = st.slider('Upfront Fee', 0.0015, 0.0040, 0.0025)

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

            # Predict the outcome
            if st.button('Predict'):
                prediction = model.predict(input_data)
                prediction_text = 'Yes' if prediction[0] == 1 else 'No'
                st.write(f'The prediction is: {prediction_text}')
        else:
            st.warning('Please train the model first.')

def train_model(data):
    # Encode categorical variables
    label_encoders = {}
    for column in ['Loan Type', 'LGD', 'Eligibility', 'Purchased From', 'Industry', 'Construction']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    # Convert percentage strings to float
    data['Association Spread'] = data['Association Spread'].str.rstrip('%').astype('float') / 100.0
    data['Upfront Fee'] = data['Upfront Fee'].str.rstrip('%').astype('float') / 100.0

    # Separate features and target
    X = data.drop('Pursue', axis=1)
    y = data['Pursue'].apply(lambda x: 1 if x == 'Yes' else 0)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a random forest classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save the model and encoders
    joblib.dump(model, 'model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')

    return model, label_encoders

def load_model():
    try:
        model = joblib.load('model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        return model, label_encoders
    except:
        return None, None

if __name__ == '__main__':
    main()

