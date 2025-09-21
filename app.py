import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- CACHING THE MODEL ---
@st.cache_data
def train_model():
    # 1. Create the synthetic customer data
    num_records = 1000
    data = {
        'age': np.random.randint(18, 75, size=num_records),
        'country': np.random.choice(['India', 'USA', 'UK', 'UAE', 'Nigeria', 'Switzerland', 'Panama'], size=num_records),
        'occupation': np.random.choice(['Salaried', 'Business Owner', 'Student', 'Freelancer', 'Politically Exposed Person (PEP)'], size=num_records),
        'annual_income': np.random.randint(20000, 500000, size=num_records)
    }
    df = pd.DataFrame(data)

    # 2. Define the logic for risk
    def assign_risk(row):
        if row['occupation'] == 'Politically Exposed Person (PEP)' or row['country'] == 'Panama':
            return 'High'
        elif row['annual_income'] > 400000 or row['country'] == 'Nigeria':
            return 'Medium'
        else:
            return 'Low'
    df['risk_level'] = df.apply(assign_risk, axis=1)

    # 3. Prepare data for the machine learning model
    df_encoded = pd.get_dummies(df, columns=['country', 'occupation'])
    X = df_encoded.drop(['risk_level'], axis=1)
    y = df_encoded['risk_level']
    
    # 4. Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, X_train.columns

# Train the model and get the column layout
model, train_columns = train_model()

# --- WEB APP INTERFACE ---

st.title("Customer Risk Prediction")
st.write("Enter the customer's details below to get a predicted risk level.")

# Create input fields for the user
age = st.slider("Age", 18, 100, 30)
country = st.selectbox("Country", ['India', 'USA', 'UK', 'UAE', 'Nigeria', 'Switzerland', 'Panama'])
occupation = st.selectbox("Occupation", ['Salaried', 'Business Owner', 'Student', 'Freelancer', 'Politically Exposed Person (PEP)'])
income = st.number_input("Annual Income (in USD)", min_value=0, value=50000)

# Create a button to make a prediction
if st.button("Predict Risk"):
    # 1. Create a DataFrame from the user's input
    new_customer_data = {
        'age': age,
        'country': country,
        'annual_income': income,
        'occupation': occupation
    }
    new_df = pd.DataFrame([new_customer_data])

    # 2. Prepare the data for the model
    new_df_encoded = pd.get_dummies(new_df)
    final_df = new_df_encoded.reindex(columns=train_columns, fill_value=0)

    # 3. Make the prediction
    prediction = model.predict(final_df)
    
    # 4. Display the result
    st.subheader("Prediction Result")
    if prediction[0] == 'High':
        st.error(f"High Risk")
    elif prediction[0] == 'Medium':
        st.warning(f"Medium Risk")
    else:
        st.success(f"Low Risk")

# --- FOOTER ---
# This line will add the centered footer to the bottom of the page
st.markdown("<div style='text-align: center; color: grey;'>Created With ❤️ by Ankit</div>", unsafe_allow_html=True)