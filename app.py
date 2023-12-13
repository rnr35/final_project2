import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    return np.where(x == 1, 1, 0)

data = {'Column1': [1, 0, 1], 'Column2': [0, 1, 1]}
test_df = pd.DataFrame(data)

cleaned_df = test_df.apply(clean_sm)
print(cleaned_df)


def clean_sm(x):
    return np.where(x == 1, 1, 0)

ss = s[['web1h', 'income', 'educ2', 'par', 'marital', 'gender', 'age']].copy()

ss['sm_li'] = clean_sm(ss['web1h'])

# Clean data
ss = pd.DataFrame({
  "sm_li": np.where(s["web1h"] == 1, 1, 0),
    "income": np.where((s["income"].between(1, 9)) | (s["income"].isna()), s["income"], np.nan),
    "education": np.where((s["educ2"].between(1, 8)) | (s["educ2"].isna()), s["educ2"], np.nan),
    "parent": np.where(s["par"] == 1, 1, 0),
    "marital": np.where(s["marital"] == 1, 1, 0),
    "gender": np.where(s["gender"] == 1, 1, 0),
    "age": np.where((s["age"] > 0) & (s["age"] <= 98), 98 - s["age"], np.nan)
})

# Drop missing values
ss = ss.dropna()

ss["age"] = np.where(ss["age"] > 98, np.nan, ss["age"])

# Target (y) and feature(s) selection (X)
y = ss["sm_li"]
X = ss[["age", "income", "education", "gender", "marital", "parent"]]

# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,     
                                                    test_size=0.2,   
                                                    random_state=123) 


# Initialize algorithm 
lr = LogisticRegression()

# Fit algorithm to training data
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

import streamlit as st

st.title("LinkedIn User Predictor")

st.markdown("### LinkedIn User or Not? :computer::smile::computer:")

income_level_options = {
    1: "Less than $10,000",
    2: "10 to under $20,000",
    3: "20 to under $30,000",
    4: "30 to under $40,000",
    5: "40 to under $50,000",
    6: "50 to under $75,000",
    7: "75 to under $100,000",
    8: "100 to under $150,000",
    9: "$150,000 or more",
}

income_level = st.selectbox("Income Level Options:", list(income_level_options.values()))

education_level_options = {
    1: "Less than high school (Grades 1-8 or no formal schooling)",
    2: "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)",
    3: "High school graduate (Grade 12 with diploma or GED certificate)",
    4: "Some college, no degree (includes some community college)",
    5: "Two-year associate degree from a college or university",
    6: "Four-year college or university degree/Bachelor’s degree",
    7: "Some postgraduate or professional schooling, no postgraduate degree",
    8: "Postgraduate or professional degree, including master’s, doctorate, medical or law degree",
}

education_level = st.selectbox("Education Level Options:", list(education_level_options.values()))

parental_status_options = {
    1: "Yes",
    2: "No",
}

is_parent = st.selectbox("Are You A Parent of a Child Under 18 Living in Your Home?", list(parental_status_options.values()))

marital_status_options = {
    1: "Married",
    2: "Living with a partner",
    3: "Divorced",
    4: "Separated",
    5: "Widowed",
    6: "Never been married",
}

marital_status = st.selectbox("Marital Status:", list(marital_status_options.values()))

gender_options = {
    1: "Male",
    2: "Female",
    3: "Other",
}

gender = st.selectbox("Gender:", list(gender_options.values()))

age = st.number_input("Numeric Age (e.g., 25)", min_value=0, max_value=97, value=25)

# Preprocess the user input data (convert categorical features to numerical values)
# This is a placeholder preprocessing function; you should replace it with your actual preprocessing logic
def preprocess_input(income_level, education_level, is_parent, marital_status, gender, age):
    income_level_encoded = list(income_level_options.keys())[list(income_level_options.values()).index(income_level)]
    education_level_encoded = list(education_level_options.keys())[list(education_level_options.values()).index(education_level)]
    is_parent_encoded = list(parental_status_options.keys())[list(parental_status_options.values()).index(is_parent)]
    marital_status_encoded = list(marital_status_options.keys())[list(marital_status_options.values()).index(marital_status)]
    gender_encoded = list(gender_options.keys())[list(gender_options.values()).index(gender)]

    # Example age preprocessing (scale to a range between 0 and 1)
    age_encoded = age / 97  # Assuming 97 is the maximum age

    user_input = [income_level_encoded, education_level_encoded, is_parent_encoded, marital_status_encoded, gender_encoded, age_encoded]

    return user_input

user_input = preprocess_input(income_level, education_level, is_parent, marital_status, gender, age)

prediction = model.predict([user_input])
probability = model.predict_proba([user_input])[:, 1]  # Probability of being classified as a LinkedIn user

# Display the prediction result
st.write("### Prediction Results")
if prediction[0] == 1:
    st.write("You are classified as a LinkedIn user.")
else:
    st.write("You are not classified as a LinkedIn user.")

st.write(f"Probability of being a LinkedIn user: {probability[0]:.2f}")
