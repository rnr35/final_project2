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

income_level = st.selectbox("Income Level Options:", ["10K - 19,999K", "20K - 29,999K", "30K - 49,999K", "50K - 59,999K",
                                                      "60K - 69,999K", "70K - 79,999K", "80K - 89,999K", "90K - 99,999K",
                                                      "100K - 109,999K", "110K - 119,999K", "120K - 129,999K", "130K - 139,999K",
                                                      "140K - 149,999K", "150K and Above"])

education_level = st.selectbox("Education Level Options:", ["Less than High School", "High School Incomplete", "High School Graduate",
                                                           "Some College, No Degree", "Two-Year Associate Degree From University or College",
                                                           "Four-year College or University Degree/Bachelor’s degree",
                                                           "Some Postgraduate or Professional Schooling, No Postgraduate Degree",
                                                           "Postgraduate or Professional Degree, including master’s, doctorate, medical or law degree"])

is_parent = st.radio("Are You A Parent of a Child Under 18 Living in Your Home?", ["Yes", "No"])

marital_status = st.selectbox("Marital Status:", ["Married", "Living with a Partner", "Divorced", "Separated", "Widowed", "Never Been Married"])

gender = st.radio("Gender:", ["Male", "Female"])

def preprocess_input(income_level, education_level, is_parent, marital_status, gender):
    # Example preprocessing (convert categorical variables to one-hot encoding)
    income_level_encoded = np.zeros(14)  # Placeholder encoding for income levels
    income_level_encoded[int(income_level) - 1] = 1

    education_level_encoded = np.zeros(8)  # Placeholder encoding for education levels
    education_level_encoded[int(education_level) - 1] = 1

    is_parent_encoded = 1 if is_parent == "Yes" else 0  # Convert Yes/No to binary

    marital_status_encoded = np.zeros(6)  # Placeholder encoding for marital status
    marital_status_encoded[int(marital_status) - 1] = 1

    gender_encoded = 1 if gender == "Male" else 0  # Convert Male/Female to binary

    # Concatenate the encoded features
    user_input = np.concatenate([income_level_encoded, education_level_encoded, [is_parent_encoded],
                                 marital_status_encoded, [gender_encoded]])

    return user_input

# Get the preprocessed user input
user_input = preprocess_input(income_level, education_level, is_parent, marital_status, gender)

# Make predictions using the loaded model
# Replace this line with the actual prediction code based on your model
prediction = model.predict([user_input])
probability = model.predict_proba([user_input])[:, 1][0]  # Probability of being classified as a LinkedIn user

# Display the prediction result
st.write("### Prediction Results")
if prediction[0] == 1:
    st.write("You are classified as a LinkedIn user.")
else:
    st.write("You are not classified as a LinkedIn user.")

st.write(f"Probability of being a LinkedIn user: {probability:.2f}")
