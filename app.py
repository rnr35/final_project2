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
linkedin_user = st.radio("Choose an option:", ["Yes", "No"])

st.markdown("### Income Level Options:")
income_level = st.selectbox(
    "Select your income level:",
    ["10K - 19,999K", "20K - 29,999K", "30K - 49,999K", "50K - 59,999K",
     "60K - 69,999K", "70K - 79,999K", "80K - 89,999K", "90K - 99,999K",
     "100K - 109,999K", "110K - 119,999K", "120K - 129,999K", "130K - 139,999K",
     "140K - 149,999K", "150K and Above"]
)

st.markdown("### Education Level Options:")
education_level = st.selectbox(
    "Select your education level:",
    ["Less than High School", "High School Incomplete", "High School Graduate",
     "Some College, No Degree", "Two-Year Associate Degree From University or College",
     "Four-year College or University Degree/Bachelor’s degree",
     "Some Postgraduate or Professional Schooling, No Postgraduate Degree",
     "Postgraduate or Professional Degree, including master’s, doctorate, medical or law degree"]
)

st.markdown("### Are You A Parent of a Child Under 18 Living in Your Home?")
is_parent = st.radio("Choose an option:", ["Yes", "No"])

st.markdown("### Marital Status:")
marital_status = st.selectbox(
    "Select your marital status:",
    ["Married", "Living with a Partner", "Divorced", "Separated", "Widowed", "Never Been Married"]
)

st.markdown("### Gender:")
gender = st.radio("Select your gender:", ["Male", "Female"])

st.write("User Profile:")
st.write(f"LinkedIn User: {linkedin_user}")
st.write(f"Income Level: {income_level}")
st.write(f"Education Level: {education_level}")
st.write(f"Parent of a Child Under 18: {is_parent}")
st.write(f"Marital Status: {marital_status}")
st.write(f"Gender: {gender}")
