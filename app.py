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

y_pred = lr.predict(x_test)

st.markdown(''''
[LinkedIn User or Not?]:computer::smile::computer:''')
st.markdown('''
[Input income level, education level, parental status, marital status, age, and gender]

st.markdown(Income Level Options:")
st.markdown("1: 10K - 19,999K)
st.markdown("2: 20K - 29,999K)
st.markdown("3: 30K - 49,999K)
st.markdown("4: 50K - 59,999K)
st.markdown("5: 60K - 69,999K)
st.markdown("6: 70K - 79,999K)
st.markdown("7: 80K - 89,999K)
st.markdown("8: 90K - 99,999K)
st.markdown("9: 100K - 109,999K)
st.markdown("10: 110K - 119,999K)
st.markdown("11: 120K - 129,999K)
st.markdown("12: 130K - 139,999K)
st.markdown("13: 140K - 149,999K)
st.markdown("14: 150K and Above)

st.markdown(Education Level Options:")
st.markdown("1: Less than High School)
st.markdown("2: High School Incomplete)
st.markdown("3: High School Graduate)
st.markdown("4: Some College, No Degree)
st.markdown("5: Two-Year Associate Degree From University or College)
st.markdown("6: Four-year College or University Degree/Bachelor’s degree)
st.markdown("7: Some Postgraduate or Professional Schooling, No Postgraduate Degree )
st.markdown("8: Postgraduate or Professional Degree, including master’s, doctorate, medical or law degree)

st.markdown(Are You A Parent of a Child Under 18 Living in Your Home:")
st.markdown("1: Yes)
st.markdown("2: No)

st.markdown(Marital Status:")
st.markdown("1: Married)
st.markdown("2: Living with a Partner)
st.markdown("3: Divorced)
st.markdown("4: Separated)
st.markdown("5: Widowed)
st.markdown("6: Never Been Married)

st.markdown(Gender:")
st.markdown("1: Male)
st.markdown("2: Female)
st.markdown("3: Other)
