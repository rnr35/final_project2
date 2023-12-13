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






