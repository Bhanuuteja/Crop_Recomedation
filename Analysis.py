import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
# import seaborn as sns
import plotly.express as px
import streamlit as st
import warnings
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

warnings.simplefilter("ignore")

data = pd.read_csv("Crop_recommendation.csv")

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
scale = StandardScaler()
minmax = MinMaxScaler()
data_scale = pd.DataFrame(scale.fit_transform(data.iloc[:,:-1]),columns = data.columns[:-1])
data_scale_min = pd.DataFrame(minmax.fit_transform(data.iloc[:,:-1]),columns = data.columns[:-1])
X = data.drop("label",axis=1)
y = data["label"]
labels = data["label"].unique()
from sklearn.model_selection import train_test_split 
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=.25, random_state = 11)
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
scores = pd.DataFrame(columns = ["Model","Accuracy"])

from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier()
RFC.fit(Xtrain, Ytrain)
print(accuracy_score(Ytest, RFC.predict(Xtest)))
# scores = scores.append({"Model":"Random Forest","Accuracy": accuracy_score(Ytest, RFC.predict(Xtest))*100},ignore_index=True)

# K-Means clustering
kmeans = KMeans(n_clusters=3)  # You can choose the number of clusters you need
data['cluster'] = kmeans.fit_predict(data_scale_min)

# Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)  # You can adjust the parameters as needed
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data_scale_min, data['label'], test_size=0.25, random_state=11)
logistic_model.fit(Xtrain, Ytrain)
logistic_accuracy = accuracy_score(Ytest, logistic_model.predict(Xtest))

# Display K-Means clusters
st.write("K-Means Clusters:")
st.write(data[['label', 'cluster']])

# Display Logistic Regression accuracy
st.write(f"Logistic Regression Accuracy: {logistic_accuracy * 100:.2f}%")

#creating a header with Header section link
# st.markdown("<h1 style='text-align: center; color: #0b0c0c;'>Crop Recommendation Model</h1>", unsafe_allow_html=True)

st.set_page_config(
    page_title="Crop Analysis",
    page_icon="🎋",
    layout="wide"
)
# st.sidebar.success("Please select page here")
st.markdown("<h1 style='text-align: center; color: green;'>Crop Recommendation Analysis</h1>", unsafe_allow_html=True)
st.write("This is a simple web application which will help in recommending the type of crop.")
st.divider()
st.write("The application is built using the Random Forest Classifier algorithm.")
st.write("The dataset used for training the model is taken from Kaggle.")
st.write("The dataset contains 22 columns and 2200 rows.")
st.write("This is the first 5 rows of the data.")
st.write(data.head())
st.divider()
st.write("Information about the Data")
st.columns((1,1,1))[1].write(data.dtypes)
st.divider()
st.write("Description of the data.")
st.write(data.describe())
st.divider()


import seaborn as sns

st.write("Checking the outliers of the data.")
col = data.columns

for column in range(len(col)):
    if data[col[column]].dtype != "object": 
        fig, ax = plt.subplots()
        sns.boxplot(data[col[column]],ax=ax)
        # fig.set_size_inches(1,1)
        ax.set_xlabel(col[column], c="r") 
        st.pyplot(fig)

st.write("Checking the distribution of the data.")
for column in range(len(col)):
    if data[col[column]].dtype != "object": 
        fig, ax = plt.subplots()
        sns.distplot(data[col[column]],ax=ax)
        # fig.set_size_inches(1,1)
        ax.set_xlabel(col[column], c="r") 
        st.pyplot(fig)
from streamlit_extras.switch_page_button import switch_page
if st.button("Predict ?"):
    switch_page("prediction")


