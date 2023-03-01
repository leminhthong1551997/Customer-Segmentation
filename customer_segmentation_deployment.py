import pickle
import streamlit as st

import numpy as np
import datetime as dt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import squarify
from datetime import datetime
# !pip install openpyxl

from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# 1. Read data
columns = ['CustomerID', 'InvoiceDate', 'Quantity', 'TotalAmountSpent']
df = pd.read_csv("CDNOW_master.txt", header = None, delim_whitespace=True, names = columns)

#--------------
# GUI
st.title("Data Science Project")
st.write("## Customer Segmentation")
# Upload file
uploaded_file = st.file_uploader("Choose a file", type=['csv', 'txt'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, header = None, delim_whitespace=True, names = columns)
    data.to_csv("CD_new.csv", index = False)

# 2. Data pre-processing
## Check Null Values.
if (df.isnull().sum().any() != 0):
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
## Dropping negative values
df = df[(df['Quantity']>0) & (df['TotalAmountSpent']>0)] 
## Removing duplicates
df=df.drop_duplicates(subset = df.columns)
## RFM analysis
string_to_date = lambda x : datetime.strptime(x, "%Y%m%d").date()

### Convert InvoiceDate from object to datetime format
df['InvoiceDate'] = df['InvoiceDate'].astype('str')
df['InvoiceDate'] = df['InvoiceDate'].apply(string_to_date)
df['InvoiceDate'] = df['InvoiceDate'].astype('datetime64[ns]')

### Drop NA values
df = df.dropna()

### RFM
### Convert string to date, get max date of dataframe
max_date = df['InvoiceDate'].max().date()

Recency = lambda x : (max_date - x.max().date()).days
Frequency  = lambda x: len(x.unique())
Monetary = lambda x : round(sum(x), 2)

rfm = df.groupby('CustomerID').agg({'InvoiceDate': Recency,
                                        'Quantity': Frequency,  
                                        'TotalAmountSpent': Monetary })

### Changing the column names
rfm.columns = ['recency', 'frequency', 'monetary']
rfm = rfm[rfm["monetary"] > 0]

"""## Save data after preprocessing
rfm.to_csv('df_RFM.csv', index = False)

## Load data to apply model
df = pd.read_csv('df_RFM.csv')"""

## Checking Outliers
### Outlier treatment for recency
Q1 = rfm.recency.quantile(0.25)
Q3 = rfm.recency.quantile(0.75)
IQR = Q3 - Q1
rfm = rfm[(rfm.recency >= Q1 - 1.5*IQR) & (rfm.recency <= Q3 + 1.5*IQR)]

### Outlier treatment for frequency
Q1 = rfm.frequency.quantile(0.25)
Q3 = rfm.frequency.quantile(0.75)
IQR = Q3 - Q1
rfm = rfm[(rfm.frequency >= Q1 - 1.5*IQR) & (rfm.frequency <= Q3 + 1.5*IQR)]

### Outlier treatment for monetary
Q1 = rfm.monetary.quantile(0.25)
Q3 = rfm.monetary.quantile(0.75)
IQR = Q3 - Q1
rfm = rfm[(rfm.monetary >= (Q1 - 1.5*IQR)) & (rfm.monetary <= (Q3 + 1.5*IQR))]

# 3. Build model (K-Means)
## Transforming the data
rfm1=rfm[['recency','frequency','monetary']]
scaler = StandardScaler()
x_scaled = scaler.fit(rfm1)
x_scaled = scaler.fit_transform(rfm1)
x_scaled

## Elbow method
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,20))
visualizer.fit(x_scaled)  

k = visualizer.elbow_value_

## Applying K-Means
kmeans_scaled = KMeans(k)
kmeans_scaled.fit(x_scaled)

#4. Save models
pkl_filename = "model_kmeans.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(model, file)

#5. Load models 
# Read model
# import pickle
with open(pkl_filename, 'rb') as file:  
    model_kmeans = pickle.load(file)

#6. GUI
menu = ["Business Objective", "Build Project", "New Prediction"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Business Objective':    
    st.subheader("Business Objective")
    st.write("""
    ###### Xây dựng hệ thống phân cụm khách
hàng dựa trên các thông tin do công ty cung cấp từ đó có
thể giúp công ty xác định các nhóm khách hàng khác nhau
để có chiến lược kinh doanh, chăm sóc khách hàng phù
hợp.
    """)  
    st.write("""###### => Problem/ Requirement: Use Machine Learning algorithms in Python for customer segmentation analysis.""")
    st.image("customer_segmentation.jpg")
elif choice == 'Build Project':
    st.subheader("Build Project")
    st.write("##### 1. Some data")
    st.dataframe(df.head(5))
    st.dataframe(df.tail(5))
    st.dataframe(df.describe())