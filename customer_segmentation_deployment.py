import pickle
import streamlit as st

import numpy as np
import datetime as dt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import squarify
import plotly.figure_factory as ff
import plotly.express as px
from datetime import datetime
import os
# !pip install openpyxl

from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# 1. Read data
columns = ['CustomerID', 'InvoiceDate', 'Quantity', 'TotalAmountSpent']
data = pd.read_csv("CDNOW_master.txt", header = None, delim_whitespace=True, names = columns)

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
if (data.isnull().sum().any() != 0):
    data_not_null = data.dropna()
    data_not_null.reset_index(drop=True, inplace=True)
else:
    data_not_null = data
## Dropping negative values
df = data_not_null[(data_not_null['Quantity']>0) & (data_not_null['TotalAmountSpent']>0)] 
## Removing duplicates
df = df.drop_duplicates(subset = df.columns)
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

## Elbow method
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,20))
visualizer.fit(x_scaled)  
visualizer.show(outpath="visualizer.png")

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

#6. Validation
identified_clusters = kmeans_scaled.fit_predict(rfm1)
clusters_scaled = rfm1.copy()
clusters_scaled['cluster_pred']=kmeans_scaled.fit_predict(x_scaled)
sil_score = silhouette_score(x_scaled, kmeans_scaled.labels_, metric='euclidean')
model = KMeans(4)
silhouette_visualizer = SilhouetteVisualizer(model)
silhouette_visualizer.fit(x_scaled)
visualizer.poof(outpath="SilhouetteVisualizer.png")

#7. Cluster Profiling
rfm1['cluster']= clusters_scaled['cluster_pred']

rfm_cluster_profiling = rfm1.groupby('cluster').agg({
    'recency' : ['mean','min','max'],
    'frequency' : ['mean','min','max'],
    'monetary' : ['mean','min','max','count']
})

#8. Calculate mean values for each segment
# Calculate average values for each RFM_Level, and return a size of each segment 
rfm_agg_kmeans = rfm1.groupby('cluster').agg({
    'recency': 'mean',
    'frequency': 'mean',
    'monetary': ['mean', 'count']}).round(0)

rfm_agg_kmeans.columns = rfm_agg_kmeans.columns.droplevel()
rfm_agg_kmeans.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
rfm_agg_kmeans['Percent'] = round((rfm_agg_kmeans['Count']/rfm_agg_kmeans.Count.sum())*100, 2)

# Reset the index
rfm_agg_kmeans = rfm_agg_kmeans.reset_index()

# Change thr Cluster Columns Datatype into discrete values
rfm_agg_kmeans['cluster'] = 'Cluster '+ rfm_agg_kmeans['cluster'].astype('str')

# Naming for each cluster
rfm_agg_kmeans_2 = rfm_agg_kmeans.copy()
customers_segments = ['At risk to lost', 'Royal customers', 'Recent customers', 'Cooling down']
rfm_agg_kmeans_2 = rfm_agg_kmeans_2.rename(columns = {'cluster' : 'customers_segments'})
rfm_agg_kmeans_2.customers_segments = customers_segments

#9. GUI
menu = ["Business Objective", "Build Project"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Business Objective':    
    st.subheader("Business Objective")
    st.write("""
    ###### Xây dựng hệ thống phân cụm khách hàng dựa trên các thông tin do công ty cung cấp từ đó có thể giúp công ty xác định các nhóm khách hàng khác nhau để có chiến lược kinh doanh, chăm sóc khách hàng phù hợp.
    """)  
    st.write("""###### => Problem/ Requirement: Use Machine Learning algorithms in Python for customer segmentation analysis.""")
    st.image("customer_segmentation.jpg")

elif choice == 'Build Project':
    st.subheader("Build Project")
    st.write("##### 1. Some data")
    st.dataframe(data.head(5))
    st.dataframe(data.tail(5))
    st.write("##### 2. Describe data before preprocessing")
    st.dataframe(data.describe())
    st.write("##### 3. Some unique values")
    invoicedate_unique = data.InvoiceDate.nunique()
    customer_unique = data.CustomerID.nunique()
    st.write("Có tổng cộng ", invoicedate_unique, " ngày bán hàng được thống kê trong bộ dữ liệu.")
    st.write("Có ", customer_unique, " ID khách hàng trong bộ dữ liệu.")
    st.write("##### 4. Corelation Check")
    corrData = data.corr()
    fig1 = sns.heatmap(corrData, 
        xticklabels=corrData.columns,
        yticklabels=corrData.columns, cmap='coolwarm_r')    
    st.pyplot(fig1.figure)
    st.write("##### 5. Dropping negative and duplicate data")
    st.write("Data after dropping")
    st.dataframe(df.head(5))
    st.write("Describing data")
    st.dataframe(df.describe())
    st.write("##### 6. RFM Analysis")
    st.write("Data after RFM analysis")
    st.dataframe(rfm.head(5))
    st.write("Visualization RFM data")
    fig2 = plt.figure(figsize = (8,10))
    plt.subplot(3, 1, 1);
    sns.distplot(rfm['recency']);
    plt.subplot(3, 1, 2);
    sns.distplot(rfm['frequency']);
    plt.subplot(3, 1, 3);
    sns.distplot(rfm['monetary']);
    plt.subplots_adjust(wspace=.25, hspace=.25)

    # save image, display it, and delete after usage.
    plt.savefig('x',dpi=400)
    st.image('x.png')
    os.remove('x.png')

    # build model
    st.write("##### 7. Build model...")
    st.write("Transforming the data")
    st.dataframe(x_scaled[5])
    st.write("Elbow method")
    st.image("visualizer.png")
    st.write("Elbow method choose k = 4")

    # validation
    st.write("##### 8. Validation")
    st.code(clusters_scaled.groupby(['cluster_pred']).count())
    fig3, ax = plt.subplots()       
    ax = sns.countplot(x="cluster_pred", data=clusters_scaled)
    st.pyplot(fig3)
    
    ## Sihouette score
    sil_score = silhouette_score(x_scaled, kmeans_scaled.labels_, metric='euclidean')
    st.code('Silhouette Score: %.3f' % sil_score) 
    st.image("SilhouetteVisualizer.png")

    # Clustering Profiling
    st.write("##### 9. Clustering Profiling")
    st.code(rfm_cluster_profiling)
    st.write("Some data after RFM + K-Means Clustering")
    st.dataframe(rfm1.head(5))

    # Visualizing the clusters
    st.write("##### 10. Visualizing the clusters")
    st.write("Average values for each RFM_Cluster")
    st.dataframe(rfm_agg_kmeans)

    st.write("Scatter plot 3D")
    fig4 = px.scatter_3d(
        clusters_scaled,
        x="recency",
        y="frequency",
        z="monetary",
        color="cluster_pred")
    st.plotly_chart(fig4, theme="streamlit", use_container_width=True)

    fig5 = px.scatter_3d(
        rfm_agg_kmeans,
        x="RecencyMean",
        y="FrequencyMean",
        z="MonetaryMean",
        color="cluster",
        hover_name="cluster",
        opacity=0.3)
    fig5.update_traces(marker=dict(size=20),selector=dict(mode='markers'))
    st.plotly_chart(fig5, theme="streamlit", use_container_width=True)

    st.write("Tree Map")
    st.write("According to Average values for each RFM_Cluster, I segments the customers to 4 cluster: at risk to lost, royal customers, recent custormers, cooling down")
    st.dataframe(rfm_agg_kmeans_2)
    st.image('Unsupervised Segments.png')
    


