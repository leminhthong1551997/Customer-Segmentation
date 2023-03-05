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
import scipy

# !pip install openpyxl

from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# GUI
menu = ["Business Objective", "Build Project", "New Prediction"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Business Objective':    
    st.title("Data Science Project")
    st.write("## Phân cụm khách hàng")
    st.subheader("Mục tiêu dự án:")
    st.write("""
    ###### Xây dựng hệ thống phân cụm khách hàng dựa trên các thông tin do công ty cung cấp từ đó có thể giúp công ty xác định các nhóm khách hàng khác nhau để có chiến lược kinh doanh, chăm sóc khách hàng phù hợp.
    """)  
    st.image("customer_segmentation.jpg")
    st.subheader("Yêu cầu: ")
    st.write("Sử dụng các thuật toán phân cụm thuộc Machine Learning hoặc Big Data kết hợp với phương pháp RFM để thực hiện phân cụm khách hàng.")
    st.subheader("Giới thiệu về phân cụm khách hàng dựa theo RFM")
    st.write("R (Recency): Khoảng thời gian mà khách hàng mua hàng gần đây nhất.")
    st.write("F (Frequency): Tần suất mua hàng của khách hàng.")
    st.write("M (Monetary Value): Giá trị mỗi lần mua hàng là gì. Chỉ số này dùng để tính toán được giá trị về vật chất mà doanh nghiệp có được mỗi khi khách hàng sử dụng dịch vụ.")
    st.image("Incontent_image.png")
    st.subheader("Thuật toán phân cụm được sử dụng trong dự án: K Means Clusterings")

elif choice == 'Build Project':
    # 1. Read data
    columns = ['CustomerID', 'InvoiceDate', 'Quantity', 'TotalAmountSpent']
    data = pd.read_csv("CDNOW_master.txt", header = None, delim_whitespace=True, names = columns)

    # Upload file
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'txt'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, header = None, delim_whitespace=True, names = columns)
        data.to_csv("CD_new.csv", index = False)

    st.subheader('Data preprocessing and RFM analysis')
    st.write("Some Data")
    st.dataframe(data.head())
    st.code(data.describe())

    # 2. Data pre-processing
    def preprocessing_data(df):
        # Dropping null data
        df_isnull = df.isnull().sum().sort_values(ascending=False)
        df_dropnull = df.dropna()
        df_dropnull = df_dropnull.reset_index(drop=True)
        
        # Dropping negative data
        df_positive = df_dropnull[(df_dropnull['Quantity']>0) & (df_dropnull['TotalAmountSpent']>0)]
        
        # Dropping duplicates
        df = df_positive.drop_duplicates(subset = df_positive.columns)
        
        return df
    df_after_preprocessing = preprocessing_data(data)
    st.write("Data after preprocessing")
    st.dataframe(df_after_preprocessing.head())
    st.code(df_after_preprocessing.describe())

    # 3. RFM analysis
    def RFM_ananysis(df):
        string_to_date = lambda x : datetime.strptime(x, "%Y%m%d").date()

        # Convert InvoiceDate from object to datetime format
        df['InvoiceDate'] = df['InvoiceDate'].astype('str')
        df['InvoiceDate'] = df['InvoiceDate'].apply(string_to_date)
        df['InvoiceDate'] = df['InvoiceDate'].astype('datetime64[ns]')

        # Drop NA values
        df = df.dropna()
        # RFM
        # Convert string to date, get max date of dataframe
        max_date = df['InvoiceDate'].max().date()

        Recency = lambda x : (max_date - x.max().date()).days
        Frequency  = lambda x: len(x.unique())
        Monetary = lambda x : round(sum(x), 2)

        rfm = df.groupby('CustomerID').agg({'InvoiceDate': Recency,
                                                'Quantity': Frequency,  
                                                'TotalAmountSpent': Monetary })
        
        # Changing the column names
        rfm.columns = ['recency', 'frequency', 'monetary']
        rfm = rfm[rfm["monetary"] > 0]
        
        return(rfm)
    df_rfm = RFM_ananysis(df_after_preprocessing)
    st.write("Data after RFM analysis")
    st.dataframe(df_rfm.head())
    st.code(df_rfm.describe())

    ## Loại bỏ các outliers
    column = [f for f in df_rfm.columns]
    for n in column:
        n_mean = df_rfm[n].mean()
        n_median = df_rfm[n].median()
        n_iqr = scipy.stats.iqr(df_rfm[n])
        
        ### Chuyển các outliers thành giá trị null
        df_rfm[n] = df_rfm[df_rfm[n].between(df_rfm[n].quantile(0.25) - 1.5*n_iqr, df_rfm[n].quantile(0.75) + 1.5*n_iqr)][n]
        
        ### Loại bỏ các giá trị null
        df_rfm = df_rfm.dropna()


    # 4. Load model kmeans and apply model
    ## Load model
    with open('model_kmeans.pkl', 'rb') as file:  
        model_kmeans = pickle.load(file)

    scaler = StandardScaler()
    x_scaled = scaler.fit(df_rfm)
    x_scaled = scaler.fit_transform(df_rfm)

    ## Applying model
    st.subheader("Applying model")
    st.image("visualizer.png")
    st.write("Elbow method choose k = 4")

    df_rfm['cluster_pred']=model_kmeans.fit_predict(x_scaled)
    st.write("Clustering profiling")
    st.code(df_rfm.groupby(['cluster_pred']).count())

    fig = plt.figure(figsize=(10, 4))
    sns.countplot(x="cluster_pred", data=df_rfm)
    st.plotly_chart(fig)

    # Calculate average values for each RFM_Level, and return a size of each segment 
    rfm_agg_kmeans = df_rfm.groupby('cluster_pred').agg({
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': ['mean', 'count']}).round(0)

    rfm_agg_kmeans.columns = rfm_agg_kmeans.columns.droplevel()
    rfm_agg_kmeans.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
    rfm_agg_kmeans['Percent'] = round((rfm_agg_kmeans['Count']/rfm_agg_kmeans.Count.sum())*100, 2)

    # Reset the index
    rfm_agg_kmeans = rfm_agg_kmeans.reset_index()
    rfm_agg_kmeans['cluster_pred'] = 'Cluster '+ rfm_agg_kmeans['cluster_pred'].astype('str')

    # Change the Cluster Columns Datatype into discrete values

    # customers_segments = ['Cooling down', 'Royal customers', 'At risk to lost', 'Recent customers']
    # rfm_agg_kmeans = rfm_agg_kmeans.rename(columns = {'cluster_pred' : 'customers_segments'})
    # rfm_agg_kmeans.customers_segments = customers_segments
    # st.write("Change the name of each cluster")
    st.dataframe(rfm_agg_kmeans)

    st.write("Scatter plot 3D")
    fig = px.scatter_3d(
                    df_rfm,
                    x="recency",
                    y="frequency",
                    z="monetary",
                    color="cluster_pred")
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    fig = px.scatter_3d(
                    rfm_agg_kmeans,
                    x="RecencyMean",
                    y="FrequencyMean",
                    z="MonetaryMean",
                    color="cluster_pred",
                    hover_name="cluster_pred",
                    opacity=0.3)
    fig.update_traces(marker=dict(size=20),selector=dict(mode='markers'))
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    st.write("Tree Map")
    st.write("According to Average values for each RFM_Cluster, I segments the customers to 4 cluster: at risk to lost, royal customers, recent custormers, cooling down")
    #Create our plot and resize it.
    fig = plt.gcf()
    ax = fig.add_subplot()
    fig.set_size_inches(14, 10)

    colors_dict = {'Cluster0':'red','Cluster1':'royalblue', 'Cluster2':'cyan',
                'Cluster3':'purple'}

    squarify.plot(sizes=rfm_agg_kmeans['Count'],
                text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
                color=colors_dict.values(),
                label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg_kmeans.iloc[i])
                        for i in range(0, len(rfm_agg_kmeans))], alpha=0.5 )


    plt.title("Customers Segments K-Means",fontsize=26,fontweight="bold")
    plt.axis('off')
    st.pyplot(fig)

elif choice == 'New Prediction':
    st.subheader("Select data")
    flag = False
    lines = None
    type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
    if type=="Upload":
        format = st.radio("Upload data", options=("With RFM Format", "Without RFM Format"))
        if format == "Without RFM Format":
            # Upload file
            uploaded_file_predict= st.file_uploader("Choose a file", type=['txt', 'csv'])
            if uploaded_file_predict is not None:
                columns = ['CustomerID', 'InvoiceDate', 'Quantity', 'TotalAmountSpent']
                df_uploaded = pd.read_csv(uploaded_file_predict, header=None, delim_whitespace=True, names = columns)
                data = df_uploaded.copy()
                st.subheader('Data preprocessing and RFM analysis')
                st.write("Some Data")
                st.dataframe(data.head())
                st.code(data.describe())

                # 2. Data pre-processing
                def preprocessing_data(df):
                    # Dropping null data
                    df_isnull = df.isnull().sum().sort_values(ascending=False)
                    df_dropnull = df.dropna()
                    df_dropnull = df_dropnull.reset_index(drop=True)
                    
                    # Dropping negative data
                    df_positive = df_dropnull[(df_dropnull['Quantity']>0) & (df_dropnull['TotalAmountSpent']>0)]
                    
                    # Dropping duplicates
                    df = df_positive.drop_duplicates(subset = df_positive.columns)
                    
                    return df
                df_after_preprocessing = preprocessing_data(data)
                st.write("Data after preprocessing")
                st.dataframe(df_after_preprocessing.head())
                st.code(df_after_preprocessing.describe())

                # 3. RFM analysis
                def RFM_ananysis(df):
                    string_to_date = lambda x : datetime.strptime(x, "%Y%m%d").date()

                    # Convert InvoiceDate from object to datetime format
                    df['InvoiceDate'] = df['InvoiceDate'].astype('str')
                    df['InvoiceDate'] = df['InvoiceDate'].apply(string_to_date)
                    df['InvoiceDate'] = df['InvoiceDate'].astype('datetime64[ns]')

                    # Drop NA values
                    df = df.dropna()
                    # RFM
                    # Convert string to date, get max date of dataframe
                    max_date = df['InvoiceDate'].max().date()

                    Recency = lambda x : (max_date - x.max().date()).days
                    Frequency  = lambda x: len(x.unique())
                    Monetary = lambda x : round(sum(x), 2)

                    rfm = df.groupby('CustomerID').agg({'InvoiceDate': Recency,
                                                            'Quantity': Frequency,  
                                                            'TotalAmountSpent': Monetary })
                    
                    # Changing the column names
                    rfm.columns = ['recency', 'frequency', 'monetary']
                    rfm = rfm[rfm["monetary"] > 0]
                    
                    return(rfm)
                df_rfm = RFM_ananysis(df_after_preprocessing)
                st.write("Data after RFM analysis")
                st.dataframe(df_rfm.head())
                st.code(df_rfm.describe())

                ## Loại bỏ các outliers
                column = [f for f in df_rfm.columns]
                for n in column:
                    n_mean = df_rfm[n].mean()
                    n_median = df_rfm[n].median()
                    n_iqr = scipy.stats.iqr(df_rfm[n])
                    
                    ### Chuyển các outliers thành giá trị null
                    df_rfm[n] = df_rfm[df_rfm[n].between(df_rfm[n].quantile(0.25) - 1.5*n_iqr, df_rfm[n].quantile(0.75) + 1.5*n_iqr)][n]
                    
                    ### Loại bỏ các giá trị null
                    df_rfm = df_rfm.dropna()


                # 4. Load model kmeans and apply model
                ## Load model
                with open('model_kmeans.pkl', 'rb') as file:  
                    model_kmeans = pickle.load(file)

                scaler = StandardScaler()
                x_scaled = scaler.fit(df_rfm)
                x_scaled = scaler.fit_transform(df_rfm)

                ## Applying model
                st.subheader("Applying model")
                df_rfm['cluster_pred']=model_kmeans.predict(x_scaled)
                st.write("Clustering profiling")
                st.code(df_rfm.groupby(['cluster_pred']).count())

                fig = plt.figure(figsize=(10, 4))
                sns.countplot(x="cluster_pred", data=df_rfm)
                st.plotly_chart(fig)

                # Calculate average values for each RFM_Level, and return a size of each segment 
                rfm_agg_kmeans = df_rfm.groupby('cluster_pred').agg({
                    'recency': 'mean',
                    'frequency': 'mean',
                    'monetary': ['mean', 'count']}).round(0)

                rfm_agg_kmeans.columns = rfm_agg_kmeans.columns.droplevel()
                rfm_agg_kmeans.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
                rfm_agg_kmeans['Percent'] = round((rfm_agg_kmeans['Count']/rfm_agg_kmeans.Count.sum())*100, 2)

                # Reset the index
                rfm_agg_kmeans = rfm_agg_kmeans.reset_index()
                rfm_agg_kmeans['cluster_pred'] = 'Cluster '+ rfm_agg_kmeans['cluster_pred'].astype('str')

                # Change the Cluster Columns Datatype into discrete values
                #customers_segments = ['Royal customers', 'At risk to lost', 'Cooling down', 'Recent customers']
                #rfm_agg_kmeans = rfm_agg_kmeans.rename(columns = {'cluster_pred' : 'customers_segments'})
                #rfm_agg_kmeans.customers_segments = customers_segments
                #st.write("Change the name of each cluster")
                st.dataframe(rfm_agg_kmeans)

                st.write("Scatter plot 3D")
                fig = px.scatter_3d(
                                df_rfm,
                                x="recency",
                                y="frequency",
                                z="monetary",
                                color="cluster_pred")
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)

                fig = px.scatter_3d(
                                rfm_agg_kmeans,
                                x="RecencyMean",
                                y="FrequencyMean",
                                z="MonetaryMean",
                                color="cluster_pred",
                                hover_name="cluster_pred",
                                opacity=0.3)
                fig.update_traces(marker=dict(size=20),selector=dict(mode='markers'))
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)

                st.write("Tree Map")
                st.write("According to Average values for each RFM_Cluster, I segments the customers to 4 cluster: at risk to lost, royal customers, recent custormers, cooling down")
                #Create our plot and resize it.
                fig = plt.gcf()
                ax = fig.add_subplot()
                fig.set_size_inches(14, 10)

                colors_dict = {'Cluster0':'red','Cluster1':'royalblue', 'Cluster2':'cyan',
                            'Cluster3':'purple'}

                squarify.plot(sizes=rfm_agg_kmeans['Count'],
                            text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
                            color=colors_dict.values(),
                            label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg_kmeans.iloc[i])
                                    for i in range(0, len(rfm_agg_kmeans))], alpha=0.5 )


                plt.title("Customers Segments K-Means",fontsize=26,fontweight="bold")
                plt.axis('off')
                st.pyplot(fig)

        if format == "With RFM Format":
            # Upload file
            uploaded_file_predict= st.file_uploader("Choose a file", type=['xlsx', 'csv'])
            if uploaded_file_predict is not None:
                df_uploaded = pd.read_csv(uploaded_file_predict)
                data = df_uploaded.copy()
                st.subheader('Data preprocessing')
                st.write("Some Data")
                st.dataframe(data.head())
                st.code(data.describe())

                # 2. Data pre-processing
                def preprocessing_data(df):
                    # Dropping null data
                    df_isnull = df.isnull().sum().sort_values(ascending=False)
                    df_dropnull = df.dropna()
                    df_dropnull = df_dropnull.reset_index(drop=True)
                    
                    # Dropping duplicates
                    df = df_dropnull.drop_duplicates(subset = df_dropnull.columns)
                    
                    return df
                df_after_preprocessing = preprocessing_data(data)
                st.write("Data after preprocessing")
                st.dataframe(df_after_preprocessing.head())
                st.code(df_after_preprocessing.describe())

                # 3. RFM analysis
                def RFM_ananysis(df):

                    rfm = df.copy()
                    # Changing the column names
                    rfm.columns = ['recency', 'frequency', 'monetary']
                    rfm = rfm[rfm["monetary"] > 0]
                    
                    return(rfm)
                df_rfm = RFM_ananysis(df_after_preprocessing)
                st.write("Data after RFM analysis")
                st.dataframe(df_rfm.head())
                st.code(df_rfm.describe())

                ## Loại bỏ các outliers
                column = [f for f in df_rfm.columns]
                for n in column:
                    n_mean = df_rfm[n].mean()
                    n_median = df_rfm[n].median()
                    n_iqr = scipy.stats.iqr(df_rfm[n])
                    
                    ### Chuyển các outliers thành giá trị null
                    df_rfm[n] = df_rfm[df_rfm[n].between(df_rfm[n].quantile(0.25) - 1.5*n_iqr, df_rfm[n].quantile(0.75) + 1.5*n_iqr)][n]
                    
                    ### Loại bỏ các giá trị null
                    df_rfm = df_rfm.dropna()


                # 4. Load model kmeans and apply model
                ## Load model
                with open('model_kmeans.pkl', 'rb') as file:  
                    model_kmeans = pickle.load(file)

                scaler = StandardScaler()
                x_scaled = scaler.fit(df_rfm)
                x_scaled = scaler.fit_transform(df_rfm)

                st.write("Applying model")

                ## Applying model
                st.subheader("Applying model")
                df_rfm['cluster_pred']=model_kmeans.predict(x_scaled)
                st.write("Clustering profiling")
                st.code(df_rfm.groupby(['cluster_pred']).count())

                fig = plt.figure(figsize=(10, 4))
                sns.countplot(x="cluster_pred", data=df_rfm)
                st.plotly_chart(fig)

                # Calculate average values for each RFM_Level, and return a size of each segment 
                rfm_agg_kmeans = df_rfm.groupby('cluster_pred').agg({
                    'recency': 'mean',
                    'frequency': 'mean',
                    'monetary': ['mean', 'count']}).round(0)

                rfm_agg_kmeans.columns = rfm_agg_kmeans.columns.droplevel()
                rfm_agg_kmeans.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
                rfm_agg_kmeans['Percent'] = round((rfm_agg_kmeans['Count']/rfm_agg_kmeans.Count.sum())*100, 2)

                # Reset the index
                rfm_agg_kmeans = rfm_agg_kmeans.reset_index()
                rfm_agg_kmeans['cluster_pred'] = 'Cluster '+ rfm_agg_kmeans['cluster_pred'].astype('str')

                # Change the Cluster Columns Datatype into discrete values
                #customers_segments = ['Royal customers', 'At risk to lost', 'Cooling down', 'Recent customers']
                #rfm_agg_kmeans = rfm_agg_kmeans.rename(columns = {'cluster_pred' : 'customers_segments'})
                #rfm_agg_kmeans.customers_segments = customers_segments
                #st.write("Change the name of each cluster")
                st.dataframe(rfm_agg_kmeans)

                st.write("Scatter plot 3D")
                fig = px.scatter_3d(
                                df_rfm,
                                x="recency",
                                y="frequency",
                                z="monetary",
                                color="cluster_pred")
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)

                fig = px.scatter_3d(
                                rfm_agg_kmeans,
                                x="RecencyMean",
                                y="FrequencyMean",
                                z="MonetaryMean",
                                color="cluster_pred",
                                hover_name="cluster_pred",
                                opacity=0.3)
                fig.update_traces(marker=dict(size=20),selector=dict(mode='markers'))
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)

                st.write("Tree Map")
                st.write("According to Average values for each RFM_Cluster, I segments the customers to 4 cluster: at risk to lost, royal customers, recent custormers, cooling down")
                #Create our plot and resize it.
                fig = plt.gcf()
                ax = fig.add_subplot()
                fig.set_size_inches(14, 10)

                colors_dict = {'Cluster0':'red','Cluster1':'royalblue', 'Cluster2':'cyan',
                            'Cluster3':'purple'}

                squarify.plot(sizes=rfm_agg_kmeans['Count'],
                            text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
                            color=colors_dict.values(),
                            label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg_kmeans.iloc[i])
                                    for i in range(0, len(rfm_agg_kmeans))], alpha=0.5 )


                plt.title("Customers Segments K-Means",fontsize=26,fontweight="bold")
                plt.axis('off')
                st.pyplot(fig)

    elif type=="Input":
    
        R = st.text_area(label="Input your Recency:")
        F = st.text_area(label="Input your Frequency:")
        M = st.text_area(label="Input your Monetary:")
        data = pd.DataFrame({'Recency': [float(R)],
                             'Frequency': [float(F)],
                             'Monetary': [float(M)]})
        st.write('### Data Input:')
        st.dataframe(data)

        with open('model_kmeans.pkl', 'rb') as file:  
            model_kmeans = pickle.load(file)
        
        data['Result'] = model_kmeans.predict(data)
        data['Result'] = 'Cluster '+ data['Result'].astype('str')
        st.write('### Result')
        st.dataframe(data)