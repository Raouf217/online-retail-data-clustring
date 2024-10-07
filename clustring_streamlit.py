import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram, linkage


# Load the data
@st.cache_data
def load_data():
    return pd.read_csv('./data/clean_data.csv')

@st.cache_data
def load_full_data():
    return pd.read_csv('./data/online_retail.csv')

try:
    customer_data = load_data()
    full_data = load_full_data()
except:
    customer_data = st.file_uploader("Upload Cleaned Data CSV", type="csv")
    full_data = st.file_uploader("Upload Full Data CSV", type="csv")

    if full_data and customer_data:
        customer_data = pd.read_csv(customer_data)
        full_data = pd.read_csv(full_data)
    else:
        st.error("Please upload the required CSV files.")


# Title and data overview
st.title("Customer Segmentation App")
st.write("This app performs customer segmentation based on the retail dataset.")
st.write("This app allows you to explore customer segmentation using KMeans clustering. Use the slider in the sidebar to choose the number of clusters, and view the corresponding visualizations below.")

# Show raw data
st.write("### The Data")
st.dataframe(full_data.head(5))

# Top Selling Products
st.subheader("Top Selling Products")
top_products = full_data.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10).reset_index()
fig_products = px.bar(top_products, x='Description', y='Quantity', title='Top 10 Selling Products')
st.plotly_chart(fig_products)

# Show raw data
st.write("### The Cleaned Data")
st.dataframe(customer_data.head(5))

# Renaming the columns for clarity
customer_data.columns = ['Customer ID', 'Monetary', 'Frequency', 'Recency']

# Visualize customer monetary distribution
st.write("### Customer Monetary Distribution")
fig, ax = plt.subplots()
sns.histplot(customer_data['Monetary'], bins=50, kde=True, ax=ax)
st.pyplot(fig)


# Visualize customer recency distribution
st.write("### Customer Recency Distribution")
fig, ax = plt.subplots()
sns.histplot(customer_data['Recency'], bins=50, kde=True, ax=ax)
st.pyplot(fig)

# Visualize customer frequency distribution
st.write("### Customer Frequency Distribution")
fig, ax = plt.subplots()
sns.histplot(customer_data['Frequency'], bins=50, kde=True, ax=ax)
st.pyplot(fig)


#visualize the correlation between the features
st.write("### Correlation between the features")
fig, ax = plt.subplots()
sns.heatmap(customer_data[['Monetary', 'Frequency', 'Recency']].corr(), annot=True, ax=ax)
st.pyplot(fig)

# visualize the pairplot of the features
st.write("### Pairplot of the features")
fig = sns.pairplot(customer_data[['Monetary', 'Frequency', 'Recency']])
st.pyplot(fig)

# visualize the boxplot of the features
st.write("### Boxplot of the features")
fig, ax = plt.subplots(1, 3, figsize=(20, 6))
sns.boxplot(x='Monetary', data=customer_data, ax=ax[0])
sns.boxplot(x='Frequency', data=customer_data, ax=ax[1])
sns.boxplot(x='Recency', data=customer_data, ax=ax[2])
st.pyplot(fig)


# Scaling the features for clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data[['Monetary', 'Frequency', 'Recency']])

# Sidebar for cluster selection
st.sidebar.title("K-Means Clustering")
num_clusters = st.sidebar.slider("Select number of clusters", 2, 10, 4)

# KMeans Clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(scaled_data)
centroids = kmeans.cluster_centers_

# Show the clustered data
st.write(f"### Clustered Data with {num_clusters} Clusters")
st.dataframe(customer_data.head(10))

# Visualization of Clusters: Scatter plot using Plotly
st.write("### Cluster Visualization")
cluster_colors ={
    0: '#4363d8',  # Blue
    1: '#f58231',  # Orange
    2: '#3cb44b',  # Green
    3: '#e6194b',  # Red
    4: '#ffe119',  # Yellow
    5: '#911eb4',  # Purple
    6: '#46f0f0',  # Cyan
    7: '#f032e6',  # Magenta
    8: '#000000',  # black
    9: '#7f7f7f',  # Gray
}



# Get unique clusters and sort them
unique_clusters = sorted(customer_data['Cluster'].unique())
# Create a list of colors in the correct order
colors_for_plot = [cluster_colors[cluster] for cluster in unique_clusters]

colors = customer_data['Cluster'].map(cluster_colors)
fig = plt.figure(figsize=(8, 8))

ax = fig.add_subplot(projection="3d")

scatter = ax.scatter(customer_data["Monetary"],
                     customer_data["Frequency"],
                     customer_data["Recency"],
                     c=colors,
                     marker='o')

ax.set_xlabel('Monetary')
ax.set_ylabel('Frequency')
ax.set_zlabel('Recency')
ax.set_title('3D Scatter Plot of Customer Data')

# add a legend
i=0
for cluster, color in cluster_colors.items():
    ax.scatter([], [], [], color=color, label=f'Cluster {cluster}')
    i+=1
    if i == num_clusters:
        break
plt.legend()

st.pyplot(fig)



st.write("### 3D Scatter Plot of Customer Data by Cluster")
# Get unique clusters and their corresponding colors
unique_clusters = sorted(customer_data['Cluster'].unique())
colors_for_plot = [cluster_colors[cluster] for cluster in unique_clusters]

# Map colors to the customer data clusters
customer_data['Color'] = customer_data['Cluster'].map(cluster_colors)

# Create 3D scatter plot using Plotly
fig = go.Figure()

# Add the scatter plot for each cluster
for cluster in unique_clusters:
    clustered_data = customer_data[customer_data['Cluster'] == cluster]
    fig.add_trace(
        go.Scatter3d(
            x=clustered_data['Monetary'],
            y=clustered_data['Frequency'],
            z=clustered_data['Recency'],
            mode='markers',
            marker=dict(
                
                color=cluster_colors[cluster],
                opacity=0.8,
            ),
            name=f'Cluster {cluster}'
        )
    )

# Update layout for better visuals
fig.update_layout(
    title="3D Scatter Plot of Customer Data by Cluster",
    scene=dict(
        xaxis_title='Monetary',
        yaxis_title='Frequency',
        zaxis_title='Recency'
    ),
    legend_title="Customer Clusters",
    width=800,
    height=700,
    margin=dict(l=0, r=0, t=50, b=0)
)

# Show the plot in 
st.plotly_chart(fig)


# Additional Visualization: Cluster distribution
st.write("### Cluster Distribution")
fig, ax = plt.subplots()
sns.countplot(x='Cluster', data=customer_data, palette=colors_for_plot, ax=ax)
st.pyplot(fig)


# Visualize recency by cluster
st.write("### Recency by Cluster")
fig, ax = plt.subplots()
sns.boxplot(x='Cluster', y='Recency', data=customer_data, palette=colors_for_plot, ax=ax)
st.pyplot(fig)


# Example dataset creation for demonstration (replace with your data)
data = customer_data.copy()
cluster_counts = data['Cluster'].value_counts().sort_index()


# Create a list of colors based on the cluster indices
cluster_colors = [cluster_colors[i] for i in cluster_counts.index]

cluster_names = [f'Cluster {i}' for i in cluster_counts.index]

# Create the pie chart
pie_chart = go.Pie(
    labels=cluster_names,
    values=cluster_counts,
    textinfo='percent',
    hoverinfo='label+percent+value',
    textposition='inside',
    hole=.4,  # Doughnut style
    marker=dict(colors=cluster_colors, line=dict(color='black', width=1)),
    pull=[0.05] * len(cluster_counts),
    name='Customer Distribution',
    domain=dict(x=[0, 0.45]),
    
    
)

# Create the bar chart
bar_chart = go.Bar(
    x=cluster_names,
    y=cluster_counts,
    text=cluster_counts,
    textposition='outside',
    marker_color=cluster_colors,
    name='Cluster Counts',
    hoverinfo='x+y',
    textfont=dict(color='orange', size=14),    
)

# Create layout
fig = go.Figure()
fig.add_trace(pie_chart)
fig.add_trace(bar_chart)

# Update layout for better visibility and spacing
fig.update_layout(
    title_text='Customer Segmentation: Pie and Bar Charts',
    
    showlegend=False,
    xaxis=dict(title='Clusters', domain=[0.55, 1],
               tickfont=dict(color='orange', size=14)
               ),
    yaxis=dict(title='Number of Customers',),
    xaxis_title_font_color='orange',
    yaxis_title_font_color='orange',
    bargap=0.3,
    width=1000,
    height=600,
    margin=dict(l=50, r=50, t=80, b=50),
    paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
    plot_bgcolor='rgba(0,0,0,0)',
    title_font_size=24,
    title_font_color='orange',
)

# Show the plot
st.plotly_chart(fig)


#pie chart of cluster and monetary
st.write("### Pie Chart of cluster and monetary")
fig, ax = plt.subplots(facecolor='none')
ax.set_facecolor('none')
cluster_monetary = customer_data.groupby('Cluster')['Monetary'].sum().reset_index()
ax.pie(cluster_monetary['Monetary'], labels=cluster_monetary['Cluster'], autopct='%1.1f%%', startangle=90, 
       colors=colors_for_plot, explode=(0, 0.05, 0, 0), shadow=True, textprops={'color': 'black'}, wedgeprops={'edgecolor': 'black'})
ax.axis('equal')
ax.set_title('Cluster Distribution by Monetary')
ax.legend(title='Cluster', loc='upper right')
st.pyplot(fig)


# Dendrogram 
st.write("### Dendrogram Explanation")
st.write("A dendrogram shows the hierarchical relationship between clusters. The height of the lines indicates the distance or dissimilarity between clusters.")
Z = linkage(centroids, method='ward')
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Dendrogram for KMeans Clusters")
plt.xlabel("Cluster Index")
plt.ylabel("Distance")
st.pyplot(plt)


# Calculate average MFR values per cluster
cluster_summary = customer_data.groupby('Cluster').agg({
    'Monetary': 'mean',
    'Frequency': 'mean',
    'Recency': 'mean'
}).reset_index()

# Rename columns for clarity
cluster_summary.columns = ['Cluster', 'Average Monetary', 'Average Frequency', 'Average Recency']
# plot the cluster_summary using plotly
st.write("### Cluster Summary Visualization")
fig = make_subplots(rows=1, cols=3, subplot_titles=("Average Monetary", "Average Frequency", "Average Recency"))
fig.add_trace(go.Bar(x=cluster_summary['Cluster'], y=cluster_summary['Average Monetary'], marker_color=colors_for_plot, name='Average Monetary'), row=1, col=1)
fig.add_trace(go.Bar(x=cluster_summary['Cluster'], y=cluster_summary['Average Frequency'], marker_color=colors_for_plot, name='Average Frequency'), row=1, col=2)
fig.add_trace(go.Bar(x=cluster_summary['Cluster'], y=cluster_summary['Average Recency'], marker_color=colors_for_plot, name='Average Recency'), row=1, col=3)

fig.update_layout(title_text="Cluster Summary Visualization", showlegend=False)
st.plotly_chart(fig)


# User Input Section for New Customer Data
st.subheader("Enter Customer Data")
monetary_input = st.number_input("Monetary Value (Total Spent)", min_value=0.0, value=500.0, step=10.0)
frequency_input = st.number_input("Frequency (Number of Purchases)", min_value=1, value=5, step=1)
recency_input = st.number_input("Recency (Days Since Last Purchase)", min_value=1, value=30, step=1)


# Button to submit the input data
if st.button("Predict Customer Cluster"):
    # Create DataFrame from the user input
    user_data = pd.DataFrame([[monetary_input, frequency_input, recency_input]], 
                             columns=['Monetary', 'Frequency', 'Recency'])
    
    # Scale the input data
    user_data_scaled = scaler.transform(user_data)
    
    # Predict the cluster for the new customer
    user_cluster = kmeans.predict(user_data_scaled)[0]
    
    # Display the result
    st.write(f"### The customer belongs to Cluster {user_cluster}")
    
    # Optional: Show some characteristics of the predicted cluster
    st.write(f"#### Cluster {user_cluster} Overview")
    cluster_overview = customer_data[customer_data['Cluster'] == user_cluster]
    st.write(cluster_overview.describe())