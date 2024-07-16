# Market Basket Analysis

## Dataset

Mall Customer Segmentation Data is a Kaggle dataset that has been used in this project for the purpose of grouping customers of a retail store based on their purchase history.

The dataset comprises of basic information of 200 users- 
- Customer ID: Unique ID assigned to the customer
- Gender: Gender of the customer
- Age: Age of the customer
- Annual Income (k$): Annual Income of the customer
- Spending Score (1-100): Score assigned by the mall based on customer behavior and spending nature

## Preprocessing

Since the 'Gender' feature is categorical, it is encoded into numerical values using LabelEncoder. Histograms, Boxplots and Scatterplots are used to understand the patterns and trends in the numerical variables. The data is then standardized using StandardScaler to ensure that all features contribute equally to the clustering process, preventing bias due to differing feature scales. This preprocessing step is crucial for improving the performance and accuracy of the clustering algorithm.

## K Means Clustering

The preprocessed data is used to perform K-means clustering to group customers based on their purchase history. The number of clusters is determined using the Elbow method, which involves plotting the inertia for a range of cluster numbers and identifying the point where the inertia starts to decrease more slowly. The elbow plot indicates that the optimal number of clusters is 5.

After selecting the optimal number of clusters, the K-means algorithm is applied to segment the customers into distinct groups. The clustering results are evaluated using the silhouette score, which measures how similar an object is to its own cluster compared to other clusters. The silhouette score for 5 clusters in this dataset is 0.55. A new column called cluster is appended to the dataframe to indicate the cluster to which each data point belongs.
