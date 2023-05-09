# CryptoClustering

Our goal for this assignment is to integrate our newly developed unsupervised machine learning theory and Python to predict if 
cryptocurrencies are affected by 24-hour or 7-day price changes. To achieve this, we start by reading in `crypto_market_data.csv` and 
setting the index as `coin_id`. 

Insert screenshot. 

Before we perform any anaylsis on the cryptocurrency data, we want to prepare the data with `StandardScalar()` and `fit_transform` to ensure 
our anaylsis is accuarate. 

Insert screenshots.

Now, we want to want calculate the elbow curve for `df_market_data_scaled` to determine the best `k-value`. From the chart below, we 
determined `k=4` is optimal.

Insert elbow 1

Now that we discovered the optimal `k-value` we want to create our model, then fit and predict our scaled values to it. We accomplish this
with the following code. 

```
# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=4)
```

```
# Fit the K-Means model using the scaled data
model.fit(df_market_data_scaled)
```

```
# Predict the clusters to group the cryptocurrencies using the scaled data
coin_clusters = model.predict(df_market_data_scaled)
```

Ultimately, we add the newly created `coin clusters` to a copied data frame and display the following scatterplot chart using the following code. 

```
df_market_data_scaled_predictions.hvplot.scatter(
    x="price_change_percentage_24h",
    y="price_change_percentage_7d", 
    by='coin_clusters',
    hover_cols=['coin_id']

)

```

Insert scatter 1

Next, we want to re-evaluate the data with Principal Component Analysis with `n_compenents=3`. We find 89.4% of the total explained variance 
is caputred by the first three principal components. We perform the same analysis on `df_pca` and discover `k=4` is the optimal number of 
clusters for analysis. This is consistent with our original analysis, and you can view our second elbow chart below. 

Insert eblow 2

We perform the same procedure as before to produce the following scatterplot chart. 

```
df_pca_predictions.hvplot.scatter(
    x="PCA1",
    y="PCA2", 
    by='pca_clusters',
    hover_cols=['coin_id']

)
```

Since we have all the charts necessary for analysis, we create two composite charts to analyse our cryptocurrency data. We discover the 
elbow charts have similar behaviour, but the vertical shift occurring when we perform the PCA analysis. Similarly, the composite scatterplot 
chart shows both analysises have similar behaviour there's a shift. 

Insert charts. 