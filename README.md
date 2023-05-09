# CryptoClustering

Our goal for this assignment is to integrate our newly developed unsupervised machine learning theory and Python to predict if 
cryptocurrencies are affected by 24-hour or 7-day price changes. To achieve this, we start by reading in `crypto_market_data.csv` and 
setting the index as `coin_id`. 

<img width="729" alt="Reading in DataFrame" src="https://github.com/acosta109/CryptoClustering/assets/119609975/be537eec-8a33-4528-8da6-029d46141c42">


Before we perform any anaylsis on the cryptocurrency data, we want to prepare the data with `StandardScalar()` and `fit_transform` to ensure 
our anaylsis is accuarate. 

<img width="1075" alt="Data Preperation" src="https://github.com/acosta109/CryptoClustering/assets/119609975/d4e03f05-7fb0-4940-b0b2-511ed5966452">
<img width="537" alt="Scaled Data" src="https://github.com/acosta109/CryptoClustering/assets/119609975/ad6b5561-5124-491c-906a-ab2d4dd17762">


Now, we want to want calculate the elbow curve for `df_market_data_scaled` to determine the best `k-value`. From the chart below, we 
determined `k=4` is optimal.

![elbow graph 1](https://github.com/acosta109/CryptoClustering/assets/119609975/2dae30af-b97e-427a-beca-9c205f97dbdf)


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

![scaled scatterplot](https://github.com/acosta109/CryptoClustering/assets/119609975/b2d858a2-eec7-4e00-bc7d-1ed4b2d393d6)


Next, we want to re-evaluate the data with Principal Component Analysis with `n_compenents=3`. We find 89.4% of the total explained variance 
is caputred by the first three principal components. We perform the same analysis on `df_pca` and discover `k=4` is the optimal number of 
clusters for analysis. This is consistent with our original analysis, and you can view our second elbow chart below. 

![eblow graph 2](https://github.com/acosta109/CryptoClustering/assets/119609975/28bf4966-e7ca-4b8c-b86e-8bed4dd0871d)


We perform the same procedure as before to produce the following scatterplot chart. 

```
df_pca_predictions.hvplot.scatter(
    x="PCA1",
    y="PCA2", 
    by='pca_clusters',
    hover_cols=['coin_id']

)
```
![pca scatterplot](https://github.com/acosta109/CryptoClustering/assets/119609975/a14b6546-61ae-4a84-bb5d-d03e6baa7da8)


Since we have all the charts necessary for analysis, we created two composite charts to analyse our cryptocurrency data. We discover the 
elbow charts have similar behaviour, but the vertical shift occurring when we perform the PCA analysis. Similarly, the composite scatterplot 
chart shows both analysises have similar behaviour there's a shift. 



