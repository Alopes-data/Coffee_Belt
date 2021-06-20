# Coffee_Belt

### Introduction
Using a mostly numerical based dataset from the International Coffee Organization (ICO) which contains information on coffee consumption and trade in the years of 1998 – 2018 with years as variables(columns) (EX., 1995, 1996, 1997) separated by countries (rows). The four datasets are the domestic consumption by all exporting countries, imports by selected importing countries, exports of all forms of coffee by all exporting countries, total production by all exporting countries. The original proposal wanted to analyze those from the coffee belt from those not though high import, low export and similarity groups but we were not able to proceed with this exact methodology and needed to make a pivot within our research. 

We started by using the domestic consumption dataset to determine if a country was a coffee belt country or not. We needed to add the coffee belt responses to the dataset as well as change or remove any non-numerical factors. The idea was to use K means to clusters our groups followed by a support vector machine and evaluate how well these groups are at predicting if a country is in the coffee belt of not.  We started by using and elbow and silhouette plots to determine the number of clusters and chose to use 3 clusters. However, when continuing with the K means model into our support vector we noticed flawed results mainly due to the dataset not being large enough. This was the main issue with the proposed plan versus the actual project. Due to the dataset being too small we were not able to make the predictions in the way that we wished. We continued by pivoting to a One hot encoded model using the Inter-quartile range values to find the top and bottom countries in imports, exports, and total production. 


![image](https://user-images.githubusercontent.com/58121111/122681787-14efff00-d1c4-11eb-9247-2b7d4951fe74.png)
 
 
![image](https://user-images.githubusercontent.com/58121111/122681810-276a3880-d1c4-11eb-92dc-a9fdf2267475.png)


![image](https://user-images.githubusercontent.com/58121111/122681836-42d54380-d1c4-11eb-9d69-ff4d827665ce.png)
 

To achieve similar results as wanted from the k means model we averaged all the years of data and used an IQR to find which countries sat on the upper and lower ends of the deviation. These variables received a 1 as a TRUE variable in our final dataset showing the averaged numbers as well as encoded factors for High/low in imports, exports, and total production.  We took this dataset and used bootstrap resampling with replacement to replicate a dataset of 1000 observations from this final created dataset. This is the dataset we use for our XGboost model in order to deal with the lack observations initially lacking in the data.

Our first XGboost model produced an error rate of .03 with a 20 iterations and a depth of 3. We tuned this model by increasing the number of iterations, scaling the imbalanced classes,  adding an early stopping point, and adding a regularization term producing and error rate of 0.018. We plotted the results down in Figure 4. The first leaf that appear has an log of -3.0164 which can be converted to around 0.046 as the first leaf representing almost 5 percent of our data. We next ran an importance matrix to evaluate how each variable is contributing to the model, shown in Figure 5. It appears the amount of exports if the biggest determining factor when determining if a country is on the coffee belt or not rather than their consumption and imports. I would have assumed that the production would have been a higher determining factor. These results do highlight how disproportionately those on the coffee belt export their coffee over other countries.

Following the same idea of using a boosted dataset, we tried boosting the consumption dataset to see if the observations with a bootstrapped resampled dataset with replacement could accurately predict of a country belongs in the coffee belt or not. Using the K means into the support vector we received and error rate of .9892 showing we are not able to predict from just the consumption alone especially when we compare it to the multi factor XGBoost model which produced a 0.018. 


![image](https://user-images.githubusercontent.com/58121111/122681853-54b6e680-d1c4-11eb-9d74-0c87f7032a01.png)


![image](https://user-images.githubusercontent.com/58121111/122681863-600a1200-d1c4-11eb-9275-433e5029a401.png)
 

