# **Rossman Sales Prediction**
![](/img/rossmann.jpg)

## 1. **Introduction**
This is an end-to-end data science project which predict sales of the next six weeks of Rossmann stores. It was used machine learning XGBoost algorithm to predict these sales. Predictions of each store can be accessed by users through Telegram as shown below.

## 2. **Business Problem**

Rossmann is a drug store chain with more than 3,000 drug stores in 7 differents countries across Europe.

CFO would like to make improvements in their stores and would like to know how much revenue they are going to make so they could decide how much they could invest now in each.

Therefore, the main goal is to predict the 6 next weeks sales for each store.


## 3. **Business Assumptions**
* Days with no sales or closed stores are not be considered in analysis

## 4. **Tools used**
* Python
* Jupyter Notebook
* Git and Github
* Flask and Python API's
* Sklearn

## **Dataset**

Dataset contains 1,017,209 rows and 17 columns of 1,115 Rossmann stores from 01/01/2013 to 06/19/2015 and it contains the following variables:

* Id - an Id that represents a (Store, Date) duple within the test set
* Store - a unique Id for each store
* Sales - the turnover for any given day (this is what you are predicting)
* Customers - the number of customers on a given day
* Open - an indicator for whether the store was open: 0 = closed, 1 = open
* StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
* SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools
* StoreType - differentiates between 4 different store models: a, b, c, d
* Assortment - describes an assortment level: a = basic, b = extra, c = extended
* CompetitionDistance - distance in meters to the nearest competitor store
* CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened
* Promo - indicates whether a store is running a promo on that day
* Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
* Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2
* PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store



## **Planning Soluction**

This project was developed following CRISP-DM (Cross-Industry Standard Process - Data Science) framework. This framework has these steps:

* Business Understanding;
* Data Collection;
* Data Cleaning;
* Exploratory Data Analysis (EDA);
* Data Preparation;
* Machine Learning Modelling and fine-tuning;
* Model and Business performance evaluation / Results;
* Model deployment.

![](/img/0%20crisp-dm.png)

Step 1. Data description and cleaning: Seeing dimensions of data, checking existence of NAs, number of rows and columns. Fillout NAs, altering columns names,  Descriptive statistics.

Step 2. Feature engineering: Creation a mindmap hypothesis with all variables and values that could have impact on sales. After that, some features was created from current ones

Step 3. Feature filtering: Drop some columns and lines that will not be used in analysis.

Step 4. Exploratory Data Analysis (EDA): Univariate, bivariate and multivariate analysis. Checking correlation between response variable and explanatory ones. Hypothesis testing.

Step 5. Data preparation: Rescaling and encoding features so they could be used in machine learning algorithms properly.  Transformation of response variable

Step 6. Feature selection: Selecting the most important features to predict sales through boruta algorithm. Also there is a split in dataset creating train and test data.

Step 7. Machine learning modelling: Testing machine learning algorithm to find out which one has best performance in prediction. Performance was evaluated using cross-validation.

Step 8. Hyperparameter tunning: Random search technique was used to find out the best hyperparameters that maximizes performance of choosen model in last step.

Step 9. Translation of modelling performance in business performance: Modelling performance was analyzed in a business perspective.

Step 10. Deploy mode to production: Model was deployed in a cloud environment so stakeholders could have access to predictions.

## Top 3 insights

 H1. Stores with competitors nearby (at most 500 meters) sell 10% less, on average, than futher competitors.

*FALSE*. Actually, stores with competitors nearby sell 12% more, on average.

![](/img/competition_distance.png)
![](/img/h1.png)


H2. Stores with longer competitors (more than 1 year) sell 10% more on average.

**FALSE**. There is almost no difference between sales of stores that have competitors with less than a year and that ones which have it.

![](/img/competition_time_month.png)
![](/img/h2.png)

H3. Stores with more consecutive sales would sell more on average

False. Actually stores that just join promotion 1 sells more than those which join consecutive promotion.

![](/img/consecutive_sales.png)
![](/img/h3.png)

## **Performance**

Performance of all models using Cross Validation

Although model with smallest error in cross validation was Random Forest, it was choosen the XGBoost model due to lower variance. Also, in hiperparameter tunning XGBoost have a lower error.

![](/img/performance_comparasion.jpg)

Performance XGBoost with hiperparameters otimized - Final model

![](/img/final_model.jpg)

## **Business Performance**

The final model got a Random Mean Squared error (RMSE) of $947.34, which means that, on average, predictions of sales for next six weeks could be futher from value predicted by almost a thousand dollar, either more or less.

In regards to the Mean Absolute Percentage Error (MAPE), model has an error of 9%, on average, meaning that predicted values could be futher from real value 9%, either more or less.  

Here is an example of 10 stores and their predictions, including best and worst scenarions and total of

![](/img/stores_predictions.png)

Considering all stores, predictions of sales for next 6 weeks was the following:

![](/img/business_prediction.jpg)

We can see in plot above the difference between true values and predictions as well how close they are.

![](/img/predictions_plot.png)
