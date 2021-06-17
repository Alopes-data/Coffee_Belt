#Optimization Final

# Coffee dataset available at
# https://www.kaggle.com/yamaerenay/ico-coffee-dataset-worldwide?select=prices-paid-to-growers.csv
# mostly numerial based
# coffee years 1998 - 2018
# datasets contain years as variables(columns) (EX., 1995, 1996, 1997) seperated by countries (rows)
# Original Source::: 
# http://www.ico.org/new_historical.asp?section=Statistics
# http://www.ico.org/new_historical.asp


# Definition of Coffee Belt countries from: "https://seasia.co/2018/01/27/the-coffee-belt-a-world-map-of-the-major-coffee-producers"
#   Central America ::: Costa Rica, Guatemala, Honduras, Mexico, Nicaragua, Panama, El Salvador
#   South America   ::: Bolivia, Brazil, Colombia, Ecuador, Peru
#   Africa & Arabia ::: Burundi, Congo, Ethiopia, Kenya, Rwanda, Tanzania, Uganda, Yemen, Zambia, Zimbabwe
#   Asia+   ::: Indonesia, India, Myanmar, Vietnam, Papua New Guinea
#   Islands & Others::: Australia, Dominican Republic, Hawaii, Jamaica, Puerto Rico


#load in Libraries
library(doParallel)
library(tidyverse)
library(stats)
library(factoextra)
library(cluster)
library(e1071)
library(caret)
library(stats)
library(Hmisc)
library(dplyr)
library(xgboost)
library(readr)
library(stringr)
library(car)
library(ROCR)
library(DiagrammeR)
library(data.table)
library(pastecs)


###Check/set working directory
getwd()
setwd("C:/Users/Alexl/Documents/JWU/Optimization_Datasets")


# domestic-consumption.csv: Domestic consumption by all exporting countries
exported_domestic_consumption <- read.csv("domestic-consumption.csv")
# imports.csv: Imports by selected importing countries
imports <- read.csv("imports.csv")
# exports-calender-year.csv: Exports of all forms of coffee by all exporting countries
exports_calender_year <- read.csv("exports-calendar-year.csv")
# re-exports.csv: Re-exports by selected importing countries
re_exports <- read.csv("re-exports.csv")
# total-production.csv: Total production by all exporting countries
total_production <- read.csv("total-production.csv")

# Define Groups of Coffee Belt vs Non Coffee Belt countries
# Create a List
# Inside will have the names of all the CoffeBelt countries 
List_Coffeebelt <- list("Costa Rica", "Guatemala", "Honduras", "Mexico", "Nicaragua", "Panama", "El Salvado", 
      "Bolivia", "Brazil", "Colombia", "Ecuador", "Peru",
      "Burundi", "Congo", "Ethiopia", "Kenya", "Rwanda", "Tanzania", "Uganda", "Yemen", "Zambia", "Zimbabwe",
      "Indonesia", "India", "Myanmar", "Vietnam", "Papua New Guinea",
      "Australia", "Dominican Republic", "Hawaii", "Jamaica", "Puerto Rico")



#   The USA is a special case where it is the only country in this dataset that belongs to the Coffee Belt country list due to Hawaii and Puerto Rico.
summary(exported_domestic_consumption)
unique(exported_domestic_consumption$domestic_consumption)


### Data transformation to create CoffeeBelt Group
# create a new Column
# If the name of the Observation in Column matches a name of a Observation in List from Above
# Provide a 1 = yes
# if not provide a 0 = no

# exported_domestic_consumption
exported_domestic_consumption$CoffeeBeltCat <- ifelse(exported_domestic_consumption$domestic_consumption %in% List_Coffeebelt, 1, 0)

# imports
imports %>%
  summary()
unique(imports$imports)
imports$CoffeeBeltCat<- ifelse(imports$imports %in% List_Coffeebelt, 1, 0)

# exports_calender_year 
exports_calender_year %>%
  summary()
unique(exports_calender_year$exports)
exports_calender_year$CoffeeBeltCat<- ifelse(exports_calender_year$exports %in% List_Coffeebelt, 1, 0)

# re_exports 
re_exports %>%
  summary()
unique(re_exports$re.exports)
re_exports$CoffeeBeltCat<- ifelse(re_exports$re.exports %in% List_Coffeebelt, 1, 0)

# total_production 
total_production %>%
  summary()
unique(total_production$total_production)
total_production$CoffeeBeltCat<- ifelse(total_production$total_production %in% List_Coffeebelt, 1, 0)




### K means pre processiong 
summary(exported_domestic_consumption)    #Review the dataset
exported_domestic_consumption

exported_domestic_consumption$domestic_consumption <- NULL   #get rid of non numerical values
names(exported_domestic_consumption)

exported_domestic_consumption <- na.omit(exported_domestic_consumption) # get rid of NA values
is.na(exported_domestic_consumption)      #Check for NA values
exported_domestic_consumption <- lapply(exported_domestic_consumption, as.numeric)  #Convert to Numeric

class(exported_domestic_consumption)  # Review the dataset
glimpse(exported_domestic_consumption)

#Convert to Dataframe in order to scale
exported_domestic_consumption <- data.frame(exported_domestic_consumption)
class(exported_domestic_consumption)  # Review the dataset

exported_domestic_consumption <- scale(exported_domestic_consumption, center = TRUE, scale = TRUE)

d1 <- exported_domestic_consumption 

##Check for null AGAIN TO BE SAFE
sapply(d1, function(x) sum(is.na(x))) 
head(d1) 


###########################################
###    K MEANS Cluster Calculation     ###
##########################################
iboot <- sample(1:nrow(exported_domestic_consumption), size = 1000, replace = TRUE)
bootdata2 <- exported_domestic_consumption[iboot,]
glimpse(bootdata2)

set.seed(123)
exported_domestic_consumption

df <- as.data.frame(bootdata2)
df

# Elbow method (look at the knee) to determine the number of clusters
# Elbow method for kmeans
fviz_nbclust(df, kmeans, method = "wss") +  
  geom_vline(xintercept = 3, linetype = 2) # 3 clusters are optimal 

# Average silhouette for kmeans
    # Not very insightful
fviz_nbclust(df, kmeans, method = "silhouette")
distance <- get_dist(df)
fviz_dist(distance, gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))

set.seed(555) # Set seed for reproduction value
smp_size <- floor(0.80 * nrow(df))    #creates an 80/20 split of the data
#traning index based off of total production and Sample Size from 80/20 split
train_ind <- sample(seq_len(nrow(df)), size = smp_size) 

train <- df[train_ind, ]
test <- df[-train_ind, ]
names(df)

k2 <- kmeans(train, centers = 3, nstart = 25)    #K means model on 3 clusters
glimpse(k2)
fviz_cluster(k2, data = train) # look this up 
summary(k2)

# assign clusters to dataset
k2_clusters <- cbind(train, as.factor(k2$cluster))
k2_clusters <- as.data.frame(k2_clusters)
#newdf with cluster variable can be used to predicting in other models
names(k2_clusters)
k2_clusters$clusters <- k2_clusters$"as.factor(k2$cluster)"
k2_clusters$"as.factor(k2$cluster)" <- NULL
names(k2_clusters)
View(k2_clusters)
######################################################
# Create K means group
# Look for countries that are misplaced i.e. Coffee Belt countries grouped with others or vice versa
# See How well the K means can predict the Last Newly created Column 

#SVM on mismatched groups


names(k2_clusters)
View(k2_clusters)
intrain <- createDataPartition(y = k2_clusters$clusters, p= 0.8, list = FALSE)    ##Review the column selected/split
training <- k2_clusters[intrain,]  ##Adjust the dataset
testing <- k2_clusters[-intrain,]   ##Adjust the dataset

dim(training) 
dim(testing)

training[["clusters"]] = factor(training[["clusters"]])       ##Review the column selected for training
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)     ###Same as above


#   cHANGE SVM linear to respective catagorical method
svm_Linear <- train(clusters ~., data = training, method = "svmLinear", 
  trControl=trctrl,
  preProcess = c("center", "scale"),
  tuneLength = 10)
svm_Linear  #review the results

## svmRadial   Required for caagorical values?

svm_Radial <- train(CoffeeBeltCat ~., data = training, method = "svmRadial", 
  trControl=trctrl,
  preProcess = c("center", "scale"),
  tuneLength = 10)
svm_Radial  #review the results

# Test the prediction
test_pred <- predict(svm_Linear, testing)
test_pred
summary(test_pred)

test_pred <- predict(svm_Radial, testing)
test_pred
summary(test_pred)

err <- mean(as.numeric(test_pred > 0.5) != testing) #Check error rate Correct?
print(paste("test-error=", err))


###################################
###     Feature Engineering     ###
###################################

imports %>%
  dim()
exports_calender_year%>%
  dim()
re_exports%>%
  dim()
total_production%>%
  dim()

#Need to define other factors 
# defined from IQRs of the 25 percentile
# High\Low Import



### Imports
glimpse(imports)
unique(imports$imports)

imports <- na.omit(imports) # get rid of NA values
is.na(imports)#Check for NA values
imports <- as.data.frame(imports)

names(imports)

imports_try <- imports %>%
  mutate(importsSums = rowSums(.[2:29])) %>%   # create sums from year columns, coulmns 2 - 29
  mutate(importsAVG = rowSums(.[2:29])/29)    # create a mean column
head(imports_try)

summary(imports_try$importsAVG)
# First Quartile:: 4.403  
# Third Quartile:: 281.265

d1 <- imports_try %>%
  select(imports, importsAVG)

# add logic based encoded columns to final dataset
head(d1) # Review
names(d1) # Review
d1$HighImports <- as.factor(ifelse(imports_try$importsAVG >= 281.265, 1, 0 ))
d1$LowImports <- as.factor(ifelse(imports_try$importsAVG <= 4.403, 1, 0))
names(d1) # Make sure new columns were added.



# total_production
names(total_production)
glimpse(total_production)
unique(total_production$total_production) # review all the different countries

total_production <- na.omit(total_production) # get rid of NA values
is.na(total_production)#Check for NA values
total_production <- as.data.frame(total_production) #convert to a dataframe

names(total_production)

total_production_try <- total_production %>%
  mutate(ProductionSums = rowSums(.[2:29])) %>%   # create sums from year columns, coulmns 2 - 29
  mutate(ProductionAVG = rowSums(.[2:29])/29)    # create a mean column
head(total_production_try)

summary(total_production_try$ProductionAVG)
# First Quartile:: 43.52
# Third Quartile:: 1428.50

d2 <- total_production_try %>%
  select(total_production, ProductionAVG)

# add logic based encoded columns to final dataset
head(d2) # Review
names(d2) # Review
d2$HighProduction <- as.factor(ifelse(total_production_try$ProductionAVG >= 1428.50, 1, 0 ))
d2$LowProduction <- as.factor(ifelse(total_production_try$ProductionAVG <= 43.52, 1, 0))
names(d2) # Make sure new columns were added.



### exports_calender_year
names(exports_calender_year)
glimpse(exports_calender_year)
unique(exports_calender_year$exports) # review all the different countries

exports_calender_year <- na.omit(exports_calender_year) # get rid of NA values
is.na(exports_calender_year)#Check for NA values
exports_calender_year <- as.data.frame(exports_calender_year) #convert to a dataframe

exports_calender_year_try <- exports_calender_year %>%
  mutate(ExportSums = rowSums(.[2:29])) %>%   # create sums from year columns, coulmns 2 - 29
  mutate(ExportAVG = rowSums(.[2:29])/29)    # create a mean column

View(exports_calender_year_try)
summary(exports_calender_year_try$AVG)
# First Quartile:: 29.2977
# Third Quartile:: 1274.335

names(exports_calender_year_try)

d3 <- exports_calender_year_try %>%
  select(exports, ExportAVG)

head(d3) # Review
names(d3) # Review
d3$HighExport <- as.factor(ifelse(exports_calender_year_try$ExportAVG >= 1274.335, 1, 0 ))
d3$LowExport <- as.factor(ifelse(exports_calender_year_try$ExportAVG <= 29.2977, 1, 0))
names(d3) # Make sure new columns were added.

#View(d3)

# Review created datasets before joining together
head(d1)
head(d2)
head(d3)
glimpse(d1)
glimpse(d2)

# Join created datasets together
dt1 <- full_join(d1, d2, by = c("imports" = "total_production" ))
dt2 <- full_join(dt1, d3, by = c("imports" = "exports"))

glimpse(dt2) # review final dataset

dt2[is.na(dt2)] <- 0 #convert all NAs into zeros, no meaningful Zeros
#View(dt2)

###Final Data SET:::
# CoffeeBelt == 1, 0  TRUE FALSE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# High exports_calender_year_higher == 1, 0  TRUE FALSE *****
# Low exports_calender_year_lower == 1, 0  TRUE FALSE ***
# High Import_higher == 1, 0  TRUE FALSE ****
# Low Import_lower == 1, 0  TRUE FALSE ****
# High total_production_higher == 1, 0  TRUE FALSE *****
# Low total_production_lower == 1, 0  TRUE FALSE  *****

# re-add coffeebelt cat to the dataset
dt2 %>%
  summary()
unique(dt2$imports)
dt2$CoffeeBeltCat <- as.factor(ifelse(dt2$imports %in% List_Coffeebelt, 1, 0)) # logic 1 if coffeebelt country
names(dt2)

dt2$imports <- as.factor(dt2$imports) # Convert from characters to factors
glimpse(dt2) #review data

# use a replicated sample with replacement 
iboot <- sample(1:nrow(dt2), size = 1000, replace = TRUE)
bootdata <- dt2[iboot,]
glimpse(bootdata)

set.seed(123) # Set sampling seed
# partition new boot strapped data into the name # of observations
smp_size <- floor(0.50 * nrow(bootdata))

# set the seed to make your partition reproducible
train_ind <- sample(seq_len(nrow(bootdata)), size = smp_size)
train <- bootdata[train_ind, ] 
test <- bootdata[-train_ind, ]

# Set test and train label as numerical factors
trainlabel <- as.numeric(as.factor(train$CoffeeBeltCat))-1
testlabel <- as.numeric(as.factor(test$CoffeeBeltCat))-1

# remove the target responses from the training and test sets
train$CoffeeBeltCat <- NULL
test$CoffeeBeltCat <- NULL

# Convert to data matric from data frame to prepare XG
trainmat <- data.matrix(train)
testmat <- data.matrix(test)

# Create a XG data matrix to input direcinty into XG Boost model
dtrain <- xgb.DMatrix(data = trainmat, label= trainlabel)
dtest <- xgb.DMatrix(data =  testmat, label= testlabel)


###############################
###     XG BOOST MODEL      ###
###############################

xgmodel <- xgboost(data = dtrain, # the data   
                   nround = 20,
                   max.depth = 3,# max number of boosting iterations
                   objective = "binary:logistic")  # the objective function
print(xgmodel)
pred <- predict(xgmodel, dtest)
#summary(pred)

# get & print the classification error
err <- mean(as.numeric(pred > 0.5) != testlabel)
print(paste("test-error=", err))

# get the number of negative & positive cases in our data
negative_cases <- sum(trainlabel == 0)
postive_cases <- sum(testlabel == 1)

model_tuned <- xgboost(data = dtrain, # the data           
                       max.depth = 3, # the maximum depth of each decision tree
                       nround = 100, # number of boosting rounds
                       early_stopping_rounds = 2, # if we dont see an improvement in this many rounds, stop
                       objective = "binary:logistic", # the objective function
                       scale_pos_weight = negative_cases/postive_cases, # control for imbalanced classes
                       gamma = 1) # add a regularization term
print(model_tuned)

# generate predictions for our held-out testing data
pred <- predict(model_tuned, dtest)

# get & print the classification error
err <- mean(as.numeric(pred > 0.5) != testlabel)
print(paste("test-error=", err))

library(DiagrammeR)
# plot model features
xgb.plot.multi.trees(feature_names = names(trainmat), 
                     model = model_tuned)
xgb.dump(model_tuned, with_stats = TRUE)

# convert log odds to probability
odds_to_probs <- function(odds){
  return(exp(odds)/ (1 + exp(odds)))
}

# probability of leaf at the top in 3rd row
odds_to_probs(-3.0164)

# get information on how important each feature is
importance_matrix <- xgb.importance(names(trainmat), model = model_tuned)

# Plot importance matrix
xgb.plot.importance(importance_matrix)

# Create our prediction probabilities
pred <- predict(model_tuned, dtest)
print(pred)
# Set our cutoff threshold
pred.resp <- ifelse(pred >= 0.5, 1, 0)


