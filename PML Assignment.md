# Practical Machine Learning Assignment - Prediction Assignment Writeup
# Author: Josué Lavandeira
==========================================================
### Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

### Introduction

The main goal of this text is to analyze these data sets and to utilize the training data set to create a prediction model that allows for accurate prediction of data that's similar to the data in the provided datasets, to test this similarity, we will use the test data set. Then we will apply the model we created to 20 other datasets and see how accurate our model really is.
To do this we will use the following:

### Methododlogy and data processing

First we download the data from the provided URL's and then we load the data into R. For this to work, make sure the data files are in the same working directory you're using in R, if not, then change the working folder to the one where the files are at, or copy the files into your current working directory. Also, the data has a lot of values labeled "#DIV/0!", we change these values into NA's so that we may manipulate the data more easily.

```{r}
training_set <- read.csv("pml-training.csv", na.strings=c("#DIV/0!") )
testing_set <- read.csv("pml-testing.csv", na.strings=c("#DIV/0!") )
```
Then we set all values as numeric values

```{r}
for(i in c(8:ncol(training_set)-1)) {training_set[,i] = as.numeric(as.character(training_set[,i]))}
for(i in c(8:ncol(testing_set)-1)) {testing_set[,i] = as.numeric(as.character(testing_set[,i]))}
```
And we also keep only complete columns (those that don't have NA values) in the data sets.

```{r}
validcolumns <- colnames(training_set[colSums(is.na(training_set)) == 0])[-(1:7)]
training_set <- training_set[validcolumns]
training_set #results hidden

validcolumns2 <- colnames(testing_set[colSums(is.na(testing_set)) == 0])[-(1:7)]
testing_set <- testing_set[validcolumns2]
testing_set #results hidden
```

Then we load the libraries needed for the data manipuation we'll do and set a seed value to make our work reproducible.

```{r}
library(Hmisc)
library(caret)
library(randomForest)
library(foreach)
library(doParallel)
set.seed(12321)
```
### Model training

And now we can slice the data to use a portion in training our model and other to validate it

```{r}
temp <- createDataPartition(y=training_set$classe, p=0.75, list=FALSE )
training <- training_set[temp,]
validating <- training_set[-temp,]
```

Now we have to build our model, we will use the random forests method and create 6 random forests with 150 trees each (we won't use more to avoid overfitting). We'll take advantage of the paralell processing tool registerDoParallel() for this.

```{r}
registerDoParallel()
x <- training[-ncol(training)]
y <- training$classe

rf <- foreach(ntree=rep(150, 6), .combine=randomForest::combine, .packages='randomForest') %dopar% {
randomForest(x, y, ntree=ntree) 
}
```
### Model testing

And now we test the model on the training set
```{r}
trainpred <- predict(rf, newdata=training)
confusionMatrix(testpred,training$classe)
# results hidden
```
And we test it on the validation data set
```{r}
validpred <- predict(rf, newdata=validating)
confusionMatrix(validpred,validating$classe)
# results hidden
```

Now we can observe the model's accuracy on the validation set and it's out of sample error
```{r}
 accuracy <- postResample(validpred, validating$classe)
 accuracy

 Accuracy     Kappa 
0.9965334   0.9956146 

oose <- 1 - as.numeric(confusionMatrix(validating$classe, validpred)$overall[1])
oose
[1] 0.003466558
```

# Conclusions
--------------------------------

We can observe in the confusion matrices for our data sets that the models seems to be very accurate. This model gives us an average accuracy of over 99% and an out of sample error of 0.34% which is great for any prediction model. We can confide that this model will give us accurate predictions for any similar dataset.

There is no need to do Cross Validation as the used model got 20 out of 20 predictions right in the testing data set, meaning our predicted accuracy and out of sample error are true values for any similar dataset.
