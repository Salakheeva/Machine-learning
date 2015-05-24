Prediction Assignment Writeup
========================================================

The goal of the project is to predict the manner in which person did the exercise. This is the "classe" variable in the training set. Use of any of the other variables to predict with is allowed.

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website [here](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset). 

## Data and project plan

For prediction data was splitted, and you can download in here:

 - [Training data](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)
 
 - [Test data](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)
 
 **Cross validation** will be performed in following steps:
 
1. Load data and reduce covariates.

2. Split training set into training part and test part.

3. Build a model on the training part.

4. Evaluate model on the test part.

5. Estimate error of the model.

6. Predict classe values for test data.
 
## Data download and processing

In the code below I download required libraries and datasets: test and training.


```r
library(caret)
library(doParallel)
set.seed(123)
train.data<-read.csv("./pml-training.csv")
test.data<-read.csv("./pml-testing.csv")
dim(train.data)
```

```
## [1] 19622   160
```

Next step is to decreise the number of variables.There are 160 columns in train data. I did data cleaning in several steps:

1. Deleted columns with a major of not available data and blanks.


```r
train.data<-train.data[,sapply(train.data, function(x){sum(is.na(x)|x=="")})/nrow(train.data)<0.9]
dim(train.data)
```

```
## [1] 19622    60
```

2. Excluded data with a near zero variance.


```r
NZV<-nearZeroVar(train.data,saveMetrics=T)$nzv
train.data<-train.data[,!NZV]
dim(train.data)
```

```
## [1] 19622    59
```

3. Exctract highly correlated variables.


```r
nums<-sapply(train.data,is.numeric)
corr <- cor(na.omit(train.data[,nums]))
diag(corr)<-0
BigCor<-findCorrelation(corrMatrix, cutoff = .80, verbose = F)
train.data<-train.data[,-BigCor]
dim(train.data)
```

```
## [1] 19622    47
```

4. Remove obvious unnecessary columns.


```r
train.data = train.data[,-1]#column x- number of rows
```

Finally, I got a data set with 46 columns.

## Training data division

R code below dividing training data on training and testing parts. 80% of data going to training part, the rest to testing.


```r
train<-createDataPartition(train.data$classe,p=0.8,list=F)
train.train<-train.data[train,]
test.train<-train.data[-train,]
```

## Model building

As a method for prediction I choose random forest. I used a special package for it to increase the speed of processing. Caret package do this method very long. Also I made this process parallel.


```r
cl <- makeCluster(detectCores())
registerDoParallel(cl)
library(randomForest)
Model<-randomForest(classe~.,data=train.train,ntree=100, importance=TRUE)
stopCluster(cl)
```

## Accuracy estimation

For prediction I applied testing part from training dataset. Then I compaired predicted results with actual data and estemated model. Accuracy of the model is equal to 99,9%.


```r
test <- predict(Model, newdata=test.train)
confusionMatrix(test,test.train$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    0    0    0    0
##          B    0  759    1    0    0
##          C    0    0  683    1    0
##          D    0    0    0  642    0
##          E    0    0    0    0  721
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9995          
##                  95% CI : (0.9982, 0.9999)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9994          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   0.9985   0.9984   1.0000
## Specificity            1.0000   0.9997   0.9997   1.0000   1.0000
## Pos Pred Value         1.0000   0.9987   0.9985   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   0.9997   0.9997   1.0000
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1935   0.1741   0.1637   0.1838
## Detection Prevalence   0.2845   0.1937   0.1744   0.1637   0.1838
## Balanced Accuracy      1.0000   0.9998   0.9991   0.9992   1.0000
```

## Prediction

Now it is time to use model for prediction on test dataset. Results you can see below.


```r
Prediction <- predict(Model, newdata=test.data,type="class")
Prediction
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
