library(glmnet)
library(cvTools)
library(tree)
library(rpart)
library(party)
library(randomForest)
library(boot)
library(fastICA)
library(ggplot2)
library(dplyr)
library(tidyr)
library(kernlab)
library(neuralnet)
library(nnet)
library(splines2)
library(fields)
library(mgcv)
library(gam)
library(pamr)
library(klaR)
library(mboost)
library(mvtnorm)
library(e1071)
library(caret)

#true, fitted values
missclass <- function(y, y_h){
  l=length(y)
  return(1-sum(diag(table(y,y_h)))/l)
}

confmatrix <- function(y, y_h){
  table(y,y_h)
}

#assignment 1.1
data1 <- read.csv("C:/Users/Daniel Bissessar/Desktop/optdigits.csv")

data1 <- data1[,-65]
PCA <- prcomp(data1)
plot(summary(PCA)$importance[2,])
sum(summary(PCA)$importance[2,1:29])
originalmatrix <- data.matrix(data1[1,])
originalmatrix <- matrix(originalmatrix, nrow=8)
heatmap(as.matrix(originalmatrix))

#assignment 1.2
data1 <- read.csv("C:/Users/Daniel Bissessar/Desktop/optdigits.csv")
n=dim(data1)[1]
set.seed(12345)
id=sample(1:n, floor(n*0.5)) 
train1=data1[id,]
id2=setdiff(1:n, id)
test1=data1[id2,]
sequence <- seq(3,50,1)
tartrain <- as.factor(train1[,65])
train1 <- train1[,-65]
tartest <- as.factor(test1[,65])
test1 <- test1[,-65]

for (i in sequence) {
  tree <- tree(tartrain ~., train1, mindev=0, minsize=i)
  predtrain <-predict(tree,train1,type='class')
  predtest <-predict(tree,test1,type='class')
  missclasstrain[i-2] <- missclass(tartrain,predtrain)
  missclasstest[i-2] <- missclass(tartest,predtest)
}
plot(missclasstrain)
points(missclasstest, col='red')

tartrain <- as.numeric(tartrain)
for (i in 1:1911) {
  if (tartrain[i] == 0) {
    tartrain[i] = 1
  } else {
    tartrain[i] = 0
  }
}
tartest <- as.numeric(tartrain)
for (i in 1:1911) {
  if (tartest[i] == 0) {
    tartest[i] = 1
  } else {
    tartest[i] = 0
  }
}

#kernel methods
dataK <- read.table("C:/Users/Daniel Bissessar/Desktop/dataKernel.txt")
class1 <- dataK[1:1500,]
class2 <- dataK[1501:2500,]
class1tr <-class1[1:900,]
class2tr <- class2[1:300,]
kernelTest <- rbind(class1[901:1200,], class2[301:400,])
KernelVa <- rbind(class1[1201:1500,], class2[401:500,])

gaussianKernel <- function(x, h) {
  return (exp(-(x**2)/(h*h*2)))
}

density1 <- function(point, h){
  dense <- 0
  for(i in 1:900) {
    dense <- dense+gaussianKernel(class1tr[i,1]-point,h)
  }
  return (dense/900)
}
density2 <- function(point, h){
  dense <- 0
  for(i in 1:300) {
    dense <- dense+gaussianKernel(class2tr[i,1]-point,h)
  }
  return (dense/300)
}
prob1 <- function(point, h) {
  probclass1 <- density1(point,h)*900/1200
  probclass2 <- density2(point,h)*300/1200
  prob1 <- probclass1/(probclass1+probclass2)
  return (prob1)
}

h <- 1
errors <- 0
for (i in 1:400) {
  prob <- prob1(KernelVa[i,1],h)
  prob <- 1-prob
  pred <- round(prob) + 1
  if (pred != KernelVa[i,2])
    errors <- errors + 1
}
errorRateVa <- errors/400


class1tr <-class1[1:1200,]
class2tr <- class2[1:400,]
density1 <- function(point, h){
  dense <- 0
  for(i in 1:1200) {
    dense <- dense+gaussianKernel(class1tr[i,1]-point,h)
  }
  return (dense/1200)
}
density2 <- function(point, h){
  dense <- 0
  for(i in 1:400) {
    dense <- dense+gaussianKernel(class2tr[i,1]-point,h)
  }
  return (dense/400)
}
prob1 <- function(point, h) {
  probclass1 <- density1(point,h)*1200/1600
  probclass2 <- density2(point,h)*400/1600
  prob1 <- probclass1/(probclass1+probclass2)
  return (prob1)
}

errors <- 0
for (i in 1:400) {
  prob <- prob1(kernelTest[i,1],h)
  prob <- 1-prob
  pred <- round(prob) + 1
  if (pred != kernelTest[i,2])
    errors <- errors + 1
}
errorRateTest <- errors/400

#nn see other file