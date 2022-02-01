library(kknn)
DF=read.csv(file= 'C:/Users/Daniel Bissessar/Desktop/TDDE01/Labs/optdigits.csv', header=FALSE)
n=dim(DF)[1]
set.seed(12345)
id=sample(1:n,floor(n*0.5))
train=DF[id,]
id1=setdiff(1:n, id)
id2=sample(id1, floor(n*0.25))
valid=DF[id2, ]
id3=setdiff(id1,id2)
test=DF[id3, ]


missclass = function(y,y_h){
  l = length(y)
  return(1-sum(diag(table(y,y_h)))/l)
}

knn_fit_train <- kknn(as.factor(V65)~., train, train, k = 30, kernel='rectangular')
train_confusion_matrix <- table(train$V65, knn_fit_train$fitted.values)

knn_fit_test <- kknn(as.factor(V65)~., train, test, k=30, kernel = 'rectangular')
test_confusion_matrix <- table(test$V65, knn_fit_test$fitted.values)

prob8s <- knn_fit_train$prob[,9]
easy1 <- which.max(prob8s)
prob8s[easy1] <- NA
easy2 <- which.max(prob8s)
prob8s[easy2] <- NA
t <- 0
vhard <- c(0,0,0)
while (t<3) {
  temp <- which.min(prob8s)
  if(train[temp, 65] == 8) {
    vhard[t+1] <- temp
    t <- t+1
  }
  prob8s[temp] <- NA
}

e1matrix <- as.matrix(train[easy1, 1:64])
v <- c(e1matrix[1:64])
e1matrix <- matrix(v, nrow = 8)

e2matrix <- as.matrix(train[easy2, 1:64])
v <- c(e2matrix[1:64])
e2matrix <- matrix(v, nrow = 8)

h1matrix <- as.matrix(train[vhard[1], 1:64])
v <- c(h1matrix[1:64])
h1matrix <- matrix(v, nrow = 8)

h2matrix <- as.matrix(train[vhard[2], 1:64])
v <- c(h2matrix[1:64])
h2matrix <- matrix(v, nrow = 8)

h3matrix <- as.matrix(train[vhard[3], 1:64])
v <- c(h3matrix[1:64])
h3matrix <- matrix(v, nrow = 8)

heatmap(e1matrix, Colv = NA, Rowv = NA)
heatmap(e2matrix, Colv = NA, Rowv = NA)
heatmap(h1matrix, Colv = NA, Rowv = NA)
heatmap(h2matrix, Colv = NA, Rowv = NA)
heatmap(h3matrix, Colv = NA, Rowv = NA)

cross_entropy <- function(x, y){
  res <- c(1:nrow(y))
  for (i in 1:nrow(y)){
    res[i] <- log(x$prob[i, (y$V65[i]+1)]+(1e-15))
  }
  return(res)
}
vemp <- c(1:30)
vmissV <- c(1:30)
vmissT <- c(1:30)
vk <- c(1:30)

for (i in 1:30){
  vk[i] <- i
  knn_fit_loopV <- kknn(as.factor(V65)~., train, valid, k = i, kernel = 'rectangular')
  knn_fit_loopT <- kknn(as.factor(V65)~., train, train, k = i, kernel = 'rectangular')
  vmissV[i] <- missclass(valid$V65, knn_fit_loopV$fitted.values)
  vmissT[i] <- missclass(train$V65, knn_fit_loopT$fitted.values)
  vemp[i] <- -1*mean(cross_entropy(knn_fit_loopV,valid))
}
plot(c(0,31), c(0,0.065), type = 'n')
points(vk, vmissV, col = 1)
points(vk, vmissT, col = 2)
print(which.min(vmissV))

knn_fit_test2 <- kknn(as.factor(V65)~., train, test, k= 3, kernel = 'rectangular')
test_error <- missclass(test$V65, knn_fit_test2$fitted.values)
print(test_error)
print(vmissT[3])
print(vmissV[3])
plot(vk, vemp)
print(which.min(vemp))



DF=read.csv(file= 'C:/Users/Daniel Bissessar/Desktop/TDDE01/Labs/parkinsons.csv')
DF <- scale(DF)
n <- dim(DF)[1]
set.seed(12345)
id <- sample(1:n, floor(n*0.6))
train <- DF[id, ]
train_motor <- train[,5]
train <- train[,-(1:6)]
test <- DF[-id,]
test_motor <- test[,5]
test <- test[,-(1:6)]
loglikelihood <- function(w,sigma) {
  return(-length(train_motor)/2*log(sigma^2)-1/(2*sigma^2)*(sum((train_motor-as.matrix(train)%*%w)^2)))
}
ridge <- function(w,sigma) {
  return(lambda*sum(w^2)-loglikelihood(w,sigma))
}  
helpridge <- function(x) {
  w <- c(x[1:16])
  sigma <- c(x[17])
  return(ridge(w,sigma))
}
ridgeOpt <- function(){
  return(optim(par=rep(1,17), helpridge, method='BFGS'))
}
DF <- function(){
  train <- as.matrix(train)
  tempT <- train%*%(t(train)%*%train+lambda*diag(ncol(train)))^-1%*%t(train)
  return(sum(diag(tempT)))
}
AIC <- function(w,sigma) {
  return(2*DF()-2*loglikelihood(w,sigma))
}

lambda <- 1
OptLambda1 <- ridgeOpt()
Opt1Var <- OptLambda1[[1]]
w1 <- Opt1Var[1:16]
sigma1 <- Opt1Var[17]
predTrain1 <- as.matrix(train)%*%w1
predTest1 <- as.matrix(test)%*%w1
MSETrain1 <- sum((predTrain1-train_motor)^2)/length(predTrain1)
MSETest1 <- sum((predTest1-test_motor)^2)/length(predTest1)
AIC1 <- AIC(w1,sigma1)


library(glmnet)
DF=read.csv(file= 'C:/Users/Daniel Bissessar/Desktop/TDDE01/Labs/tecator.csv', header=TRUE)
set.seed(12345)
n <- dim(DF)[1]
id <- sample(1:n, floor(n*0.5))
trainData <- DF[id,]
testData <- DF[-id,]

fitLm <- lm(formula=Fat~., data=trainData)
summary(fitLm)
predTrain <- predict(fitLm, newdata = trainData)
predTest <- predict(fitLm, newdata = testData)

covariates <- trainData[,2:101]
response <- trainData$Fat
fitLasso <- cv.glmnet(as.matrix(covariates), response, alpha=1, family='gaussian')
plot(fitLasso$glmnet.fit, xvar = 'lambda')

plot(fitLasso$lambda, fitLasso$glmnet.fit$df)
fitRidge <- cv.glmnet(as.matrix(covariates), response, alpha = 0, family='gaussian')
plot(fitRidge$glmnet.fit, xvar = 'lambda')
