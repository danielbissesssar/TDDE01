library(datasets)
library(dplyr)
library(MASS)
library("nnet")
library("mvtnorm")
library("OceanView")
data(iris)
setosa <- filter(iris, Species == 'setosa')
versicolor <- filter(iris, Species == 'versicolor')
virginica <- filter(iris, Species == 'virginica')
plot(iris$Sepal.Length, iris$Sepal.Width)
points(setosa$Sepal.Length, setosa$Sepal.Width, col = 2)
points(versicolor$Sepal.Length, versicolor$Sepal.Width, col = 3)
points(virginica$Sepal.Length, virginica$Sepal.Width, col = 4)

prior_prob <- function(x){
  return(nrow(x)/nrow(iris))
}

classMean <- function(x){
  return(c(mean(x$Sepal.Length), mean(x$Sepal.Width)))
}

cov_matrix <- function(x) {
  mat <- cbind(x$Sepal.Length, x$Sepal.Width)
  return(cov(mat, method = 'pearson'))
}

setosa_prior_prob <- prior_prob(setosa)
versicolor_prior_prob <- prior_prob(versicolor)
virginica_prior_prob <- prior_prob(virginica)

setosa_meanLW <- classMean(setosa)
versicolor_meanLW <- classMean(versicolor)
virginica_meanLW <- classMean(virginica)

setosa_cov <- cov_matrix(setosa)
versicolor_cov <- cov_matrix(versicolor)
virginica_cov <- cov_matrix(virginica)

pooled_cov <- (setosa_cov+versicolor_cov+virginica_cov)/3

weights <- function(meanLW, prob){
  wix <- solve(pooled_cov)%*%meanLW
  w0 <- -0.5*t(meanLW)%*%solve(pooled_cov)%*%meanLW+log(prob)
  res <- c(wix,w0)
  return(res)
}

setosa_disc_fun <- function(x){
  w <- weights(setosa_meanLW, setosa_prior_prob)
  res <- x%*%c(w[1],w[2])+w[3]
  return(res)
}

versicolor_disc_fun <- function(x){
  w <- weights(versicolor_meanLW, versicolor_prior_prob)
  res <- x%*%c(w[1],w[2])+w[3]
  return(res)
}

virginica_disc_fun <- function(x){
  w <- weights(virginica_meanLW, virginica_prior_prob)
  res <- x%*%c(w[1],w[2]) + w[3]
  return(res)
}

set_ver_eq <- function(x){
  set_w <- weights(setosa_meanLW, setosa_prior_prob)
  ver_w <- weights(versicolor_meanLW, versicolor_prior_prob)
  res <- t(set_W[1:2]-ver_w[1:2])%*%x+(set_w[3]-ver_w[3])
  return(res)
}

set_vir_eq <- function(x){
  set_w <- weights(setosa_meanLW, setosa_prior_prob)
  vir_w <- weights(virginica_meanLW, virginica_prior_prob)
  res <- t(set_w[1:2]-vir_w[1:2])%*%x+(set_w[3]-vir_w[3])
  return(res)
}

ver_vir_eq <- function(x){
  ver_w <- weights(versicolor_meanLW, versicolor_prior_prob)
  vir_w <- weights(virginica_meanLW, virginica_prior_prob)
  res <- t(ver_w[1:2]-vir_w[1:2])%*%x+(ver_w[3]-vir_w[3])
  return(res)
}

seW <- weights(setosa_meanLW, setosa_prior_prob)
veW <- weights(versicolor_meanLW, setosa_prior_prob)
viW <- weights(virginica_meanLW, virginica_prior_prob)
coefSeVe <- seW-veW
coefSeVi <- seW-viW
coefVeVi <- veW-viW
width.intercept <- coefSeVe[3]/coefSeVe[2]
length.intercept <- coefSeVe[3]/coefSeVe[1]
m <- width.intercept/length.intercept
x <- seq(4,8,0.01)
y <- m*x+width.intercept
plot(x,y)

width.intercept <- coefSeVi[3]/coefSeVi[2]
length.intercept <- coefSeVi[3]/coefSeVi[1]
m <- width.intercept/length.intercept
x <- seq(4,8,0.01)
y <- m*x+width.intercept
points(x,y)

width.intercept <- coefVeVi[3]/coefVeVi[2]
length.intercept <- coefVeVi[3]/coefVeVi[1]
m <- width.intercept/length.intercept
x <- seq(4,8,0.01)
y <- m*x+width.intercept
points(x,y)

my_predict <- function(x){
  se <- setosa_disc_fun(x)
  ve <- versicolor_disc_fun(x)
  vi <- virginica_disc_fun(x)
  disc_mat <- cbind(se,ve,vi)
  y_hat <- rep("???",dim(disc_mat)[1])
  for (i in 1:dim(disc_mat)[1]){
    pred <- which.max(disc_mat[i,])
    if(pred == 1) {
      y_hat[i] <- 'setosa'
    } else if (pred == 2) {
      y_hat[i] <- 'versicolor'
    } else if (pred == 3) {
      y_hat[i] <- 'virginica'
    }
  }
  return(y_hat)
}

y_hat <- my_predict(cbind(iris$Sepal.Length, iris$Sepal.Width))
plot(iris$Sepal.Length,iris$Sepal.Width)
for (i in 1:length(y_hat)) {
  color <- y_hat[i]
  points(iris$Sepal.Length[i], iris$Sepal.Width[i], col = ifelse(color=='setosa','blue',ifelse(color=='versicolor','green', ifelse(color=='virginica','red','black'))))
}
missclass_rate_fun <- function(y,y_hat){
  wrong <- 0
  for (i in length(y)){
    if (y[i]!=y_hat[i]){
      wrong <- wrong + 1
    }
  }
  res <- wrong/length(y)
  return(res)
}
missclass_rate <- missclass_rate_fun(iris$Species, y_hat)
table(iris$Species, y_hat)

lda_fit <- lda(Species~Sepal.Length+Sepal.Width, iris)
pred <- predict(lda_fit, iris)
table(iris$Species, pred$class)

setosa_new <- rmvnorm(50, mean = setosa_meanLW, sigma = pooled_cov)
versicolor_new <- rmvnorm(50, mean = versicolor_meanLW, sigma = pooled_cov)
virginica_new <- rmvnorm(50, mean = virginica_meanLW, sigma = pooled_cov)
iris_new <- rbind(setosa_new,versicolor_new,virginica_new)

plot(iris_new)
points(setosa_new[,1],setosa_new[,2], col = 2)
points(versicolor_new[,1],versicolor_new[,2], col = 3)
points(virginica_new[,1],virginica_new[,2], col = 4)

log_reg_model <- multinom(Species~Sepal.Length+Sepal.Width, iris, model = TRUE)
y_hat_log_reg <- rep("???",dim(log_reg_model$fitted.values)[1])
for (i in 1:dim(log_reg_model$fitted.values)[1]){
  pred <- which.max(log_reg_model$fitted.values[i,])
  if(pred == 1){
    y_hat_log_reg[i] <- 'setosa'
  } else if (pred == 2) {
    y_hat_log_reg[i] <- 'versicolor'
  } else if (pred == 3) {
    y_hat_log_reg[i] <- 'virginica'
  }
}
plot(iris$Sepal.Length, iris$Sepal.Width)
for (i in 1:length(y_hat_log_reg)){
  color <- y_hat_log_reg[i]
  points(iris$Sepal.Length[i], iris$Sepal.Width[i], col = ifelse(color=="setosa", "blue", ifelse(color=="versicolor", "green", ifelse(color=="virginica", "red", "black"))))
}
plot(iris$Sepal.Length, iris$Sepal.Width)
for (i in 1:length(y_hat)){
  color <- y_hat[i]
  points(iris$Sepal.Length[i], iris$Sepal.Width[i], col = ifelse(color=="setosa", "blue", ifelse(color=="versicolor", "green", ifelse(color=="virginica", "red", "black"))))
}

log_reg_missclass_rate <- missclass_rate_fun(iris$Species, y_hat_log_reg)
print(log_reg_missclass_rate)

data=read.csv2(file= 'C:/Users/Daniel Bissessar/Desktop/TDDE01/Labs/bank-full.csv')
data[,2] = as.factor(data[,2])
data[,3] = as.factor(data[,3])
data[,4] = as.factor(data[,4])
data[,5] = as.factor(data[,5])
data[,7] = as.factor(data[,7])
data[,8] = as.factor(data[,8])
data[,9] = as.factor(data[,9])
data[,11] = as.factor(data[,11])
data[,16] = as.factor(data[,16])
data[,17] = as.factor(data[,17])


data = as.data.frame(data)
data = data[,-12]
library(tree)
library(naivebayes)
library(rpart)
library(rpart.plot)
library(caret)
library(e1071)

n=dim(data)[1]
set.seed(12345) 
id=sample(1:n, floor(n*0.4)) 
train=data[id,] 
id1=setdiff(1:n, id)
set.seed(12345) 
id2=sample(id1, floor(n*0.3)) 
valid=data[id2,]
id3=setdiff(id1,id2)
test=data[id3,]

fit <- tree(y~.,data=train)
plot(fit)
text(fit, pretty = 0)
summary(fit)
yfitoriginal <- predict(fit, newdata = valid, type = 'class')
confusionmatrixoriginal <- data.matrix(table(valid$y,yfitoriginal))
missclassoriginal <- 1-sum(diag(confusionmatrixoriginal)/sum(confusionmatrixoriginal))

fit7000 <- tree(y~.,data=train, minsize=7000)
plot(fit7000)
text(fit7000, pretty=0)
summary(fit7000)

fitmindev <- tree(y~.,data=train,mindev=0.0005)
plot(fitmindev)
text(fitmindev, pretty=0)
summary(fitmindev)
yfitmindev <- predict(fitmindev, newdata=valid, type='class')
confusionmatrixmindev <- data.matrix(table(valid$y,yfitmindev))
missclassmindev <- 1-sum(diag(confusionmatrixmindev))/sum(confusionmatrixmindev)

trainScore <- rep(0,50)
valScore <- rep(0,50)
for (i in 2:50) {
  prunedTree <- prune.tree(fitmindev, best = i)
  pred <- predict(prunedTree, newdata = valid, type = 'tree')
  trainScore[i] <- deviance(prunedTree)
  valScore[i] <- deviance(pred)
}
plot(2:50, trainScore[2:50], type='b', col = 2, ylim=c(0,20000))
points(2:50,valScore[2:50], type = 'b', col = 3)

finalTree <- prune.tree(fitmindev, best=21)
yfit <- predict(finalTree, newdata = valid, type = 'class')
confusionmatrix <- data.matrix(table(valid$y, yfit))
missclass <- 1-sum(diag(confusionmatrix))/sum(confusionmatrix)

fitNaive <- naiveBayes(formula=y~., data=train)
finalPredTest <- predict(finalTree, newdata = test, type = 'vector')
naivePredTest <- predict(fitNaive, newdata = test, type = 'raw')

sequence <- seq(0,0.95,0.05)
TPRFinal <- rep(0,length(sequence))
TPRNaive <- rep(0,length(sequence))
FPRFinal <- rep(0,length(sequence))
FPRNaive <- rep(0,length(sequence))

for (i in 1:length(sequence)) {
  TP1 = 0
  TP2 = 0
  FP1 = 0
  FP2 = 0
  for (j in 1:length(test$y)) {
    if(finalPredTest[j,2] > sequence[i]) {
      if(test$y[j] == 'yes') {
        TP1 <- TP1 +1
      } else {
        FP1 <- FP1 + 1
      }
    }
    if(naivePredTest[j,2] > sequence[i]){
      if(test$y[j] == "yes"){
        TP2 <- TP2 + 1
      }
      else{
        FP2 <- FP2 + 1
      }
    }
    
  }
  TPRFinal[i] = TP1/sum(test$y == "yes")
  TPRNaive[i] = TP2/sum(test$y == "yes")
  FPRFinal[i] = FP1/sum(test$y == "no")
  FPRNaive[i] = FP2/sum(test$y == "no")
  
}
plot(FPRFinal,TPRFinal)
plot(FPRNaive, TPRNaive)  


library(ggplot2)
library(boot)
data=read.csv(file= 'C:/Users/Daniel Bissessar/Desktop/TDDE01/Labs/communities.csv')

xData <- data
xData$ViolentCrimesPerPop <- c()

centered.xData <- scale(xData, scale = FALSE)
centered.scaled.xData <- scale(xData)
covData <- cov(centered.scaled.xData)
eigenX <- eigen(covData)

print(sum(eigenX$values[1:34]))
print(sum(eigenX$values[1:35]))

print(eigenX$values[1])
print(eigenX$values[2])

princompx <- princomp(xData, cor = TRUE, scores = TRUE)
plot(princompx$scores[,1])
plot(princompx$loadings[,1])
absScores <- abs(princompx$loadings[,1])
top5Scores <- tail(sort(unlist(absScores)),5)
absScores
top5Scores

pcs <- data.frame(Pc1=princompx$scores[,1],Pc2=princompx$scores[,2])
pcsbind <- cbind(pcs,ViolentCrimesPerPop=data$ViolentCrimesPerPop)
ggplot(pcsbind, mapping = aes(x=Pc1,y=Pc2, colour=ViolentCrimesPerPop))+geom_point()

pcsbind3 <- pcsbind
pcsbind3[,2] <- c()
fit <- lm(formula = pcsbind3[,2]~poly(pcsbind3[,1],degree = 2), data=pcsbind3)
fitPred <- predict(fit)

ggplot(pcsbind3, mapping = aes(x=Pc1, y=ViolentCrimesPerPop))+geom_point()+geom_line(aes(y=fitPred),color=2)

mle <- fit
data2 <- data.frame(Crimes=pcsbind3[,2],Pc1=pcsbind3[,1])

rng <- function(data,mle) {
  data1 <- data.frame(Crimes=data$Crimes, Pc1=data$Pc1)
  n <- length(data$Crimes)
  data1$Crimes=rnorm(n,predict(mle, newdata = data1),sd(mle$residuals))
  return(data1)
}
f1 <- function(data1){
  res <- lm(formula = Crimes~poly(Pc1, degree = 2), data = data1)
  crimePred <- predict(res, newdata=data2)
  return(crimePred)
}
res <- boot(data=data2, statistic=f1, R=1000, mle=mle, ran.gen = rng, sim = 'parametric')
e <- envelope(res, level = 0.95)
fitT4 <- lm(Crimes~poly(Pc1, degree = 2), data=data2)
crimesP <- predict(fitT4)
ggplot(data2, aes(x=Pc1,y=Crimes))+geom_point()+geom_ribbon(aes(ymin=e$point[2,],ymax=e$point[1,]), colour = 'red')+geom_smooth(mapping = aes(x=Pc1, y=crimesP))

f2 <- function(data1) {
  res <- lm(formula = Crimes~poly(Pc1, degree=2), data = data1)
  crimePred <- predict(res, newdata=data2)
  n <- length(data1$Crimes)
  predictedC <- rnorm(n,crimePred, sd(mle$residuals))
  return(predictedC)
}
res2=boot(data2, statistic = f2, R=1000, mle=mle, ran.gen = rng, sim="parametric")
e2 <- envelope(res2, level = 0.95)


ggplot(data2, aes(x=Pc1,y=Crimes))+
  geom_ribbon(aes(ymin=e2$point[2,], ymax=e2$point[1,]), colour="red")+
  geom_point()+
  geom_smooth( mapping = aes(x=Pc1, y=crimesP))
