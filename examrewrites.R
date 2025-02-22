#examrewrites
set.seed(1234567890)

N_class1 <- 1500
N_class2 <- 1000

data_class1 <- NULL
for(i in 1:N_class1) {
  a <- rbinom(1,1,0.3)
  b <- rnorm(1,15,3)*a+(1-a)*rnorm(1,4,2)
  data_class1 <- c(data_class1, b)
}

data_class2 <- NULL
for (i in 1:N_class2){
  a <- rbinom(1,1,0.4)
  b <- rnorm(1,15,3)*a+(1-a)*rnorm(1,4,2)
  data_class2 <- c(data_class2, b)
}
gaussian_k <- function(x,h) {
  return(exp(-(x^2)/(2*h*h)))
}

conditional_class1 <- function(t,h) {
  d <- 0
  for (i in 1:N_class1) {
    d <- d+gaussian_k(data_class1[i]-t,h)
  }
  return(d/N_class1)
}
h <- .5
hist(data_class1, xlim=c(-5,25), probability = TRUE)
xfit <- seq(-5,25,0.1)
yfit <- conditional_class1(xfit, h)
lines(xfit,yfit, lwd=2)

conditional_class2 <- function(t,h) {
  d <- 0
  for (i in 1:N_class2) {
    d <- d+gaussian_k(data_class2[i]-t,h)
  }
  return (d/N_class2)
}
h <- 0.5
hist(data_class2, xlim=c(-5,25), probability = TRUE)
xfit <- seq(-5,25, 0.1)
yfit <- conditional_class2(xfit, h)
lines(xfit,yfit, lwd=2)

prob_class1 <- function(t,h) {
  prob_class1 <- conditional_class1(t,h)*N_class1/(N_class1+N_class2)
  prob_class2 <- conditional_class2(t,h)*N_class2/(N_class1+N_class2)
  return(prob_class1/(prob_class1+prob_class2))
}

h <- 0.5
xfit <- seq(-5,25,0.1)
yfit <- probclass(xfit, h)
plot(xfit,yfit,lwd=2)
abline(h=0.5)

library(neuralnet)

da <- iris[sample(1:150, 150),]
tr <- da[1:50,]
va <- da[51:100,]
te <- da[101:150,]
labels <- unique(tr[,5])

nn <- neuralnet(Species ~ Sepal.Length, tr, linear.output = FALSE)
right <- 0
for(i in 1:50)
  if(labels[which.max(predict(nn,va[i,]))]==va[i,5])
    right <- right+1
right/50

nn <- neuralnet(Species ~ Sepal.Width, tr, linear.output = FALSE)
right <- 0
for(i in 1:50)
  if(labels[which.max(predict(nn,va[i,]))]==va[i,5])
    right <- right+1
right/50

nn <- neuralnet(Species ~ Petal.Length, tr, linear.output = FALSE)
right <- 0
for(i in 1:50)
  if(labels[which.max(predict(nn,va[i,]))]==va[i,5])
    right <- right+1
right/50

nn <- neuralnet(Species ~ Petal.Width, tr, linear.output = FALSE)
right <- 0
for(i in 1:50)
  if(labels[which.max(predict(nn,va[i,]))]==va[i,5])
    right <- right+1
right/50

nn <- neuralnet(Species ~ Sepal.Length + Sepal.Width, tr, linear.output = FALSE)
right <- 0
for(i in 1:50)
  if(labels[which.max(predict(nn,va[i,]))]==va[i,5])
    right <- right+1
right/50

nn <- neuralnet(Species ~ Petal.Width, cbind(tr,va), linear.output = FALSE)
right <- 0
for(i in 1:50)
  if(labels[which.max(predict(nn,te[i,]))]==te[i,5])
    right <- right+1
right/50





