################
# Kernel methods for both 732A99 and TDDE01
################

set.seed(1234567890)

# N_class1 <- 1500
# N_class2 <- 1000

# data_class1 <- NULL
# for(i in 1:N_class1){
#   a <- rbinom(n = 1, size = 1, prob = 0.3)
#   b <- rnorm(n = 1, mean = 15, sd = 3) * a + (1-a) * rnorm(n = 1, mean = 4, sd = 2)
#   data_class1 <- c(data_class1,b)
# }

# data_class2 <- NULL
# for(i in 1:N_class2){
#   a <- rbinom(n = 1, size = 1, prob = 0.4)
#   b <- rnorm(n = 1, mean = 10, sd = 5) * a + (1-a) * rnorm(n = 1, mean = 15, sd = 2)
#   data_class2 <- c(data_class2,b)
# }

# data <- rbind(cbind(data_class1,1),cbind(data_class2,2))
# write.table(data,"dataKernel.txt")

data <- read.table("dataKernel.txt")
index <- sample(1:2500)
tr <- data[index[1:1500],]
va <- data[index[1501:2000],]
te <- data[index[2001:2500],]

classify <- function(t, h, data){
  cc1 <- 0
  n1 <- 0
  cc2 <- 0
  n2 <- 0
  for(i in 1:nrow(data)){
    if(data[i,2]==1){
      cc1 <- cc1+dnorm(x=data[i,1]-t,sd=h)
      n1 <- n1+1
    }
    else
    {
      cc2 <- cc2+dnorm(x=data[i,1]-t,sd=h)
      n2 <- n2+1
    }
  }
  
  cc1 <- cc1/n1
  cc2 <- cc2/n2
  c <- 1
  if(cc1*n1/nrow(data)<cc2*n2/nrow(data))
    c <- 2
  
  return (c)
}

acc <- 0
for(i in 1:nrow(va))
  if(classify(va[i,1],0.5,tr)==va[i,2])
    acc <- acc+1
acc/nrow(va)

acc <- 0
for(i in 1:nrow(va))
  if(classify(va[i,1],1,tr)==va[i,2])
    acc <- acc+1
acc/nrow(va)

acc <- 0
for(i in 1:nrow(va))
  if(classify(va[i,1],5,tr)==va[i,2])
    acc <- acc+1
acc/nrow(va)

acc <- 0
for(i in 1:nrow(va))
  if(classify(va[i,1],10,tr)==va[i,2])
    acc <- acc+1
acc/nrow(va)

# h=1 is selected sincee it achieves higher accuracy in the validation set.
# To estimate the accuracy on the test set, use tr and va as training data.

acc <- 0
for(i in 1:nrow(te))
  if(classify(te[i,1],1,rbind(tr,va))==te[i,2])
    acc <- acc+1
acc/nrow(te)

############################################################################################
# Neural networks for TDDE01 only
############################################################################################

set.seed(1234567)

Var <- runif(100, -2,2)
trva <- data.frame(Var, Sin=abs(Var))
tr <- trva[1:50,] # Training
va <- trva[51:100,] # Validation

# plot(trva)
# plot(tr)
# plot(va)

w_j <- runif(10, -1, 1)
b_j <- runif(10, -1, 1)
w_k <- runif(10, -1, 1)
b_k <- runif(1, -1, 1)

l_rate <- 1/nrow(tr)^2
n_ite = 1000
error <- rep(0, n_ite)
error_va <- rep(0, n_ite)

for(i in 1:n_ite) {
  
  for(n in 1:nrow(tr)) {
    
    z_j <- pmax(0,w_j * tr[n,]$Var + b_j)
    y_k <- sum(w_k * z_j) + b_k
    
    error[i] <- error[i] + (y_k - tr[n,]$Sin)^2
    
  }
  
  for(n in 1:nrow(va)) {
    
    z_j <- pmax(0,w_j * va[n,]$Var + b_j)
    y_k <- sum(w_k * z_j) + b_k
    
    error_va[i] <- error_va[i] + (y_k - va[n,]$Sin)^2
    
  }
  
  cat("i: ", i, ", error: ", error[i]/2, ", error_va: ", error_va[i]/2, "\n")
  flush.console()
  
  for(n in 1:nrow(tr)) {
    
    # forward propagation
    
    z_j <- pmax(0,w_j * tr[n,]$Var + b_j)
    y_k <- sum(w_k * z_j) + b_k
    
    # backward propagation
    
    d_k <- y_k - tr[n,]$Sin
    d_j <- (z_j>0) * w_k * d_k
    partial_w_k <- d_k * z_j
    partial_b_k <- d_k
    partial_w_j <- d_j * tr[n,]$Var
    partial_b_j <- d_j
    w_k <- w_k - l_rate * partial_w_k
    b_k <- b_k - l_rate * partial_b_k
    w_j <- w_j - l_rate * partial_w_j
    b_j <- b_j - l_rate * partial_b_j
    
  }
  
}

w_j
b_j
w_k
b_k

plot(error/2, ylim=c(0, 10))
points(error_va/2, col = "red")

# prediction on training data

pred <- matrix(nrow=nrow(tr), ncol=2)

for(n in 1:nrow(tr)) {
  
  z_j <- pmax(0,w_j * tr[n,]$Var + b_j)
  y_k <- sum(w_k * z_j) + b_k
  pred[n,] <- c(tr[n,]$Var, y_k)
  
}

plot(pred)
points(tr, col = "red")

# prediction on validation data

pred <- matrix(nrow=nrow(va), ncol=2)

for(n in 1:nrow(va)) {
  
  z_j <- pmax(0,w_j * va[n,]$Var + b_j)
  y_k <- sum(w_k * z_j) + b_k
  pred[n,] <- c(va[n,]$Var, y_k)
  
}

plot(pred)
points(va, col = "red")