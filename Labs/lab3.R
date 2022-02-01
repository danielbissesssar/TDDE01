set.seed(1234567890)
library(geosphere)
stations <- read.csv(file= 'C:/Users/Daniel Bissessar/Desktop/TDDE01/Labs/stations.csv')
temps <- read.csv(file= 'C:/Users/Daniel Bissessar/Desktop/TDDE01/Labs/temps50k.csv')


st <- merge(stations,temps,by="station_number")
h_distance <- 100000 
h_date <- 15  
h_time <- 4 
a <- 58.4108 # The point to predict (up to the students) Linköping: Breddgraden: 58.41080700000001 Längdgrad: 15.621372699999938
b <- 15.6213 
my_date <- "2013-11-04" # The date to predict (up to the students)
times <- c("04:00:00", "06:00:00", "08:00:00", "10:00:00", "12:00:00", "14:00:00", "16:00:00", "18:00:00", "20:00:00", "22:00:00", "24:00:00")
temp <- vector(length=length(times))
### Students' code here ###
library(dplyr)

filter_data <- function(){
  filtered_data <- filter(st, date < my_date)
  return(filtered_data)
}

dist_kernel <- function(fd){
  res1 <- vector(length=dim(fd)[1])
  diff <- vector(length=dim(fd)[1])
  for (i in 1:dim(fd)[1]){
    diff[i] <- distHaversine(c(b, a), c(fd$longitude[i], fd$latitude[i]))
    res1[i] <- exp(-1*(diff[i]/h_distance)^2)
  }
  return(cbind(res1, diff))
}

day_kernel <- function(fd){
  res1 <- vector(length=dim(fd)[1])
  diff <- vector(length=dim(fd)[1])
  split <- strsplit(my_date, "\\-")
  day_x <- as.numeric(split[[1]][3])
  month_X <- as.numeric(split[[1]][2])
  days_x <- ((month_X-1)*30) + day_x
  for (i in 1:dim(fd)[1]){
    split <- strsplit(fd$date[i], "\\-")
    day_fd <- as.numeric(split[[1]][3])
    month_fd <- as.numeric(split[[1]][2])
    days_fd <- ((month_fd-1)*30) + day_fd
    diff[i] <- abs(days_x - days_fd)
    res1[i] <- exp(-1*(diff[i]/h_date)^2)
  }
  return(cbind(res1, diff)) 
}

hour_kernel <- function(fd, time){
  res1 <- vector(length=dim(fd)[1])
  diff <- vector(length=dim(fd)[1])
  split <- strsplit(time, "\\:")
  hour_x <- as.numeric(split[[1]][1])
  for (i in 1:dim(fd)[1]){
    time_i <- strsplit(fd$time[i], "\\:")
    hour <- as.numeric(time_i[[1]][1])
    diff[i] <- abs(hour_x - hour)
    res1[i] <- exp(-1*(diff[i]/h_time)^2)
  }
  return(cbind(res1, diff))
}

temperatures1 <- function(fd){
  dist_k <- dist_kernel(fd)
  day_k <- day_kernel(fd)
  for (i in 1:length(times)){
    hour_k <- hour_kernel(fd, times[i])
    kernels <- dist_k[,1] + day_k[,1] + hour_k[,1]
    temp[i] <- sum(kernels*fd$air_temperature)/sum(kernels)
  }
  return(temp)
}

temperatures2 <- function(fd){
  dist_k <- dist_kernel(fd)
  day_k <- day_kernel(fd)
  for (i in 1:length(times)){
    hour_k <- hour_kernel(fd, times[i])
    kernels <- dist_k[,1]*day_k[,1]*hour_k[,1]
    temp[i] <- sum(kernels*fd$air_temperature)/sum(kernels)
  }
  return(temp)
}


test_data <- filter_data() 
temp_list_test <- temperatures1(test_data)
temp_list_test2 <- temperatures2(test_data)

plot(temp_list_test, type="o", xlab = "Time", ylab = "Temperature", xaxt="n")
axis(1, at=1:length(times), labels = times)
plot(temp_list_test2, type="o", xlab = "Time", ylab = "Temperature", xaxt="n")
axis(1, at=1:length(times), labels = times)

dis_gau_mat <- dist_kernel(test_data)
plot(dis_gau_mat[,2], dis_gau_mat[,1],xlim=c(0,500000))
day_gau_mat <- day_kernel(test_data)
plot(day_gau_mat[,2], day_gau_mat[,1], xlim=c(0, 100), xlab="Distance (Days)", ylab="Kernel values")

hou_gau_mat <- hour_kernel(test_data, times[1])
plot(hou_gau_mat[,2], hou_gau_mat[,1], xlab="Distance (Hours)", ylab="Kernel values")

library(neuralnet)
Var <- runif(500,0,10)
mydata <- data.frame(Var, Sin=sin(Var))
tr <- mydata[1:25,]
te <- mydata[26:500,]
winit <- runif(19,-1,1)
nn <- neuralnet(formula=Sin~Var, data=tr, hidden = 6, startweights = winit)
plot(tr, cex=2, xlim=c(-5,25),ylim=c(-2,2))
points(te, col = 'blue', cex=1)
points(te[,1],predict(nn,te),col='red', cex=1)

Var2 <- runif(500,0,20)
mydata2 <- data.frame(Var=Var2, Sin=sin(Var2))
plot(mydata2, cex=2, xlim=c(-5,25),ylim=c(-2,2))
points(mydata2[,1],predict(nn,mydata2),col='red',cex=1)

winit<- runif(19,-1,1)
Var3 <- runif(500,0,10)
mydata3 <- data.frame(Var=Var3, Sin=sin(Var3))
nn3 <- neuralnet(formula=Var~Sin,data=mydata3,hidden=6, startweights = winit)
plot(y=mydata3$Var, x=mydata3$Sin, cex=2)
points(mydata3$Sin,predict(nn3,mydata3),col = 'red', cex=1)
plot(nn3)
