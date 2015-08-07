
compute_rf <- function(co2conc, initial = 283.9660) {
  alpha <- 5.35
  
  rf <- alpha * log(co2conc / initial)
  
  return(rf)
}


compute_delta_T <- function(time, rf) {
  # Compute temperature change from radiative forcing
  # time<-years<-2000:2002
  # myCO2<-seq(from=290,to=300,len=length(time))
  # rf<-compute_rf(myCO2)
  c1 <- 0.631
  c2 <- 0.429
  d1 <- 8.4
  d2 <- 409.5
  
  # Adjust c1 and d1 values to better correspond with CMIP5 results
  sf <- 0.1054275
  c1 <- c1 - (sf * c1)
  d1 <- d1 - (sf * d1)
  
  int_big_delta_T    <- double(length(time))
  int_big_delta_T[1] <- 0.
  for (j in 2:length(time)) {
    # for year j: j<-2; j<-3
    k <- rev(1:j) - 1
    delta_T <- (c1/d1) * exp(-k/d1) + (c2/d2) * exp(-k/d2)
    int_big_delta_T[j] <- rf[1] * delta_T[1]
    for (i in 2:j) {
      # i<-2
      int_big_delta_T[j] <- int_big_delta_T[j] + rf[i] * delta_T[i]
    }
  }
  
  return(int_big_delta_T)
}



######################################################################
### comparing original R code to python code for accurracy ###########

### using simulated data
time<-seq(from=2000,to=2010,by=1)
co2<-seq(from=290,to=300,len=11)

rf=compute_rf(co2)
compute_delta_T(time, rf)

### using real data downloaded, cleaned, and saved to csv in python
setwd("~/Documents/PROJECTS/Sesync/model_py")
getwd()
system('ls')
climateDat<-read.table("climateDat.csv",sep=",",header=TRUE)
climateDat$Year <- as.numeric(sub(' *\\-.*$', '', as.character(climateDat$Year)))
head(climateDat)

model_rf <- compute_rf(climateDat$annual_CO2)
model_delta_T <- compute_delta_T(climateDat$Year, model_rf)

# examining beginning and end of output for comparison to R
head(model_rf)
# 0.5714259 0.5871915 0.5995151 0.6131547 0.6223029 0.6327609
tail(model_rf)
# 1.661221 1.696176 1.720112 1.750476 1.786181 1.814206

head(model_delta_T)
# 0.00000000 0.08289799 0.11837983 0.15055578 0.17949723 0.20570794
tail(model_delta_T)
# 0.9424666 0.9607743 0.9788250 0.9971429 1.0161051 1.0350501



### comparing R code to mathematica results ###########
co2<-read.table("mlDataAnnual.txt")
co2<-matrix(co2[,1],nrow=39,ncol=2,byrow=T)
co2<-as.data.frame(co2)
myTime<-co2[,1]
myCO2<-co2[,2]
model_rf <- compute_rf(myCO2)
model_delta_T <- compute_delta_T(myTime, model_rf)
results<-cbind(myTime,myCO2,model_delta_T)
plot(myTime,model_delta_T)

model_rf <- compute_rf(c(myCO2[1],myCO2[length(myCO2)]))
model_delta_T <- compute_delta_T(c(myTime[1],myTime[length(myTime)]), model_rf)

c(myCO2[1],myCO2[length(myCO2)])


myTime<-seq(from=2000,to=2100,by=1)
myCO2<-seq(from=285,to=350,len=length(myTime))
model_rf <- compute_rf(myCO2)
model_delta_T <- compute_delta_T(myTime, model_rf)
plot(myTime,model_delta_T)


######################################################################
### exploring code prior to translating to Mathematica

j<-10 # number of years in time series
k <- rev(1:j) - 1 # 9 8 7 6 5 4 3 2 1 0
delta_T <- (c1/d1) * exp(-k/d1) + (c2/d2) * exp(-k/d2)
# [1] 0.02370239 0.02693279 0.03062265 0.03483736 0.03965164 0.04515083 0.05143241 0.05860775
# [9] 0.06680406 0.07616667



