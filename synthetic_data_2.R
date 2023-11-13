###### Packages & Data #####

setwd("/Users/Ryan/PycharmProjects/master_thesis")

library(tidyverse)
library(tseries)
library(corrplot)
library(dlookr)
library(forecast)
library(ggfortify)
library(fabricatr)
library(ggdag)



##### SYNTHETIC DATA ######

#Synthetic Data Creation

synth_data <- function(num_groups, te_ind){
  data_list <- list()
  for(i in 1:num_groups){
    #seed changes for each change in group
    set.seed(num_groups*2 + i)
    #Covariate Generation
    N <- 100#number of obvs
    sb_rand = sample.int(99, 1)
    sb_rand_comp = 100 - sb_rand
    
    
    x1 <- simulate(
      Arima(ts(rnorm(N, 5, 1.5)),
            order = c(1, 0, 1),
            method = "ML"))
    
    x2 <- simulate(
      Arima(ts(rpois(N,1)),
            order = c(1, 0, 1),
            method = "ML"))
    
    x3 <- simulate(
      Arima(ts(runif(N, min = 0, max = 1)),
            order = c(1,0,1),
            method = "ML"))
    
    x4 <- simulate(
      Arima(ts(rnorm(N,0,.5) + x3^2),
            order = c(1, 0, 1),
            method = "ML"))
    
    x5 <- simulate(
      Arima(ts(3*x1/(cos(x3))),
            order = c(1, 0, 1),
            method = "ML"))
    
    x6 <- simulate(
      #Arima(ts(exp((1 + x2)/(x3 + sqrt(abs(x1)) ))),
      Arima(ts((1 + x2)/(x3) + sqrt(abs(x1)) )),
            order = c(1, 0, 1),
            method = "ML")
    
    cov_data <- cbind(x1,x2,x3,x4,x5,x6)
    
    cov_data <- as.data.frame(cov_data) 
    cov_data <- cov_data %>%
      mutate(x5 = imputate_outlier(cov_data, x5, method = "capping", cap_ntiles = c(0.03, 0.97)))
    #Creating Dataset, Target Parameter
    data <- fabricate(
      years = add_level(
        N = N, #number of years
        ts_year = 0:(N-1) 
        #year_shock = rnorm(N, -2, 0.3) #data that varies by year, static by group
      ),
      groups = add_level(
        N = 1, #number of groups
        group_name = i,
        base_te = runif(N, 3, 12),
        t_trend = runif(N, -.02, .02), #put if statement: if group_name divisable by 3, then ttrend = ...
        t_trend2 = runif(N, -.01, .01),
        t_trend3 = runif(N, -.03, .03),
        t_trend4 = runif(N, -.05, .05),
       
        #base_y = runif(N, 30, 50), #data that varies by group, static by year
        nest = FALSE
      ),
      group_years = cross_levels(
        by = join_using(years, groups), #data that varies by year and group
        t = ifelse(x1 + x5 < 20, 1, 0),
        t_cf = ifelse(t == 1, 0, 1),
        sb_dummy = c(rep(1,sb_rand),rep(-2,sb_rand_comp)),
        x1 = cov_data$x1 + ts_year*t_trend,
        x2 = cov_data$x2 + ts_year*t_trend2,
        x3 = (cov_data$x3 + ts_year*t_trend3)*sb_dummy,
        x4 = cov_data$x4 + ts_year*t_trend4,
        x5 = cov_data$x5 + ts_year*t_trend2,
        x6 = cov_data$x6 + ts_year*t_trend3,
        te_1 = exp(x3) + base_te + rnorm(N,0,2),
        te_2 = sqrt(abs(x1)) + 3*x4^2 + sin(2*x2) +.2*x5 + base_te + rnorm(N,0,2),
        te_3 = ((abs(ts_year - 50))^(1.008) + x6)*(x4+1)*0.5 + rnorm(N,0,2),
        
        y1 = 5 + sqrt(abs(x1)) + x1 + 2*abs(x2) + (-3*x3) + 3*x2*x4 + 2*abs(x2) + (exp(x3))/x5 + 
          (x4+1)^2 + 0.5*x5 + x6 + te_1*t + rnorm(N,0,3),
        
        y2 = 5 + sqrt(abs(x1)) + x1 + 2*abs(x2) + (-3*x3) + 3*x2*x4 + 2*abs(x2) + (exp(x3))/x5 + 
          (x4+1)^2 + 0.5*x5 + x6 + te_2*t + rnorm(N,0,3),
        
        y3 = 5 + sqrt(abs(x1)) + x1 + 2*abs(x2) + (-3*x3) + 3*x2*x4 + 2*abs(x2) + (exp(x3))/x5 + 
          (x4+1)^2 + 0.5*x5 + x6 + te_3*t + rnorm(N,0,3)
      )
    )
    
    
    data <- as.data.frame(sapply(data, as.numeric))
    data_list[[i]] <- data
    print(paste0("TS number ", i, " complete"))
  }
  
  merged_data <- do.call(rbind, data_list) %>%
    select(-c(years,groups,group_years)) %>%
    mutate(ts_year = ts_year + 1900) %>%
    select(ts_year,group_name,t,t_cf,x1,x2,x3,x4,x5,x6,y1,y2,y3,te_1,te_2,te_3)
  
  
  
  
  if(te_ind == 1){
  merged_data <- subset(merged_data, select = -c(y2,y3,te_2,te_3))
  }

  if(te_ind == 2){
  merged_data <- subset(merged_data, select = -c(y1,y3,te_1,te_3))
  }

  if(te_ind == 3){
  merged_data <- subset(merged_data, select = -c(y1,y2,te_1,te_2))
  }


                     
names(merged_data)[11] <- "y"
names(merged_data)[12] <- "te_true"
  #removing datasets with extreme outliers
  lb_t <- quantile(merged_data$y, 0.01)
  ub_t <- quantile(merged_data$y, 0.99)

  outliers <- merged_data %>%
    mutate( y = ifelse(y > ub_t | y < lb_t, NA, y)) %>%
    filter(is.na(y))

  #how many datasets with heavily skewed data
  length(unique(outliers$group_name))

  #vector of the above datasets
  out_group <- unique(outliers$group_name)

  #new merged data without as heavy skew on p5
  merged_data <- merged_data %>%
    dplyr::filter(!(group_name %in% out_group))
  return(merged_data)
}

#creating parameters for large datasets
num_groups <- c(30, 300, 3000)
te_ind <-  c(1,2,3) #can select 1, 2, 3
args <- expand.grid(num_groups, te_ind)
list2env(args, envir=.GlobalEnv)
#creating data
te1_list <- lapply(num_groups, 1, FUN = synth_data)
te2_list <- lapply(num_groups, 2, FUN = synth_data)
te3_list <- lapply(num_groups, 3, FUN = synth_data)
final_list <- c(te1_list, te2_list, te3_list)

#writing csvs
write.csv(final_list[[1]], "datasets/dataset_1.csv")
write.csv(final_list[[2]], "datasets/dataset_2.csv")
write.csv(final_list[[3]], "datasets/dataset_3.csv")
write.csv(final_list[[4]], "datasets/dataset_4.csv")
write.csv(final_list[[5]], "datasets/dataset_5.csv")
write.csv(final_list[[6]], "datasets/dataset_6.csv")
write.csv(final_list[[7]], "datasets/dataset_7.csv")
write.csv(final_list[[8]], "datasets/dataset_8.csv")
write.csv(final_list[[9]], "datasets/dataset_9.csv") 
