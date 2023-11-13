##### EMPIRICAL DATA ######

library(tidyverse)

#install.packages("prediction")
library(prediction)

#install.packages("grf")
library(grf)

#install.packages("Hmisc")
library(Hmisc)

library(tseries)

#install.packages("corrplot")
library(corrplot)

#install.packages("dlookr")
library(dlookr)

library(forecast)

#install.packages("ggfortify")
library(ggfortify)

#install.packages("fabricatr")
library(fabricatr)
#install.packages("ggdag")
library(ggdag)


#Load data
load("/Users/Ryan/Downloads/Acemoglu.RData")



#Data Cleaning

colnames(d2)[7] <- "pop14"
colnames(d2)[8] <- "pop1564"
d2 <- d2 %>%
  group_by(wbcode2) %>%
  mutate(l1 = lag(y,1), l2 = lag(y,2), l3 = lag(y,3), l4 = lag(y,4)) %>%
  drop_na()


#write csv to conduct analysis in python
setwd("/Users/Ryan/PycharmProjects/master_thesis")
write_csv(d2, "datasets/data_emp.csv")

#reload final dataset
acc_emp <- read.csv("acc_emp.csv")


y_emp <- acc_emp %>%
  ggplot(aes(x = ts_year, y = y)) +
  geom_line(color = "#69b3a2") +
  #scale_color_manual(values="#69b3a2") +
  scale_x_continuous(breaks = c(1962,1980,1990,2000,2010)) +
  labs(x = "Time",
       y = "Y",
       title = "Average Y Value vs Time")
y_emp

te_emp <- acc_emp %>%
  ggplot(aes(x = ts_year, y = te_hat)) +
  geom_line(color = "#404080") +
  #scale_color_manual(values="#69b3a2") +
  scale_x_continuous(breaks = c(1960,1980,1990,2000,2010)) +
  labs(x = "Time",
       y = "Treatment Effect",
       title = "Average Treatment Effect vs Time")
te_emp
ggsave("tehat_emp.pdf")


scale <- 25
dual_emp <- acc_emp %>%
  mutate(te_hat = te_hat*scale) %>%
  ggplot(aes(x = ts_year, y = y)) +
  geom_line(aes(color = "Y Value")) +
  geom_line(aes(y = te_hat, color = "Treatment Effect")) +
  #scale_x_continuous(breaks = c(1960,1980,1990,2000,2010)) +
  scale_y_continuous(sec.axis = sec_axis(~.* (1/scale), name="Treatment Effect")) +
  labs(x = "Time",
       y = "Y Value",
       color = "") +
  scale_color_manual(values = c("orange2", "gray30")) +
  theme(
    axis.title.y = element_text(color = "gray30"),
    axis.title.y.right = element_text(color = "orange3"),
    legend.position = "bottom"
  )
dual_emp
  


