###### Packages & Data #####

setwd("/Users/Ryan/PycharmProjects/master_thesis")

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


#What to include
  #DAG
  #bias plots
  #error plots
  #time plots

#DAG

dag <- ggdag::dagify(Y ~ x1 + x2 + x3 + x4 + x5 + x6 + D,
                     D ~ x1 + x5,
                     x4 ~ x3,
                     x5 ~ x1 + x3,
                     x6 ~ x1 + x2 + x3)

ggdag::ggdag(dag) +
  theme_void()

#Synth Data Load
data_list <- c("dataset_1.csv", "dataset_2.csv", "dataset_3.csv", "dataset_4.csv",
               "dataset_5.csv", "dataset_6.csv", "dataset_7.csv", "dataset_8.csv", 
               "dataset_9.csv")



#create average dataset for corrplot, varplot, treatplot, etc
data_master <- do.call(rbind, lapply(data_list, read_csv))
data_master2 <- data_master %>%
  group_by(ts_year) %>%
  summarise(across(everything(), mean))






#dataset of relevant covariates

corr_data <-data_master2 %>%
  select(ts_year, group_name, t, x1, x2, x3, x4, x5, x6, y, te_true)



#converting dataset to long format for ggplot
fab_long <- corr_data %>% 
  pivot_longer(
    cols = 3:11, 
    names_to = c("variable"),
    values_to = "value")

#averaging over years
avg_group <- fab_long %>%
  group_by(ts_year,variable) %>%
  dplyr::summarize(value = mean(value))

#wide version

avg_group_wide <- avg_group %>%
  pivot_wider(
    names_from = variable,
    values_from = value
  )


#ALL VARIABLES PLOT
#plot of all variables against mean
var_plot <- ggplot() + 
  geom_line(data = fab_long, aes(x = ts_year, y = value, group = variable), color = "grey80", alpha = .7) +
  geom_line(data = avg_group, mapping = aes(x = ts_year, y = value, color = variable)) +
  facet_wrap(.~variable, scales = "free_y")
var_plot


#TREATMENT PLOT
#treatment (0/1) vs target value (density)
plot_01 <- ggplot(corr_data, aes(x = y, fill = as.factor(t))) +
  geom_density(alpha = 0.5) +
  scale_fill_discrete(name="Treatment\nAssignment") + 
  scale_fill_manual(values=c("#69b3a2", "#404080")) +
  labs(title = "Treated (1) vs. Non-Treated (0) Y-Value") +
  guides(fill=guide_legend(title="Treatment"))
plot_01
ggsave("treat_density_d0.pdf")


#CORRPLOT
#correlation matrix
corr_matrix_data <- avg_group_wide %>%
  select(-c(te_true,ts_year))
corr_matrix <- cor(corr_matrix_data)
pdf(file = "corr_plot_d0.pdf")
corr_plot <- corrplot(corr_matrix) 
corr_plot
dev.off()


#TE PLOT TRUE
te_sample <- avg_group %>%
  filter(variable == "te_true")

te_plot <- ggplot(te_sample, aes(x = ts_year, y = value)) +
  geom_line(color = "#69b3a2") +
  geom_smooth(color = "#404080") +
  labs(x = "year",
       y = "value",
       title = "Average TE (true) vs Time")
te_plot
ggsave("te_d0.pdf")



#read result frames

acc_total <- read.csv("acc_total.csv") %>%
  mutate(model = replace(model, model == "twfe", "TWFE")) %>%
  mutate(model = replace(model, model == "knn", "KNN")) %>%
  mutate(model = replace(model, model == "dml", "DML-L")) %>%
  mutate(model = replace(model, model == "cf", "DML-RF")) %>%
  mutate(model = replace(model, model == "nn", "TNet")) %>%
  mutate(model = replace(model, model == "gan", "GANITE"))
all_metrics <- read.csv("all_metrics.csv") %>%
  mutate(method = replace(method, method == "twfe", "TWFE")) %>%
  mutate(method = replace(method, method == "knn", "KNN")) %>%
  mutate(method = replace(method, method == "dml", "DML-L")) %>%
  mutate(method = replace(method, method == "cf", "DML-RF")) %>%
  mutate(method = replace(method, method == "nn", "TNet")) %>%
  mutate(method = replace(method, method == "gan", "GANITE"))
final_table <- read.csv("final_table.csv") %>%
  mutate(method = replace(method, method == "twfe", "TWFE")) %>%
  mutate(method = replace(method, method == "knn", "KNN")) %>%
  mutate(method = replace(method, method == "dml", "DML-L")) %>%
  mutate(method = replace(method, method == "cf", "DML-RF")) %>%
  mutate(method = replace(method, method == "nn", "TNet")) %>%
  mutate(method = replace(method, method == "gan", "GANITE")) %>%
  mutate(across(where(is.numeric), ~ round(., 3))) %>%
  arrange(rmse)

write_csv(final_table, "agg_table_final.csv")
  


#add data, n, delta columns
acc_total <- acc_total %>%
  mutate(data = c(rep(1,100),rep(2,100),rep(3,100),rep(4,100),rep(5,100),rep(6,100),
                  rep(7,100),rep(8,100),rep(9,100),
                  rep(1,100),rep(2,100),rep(3,100),rep(4,100),rep(5,100),rep(6,100),
                  rep(7,100),rep(8,100),rep(9,100),
                  rep(1,100),rep(2,100),rep(3,100),rep(4,100),rep(5,100),rep(6,100),
                  rep(7,100),rep(8,100),rep(9,100),
                  rep(1,100),rep(2,100),rep(3,100),rep(4,100),rep(5,100),rep(6,100),
                  rep(7,100),rep(8,100),rep(9,100),
                  rep(1,100),rep(2,100),rep(3,100),rep(4,100),rep(5,100),rep(6,100),
                  rep(7,100),rep(8,100),rep(9,100),
                  rep(1,100),rep(2,100),rep(3,100),rep(4,100),rep(5,100),rep(6,100),
                  rep(7,100),rep(8,100),rep(9,100)
                 )) %>%
  mutate(n = ifelse(data == 1 | data == 4 | data == 7, 25,
                    ifelse(data == 2 | data == 5 | data == 8, 250, 2500))) %>%
  mutate(delta = ifelse(data == 1 | data == 2 | data == 3, 1,
                        ifelse(data == 4 | data == 5 | data == 6, 2, 3)))

all_metrics <- all_metrics %>%
  mutate(n = ifelse(data == 1 | data == 4 | data == 7, 25,
                    ifelse(data == 2 | data == 5 | data == 8, 250, 2500))) %>%
  mutate(delta = ifelse(data == 1 | data == 2 | data == 3, 1,
                        ifelse(data == 4 | data == 5 | data == 6, 2, 3)))


testmet <- all_metrics %>%
  filter(method == "TWFE" | method == "DML-L")

#individual tables

indtables <- function(model){
  
  model_table <- all_metrics %>%
    filter(method == model) %>%
    select(delta, n, rmse, bias, comp_time)
  
}

table_twfe <- indtables("TWFE")
table_knn <- indtables("KNN")
table_dmll <- indtables("DML-L")
table_dmlrf <- indtables("DML-RF")
table_tnet <- indtables("TNet")
table_ganite <- indtables("GANITE")

write_csv(table_twfe, "table_twfe.csv")
write_csv(table_knn, "table_knn.csv")
write_csv(table_dmll, "table_dmll.csv")
write_csv(table_dmlrf, "table_dmlrf.csv")
write_csv(table_tnet, "table_tnet.csv")
write_csv(table_ganite, "table_ganite.csv")


#delta 3 table

d3_table <- all_metrics %>%
  filter(delta == 3) %>%
  filter(method != "KNN" & method != "GANITE") %>%
  mutate(across(where(is.numeric), ~ round(., 3))) %>%
  arrange(method, n) %>%
  select(method, n, rmse, bias, comp_time)
write_csv(d3_table, "d3_table.csv")

#n = 2500 table

n2500_table <- all_metrics %>%
  filter(n == 2500) %>%
  mutate(across(where(is.numeric), ~ round(., 3))) %>%
  arrange(method, n) %>%
  select(method, delta, rmse, bias, comp_time)
write_csv(n2500_table, "n2500_table.csv")



#create comp time table
comp_table <- all_metrics %>%
  select(method,comp_time, n) %>%
  group_by(method, n) %>%
  dplyr::summarize(comp_time = mean(comp_time)) %>%
  pivot_wider(names_from = method, values_from = comp_time)


#Line Chart

label_names <- c(
  '1' = "Delta (1)",
  '2' = "Delta (2)",
  '3' = "Delta (3)"
)

#line graph (d = all)
rmse_graph <- all_metrics %>%
  mutate(n = log(n)) %>%
  mutate_at("n", as.factor) %>%
  # group_by(method, n) %>%
  # dplyr::summarize(rmse = mean(rmse)) %>%
  ggplot(aes(x=n, y=rmse, group=method, color=method)) +
  geom_line() +
  #scale_color_viridis(discrete = TRUE) +
  ggtitle("RMSE vs Sample Size") +
  scale_x_discrete(labels = c('25','250','2500')) +
  theme(plot.title = element_text(hjust = 0.5)) +
  guides(color=guide_legend(title="Model")) +
  #theme_ipsum() +
  ylab("RMSE") +
  xlab("Log(N)") +
  facet_wrap(~delta, labeller = as_labeller(label_names))
rmse_graph
ggsave("rmse_graph.pdf")



#Ridgeline Plot
#install.packages("ggridges")
library(ggridges)


#test w acc_total (delta = 1,2,3)
ridge_delta <- ggplot(acc_total, aes(x = error, y = model, fill = model)) +
  geom_density_ridges() +
  theme_ridges() + 
  theme(legend.position = "none") +
  geom_vline(xintercept = 0, linetype = "dashed") +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(title = "Bias Distribution",
       x = "Absolute Bias",
       y = "Method") +
  facet_grid(~delta, labeller = as_labeller(label_names)) +
  theme(panel.spacing.x = unit(8, "mm"))
ridge_delta

#total average ridge (d = agg)
ridge_avg <- ggplot(acc_total, aes(x = error, y = model, fill = model)) +
  geom_density_ridges() +
  theme_ridges() + 
  theme(legend.position = "none") +
  geom_vline(xintercept = 0, linetype = "dashed") +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(title = "Bias Distribution (Aggregated)",
       x = "Absolute Bias",
       y = "Method")
ridge_avg


#Time Series

#frame with all methods (d = 7)


ts_all <- acc_total %>%
  select(ts_year, model, te_true, te_hat, n, delta) %>%
  filter(n == 2500) %>%
  #filter(model != "KNN" & model != "GANITE") %>%
  ggplot(aes(x = ts_year, y = te_hat, group = model, color = model)) +
  geom_line() +
  geom_smooth(aes(y = te_true, color = "te_true"), color = "black") +
  scale_x_discrete(breaks = c(1900,1925,1950,1975,1999)) +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(x = "Time",
       y = "Value",
       title = "All Methods vs. True Treatment Effect",
       color = "Method") +
  facet_grid(delta~model) +
  theme(panel.spacing.x = unit(7, "mm"))
ts_all

#best method + twfe (d = 7)
ts_top1 <- acc_total %>%
  select(ts_year, model, te_true, te_hat, n, delta) %>%
  filter(delta == 3 & n == 250) %>%
  filter(model == "DML-RF" | model == "TWFE") %>%
  ggplot(aes(x = ts_year, y = te_hat, group = model, color = model)) +
  geom_line() +
  geom_smooth(aes(y = te_true, color = "te_true"), color = "black") +
  scale_x_discrete(breaks = c(1900,1925,1950,1975,1999)) +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(x = "Time",
       y = "Value",
       title = "Predicted (CF, TWFE) vs True TE",
       color = "Model")
ts_top1


#best method (d = all)
ts_top1_all <- acc_total %>%
  select(ts_year, model, te_true, te_hat, n, delta) %>%
  filter(model == "DML-RF" | model == "TWFE") %>%
  ggplot(aes(x = ts_year, y = te_hat, group = model, color = model)) +
  geom_line() +
  geom_smooth(aes(y = te_true, color = "te_true"), color = "black") +
  facet_grid(n~delta) +
  scale_x_discrete(breaks = c(1900,1925,1950,1975)) +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(x = "Time",
       y = "Value",
       title = "Predicted (CF, TWFE) vs True TE",
       color = "Model")
ts_top1_all




#loop for all methods (d = all)

model_list <- c("DML-RF", "DML-L", "GANITE", "KNN", "TNet", "TWFE")

for (i in model_list){
  ts_ind <- acc_total %>%
    select(ts_year, model, te_true, te_hat, n, delta) %>%
    filter(model == i) %>%
    ggplot(aes(x = ts_year, y = te_hat, group = model, color = model)) +
    geom_line() +
    geom_smooth(aes(y = te_true, color = "te_true")) +
    scale_x_discrete(breaks = c(1900,1925,1950,1975)) +
    theme(plot.title = element_text(hjust = 0.5)) +
    facet_grid(n~delta) +
    labs(x = "Time",
         y = "Value",
         title = paste0("Actual/Predicted TE (", i, ") vs Time"))
  print(ts_ind)
  ggsave(paste0("ts_",i,".pdf"))
  
}










#RESULTS DATA LOAD
setwd("/Users/Ryan/PycharmProjects/master_thesis")
temp = list.files(pattern="acc_*")
temp
myfiles = lapply(temp, read.csv)

long_convert <- function(x){
  
  x <- x %>%
    pivot_longer(cols = -c(ts_year, model), names_to = "metric", values_to = "value")
  
}

myfiles_long <- lapply(myfiles, long_convert)

acc_final <- do.call(rbind,myfiles_long)

methods <- c("twfe", "knn", "dml", "cf", "nn", "gan")

acc_twfe <- as.data.frame(myfiles_long[6])
acc_knn <- as.data.frame(myfiles_long[4])
acc_dml <- as.data.frame(myfiles_long[2])
acc_cf <- as.data.frame(myfiles_long[1])
acc_nn <- as.data.frame(myfiles_long[5])
acc_gan <- as.data.frame(myfiles_long[3])


setwd("/Users/Ryan/PycharmProjects/master_thesis/plots")


#TE true vs TE hat
hist_twfe <- acc_twfe %>%
  filter(metric == "te_true" | metric == "te_hat") %>%
  ggplot(aes(x=value, fill = metric)) +
  geom_histogram(color = "#e9ecef", alpha = .6, position = "identity") +
  scale_fill_manual(values=c("#69b3a2", "#404080")) +
  labs(fill="", title = "TWFE Treatment Effect vs True")
hist_twfe
ggsave("twfe_hist_d0.pdf")

hist_knn <- acc_knn %>%
  filter(metric == "te_true" | metric == "te_hat") %>%
  ggplot(aes(x=value, fill = metric)) +
  geom_histogram(color = "#e9ecef", alpha = .6, position = "identity") +
  scale_fill_manual(values=c("#69b3a2", "#404080")) +
  labs(fill="", title = "KNN Treatment Effect vs True")
hist_knn 
ggsave("knn_hist_d0.pdf")

hist_dml <- acc_dml %>%
  filter(metric == "te_true" | metric == "te_hat") %>%
  ggplot(aes(x=value, fill = metric)) +
  geom_histogram(color = "#e9ecef", alpha = .6, position = "identity") +
  scale_fill_manual(values=c("#69b3a2", "#404080")) +
  labs(fill="", title = "DML Treatment Effect vs True")
hist_dml 
ggsave("dml_hist_d0.pdf")

hist_cf <- acc_cf %>%
  filter(metric == "te_true" | metric == "te_hat") %>%
  ggplot(aes(x=value, fill = metric)) +
  geom_histogram(color = "#e9ecef", alpha = .6, position = "identity") +
  scale_fill_manual(values=c("#69b3a2", "#404080")) +
  labs(fill="", title = "CF Treatment Effect vs True")
hist_cf 
ggsave("cf_hist_d0.pdf")

hist_nn <- acc_nn %>%
  filter(metric == "te_true" | metric == "te_hat") %>%
  ggplot(aes(x=value, fill = metric)) +
  geom_histogram(color = "#e9ecef", alpha = .6, position = "identity") +
  scale_fill_manual(values=c("#69b3a2", "#404080")) +
  labs(fill="", title = "NN Treatment Effect vs True")
hist_nn 
ggsave("nn_hist_d0.pdf")

hist_gan <- acc_gan %>%
  filter(metric == "te_true" | metric == "te_hat") %>%
  ggplot(aes(x=value, fill = metric)) +
  geom_histogram(color = "#e9ecef", alpha = .6, position = "identity") +
  scale_fill_manual(values=c("#69b3a2", "#404080")) +
  labs(fill="", title = "GAN Treatment Effect vs True")
hist_gan 
ggsave("gan_hist_d0.pdf")


#Error Histograms
hist_twfe_error <- acc_twfe %>%
  filter(metric == "error") %>%
  ggplot(aes(x=value, fill = metric)) +
  geom_histogram(bins = 15, color = "#e9ecef", alpha = .9, position = "identity") +
  scale_fill_manual(values="#404080") +
  labs(fill="", title = "TWFE Error Plot")
hist_twfe_error
ggsave("twfe_error_d0.pdf")

hist_knn_error <- acc_knn %>%
  filter(metric == "error") %>%
  ggplot(aes(x=value, fill = metric)) +
  geom_histogram(bins = 15, color = "#e9ecef", alpha = .9, position = "identity") +
  scale_fill_manual(values="#404080") +
  labs(fill="", title = "KNN Error Plot")
hist_knn_error 
ggsave("knn_error_d0.pdf")

hist_dml_error <- acc_dml %>%
  filter(metric == "error") %>%
  ggplot(aes(x=value, fill = metric)) +
  geom_histogram(bins = 15, color = "#e9ecef", alpha = .9, position = "identity") +
  scale_fill_manual(values="#404080") +
  labs(fill="", title = "DML Error Plot")
hist_dml_error 
ggsave("dml_error_d0.pdf")

hist_cf_error <- acc_cf %>%
  filter(metric == "error") %>%
  ggplot(aes(x=value, fill = metric)) +
  geom_histogram(bins = 15, color = "#e9ecef", alpha = .9, position = "identity") +
  scale_fill_manual(values="#404080") +
  labs(fill="", title = "CF Error Plot")
hist_cf_error 
ggsave("cf_error_d0.pdf")

hist_nn_error <- acc_nn %>%
  filter(metric == "error") %>%
  ggplot(aes(x=value, fill = metric)) +
  geom_histogram(bins = 15, color = "#e9ecef", alpha = .9, position = "identity") +
  scale_fill_manual(values="#404080") +
  labs(fill="", title = "NN Error Plot")
hist_nn_error 
ggsave("nn_error_d0.pdf")

hist_gan_error <- acc_gan %>%
  filter(metric == "error") %>%
  ggplot(aes(x=value, fill = metric)) +
  geom_histogram(bins = 15, color = "#e9ecef", alpha = .9, position = "identity") +
  scale_fill_manual(values="#404080") +
  labs(fill="", title = "GAN Error Plot")
hist_gan_error 
ggsave("gan_error_d0.pdf")



#Add Time Series Plots Here

te_hat_twfe <- acc_twfe %>%
  mutate(across('ts_year', str_replace, '-01-01', '')) %>%
  filter(metric == "te_hat" | metric == "te_true") %>%
  ggplot(aes(x = ts_year, y = value, group = metric, color = metric)) +
  geom_line() +
  scale_color_manual(values=c("#69b3a2", "#404080")) +
  scale_x_discrete(breaks = c("1900", "1925", "1950", "1975", "1999")) +
  labs(x = "Time",
       y = "Value",
       title = "Average TE (TWFE) vs Time")
te_hat_twfe
ggsave("tehat_twfe_d0.pdf")

te_hat_knn <- acc_knn %>%
  filter(metric == "te_hat" | metric == "te_true") %>%
  ggplot(aes(x = ts_year, y = value, group = metric, color = metric)) +
  geom_line() +
  scale_color_manual(values=c("#69b3a2", "#404080")) +
  scale_x_continuous(breaks = c(1900,1925,1950,1975,1999)) +
  labs(x = "Time",
       y = "Value",
       title = "Average TE (KNN) vs Time")
te_hat_knn
ggsave("tehat_knn_d0.pdf")

te_hat_dml <- acc_dml %>%
  filter(metric == "te_hat" | metric == "te_true") %>%
  ggplot(aes(x = ts_year, y = value, group = metric, color = metric)) +
  geom_line() +
  scale_color_manual(values=c("#69b3a2", "#404080")) +
  scale_x_continuous(breaks = c(1900,1925,1950,1975,1999)) +
  labs(x = "Time",
       y = "Value",
       title = "Average TE (DML) vs Time")
te_hat_dml
ggsave("tehat_dml_d0.pdf")

te_hat_cf <- acc_cf %>%
  filter(metric == "te_hat" | metric == "te_true") %>%
  ggplot(aes(x = ts_year, y = value, group = metric, color = metric)) +
  geom_line() +
  scale_color_manual(values=c("#69b3a2", "#404080")) +
  scale_x_continuous(breaks = c(1900,1925,1950,1975,1999)) +
  labs(x = "Time",
       y = "Value",
       title = "Average TE (CF) vs Time")
te_hat_cf
ggsave("tehat_cf_d0.pdf")

te_hat_nn <- acc_nn %>%
  filter(metric == "te_hat" | metric == "te_true") %>%
  ggplot(aes(x = ts_year, y = value, group = metric, color = metric)) +
  geom_line() +
  scale_color_manual(values=c("#69b3a2", "#404080")) +
  scale_x_continuous(breaks = c(1900,1925,1950,1975,1999)) +
  labs(x = "Time",
       y = "Value",
       title = "Average TE (NN) vs Time")
te_hat_nn
ggsave("tehat_nn_d0.pdf")

te_hat_gan <- acc_gan %>%
  filter(metric == "te_hat" | metric == "te_true") %>%
  ggplot(aes(x = ts_year, y = value, group = metric, color = metric)) +
  geom_line() +
  scale_color_manual(values=c("#69b3a2", "#404080")) +
  scale_x_continuous(breaks = c(1900,1925,1950,1975,1999)) +
  labs(x = "Time",
       y = "Value",
       title = "Average TE (GAN) vs Time")
te_hat_gan
ggsave("tehat_gan_d0.pdf")




