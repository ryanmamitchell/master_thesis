# final metric dataframe
import pandas as pd


data_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
total_metrics = []
for i  in data_list:
    gan_m = pd.read_csv("metrics/gan_metrics_" + "dataset_" + i + ".csv")
    gan_m["data"] = i
    nn_m = pd.read_csv("metrics/nn_metrics_" + "dataset_" + i + ".csv")
    nn_m["data"] = i
    cf_m = pd.read_csv("metrics/cf_metrics_" + "dataset_" + i + ".csv")
    cf_m["data"] = i
    dml_m = pd.read_csv("metrics/dml_metrics_" + "dataset_" + i + ".csv")
    dml_m["data"] = i
    knn_m = pd.read_csv("metrics/knn_metrics_" + "dataset_" + i + ".csv")
    knn_m["data"] = i
    twfe_m = pd.read_csv("metrics/twfe_metrics_" + "dataset_" + i + ".csv")
    twfe_m["data"] = i

    metrics_final = pd.concat([twfe_m, knn_m, dml_m, cf_m, nn_m, gan_m], ignore_index=True)
    metrics_final = metrics_final.drop(metrics_final.columns[[0]], axis=1)

    total_metrics.append(metrics_final)


total_metrics_df = pd.concat(total_metrics, ignore_index=True)
print(total_metrics_df)
total_metrics_df.to_csv('metrics/all_metrics.csv')

grouped_total = total_metrics_df.drop("data", axis=1)
grouped_total = grouped_total.groupby(["method"]).mean()
print(grouped_total)
grouped_total.to_csv('metrics/final_table.csv')


# acc_frame aggregation

data_list = ["dataset_1.csv", "dataset_2.csv", "dataset_3.csv", "dataset_4.csv", "dataset_5.csv",
             "dataset_6.csv", "dataset_7.csv", "dataset_8.csv", "dataset_9.csv"]

method_list = ["cf", "dml", "knn", "twfe", "gan", "nn"]


df_list = []

for i in method_list:
    for j in data_list:
        df_list.append(pd.read_csv("acc_" + i + "_" + j))
acc_total = pd.concat(df_list, ignore_index=True)
print(acc_total)
acc_total.to_csv("acc/acc_total.csv")


