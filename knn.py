# K-Nearest Neighbors Regression Method (KNN)

# importing libraries
import pandas as pd
import numpy as np
from math import sqrt
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time

data_list = ["dataset_1.csv", "dataset_2.csv", "dataset_3.csv", "dataset_4.csv", "dataset_5.csv",
             "dataset_6.csv", "dataset_7.csv", "dataset_8.csv", "dataset_9.csv"]


for i in data_list:
    start = time.time()

    # importing synthetic data
    df = pd.read_csv('datasets/' + i)
    df = df.drop(df.columns[0], axis=1)
    df['ts_year'] = df['ts_year'] - 1899

    # creating train, validate, test sets

    aa = df
    pd_id = aa.drop_duplicates(subset='group_name')
    pd_id = pd_id[['group_name']]

    np.random.seed(69)
    pd_id['wookie'] = (np.random.randint(0, 10000, pd_id.shape[0])) / 10000
    pd_id = pd_id[['group_name', 'wookie']]


    pd_id['model_group'] = np.where(pd_id.wookie <= 0.70, 'training', 'test')

    tips_summed = pd_id.groupby(['model_group'])['wookie'].count()

    df = df.sort_values(by=['group_name'], ascending=[True])
    pd_id = pd_id.sort_values(by=['group_name'], ascending=[True])
    df = df.merge(pd_id, on=['group_name'], how='inner')

    # K-Nearest Neighbors Regression
    df_knn = df
    df_knn['model'] = 'knn'

    # training data
    knn_train = df_knn[df_knn.model_group == 'training']

    cov_list = ['ts_year', 't', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    cov_train = knn_train[cov_list]
    y_train = knn_train['y']

    # validation data
    knn_val = df_knn[df_knn.model_group == 'val']

    cov_list_val = ['ts_year', 't_cf', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    cov_val = knn_val[cov_list_val]
    y_val = knn_val['y']

    K = 5

    # predictions with test set

    # test data
    knn_test = df_knn[df_knn.model_group == 'test']

    cov_list_test = ['ts_year', 't_cf', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    cov_test = knn_test[cov_list_test]

    y_test = knn_test['y']

    # model fit
    model = neighbors.KNeighborsRegressor(n_neighbors=K)
    model.fit(cov_train, y_train)  # fit the model

    # model predictions
    cov_test.columns = cov_test.columns.str.replace('t_cf', 't')
    knn_test['y_hat'] = model.predict(cov_test)  # make prediction on test set
    knn_test['te_hat'] = np.where(knn_test['t'] == 1,
                                  knn_test['y'] - knn_test['y_hat'],
                                  knn_test['y_hat'] - knn_test['y'])
    knn_test["ts_year"] = knn_test["ts_year"] + 1899

    # plots frame
    acc_knn = knn_test.groupby(['ts_year', 'model']).mean('te_true', 'te_hat')
    acc_knn = acc_knn[["te_true", "te_hat"]]
    acc_knn["error"] = acc_knn["te_hat"] - acc_knn["te_true"]

    # rmse
    rmse = sqrt(mean_squared_error(acc_knn["te_true"], acc_knn["te_hat"]))

    # bias
    bias = acc_knn["error"].mean()

    # computation time
    end = time.time()

    comp_time = end - start

    print("RMSE: ", rmse)  # 14.96
    print("Bias: ", bias)  # -14.69
    print("Computation Time: ", comp_time)  # 1.471 sec

    # save data as csv
    acc_knn.to_csv('acc/acc_knn_' + i)

    # save metrics
    metric_val = [["knn", rmse, bias, comp_time]]
    knn_metrics = pd.DataFrame(metric_val, columns=['method', 'rmse', 'bias', 'comp_time'])
    knn_metrics.to_csv('metrics/knn_metrics_' + i)




# next steps

# write code to run method for each dataset variation made in R
# fix settings with copy warning (need to remove t01 before renaming t_cf
