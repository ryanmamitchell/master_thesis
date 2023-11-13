# Neural Network

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from catenets.models.jax import TNet

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

    # pd_id['model_group'] = np.where(pd_id.wookie <= 0.35,
    #                                'training', np.where(pd_id.wookie <= 0.65,
    #                                                     'val', 'test'))

    pd_id['model_group'] = np.where(pd_id.wookie <= 0.70, 'training', 'test')

    tips_summed = pd_id.groupby(['model_group'])['wookie'].count()

    df = df.sort_values(by=['group_name'], ascending=[True])
    pd_id = pd_id.sort_values(by=['group_name'], ascending=[True])
    df = df.merge(pd_id, on=['group_name'], how='inner')

    # Neural Network:

    df_nn = df
    df_nn['model'] = 'nn'

    # training data
    nn_train = df_nn[df_nn.model_group == 'training']

    cov_list = ['ts_year', 'group_name', 't', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    cov_train = nn_train[cov_list]

    # validation data
    nn_val = df_nn[df_nn.model_group == 'val']

    cov_list_val = ['ts_year', 'group_name', 't', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    cov_val = nn_val[cov_list_val]
    y_val = nn_val['y']

    # test data
    nn_test = df_nn[df_nn.model_group == 'test']

    cov_list_test = ['ts_year', 'group_name', 't', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    cov_test = nn_test[cov_list_test]
    y_test = nn_test['y']

    # SNet
    # model fit and prediction

    y_train = nn_train['y'].to_numpy()
    t_train = cov_train["t"].to_numpy()
    x_train = cov_train[['ts_year', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']].to_numpy()
    x_test = cov_test[['ts_year', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']].to_numpy()

    t = TNet()

    t.fit(x_train, y_train, t_train)
    nn_test["te_hat"] = t.predict(x_test)
    nn_test["ts_year"] = nn_test["ts_year"] + 1899

    nn_results = nn_test[["ts_year", "model", "te_true", "te_hat"]]
    acc_nn = nn_results.groupby(["ts_year", "model"]).mean()
    acc_nn["error"] = acc_nn["te_hat"] - acc_nn["te_true"]

    # rmse
    rmse = sqrt(mean_squared_error(acc_nn["te_true"], acc_nn["te_hat"]))

    # bias
    bias = acc_nn["error"].mean()

    # computation time
    end = time.time()
    comp_time = end - start

    print("RMSE: ", rmse)  # 2.59
    print("Bias: ", bias)  # -0.280
    print("Computation Time: ", comp_time)  # 24.95 sec

    # save data as csv
    acc_nn.to_csv('acc/acc_nn_' + i)

    # save metrics
    metric_val = [["nn", rmse, bias, comp_time]]
    nn_metrics = pd.DataFrame(metric_val, columns=['method', 'rmse', 'bias', 'comp_time'])
    nn_metrics.to_csv('metrics/nn_metrics_' + i)


# Visualizations

# all te and te_hat
bins = np.linspace(12, 18, 30)

plt.hist(acc_nn['te_true'], bins, alpha=0.5, label='true')
plt.hist(acc_nn['te_hat'], bins, alpha=0.5, label='pred')
plt.legend(loc='upper right')
# plt.show()

# basic error plot
plt.hist(acc_nn['error'], bins=20, density=True)
plt.xlim([-4, 4])
# plt.show()

# group vs. te line
acc_nn['te_hat'].plot(label='te_hat', color='blue')
acc_nn['te_true'].plot(label='te', color='black')
# plt.show()

