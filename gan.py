# Generative Adversarial Network

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import pandas as pd
import numpy as np
from ganite import Ganite
from math import sqrt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler

data_list = ["dataset_1.csv", "dataset_2.csv", "dataset_3.csv", "dataset_4.csv", "dataset_5.csv",
             "dataset_6.csv", "dataset_7.csv", "dataset_8.csv", "dataset_9.csv"]

for i in data_list:
    start = time.time()

    # importing synthetic data
    df = pd.read_csv('datasets/' + i)
    df = df.drop(df.columns[0], axis=1)
    df['ts_year'] = df['ts_year'] - 1899
    # scaler = MinMaxScaler()
    # df = pd.DataFrame(scaler.fit_transform(df.values), columns=df.columns, index=df.index)
    # print(df)

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

    # GAN

    df_gan = df
    df_gan['model'] = 'gan'

    # training data
    gan_train = df_gan[df_gan.model_group == 'training']
    cov_list = ['ts_year', 'group_name', 't', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    cov_train = gan_train[cov_list]

    # validation data
    gan_val = df_gan[df_gan.model_group == 'val']
    cov_list_val = ['ts_year', 'group_name', 't', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    cov_val = gan_val[cov_list_val]
    y_val = gan_val['y']

    # test data
    gan_test = df_gan[df_gan.model_group == 'test']
    cov_list_test = ['ts_year', 'group_name', 't', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    cov_test = gan_test[cov_list_test]
    y_test = gan_test['y']

    # model fit and prediction

    y_train = gan_train['y'].to_numpy()
    t_train = cov_train["t"].to_numpy()
    x_train = cov_train["group_name"].to_numpy().reshape(-1, 1)
    w_train = cov_train[['ts_year', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']].to_numpy()
    x_test = cov_test["group_name"].to_numpy().reshape(-1, 1)
    w_test = cov_test[['ts_year', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']].to_numpy()

    model = Ganite(w_train, t_train, y_train, num_iterations=500)
    gan_test["te_hat"] = model(w_test).numpy()
    gan_test["ts_year"] = gan_test["ts_year"] + 1899

    gan_results = gan_test[["ts_year", "model", "te_true", "te_hat"]]
    acc_gan = gan_results.groupby(["ts_year", "model"]).mean()
    acc_gan["error"] = acc_gan["te_hat"] - acc_gan["te_true"]

    # rmse
    rmse = sqrt(mean_squared_error(acc_gan["te_true"], acc_gan["te_hat"]))

    # bias
    bias = acc_gan["error"].mean()

    # computation time

    end = time.time()
    comp_time = end - start

    print("RMSE: ", rmse)  # 4.067
    print("Bias: ", bias)  # -3.190
    print("Computation Time: ", comp_time)  # 3.414

    # save data as csv
    acc_gan.to_csv('acc/acc_gan_' + i)

    # saving metrics
    metric_val = [["gan", rmse, bias, comp_time]]
    gan_metrics = pd.DataFrame(metric_val, columns=['method', 'rmse', 'bias', 'comp_time'])
    gan_metrics.to_csv('metrics/gan_metrics_' + i)


# Visualizations
# all te and te_hat
bins = np.linspace(12, 18, 30)

plt.hist(acc_gan['te_true'], bins, alpha=0.5, label='true')
plt.hist(acc_gan['te_hat'], bins, alpha=0.5, label='pred')
plt.legend(loc='upper right')
#plt.show()

#basic error plot
plt.hist(acc_gan['error'], bins=20, density=True)
plt.xlim([-4, 4])
# plt.show()

# group vs. te line

acc_gan['te_hat'].plot(label='te_hat', color='blue')
acc_gan['te_true'].plot(label='te', color='black')
# plt.show()


# Next steps:

# write code to run method for each dataset variation made in R





