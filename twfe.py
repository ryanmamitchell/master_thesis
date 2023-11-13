# Two Way Fixed Effects Method (TWFE)

# importing libraries
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from math import sqrt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time
import os

data_list = ["dataset_1.csv", "dataset_2.csv", "dataset_3.csv", "dataset_4.csv", "dataset_5.csv",
             "dataset_6.csv", "dataset_7.csv", "dataset_8.csv", "dataset_9.csv"]

for i in data_list:

    start = time.time()

    # importing synthetic data
    df = pd.read_csv("datasets/" + i)
    df = df.drop(df.columns[0], axis=1)
    df['ts_year'] = pd.to_datetime(df['ts_year'], format='%Y')

    # creating train, validate, test sets

    aa = df
    pd_id = aa.drop_duplicates(subset='group_name')
    pd_id = pd_id[['group_name']]

    np.random.seed(69)
    pd_id['wookie'] = (np.random.randint(0, 10000, pd_id.shape[0])) / 10000
    pd_id = pd_id[['group_name', 'wookie']]

    # pd_id['model_group'] = np.where(pd_id.wookie <= 0.35,
    #                                'training', np.where(pd_id.wookie <= 0.65,
    #                                                    'val', 'test'))

    pd_id['model_group'] = np.where(pd_id.wookie <= 0.70, 'training', 'test')

    tips_summed = pd_id.groupby(['model_group'])['wookie'].count()

    df = df.sort_values(by=['group_name'], ascending=[True])
    pd_id = pd_id.sort_values(by=['group_name'], ascending=[True])
    df = df.merge(pd_id, on=['group_name'], how='inner')

    # Two Way Fixed Effects Model

    df_twfe = df.set_index(['group_name', 'ts_year'])
    df_twfe = df_twfe.sort_values(['group_name', 'ts_year'],
                                  ascending=[True, True])
    df_twfe['model'] = 'twfe'

    # training data
    twfe_train = df_twfe[df_twfe.model_group == 'training']

    # test data
    twfe_test = df_twfe[df_twfe.model_group == 'test']

    # regression train
    cov_list = ['t', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    covs_train = sm.add_constant(twfe_train[cov_list])

    twfe = PanelOLS(twfe_train.y, covs_train,
                    entity_effects=True,
                    time_effects=True)

    twfe_fit = twfe.fit()

    # counterfactual prediction

    cov_test_list = ['t_cf', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    covs_test = sm.add_constant(twfe_test[cov_test_list])
    covs_test.columns = covs_test.columns.str.replace('t_cf', 't')
    y_hat = twfe_fit.predict(covs_test)

    df_twfe = pd.merge(df_twfe, y_hat, on=["group_name", "ts_year"])
    df_twfe.columns = df_twfe.columns.str.replace('predictions', 'y_hat')
    df_twfe['te_hat'] = np.where(df_twfe['t'] == 1,
                                 df_twfe['y'] - df_twfe['y_hat'],
                                 df_twfe['y_hat'] - df_twfe['y'])

    # plots frame

    acc_twfe = df_twfe.groupby(['ts_year', 'model']).mean(True)
    acc_twfe = acc_twfe[["te_true", "te_hat"]]
    acc_twfe["error"] = acc_twfe["te_hat"] - acc_twfe["te_true"]

    # rmse
    rmse = sqrt(mean_squared_error(acc_twfe["te_true"], acc_twfe["te_hat"]))

    # bias
    bias = acc_twfe["error"].mean()

    end = time.time()

    comp_time = end - start

    print("RMSE: ", rmse)  # 3.69
    print("Bias: ", bias)  # -2.78
    print("Computation Time: ", comp_time)  # 0.468 sec

    # save data as csv
    acc_twfe.to_csv('acc/acc_twfe_' + i)

    # save metrics
    twfe_metric_val = [["twfe", rmse, bias, comp_time]]
    twfe_metrics = pd.DataFrame(twfe_metric_val, columns=['method', 'rmse', 'bias', 'comp_time'])
    twfe_metrics.to_csv('metrics/twfe_metrics_' + i)



# Visualizations
# all te and te_hat
bins = np.linspace(12, 18, 30)

plt.hist(acc_twfe['te_true'], bins, alpha=0.5, label='true')
plt.hist(acc_twfe['te_hat'], bins, alpha=0.5, label='pred')
plt.legend(loc='upper right')
# plt.show()

# basic error plot
plt.hist(acc_twfe['error'], bins=20, density=True)
plt.xlim([-4, 4])
# plt.show()

# group vs. te line

acc_twfe['te_hat'].plot(label='te_hat', color='blue')
acc_twfe['te_true'].plot(label='te', color='black')
# plt.show()


