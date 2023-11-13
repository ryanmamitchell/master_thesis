# Double Machine Learning Method (DML)
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
# importing libraries
import pandas as pd
import numpy as np
from math import sqrt
from econml.dml import LinearDML
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures
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

    # pd_id['model_group'] = np.where(pd_id.wookie <= 0.35,
    #                                'training', np.where(pd_id.wookie <= 0.65,
    #                                                     'val', 'test'))

    pd_id['model_group'] = np.where(pd_id.wookie <= 0.70, 'training', 'test')

    tips_summed = pd_id.groupby(['model_group'])['wookie'].count()

    df = df.sort_values(by=['group_name'], ascending=[True])
    pd_id = pd_id.sort_values(by=['group_name'], ascending=[True])
    df = df.merge(pd_id, on=['group_name'], how='inner')

    # Double Machine Learning

    df_dml = df
    df_dml['model'] = 'dml'

    # training data
    dml_train = df_dml[df_dml.model_group == 'training']

    cov_list = ['ts_year', 'group_name', 't', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    cov_train = dml_train[cov_list]

    # validation data
    dml_val = df_dml[df_dml.model_group == 'val']

    cov_list_val = ['ts_year', 'group_name', 't', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    cov_val = dml_val[cov_list_val]
    y_val = dml_val['y']

    # test data
    dml_test = df_dml[df_dml.model_group == 'test']

    cov_list_test = ['ts_year', 'group_name', 't', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    cov_test = dml_test[cov_list_test]
    y_test = dml_test['y']

    # model fit and prediction

    y_train = dml_train['y'].to_numpy()
    t_train = cov_train["t"].to_numpy()
    x_train = cov_train[['ts_year', 'x2', 'x3', 'x4', 'x6']].to_numpy()  # all other x variables
    w_train = cov_train[['x1', 'x5']].to_numpy()  # variables that affect propensity score
    x_test = cov_test[['ts_year', 'x2', 'x3', 'x4', 'x6']].to_numpy()

    model_y = LassoCV(max_iter=10000)
    model_t = LassoCV(max_iter=10000)
    est = LinearDML(
        model_y=model_y,
        model_t=model_t,
        featurizer=PolynomialFeatures(degree=2),
        fit_cate_intercept=False,
    )
    est.fit(y_train, t_train, X=x_train, W=w_train)
    # lb, ub = est.effect_interval(x_test, alpha = 0.01)
    dml_test["te_hat"] = est.effect(x_test)
    dml_test["ts_year"] = dml_test["ts_year"] + 1899

    dml_results = dml_test[["ts_year", "model", "te_true", "te_hat"]]
    acc_dml = dml_results.groupby(["ts_year", "model"]).mean()
    acc_dml["error"] = acc_dml["te_hat"] - acc_dml["te_true"]
    print(acc_dml)
    # rmse
    rmse = sqrt(mean_squared_error(acc_dml["te_true"], acc_dml["te_hat"]))

    # bias
    bias = acc_dml["error"].mean()

    # computation time
    end = time.time()

    comp_time = end - start

    print("RMSE: ", rmse)  # 3.85
    print("Bias: ", bias)  # -2.36
    print("Computation Time: ", comp_time)  # 0.577 sec

    # save data as csv
    acc_dml.to_csv('acc/acc_dml_' + i)

    # save metrics
    metric_val = [["dml", rmse, bias, comp_time]]
    dml_metrics = pd.DataFrame(metric_val, columns=['method', 'rmse', 'bias', 'comp_time'])
    dml_metrics.to_csv('metrics/dml_metrics_' + i)


# Visualizations
# all te and te_hat
bins = np.linspace(12, 18, 30)

plt.hist(acc_dml['te_true'], bins, alpha=0.5, label='true')
plt.hist(acc_dml['te_hat'], bins, alpha=0.5, label='pred')
plt.legend(loc='upper right')
# plt.show()

#basic error plot
plt.hist(acc_dml['error'], bins=20, density=True)
plt.xlim([-4, 4])
# plt.show()

# group vs. te line

acc_dml['te_hat'].plot(label='te_hat', color='blue')
acc_dml['te_true'].plot(label='te', color='black')
# plt.show()



# next steps:

# write code to run method for each dataset variation made in R
