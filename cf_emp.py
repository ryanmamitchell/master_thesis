# Causal Forest Method - Empirical Application

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from econml.dml import CausalForestDML
import time

start = time.time()

# importing synthetic data
df = pd.read_csv('datasets/data_emp.csv')
df = df.drop(["democ", "rever"], axis=1)
df = df.rename(columns={"wbcode2": "group_name", "year": "ts_year", "dem": "t", "logpop": "x1", "pop14": "x2",
                        "pop1564": "x3", "tradewb": "x4", "nfagdp": "x5", "l1": "x6", "l2": "x7", "l3": "x8",
                        "l4": "x9", "unrest": "x10"})
df["ts_year"] = df["ts_year"] - 1969

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

# Causal Forest

df_cf = df
df_cf['model'] = 'cf'

# training data
cf_train = df_cf[df_cf.model_group == 'training']

cov_list = ['ts_year', 'group_name', 't', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
cov_train = cf_train[cov_list]

# test data
cf_test = df_cf[df_cf.model_group == 'test']

cov_list_test = ['ts_year', 'group_name', 't', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
cov_test = cf_test[cov_list_test]
y_test = cf_test['y']

# model fit and prediction

y_train = cf_train['y'].to_numpy()
t_train = cov_train["t"].to_numpy()
x_train = cov_train[['ts_year', 'x1', 'x2', 'x3', 'x4', 'x5', 'x10']].to_numpy()  # all other x variables
# x_train = cov_train[['ts_year', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']].to_numpy()
w_train = cov_train[['ts_year', 'x1', 'x2', 'x3', 'x4', 'x5', 'x10']].to_numpy()  # variables that affect propensity score
x_test = cov_test[['ts_year', 'x1', 'x2', 'x3', 'x4', 'x5', 'x10']].to_numpy()

model_y = RandomForestRegressor(random_state=0)
model_t = RandomForestRegressor(random_state=0)

est = CausalForestDML(model_y=model_y,
                      model_t=model_t,
                      criterion='mse', n_estimators=100,
                      min_impurity_decrease=0.001, random_state=0)

est.fit(y_train, t_train, X=x_train, W=w_train)
# note: _ensemble.py needed to be changed: line 161 dtype=int

lb, ub = est.effect_interval(x_test, alpha = 0.01)
cf_test["te_hat"] = est.effect(x_test)
cf_test["ts_year"] = cf_test["ts_year"] + 1969

cf_results = cf_test[["ts_year", "y", "t", "te_hat"]]
acc_cf = cf_results.groupby(["ts_year"]).mean()
print(acc_cf)
end = time.time()

comp_time = end - start
print(comp_time)
#write csv
acc_cf.to_csv('acc/acc_emp.csv')

