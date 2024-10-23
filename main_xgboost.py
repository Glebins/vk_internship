from extract_features import *

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import xgboost as xgb

pd.set_option('display.max_columns', 100)

path = "/"
train_data = pd.read_parquet(path + "train.parquet")
test_data = pd.read_parquet(path + "test.parquet")

y = train_data['label']

is_read = True

if is_read:
    X = pd.read_csv('out.csv')
else:
    X = train_data.loc[:, ['dates', 'values']]
    X = extract_features(X)
    X.to_csv('featured_X.csv')

# Feature selection. Approximately the same results for any set of features.
X = X.iloc[:, [0, 1, 2, 3, 14, 15]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 4,
    'eta': 0.1,
    'seed': 42
}

train_data_xgb = xgb.DMatrix(X_train, label=y_train)
test_data_xgb = xgb.DMatrix(X_test)

model = xgb.train(params, train_data_xgb, num_boost_round=300)

y_pred_prob = pd.DataFrame(model.predict(test_data_xgb))
y_pred = pd.DataFrame((y_pred_prob > 0.5) * 1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

fi = model.get_score(importance_type='gain')
fi = {k: v for k, v in sorted(fi.items(), key=lambda item: -item[1])}

print(accuracy, precision, recall, f1, roc_auc)
print(fi) # 'mean' always is bigger than any other feature at least 3 times

# Typical value:
# 0.8211875 0.7377405338299193 0.5410880946961074 0.6242941562705188 0.8664000002667213
# Maximum value of f1 (for different hyperparameters' values) is 0.67
