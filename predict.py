#!/usr/bin/env python
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split

def get_data():
  df = pd.read_csv('train.csv', index_col='Id')

  return df

def process_data():
  df = get_data()

  labels = df['Hazard']

  df.drop('Hazard', axis=1, inplace=True)

  feature_names = list(df.columns.values)

  features = df.as_matrix()
  labels = labels.as_matrix()

  # Vectorize features
  features = [dict(zip(feature_names, feature)) for feature in features]
  vectorizer = DictVectorizer(sparse=False)
  features = vectorizer.fit_transform(features)

  print features.shape
  print labels.shape

  train_features, test_features, train_labels, test_labels = \
    train_test_split(features, labels, test_size=.3)

  return train_features, test_features, train_labels, test_labels

def rf_score(train_features, train_labels, test_features, test_labels):
  forest_regressor = RandomForestRegressor(n_estimators=200, n_jobs=-1, verbose=3, oob_score=True)

  forest_regressor.fit(train_features, train_labels)

  pred_labels = forest_regressor.predict(test_features)

  print "Score: " + str(gini(test_labels, pred_labels))

def linear_reg_score(train_features, train_labels, test_features, test_labels):
  linear_regressor = LinearRegression(n_jobs=-1)

  linear_regressor.fit(train_features, train_labels)

  pred_labels = linear_regressor.predict(test_features)

  print "Score: " + str(gini(test_labels, pred_labels))

def xgb_score(train_features, train_labels, test_features, test_labels):
  xg_train = xgb.DMatrix(train_features, label=train_labels)
  xg_val = xgb.DMatrix(train_features, label=train_labels)
  xg_test = xgb.DMatrix(test_features)

  params = {'objective': "reg:linear", 'eta': 1, 'silent': 1, 'max_depth': 9}

  watchlist = [(xg_train, 'train'), (xg_val, 'val')]

  plst = list(params.items())

  model = xgb.train(list(params.items()), xg_train, 10000, watchlist, early_stopping_rounds=120)

  pred_labels_1 = model.predict(xg_test, ntree_limit=model.best_iteration)

  # Reverse
  train_features = train_features[::-1,:]
  train_labels = np.log(train_labels[::-1])

  watchlist = [(xg_train, 'train'), (xg_val, 'val')]

  xg_train = xgb.DMatrix(train_features, label=train_labels)
  xg_val = xgb.DMatrix(train_features, label=train_labels)

  model = xgb.train(list(params.items()), xg_train, 10000, watchlist, early_stopping_rounds=120)

  pred_labels_2 = model.predict(xg_test, ntree_limit=model.best_iteration)

  # Combine predictions
  pred_labels = (pred_labels_1) * 1.4 + (pred_labels_2) * 8.6

  print "Score: " + str(gini(test_labels, pred_labels))

def gini(true_labels, pred_labels):
  # Get number of samples
  num_samples = true_labels.shape[0]

  # Sort rows on prediction columns (largest to smallest)
  arr = np.array([true_labels, pred_labels]).transpose()
  true_order = arr[arr[:,0].argsort()][::-1,0]
  pred_order = arr[arr[:,1].argsort()][::-1,0]

  # Get Lorenz curves
  L_true = np.cumsum(true_order) / np.sum(true_order)
  L_pred = np.cumsum(pred_order) / np.sum(pred_order)
  L_ones = np.linspace(1 / num_samples, 1, num_samples)

  # Get Gini coefficients (area between curves)
  G_true = np.sum(L_ones - L_true)
  G_pred = np.sum(L_ones - L_pred)

  # Normalize coefficient
  return G_pred / G_true

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("model", help="the model to use, one of: linreg, rf, xgb", type=str)

  args = parser.parse_args()

  if args.model == "linreg":
    train_features, test_features, train_labels, test_labels = process_data()
    linear_reg_score(train_features, train_labels, test_features, test_labels)
  elif args.model == "rf":
    train_features, test_features, train_labels, test_labels = process_data()
    rf_score(train_features, train_labels, test_features, test_labels)
  elif args.model == "xgb":
    train_features, test_features, train_labels, test_labels = process_data()
    xgb_score(train_features, train_labels, test_features, test_labels)
  else:
    parser.print_help()