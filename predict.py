#!/usr/bin/env python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split

def read_properties_as_matrix(filename):
  df = pd.read_csv(filename, index_col='Id')

  return df.as_matrix()

def predict():
  features = read_properties_as_matrix('train.csv')
  labels = features[:,0]
  features = features[:,1:]

  # Vectorize features
  features = [dict(enumerate(feature)) for feature in features]
  vectorizer = DictVectorizer(sparse=False)
  features = vectorizer.fit_transform(features)

  train_features, test_features, train_labels, test_labels = \
    train_test_split(features, labels, test_size=.3)

  print train_features.shape
  print train_labels.shape
  print test_features.shape
  print test_labels.shape

  forest_regressor = RandomForestRegressor(n_estimators=8, n_jobs=-1, verbose=3, oob_score=True, max_features=None)

  forest_regressor.fit(train_features, train_labels)

  print forest_regressor.score(test_features, test_labels)
  print forest_regressor.oob_score_

if __name__ == "__main__":
  predict()