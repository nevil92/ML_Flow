import json
import numpy as np 
import pandas as pd 
from sys import argv
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from math import sqrt

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

def eval_metrics(y_test, y_pred):

	# metric 1
	mae = mean_absolute_error(y_test, y_pred)
	# metric 2
	rmse = mean_squared_error(y_test, y_pred)
	# metric 3
	r2 = r2_score(y_test, y_pred)
	return mae, rmse, r2

def data_sclicing(data):
	# data sclicing
	training_df = data.loc[np.r_[1:45,52:85,101:145],data.columns]
	testing_df = data.loc[np.r_[45:51,85:101,145:150],data.columns]

	print(data.columns)
	training_df.to_parquet('training_data.parquet', partition_cols=list(data.columns))
	testing_df.to_parquet('testing_data.parquet', partition_cols=list(data.columns))		

# parse Command Line Arguments
alpha = float(argv[1]) if len(argv) > 1 else 0
l1_ratio = float(argv[2]) if len(argv) > 1 else 0
mlflow.log_param('alpha', alpha)
mlflow.log_param('l1_ratio', l1_ratio)

# C_value = float(argv[1]) if len(argv) > 1 else 0.5
# penalty = str(argv[2]) if len(argv) > 2 else 'l2'

# data section
data = pd.read_csv('data/iris_csv.csv')

# data_sclicing(data)
# data['sepallength'] = np.random.random(len(data))
# data['sepalwidth'] = np.random.random(len(data))
# data['petallength'] = np.random.random(len(data))	

data['target'] = data.apply(lambda x: 0 if x['class']=='Iris-setosa' else 1 if x['class']=='Iris-versicolour' else 2, axis=1)
print(data.columns)

X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,0:3].values, data['target'].values, test_size=0.2, random_state=1)

# model section 
model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
signature = infer_signature(X_train, model.predict(X_train))

(mae, rmse, r2) = eval_metrics(y_test, y_pred)

mlflow.log_metric("MAE",mae)
mlflow.log_metric("RMSE",rmse)
mlflow.log_metric("R2",r2)

mlflow.sklearn.log_model(model, "iris_lr")
# mlflow.sklearn.log_model(model)

# # model section
# model = LogisticRegression(C=C_value, penalty=penalty)

# model.fit(X_train, y_train)

# predictions = model.predict(X_test)		
# print("Accuracy: %.3f" % accuracy_score(y_test, predictions))
 
# scores = cross_val_score(model, X_train, y_train, cv=10)
# print(np.mean(scores))







