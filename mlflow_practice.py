import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models.signature import infer_signature
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

random_state = 50

iris = datasets.load_iris()
X = iris.data
y = iris.target





