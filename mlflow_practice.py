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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

experiments = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 150, "max_depth": 7}
]

for i, params in enumerate(experiments):
    with mlflow.start_run(run_name=f"Experiment {i + 1}"):
        rf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                    max_depth=params['max_depth'],
                                    random_state=random_state)

        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)

        mlflow.log_param("n_estimators", params['n_estimators'])
        mlflow.log_param("max_depth", params['max_depth'])

        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        input_example = pd.DataFrame(X_train[:1], columns=iris.feature_names)
        signature = infer_signature(X_train, rf.predict(X_train))

        mlflow.sklearn.log_model(rf, "random_forest_model", signature=signature, input_example=input_example)
        print(f"Experiment {i + 1}: Accuracy = {accuracy}")
