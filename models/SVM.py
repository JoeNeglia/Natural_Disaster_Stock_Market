import pandas as pd
import numpy as py
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def importData():
    # Read in datasetseverity
    df = pd.read_csv(
        "C:\\Users\\battl\\Documents\\DataMining\\ClassProject\\Natural_Disaster_Stock_Market\\data\\sector_datasets\\conceptual_complete_dataset.csv"
    )

    df = df[
        [
            "energy_prev_5d_return",
            "financials_prev_5d_return",
            "industrials_prev_5d_return",
            "utilities_prev_5d_return",
            "hurricane_name",
            "severity",
            "deaths",
            "damage_billions",
        ]
    ]

    # Add CLassifiers
    df["industrials_prev_5d_classifier"] = df["industrials_prev_5d_return"] >= 0
    df["energy_prev_5d_classifier"] = df["energy_prev_5d_return"] >= 0
    df["utilities_prev_5d_classifier"] = df["utilities_prev_5d_return"] >= 0
    df["financials_prev_5d_classifier"] = df["financials_prev_5d_return"] >= 0

    df["severity"] = df["severity"].map(
        {
            "Category 1": 0,
            "Category 2": 1,
            "Category 3": 2,
            "Category 4": 3,
            "Category 5": 4,
        }
    )

    df = df.set_index("hurricane_name")

    return df


def runSVM(data):
    model = SVC()
    X = data.drop(
        columns=[
            "industrials_prev_5d_classifier",
            "financials_prev_5d_classifier",
            "utilities_prev_5d_classifier",
            "energy_prev_5d_classifier",
        ]
    )

    y = data["industrials_prev_5d_classifier"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline for scaling + SVM
    pipeline = Pipeline([("scaler", StandardScaler()), ("svm", SVC())])

    # Parameters to test
    param_grid = [
        {"svm__kernel": ["linear"], "svm__C": [0.01, 0.1, 1, 10, 100]},
        {
            "svm__kernel": ["rbf"],
            "svm__C": [0.01, 0.1, 1, 10, 100],
            "svm__gamma": [0.001, 0.01, 0.1, 1, "scale"],
        },
        {
            "svm__kernel": ["poly"],
            "svm__C": [0.1, 1, 10],
            "svm__gamma": ["scale", 0.01, 0.1],
            "svm__degree": [2, 3, 4],
        },
    ]

    # Grid search
    grid = GridSearchCV(
        pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
    )

    grid.fit(X_train, y_train)

    # Best parameters
    print("Best parameters:", grid.best_params_)
    print("Best CV accuracy:", grid.best_score_)

    # Evaluate on held-out test data
    print("Test accuracy:", grid.score(X_test, y_test))

    return None


def main():
    data = importData()
    runSVM(data)
    return None


main()
