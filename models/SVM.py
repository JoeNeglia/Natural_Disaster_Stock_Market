import pandas as pd
import numpy as py
import sklearn.svm as SVC

model1 = SVC()


def formatData(data):

    return data


def runSVM(data):

    return


def main():
    # Read in datasetseverity

    data = pd.read_csv(
        "C:\\Users\\battl\\Documents\\DataMining\\ClassProject\\Natural_Disaster_Stock_Market\\data\\complete_dataset.csv"
    )

    data = data[[
        "energy_prev_5d_return",
        "financials_prev_5d_return",
        "industrials_prev_5d_return",
        "utilities_prev_5d_return",
        "hurricane_name",
        "severity",
        "deaths",
        "damage_billions",]
    ]

    data["industrials_prev_5d_classifier"] = data["industrials_prev_5d_return"] >= 0
    data["energy_prev_5d_classifier"] = data["energy_prev_5d_return"] >= 0
    data["utilities_prev_5d_classifier"] = data["utilities_prev_5d_return"] >= 0
    data["financials_prev_5d_classifier"] = data["financials_prev_5d_return"] >= 0

    runSVM(data)

    return None


main()
