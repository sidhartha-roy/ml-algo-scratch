import pandas as pd

url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/abalone/abalone.data"
)

abalone = pd.read_csv(url, header=None)

print(abalone)