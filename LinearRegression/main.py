import os

import calc_lsr
import pandas as pd

WORKING_DIR = os.getcwd()


def main():
    df = pd.read_csv("../test_data/iris.csv")
    df = pd.read_csv(f"{WORKING_DIR}" + "/data/iris.csv")

    x: pd.Series = pd.Series(df["Sepal.Length"])
    y: pd.Series = pd.Series(df["Sepal.Width"])

    lr = calc_lsr.LinearRegression(x, y)
    lr.fit()
    beta1, beta0 = lr.get_output()

    print(f"beta1: {beta1}, beta0: {beta0}")


if __name__ == "__main__":
    main()
