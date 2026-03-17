import pandas as pd

import calc_lsr


def main():
    df = pd.read_csv("../test_data/iris.csv")
    x: pd.Series = pd.Series(df["Sepal.Length"])
    y: pd.Series = pd.Series(df["Sepal.Width"])

    lsr = calc_lsr.LeastSquaresRegression(x, y)
    beta1, beta0 = lsr.get_output()

    print(f"beta1: {beta1}, beta0: {beta0}")


if __name__ == "__main__":
    main()
