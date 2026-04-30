import calc_lsr
import pandas as pd
import os

WORKING_DIR = os.getcwd()
def main():
<<<<<<< HEAD
    df = pd.read_csv("../test_data/iris.csv")
=======
    df = pd.read_csv(f"{WORKING_DIR}" + "/data/iris.csv")
>>>>>>> b18e83f (adapts calc_lsr.py and main.py to directory updates)
    x: pd.Series = pd.Series(df["Sepal.Length"])
    y: pd.Series = pd.Series(df["Sepal.Width"])

    lsr = calc_lsr.LeastSquaresRegression(x, y)
    lsr.fit()
    beta1, beta0 = lsr.get_output()

    print(f"beta1: {beta1}, beta0: {beta0}")


if __name__ == "__main__":
    main()
