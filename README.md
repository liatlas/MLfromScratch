# Least Squares Regression (Python)

This project implements **Simple Linear Regression using the Least
Squares method** from scratch in Python using `pandas` and `numpy`.

The program reads the **Iris dataset**, extracts two variables, and
computes the regression coefficients:

-   **β₁ (slope)**
-   **β₀ (intercept)**

The regression model produced is:

y = β₀ + β₁x

------------------------------------------------------------------------

# Project Structure

. ├── main.py ├── calc_lsr.py ├── data │ └── iris.csv └── README.md

------------------------------------------------------------------------

# Features

This implementation manually computes the components required for linear
regression:

-   Mean
-   Variance
-   Covariance
-   Regression slope (β₁)
-   Regression intercept (β₀)

### Mean

x̄ = (1/n) Σ xᵢ\
ȳ = (1/n) Σ yᵢ

### Variance

Var(x) = Σ(xᵢ - x̄)² / (n - 1)

### Covariance

Cov(x, y) = Σ(xᵢ - x̄)(yᵢ - ȳ) / (n - 1)

### Regression Coefficients

Slope:

β₁ = Cov(x, y) / Var(x)

Intercept:

β₀ = ȳ − β₁x̄

------------------------------------------------------------------------

# Installation

Install dependencies:

``` bash
pip install pandas numpy
```

------------------------------------------------------------------------

# Running the Program

``` bash
python main.py
```

Example output:

    beta1: -0.061884
    beta0: 3.418947

------------------------------------------------------------------------

# Example Usage

``` python
lsr = calc_lsr.LeastSquaresRegression(x, y)
beta1, beta0 = lsr.get_output()

print(beta1, beta0)
```

------------------------------------------------------------------------

# Dataset

The project uses the **Iris dataset**, specifically:

-   `Sepal.Length`
-   `Sepal.Width`

to demonstrate simple linear regression.

------------------------------------------------------------------------

# Purpose

This project is intended for:

-   Understanding the **mathematics behind linear regression**
-   Practicing **Python class design**
-   Learning how regression works **without using machine learning
    libraries**
