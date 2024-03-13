import pandas as pd
import numpy as np

"""Write your code here """
df = pd.read_csv(
    "/Users/mohamedbarry/Downloads/CS4341_Assignment2/Life Expectancy Data.csv"
)

"""end of student code"""
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

total_df = df[
    [
        "Year",
        "GDP",
        "Adult Mortality",
        "Alcohol",
        " BMI ",
        "Schooling",
        "Life expectancy ",
        "Status",
        "Country",
    ]
]
total_df = total_df.dropna()

for status in ["Developing", "Developed"]:
    x_dim = 5
    # Step 1:You should define the train_x, each row of it represents a year of the 5 features of a country with 5 columns. The Year column should be used to select the samples.
    # Step 2:Define train_y, each row of it represents a year of Life expectancy of a country. The Year column should be used to select the samples.
    # Step 3:Define a LinearRegression model, and fit it using train_X and train_y.
    # Step 4:Calculate rmse and r2_score using the fitted model.
    # Step 5:Print the coefficients of the linear regression model
    """Write your code here """

    train_x = total_df[(total_df.Year <= 2013) & (total_df.Status == status)][
        ["GDP", "Adult Mortality", "Alcohol", " BMI ", "Schooling"]
    ].values
    train_y = total_df[(total_df.Year <= 2013) & (total_df.Status == status)][
        "Life expectancy "
    ].values

    model = LinearRegression()
    model.fit(train_x, train_y)

    rmse_train = np.sqrt(mean_squared_error(model.predict(train_x), train_y))
    r2_train = model.score(train_x, train_y)

    """end of student code"""
    print(f"Status = {status}, Training data, RMSE={rmse_train:.3f}, R2={r2_train:.3f}")
    for feature_i, feature in enumerate(
        ["GDP", "Adult Mortality", "Alcohol", " BMI ", "Schooling"]
    ):
        print(f"coef for {feature} = {model.coef_[feature_i]:.7f}")

    # Step 1: Define test_x and test_y by selecting the remaining years of the data
    # Step 2: Use model.predict to generate the prediction
    # Step 3: Calculate rmse and r2_score on test_x and test_y.
    """Write your code here """

    test_x = total_df[(total_df.Year > 2013) & (total_df.Status == status)][
        ["GDP", "Adult Mortality", "Alcohol", " BMI ", "Schooling"]
    ].values
    test_y = total_df[(total_df.Year > 2013) & (total_df.Status == status)][
        "Life expectancy "
    ].values

    rmse_test = np.sqrt(mean_squared_error(model.predict(test_x), test_y))
    r2_test = model.score(test_x, test_y)

    """end of student code"""
    print(f"Status = {status}, Testing data, RMSE={rmse_test:.3f}, R2={r2_test:.3f}")
    print("")
