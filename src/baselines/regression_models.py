from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def get_regression_models():
    return {
        "Linear": LinearRegression(),
        "Poly2": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("lr", LinearRegression())
        ]),
        "Ridge": Ridge(alpha=1.0),
        "SVR": SVR(kernel="rbf"),
        "Tree": DecisionTreeRegressor(max_depth=4, random_state=42),
        "RF": RandomForestRegressor(n_estimators=200, random_state=42),
    }
