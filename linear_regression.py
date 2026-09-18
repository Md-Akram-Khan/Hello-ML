"""A minimal simple linear regression example."""

import numpy as np


def fit_linear_regression(x_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float]:
    """Return the slope and intercept for y = slope * x + intercept."""
    x_mean = np.mean(x_values)
    y_mean = np.mean(y_values)
    slope = np.sum((x_values - x_mean) * (y_values - y_mean)) / np.sum(
        (x_values - x_mean) ** 2
    )
    intercept = y_mean - slope * x_mean
    return float(slope), float(intercept)


def main() -> None:
    hours_studied = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    exam_scores = np.array([52, 55, 61, 66, 72, 78], dtype=float)

    slope, intercept = fit_linear_regression(hours_studied, exam_scores)
    predictions = slope * hours_studied + intercept
    mean_squared_error = np.mean((exam_scores - predictions) ** 2)

    new_hours = 7.0
    predicted_score = slope * new_hours + intercept

    print(f"Regression equation: score = {slope:.2f} * hours + {intercept:.2f}")
    print(f"Mean squared error: {mean_squared_error:.2f}")
    print(f"Predicted score for {new_hours:.0f} hours: {predicted_score:.2f}")


if __name__ == "__main__":
    main()