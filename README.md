# Hello ML

This project contains a minimal example of simple linear regression in Python.

The example predicts an exam score from hours studied using the equation:

```text
score = slope * hours + intercept
```

## Run it

```powershell
python -m pip install -r requirements.txt
python linear_regression.py
```

The script prints the fitted equation, mean squared error, and a prediction for a student who studies for seven hours.