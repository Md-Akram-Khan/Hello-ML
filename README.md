# Hello ML

This project implements logistic regression from scratch using NumPy to classify images as cat or non-cat.

## Features

- Loads cat and non-cat images from HDF5 datasets
- Flattens and normalizes image data
- Trains logistic regression with gradient descent
- Reports training and test accuracy
- Displays predictions and learning curves
- Classifies a custom image

## Project Structure

```text
ML project/
├── datasets/
│   ├── train_catvnoncat.h5
│   └── test_catvnoncat.h5
├── images/
├── linear_regression.py
├── lr_utils.py
└── requirements.txt
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

The dataset files must be located in the `datasets` folder. The loader expects:

```text
datasets/train_catvnoncat.h5
datasets/test_catvnoncat.h5
```

## Run the Project

```powershell
python .\linear_regression.py
```

The script prints the model cost, training accuracy, test accuracy, and image predictions. It also displays the sample images and learning curves.

## Classify a Custom Image

Place an image in the `images` folder, then update the filename in `linear_regression.py`:

```python
my_image = "goat.avif"
```

Run the script again to see the predicted class.

> The filename `linear_regression.py` is retained from the original exercise, but the model implemented in the file is logistic regression.