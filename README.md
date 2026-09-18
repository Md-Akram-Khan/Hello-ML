# Hello ML

This project implements logistic regression from scratch using NumPy to classify images as cat or non-cat. It also includes a CatVision React interface for uploading images and viewing predictions from a local Python API.

## Features

- Loads cat and non-cat images from HDF5 datasets
- Flattens and normalizes image data
- Trains logistic regression with gradient descent
- Reports training and test accuracy
- Displays predictions and learning curves
- Classifies a custom image in JPG, JPEG, PNG, or AVIF format
- Provides a Flask API at `POST /predict`
- Includes a responsive React/Vite dashboard with drag-and-drop upload

## Project Structure

```text
ML project/
├── datasets/
│   ├── train_catvnoncat.h5
│   └── test_catvnoncat.h5
├── images/
├── backend.py
├── frontend/
│   ├── src/
│   └── package.json
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

## Run the Python Script

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

## Run CatVision

Start the Python API from the project root:

```powershell
python .\backend.py
```

In a second terminal, start the React frontend:

```powershell
cd frontend
npm install
npm run dev
```

Open the URL shown by Vite, normally `http://localhost:5173`. The frontend sends images to `http://localhost:5000/predict` by default. To use another backend URL, create `frontend/.env` from `frontend/.env.example` and set:

```text
VITE_API_URL=http://localhost:5000
```

The API accepts a multipart form field named `file` in JPG, JPEG, PNG, or AVIF format and returns:

```json
{
	"prediction": "cat",
	"confidence": 0.947
}
```

> The filename `linear_regression.py` is retained from the original exercise, but the model implemented in the file is logistic regression.