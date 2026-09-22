# CatVision: Cat vs Non-Cat Classifier

CatVision is the initial version of an image classification system that predicts whether an uploaded image contains a cat. It uses a Logistic Regression model implemented with NumPy, a Flask API and a React/Vite frontend.

The application follows this architecture:

```text
React Frontend -> Flask API -> Logistic Regression Model
```

The React frontend is deployed through GitHub Pages, while the Flask backend is deployed separately through Render.

## Current Version and Scope

This is a basic, initial version of the classifier. The model is trained on the provided cat-versus-non-cat HDF5 dataset and uses a saved `model_weights.npz` file when the API starts. It is intended as a learning project and working prototype, not as a production-grade computer vision system.

Future versions may improve the model, accuracy, interface, dataset and functionality. Those improvements are not part of the current implementation.

## Features

- Upload JPG, JPEG, PNG, or AVIF images through the web interface.
- Drag and drop an image or select one from the computer.
- Preview the selected image before classification.
- Predict `CAT` or `NOT A CAT`.
- Display the prediction confidence.
- Use a separate React frontend and Flask backend.
- Send predictions through the Flask `POST /predict` endpoint.
- Load pre-trained model weights during backend startup instead of retraining on every request.
- Deploy the frontend with GitHub Pages and the backend with Render.

## Machine Learning Approach

The model is Logistic Regression implemented from scratch in `linear_regression.py`.

1. **Dataset:** `lr_utils.py` loads the training and test images and labels from the HDF5 files in `datasets/`.
2. **Resize:** Uploaded images are converted to RGB and resized to the dataset image size, currently 64 by 64 pixels.
3. **Flatten:** Each image is reshaped into one feature column.
4. **Normalize:** Pixel values are divided by `255.0`, converting them to values between 0 and 1.
5. **Sigmoid:** The sigmoid function converts the model score into a value that can be interpreted as a cat probability.
6. **Cost function:** The model uses binary cross-entropy cost.
7. **Gradient descent:** The weights and bias are updated repeatedly using the calculated gradients.
8. **Prediction threshold:** A sigmoid output greater than `0.5` is classified as a cat; otherwise it is classified as non-cat.

The trained weights and bias are stored in `model_weights.npz` so the deployed API can start without retraining.

## Project Structure

```text
ML project/
├── datasets/
│   ├── train_catvnoncat.h5
│   └── test_catvnoncat.h5
├── .github/workflows/deploy-pages.yml
├── datasets/
├── frontend/
├── images/
├── backend.py
├── linear_regression.py
├── lr_utils.py
├── model_weights.npz
├── render.yaml
└── requirements.txt
```

| Path | Purpose |
| --- | --- |
| `backend.py` | Flask application that loads the saved model, preprocesses uploads and exposes `/health` and `/predict`. |
| `linear_regression.py` | Contains the Logistic Regression functions. Running it trains the model and creates `model_weights.npz`. |
| `lr_utils.py` | Loads the training and test HDF5 datasets, labels and class names. |
| `model_weights.npz` | Saved weights and bias used by the backend at startup. |
| `datasets/` | Contains `train_catvnoncat.h5` and `test_catvnoncat.h5`. |
| `images/` | Contains image files included with the repository for local reference or testing. |
| `frontend/src/App.jsx` | React interface for selecting, previewing, uploading and displaying results. |
| `frontend/src/api.js` | Sends image files to the backend and maps the API response. |
| `frontend/src/main.jsx` | React entry point. |
| `frontend/src/styles.css` | Frontend styling. |
| `frontend/package.json` | Frontend dependencies and Vite scripts. |
| `frontend/vite.config.js` | Vite configuration, including the GitHub Pages base path. |
| `frontend/.env.example` | Example local configuration for `VITE_API_URL`. |
| `requirements.txt` | Python dependencies for the backend and model. |
| `render.yaml` | Render configuration for deploying the Flask API with Gunicorn. |
| `.github/workflows/deploy-pages.yml` | GitHub Actions workflow that deploys the frontend to GitHub Pages. |

## How the Project Works

1. The user selects or drops an image into the React frontend.
2. The frontend checks that the image is JPG, JPEG, PNG, or AVIF and is no larger than 10 MB.
3. The frontend sends a multipart request with the image in a field named `file`.
4. Flask receives the request at `POST /predict`.
5. The backend converts the image to RGB, resizes it to 64 by 64 pixels, flattens the pixels and normalizes them.
6. The backend calls the model's prediction and sigmoid functions using the saved weights.
7. Flask returns JSON containing `prediction` and `confidence`.
8. React displays the classification result and confidence.

Example response:

```json
{
	"prediction": "cat",
	"confidence": 0.947
}
```

The API also provides `GET /health`, which returns the backend status.

## Installation and Setup

### Backend Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

The dataset files must be present at:

```text
datasets/train_catvnoncat.h5
datasets/test_catvnoncat.h5
```

Start the Flask API:

```powershell
python .\backend.py
```

The local API normally runs at `http://localhost:5000`.

To retrain the model locally and regenerate the saved weights:

```powershell
python .\linear_regression.py
```

Commit the updated `model_weights.npz` when new weights should be used by Render.

### Frontend Setup

Open a second terminal and install the frontend dependencies:

```powershell
cd frontend
npm install
```

Create a local environment file from the example:

```powershell
Copy-Item .env.example .env
```

For local development, `.env` should contain:

```text
VITE_API_URL=http://localhost:5000
```

Start the Vite development server:

```powershell
npm run dev
```

Open the local URL shown by Vite, normally `http://localhost:5173`.



## How to Use

1. Open the application.
2. Select or drag an image into the upload area.
3. Confirm the preview and click **Classify image**.
4. View the predicted class and confidence.

The frontend accepts JPG, JPEG, PNG and AVIF images smaller than 10 MB. The backend also validates the file extension and image content.

## Public Deployment

GitHub Pages serves the static React frontend, while Render runs the Python Flask API.

**Live Demo:** [\[GitHub Pages URL\]](https://md-akram-khan.github.io/Hello-ML/)

**Backend Health Check:** [\[Render URL\]/health](https://catvision-ams9.onrender.com/health)

Render free services may sleep when idle, so the first request after inactivity can take longer.

## Limitations

- This is an initial, basic Logistic Regression image classifier.
- It performs binary cat-versus-non-cat classification only.
- The provided dataset is limited in size and variety compared with production computer vision datasets.
- The model can make incorrect predictions, especially for unusual images or images unlike the training data.
- It uses a fixed image size and simple pixel-based features; it does not use a convolutional neural network or transfer learning.
- Prediction quality depends on the committed `model_weights.npz` artifact.
- The Render free tier may sleep when unused, causing a delay on the first request after inactivity.
- The frontend limits selected files to 10 MB and supports JPG, JPEG, PNG and AVIF formats.

