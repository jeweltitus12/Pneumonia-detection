# PneumoDetect - Pneumonia Detection AI

A full-stack web application that uses deep learning to detect pneumonia from chest X-ray images.

## Features

- **AI prediction**: Upload chest X-rays and receive Normal/Pneumonia predictions with confidence scores
- **Medical dashboard**: View scan statistics, result distribution chart, and prediction history
- **Modern UI**: React + Tailwind CSS with drag-and-drop upload, dark mode, and backend status indicator
- **REST API**: Flask backend with health check, prediction, history, and stats endpoints

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Frontend | React 19, Vite, Tailwind CSS 4, Axios, Recharts, Lucide React |
| Backend | Python, Flask, TensorFlow/Keras, SQLite, Pillow |
| Model | MobileNetV2 transfer learning (binary classification) |

## Architecture

```
frontend (React/Vite :5173)
    │  POST /api/predict  (multipart image upload)
    │  GET  /api/history, /api/stats, /api/health
    ▼
backend (Flask :5000)
    ├── routes/api.py        → API endpoints
    ├── services/ai_model.py → TensorFlow inference
    ├── models/database.py   → SQLite persistence
    └── weights/pneumonia_model.h5
```

### Data flow

1. User selects or drops an X-ray in the frontend
2. Frontend sends the image as `multipart/form-data` to `/api/predict`
3. Backend saves the file, runs TensorFlow inference, stores the result in SQLite
4. Frontend displays diagnosis, confidence bar, and updates the dashboard

## Prerequisites

- **Python 3.10+** (3.11 recommended)
- **Node.js 18+**
- **npm**

## Step-by-step: Run locally

### 1. Clone and open the project

```powershell
cd C:\Users\jewel\OneDrive\Desktop\pneumonia
```

### 2. Backend setup

```powershell
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

### 3. Create the AI model

Choose **one** of the following:

**Option A — Bootstrap model (quick start, uses sample images in `uploads/`)**

```powershell
python scripts/bootstrap_model.py
```

**Option B — Full training (recommended for real accuracy)**

Download the [Kaggle Chest X-Ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), then:

```powershell
python scripts/train_model.py --dataset path\to\chest_xray
```

This saves the trained model to `backend/weights/pneumonia_model.h5`.

### 4. Start the backend

```powershell
python app.py
```

The API runs at `http://localhost:5000`. You should see `Model loaded from ...` in the console.

Verify:

```powershell
curl http://localhost:5000/api/health
```

### 5. Frontend setup (new terminal)

```powershell
cd frontend
npm install
copy .env.example .env
npm run dev
```

Open **http://localhost:5173** in your browser.

### 6. Test the app

1. Go to **Scan Image**
2. Upload a chest X-ray (JPG or PNG)
3. Click **Analyze Image**
4. View the prediction and confidence score
5. Switch to **Dashboard** to see history and stats

## Environment variables

### Backend (`backend/.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5000` | Flask server port |
| `FLASK_DEBUG` | `true` | Enable Flask debug mode |
| `CORS_ORIGINS` | `http://localhost:5173,...` | Allowed frontend origins |
| `MODEL_PATH` | `weights/pneumonia_model.h5` | Path to the Keras model file |

### Frontend (`frontend/.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_URL` | `/api` | API base URL (Vite dev proxy handles routing) |

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API status and model info |
| GET | `/api/health` | Health check with model load status |
| POST | `/api/predict` | Upload image (`file` field), returns prediction |
| GET | `/api/history` | List all past predictions |
| GET | `/api/stats` | Aggregate counts (total, pneumonia, normal) |

### Example predict response

```json
{
  "message": "Prediction successful",
  "prediction": "Pneumonia",
  "confidence": 87.42,
  "filename": "xray.jpeg"
}
```

## Project structure

```
pneumonia/
├── backend/
│   ├── app.py                 # Flask entry point
│   ├── routes/api.py          # REST endpoints
│   ├── services/ai_model.py   # Model loading & inference
│   ├── models/database.py     # SQLite helpers
│   ├── scripts/
│   │   ├── bootstrap_model.py # Quick dev model trainer
│   │   └── train_model.py     # Full dataset trainer
│   ├── weights/               # Saved model (.h5)
│   ├── uploads/               # Uploaded X-rays
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── api/client.js      # Axios API client
│   │   ├── components/
│   │   │   ├── UploadSection.jsx
│   │   │   └── Dashboard.jsx
│   │   └── App.jsx
│   └── vite.config.js         # Dev proxy to backend
└── README.md
```

## Production deployment notes

**Backend (Gunicorn):**

```powershell
cd backend
gunicorn -w 1 -b 0.0.0.0:5000 --timeout 120 app:app
```

Use `-w 1` because TensorFlow models are memory-heavy. Set `FLASK_DEBUG=false` in production.

**Frontend:**

```powershell
cd frontend
npm run build
npm run preview
```

Set `VITE_API_URL` to your deployed backend URL before building.

## Issues fixed in this version

- Replaced mock random predictions with real TensorFlow inference
- Fixed fragile `os.getcwd()` paths — all paths now resolve from the backend root
- Added missing `scipy` and `python-dotenv` dependencies
- Removed unused `SQLAlchemy` dependency
- Added Vite dev proxy and centralized API client (no hardcoded URLs)
- Added `/api/health` endpoint and frontend backend-status indicator
- Added confidence progress bar, dashboard chart, and error/retry states
- Added model bootstrap and full training scripts
- Added `.env.example` files for both frontend and backend

## Disclaimer

This application is for **demonstration and educational purposes only**. It is not a substitute for professional medical diagnosis.
