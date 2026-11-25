# SmartBin City

A waste-management platform. Citizens report overflowing or dirty bins with a photo, officers receive cleanup tasks, and admins see live bin status on a city map.

The project has three parts that run together:

| Part | Folder | Port | Tech |
|---|---|---|---|
| Backend API | `server/` | 5001 | Node.js, Express, Socket.io, MongoDB |
| Frontend | `client/` | 5173 | React, Vite |
| ML service | `ml/service/` | 8000 | Python, FastAPI, YOLO11 |

The ML service classifies bin photos as **empty**, **half-full**, or **full**, and converts that into a fill percentage (0%, 50%, 100%). The backend can use this model or Gemini for photo analysis (see [Choosing the AI provider](#choosing-the-ai-provider)).

---

## 1. Prerequisites

Install these first:

- **Git**
- **Node.js** 18 or newer (includes npm): https://nodejs.org
- **Python** 3.10 or newer: https://www.python.org/downloads (tick *Add python.exe to PATH*)
- **MongoDB Community Server** (runs locally) and optionally **MongoDB Compass** to view the data: https://www.mongodb.com/try/download/community
- Accounts for **Cloudinary** (photo storage) and **Google AI Studio** (Gemini API key). Both have free tiers.

---

## 2. Clone the project

```bash
git clone <your-repo-url>
cd Final-Year-Project
```

---

## 3. Start MongoDB

Start the MongoDB service. On Windows it usually starts automatically; otherwise run `mongod`. The backend connects to `mongodb://127.0.0.1:27017/smartbin` and creates the database on first use. Compass can connect to `mongodb://127.0.0.1:27017` to browse it.

---

## 4. Set up the ML service

The ML service loads the trained model from `ml/models/<version>/best.pt`. The trained model files are already in the repo, so **you don't need to train anything**.

Create a virtual environment and install the dependencies (this downloads PyTorch, so it takes a while):

**Windows (PowerShell)**
```powershell
python -m venv ml\.venv
ml\.venv\Scripts\Activate.ps1
python -m pip install -r ml\service\requirements.txt
```

**macOS / Linux**
```bash
python3 -m venv ml/.venv
source ml/.venv/bin/activate
python -m pip install -r ml/service/requirements.txt
```

Check that the models are there: `ml/models/v1/best.pt` should exist.

---

## 5. Configure the backend

Copy the template and fill in your values:

**Windows (PowerShell)**
```powershell
copy server\.env.example server\.env
```

**macOS / Linux**
```bash
cp server/.env.example server/.env
```

Then edit `server/.env`:

| Variable | What to put |
|---|---|
| `PORT` | Keep `5001` (the frontend expects it) |
| `MONGO_URI` | Keep `mongodb://127.0.0.1:27017/smartbin` |
| `JWT_SECRET` | Any long random string. Generate one with `node -e "console.log(require('crypto').randomBytes(48).toString('hex'))"` |
| `GEMINI_API_KEY` | Your key from https://aistudio.google.com/app/apikey |
| `GEMINI_MODEL` | `gemini-3.5-flash` (or another model your key can use) |
| `GEMINI_CHECK_MODEL` | `gemini-3.5-flash-lite` |
| `CLOUDINARY_CLOUD_NAME` | From your Cloudinary dashboard |
| `CLOUDINARY_API_KEY` | From your Cloudinary dashboard |
| `CLOUDINARY_API_SECRET` | From your Cloudinary dashboard |
| `AI_PROVIDER` | `local` (default, uses the ML service) or `gemini` |
| `ML_SERVICE_URL` | `http://localhost:8000` |
| `ML_MODEL_VERSION` | `v1` |

Install the backend packages:

```bash
cd server
npm install
cd ..
```

Never commit `server/.env`. It's already listed in `.gitignore`.

---

## 6. Configure the frontend

```bash
copy client\.env.example client\.env     # Windows
cp client/.env.example client/.env       # macOS / Linux
```

The defaults point at `http://localhost:5001`, so you usually don't need to change anything. Then install the packages:

```bash
cd client
npm install
cd ..
```

---

## 7. Run the project

Open three terminals in the project folder.

**Terminal 1: ML service** (with the virtual environment active)
```bash
python -m uvicorn app:app --port 8000 --app-dir ml/service
```
Check it at http://localhost:8000/versions. It should return `["v1"]`.

**Terminal 2: backend**
```bash
cd server
npm run dev
```
You should see `MongoDB Connected` and `running on port 5001`.

**Terminal 3: frontend**
```bash
cd client
npm run dev
```
Open the address it shows, usually http://localhost:5173.

---

## 8. First use

1. Go to `/register` and create a **citizen** account, then log in.
2. Report a bin: choose a bin, upload a photo, and submit.
3. To test admin or officer features, change the `role` of a user in the `users` collection in Compass to `admin` or `officer`. Set `city` on officers so they receive tasks for that city.

---

## Choosing the AI provider

Set `AI_PROVIDER` in `server/.env`:

- **`local`** (default): photos are classified by the local YOLO11 model in `ml/service`. Gemini is used only to check that an uploaded photo shows a bin, using `GEMINI_CHECK_MODEL`. Without a Gemini key, that check is skipped with a warning in the backend log.
- **`gemini`**: photos are analysed by Gemini with `GEMINI_MODEL`, which also checks for bins. The ML service isn't needed.

To use a different model version, set `ML_MODEL_VERSION` in `server/.env` and restart the backend, and make sure the ML service is started with the same version. Available versions are the folders in `ml/models/`.

---

## Troubleshooting

- **"Photo upload failed"**: check the backend terminal for an error. Usually the Cloudinary keys are wrong, or the file type isn't JPG, PNG, or WebP.
- **Reports don't use the model**: restart the backend after changing `server/.env`.
- **The map shows no tiles**: check your internet connection. The map uses OpenStreetMap.
- **Gemini errors with 404**: the model name in `GEMINI_MODEL` or `GEMINI_CHECK_MODEL` is retired. List the models your key supports and update the name.
- **Gemini errors with 503**: Google is busy. The app retries automatically.
- **Backend can't connect to MongoDB**: make sure the MongoDB service is running.

---

## Project structure

```
client/         React frontend (Vite)
server/         Express backend, MongoDB models, Socket.io
ml/service/     FastAPI prediction service
ml/models/      Approved trained model versions (weights + model-info.json)
ml/train.py     Trains a new model version (not needed to run the app)
ml/evaluate_real.py  Evaluates a model on labelled test photos
```

Training data, training runs, and the synthetic image generator are kept out of git. You only need them to train new models.
