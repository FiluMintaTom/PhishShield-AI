
# PhishShield AI — real web application baseline

This version converts the single-file prototype into a small Flask + SQLite web application.

## What changed
- Real account registration and login.
- Passwords are hashed; no password is stored as plain text.
- Server-side validation for registration/login and scan input.
- User-specific persistent scan history.
- Dashboard and reports are calculated only from stored scans.
- No seeded fake scan records and no random counters.
- URL scans perform server-side checks for HTTPS, URL structure, redirects, HTTP status and RDAP registration date when available.
- Scan evidence is stored with every scan.
- Settings are persisted.
- The result page explicitly calls the method a transparent feature-based risk analysis instead of falsely claiming validated ML/SHAP output.

## Important limitation
A normal hosted website cannot directly open Windows Sandbox on your personal computer. Windows Sandbox is a local Windows isolation feature. For a public deployment, use a dedicated isolated sandbox service/worker if you want dynamic detonation of suspicious pages/files. Do not execute arbitrary submitted URLs inside the same web server.

## Run on Windows
1. Install Python 3.11+.
2. Open a terminal in this folder.
3. Create a virtual environment:
   `python -m venv .venv`
4. Activate:
   `.venv\Scripts\activate`
5. Install:
   `pip install -r requirements.txt`
6. Start:
   `python app.py`
7. Open:
   `http://127.0.0.1:5000`

## Deploy
Use a Python host that supports Flask/Gunicorn. Set a strong `PHISHSHIELD_SECRET` environment variable. For production, use PostgreSQL instead of SQLite if you expect multiple server instances/users.

## Accuracy note
Do not present the heuristic risk score as "95.4% AI accuracy" unless you have a tested model, a held-out test set, and measured metrics. The original prototype displayed a fixed accuracy value; this version removes that unsupported claim.


## Add the ML + SHAP model
After installing requirements, run:

`python train_model.py`

This creates:
- `models/url_model.joblib`
- `models/feature_names.joblib`
- `training_urls.csv`

The web application automatically loads the saved RandomForest model if those files exist. SHAP explanations are generated from the same URL feature vector.

### Do not claim "100% accuracy"
The training script prints a held-out test result, but the included starter dataset is small and synthetic/curated. For an honours-project accuracy claim, train on a larger reputable dataset, keep an untouched test set, and report accuracy, precision, recall, F1 and ROC-AUC.
