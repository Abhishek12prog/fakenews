# Fake News Detection System (Advanced Web App)

Complete final year project built with Flask, SQLite, Scikit-learn, TF-IDF, Logistic Regression, and Naive Bayes.

## Project Structure

```text
fake detection/
│── app.py
│── README.md
│── requirements.txt
│── data/
│   ├── sample_news.csv
│   └── kaggle/
│       ├── Fake.csv
│       └── True.csv
│── database/
│── model/
│   ├── train_model.py
│   ├── model_bundle.pkl
│   └── metrics.json
│── static/
│   ├── css/
│   │   └── styles.css
│   ├── images/
│   │   └── confusion_matrix.png
│   └── js/
│       └── main.js
└── templates/
    ├── admin_predictions.html
    ├── admin_users.html
    ├── admin_dashboard.html
    ├── base.html
    ├── dashboard.html
    ├── error.html
    ├── history.html
    ├── index.html
    ├── login.html
    ├── metrics.html
    └── register.html
```

## Features

- User registration and login
- Multi-page user flow: Home, Detect News, History, Metrics
- Multi-page admin flow: Admin Home, Users, Predictions
- Admin dashboard with user management
- News prediction with `REAL` / `FAKE` output
- Confidence score and probability graph
- Prediction history stored in SQLite
- API endpoint at `/predict`
- Automatic ML artifact generation
- Accuracy, precision, recall, F1-score, and confusion matrix

## Setup Instructions

1. Open the project folder in VS Code:
   ```bash
   cd "/Users/abhi/Desktop/fake detection"
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the environment:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Optional but recommended: place Kaggle dataset files inside `data/kaggle/` as:
   - `data/kaggle/Fake.csv`
   - `data/kaggle/True.csv`
6. Train the model manually if you want:
   ```bash
   python model/train_model.py
   ```
7. Run the Flask app:
   ```bash
   python app.py
   ```
8. Open the browser:
   ```text
   http://127.0.0.1:5000
   ```

## Default Admin Login

- Email: `admin@fakenews.local`
- Password: `Admin@123`

Change these values in environment variables before deployment:

```bash
export ADMIN_EMAIL="your-admin@example.com"
export ADMIN_PASSWORD="StrongAdminPassword"
export SECRET_KEY="replace-with-a-secure-secret"
```

## How To Run In VS Code

1. Open the folder in VS Code.
2. Open the integrated terminal.
3. Create and activate the virtual environment.
4. Install dependencies with `pip install -r requirements.txt`.
5. Run `python app.py`.
6. Open the local server URL shown in the terminal.

## How To Run In Jupyter Notebook

Use these commands in notebook cells:

```python
!pip install -r requirements.txt
```

```python
!python model/train_model.py
```

```python
!python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## API Usage

### Endpoint

`POST /predict`

### JSON Request

```json
{
  "news_text": "Paste a long news article or paragraph here..."
}
```

### JSON Response

```json
{
  "confidence": 91.32,
  "detected_at": "2026-04-16T18:45:12.123456",
  "model_accuracy": 0.94,
  "prediction": "REAL",
  "probabilities": {
    "logistic_regression": 91.32,
    "naive_bayes": 84.17
  }
}
```

## Kaggle Dataset Link

- Fake and Real News Dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

## Notes

- If Kaggle files are not added, the app uses `data/sample_news.csv` so the project can still run for demo purposes.
- `model_bundle.pkl`, `metrics.json`, and `static/images/confusion_matrix.png` are generated automatically the first time the app starts or when training runs.
