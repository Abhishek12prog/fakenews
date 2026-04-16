import json
import re
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from sklearn.svm import LinearSVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_FILE = "model_bundle.pkl"
METRICS_FILE = "metrics.json"
REQUIRED_MODELS = {
    "logistic_regression",
    "naive_bayes",
    "linear_svm",
    "passive_aggressive",
    "sgd_classifier",
}


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [token for token in text.split() if token not in ENGLISH_STOP_WORDS and len(token) > 2]
    return " ".join(tokens)


def load_dataset(base_dir: Path) -> pd.DataFrame:
    data_dir = base_dir / "data"
    fake_path = data_dir / "kaggle" / "Fake.csv"
    true_path = data_dir / "kaggle" / "True.csv"
    sample_path = data_dir / "sample_news.csv"

    if fake_path.exists() and true_path.exists():
        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)
        fake_df["label"] = 0
        true_df["label"] = 1
        df = pd.concat([fake_df, true_df], ignore_index=True)
        df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
        return df[["text", "label"]].dropna()

    if sample_path.exists():
        df = pd.read_csv(sample_path)
        return df[["text", "label"]].dropna()

    raise FileNotFoundError(
        "Dataset not found. Add Fake.csv and True.csv inside data/kaggle/ or keep data/sample_news.csv."
    )


def build_pipeline(estimator):
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    max_features=12000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            ("classifier", estimator),
        ]
    )


def export_confusion_matrix(y_true, y_pred, output_file: Path):
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(image, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Fake", "Real"],
        yticklabels=["Fake", "Real"],
        ylabel="Actual",
        xlabel="Predicted",
        title="Confusion Matrix",
    )

    threshold = matrix.max() / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                format(matrix[i, j], "d"),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )

    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close(fig)


def train_models(base_dir: Path):
    df = load_dataset(base_dir)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    models = {
        "logistic_regression": build_pipeline(LogisticRegression(max_iter=1000)),
        "naive_bayes": build_pipeline(MultinomialNB()),
        "linear_svm": build_pipeline(
            CalibratedClassifierCV(LinearSVC(random_state=42), cv=3)
        ),
        "passive_aggressive": build_pipeline(
            CalibratedClassifierCV(PassiveAggressiveClassifier(random_state=42, max_iter=1000), cv=3)
        ),
        "sgd_classifier": build_pipeline(
            SGDClassifier(loss="log_loss", random_state=42, max_iter=1000)
        ),
    }

    trained_models = {}
    scores = {}

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        trained_models[name] = pipeline
        scores[name] = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        }

    champion_name = max(scores, key=lambda model_name: scores[model_name]["f1_score"])
    champion_model = trained_models[champion_name]
    champion_predictions = champion_model.predict(X_test)

    export_confusion_matrix(
        y_true=y_test,
        y_pred=champion_predictions,
        output_file=base_dir / "static" / "images" / "confusion_matrix.png",
    )

    bundle = {
        "models": trained_models,
        "scores": scores,
        "champion_model": champion_name,
        "dataset_size": int(len(df)),
    }
    joblib.dump(bundle, base_dir / "model" / MODEL_FILE)

    metrics_payload = {
        "dataset_size": int(len(df)),
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
        "champion_model": champion_name,
        "scores": scores,
    }

    with (base_dir / "model" / METRICS_FILE).open("w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, indent=2)

    return bundle


def ensure_model_artifacts(base_dir: Path):
    model_path = base_dir / "model" / MODEL_FILE
    metrics_path = base_dir / "model" / METRICS_FILE
    if not model_path.exists() or not metrics_path.exists():
        train_models(base_dir)
        return

    bundle = joblib.load(model_path)
    bundle_models = set(bundle.get("models", {}).keys())
    if not REQUIRED_MODELS.issubset(bundle_models):
        train_models(base_dir)


def load_model_bundle(base_dir: Path):
    ensure_model_artifacts(base_dir)
    return joblib.load(base_dir / "model" / MODEL_FILE)


def predict_news(news_text: str, bundle: dict, selected_model: str = "auto"):
    champion_name = bundle["champion_model"]
    chosen_model_name = champion_name if selected_model == "auto" else selected_model
    if chosen_model_name not in bundle["models"]:
        raise ValueError("Selected model is not available.")

    chosen_model = bundle["models"][chosen_model_name]
    chosen_proba = float(chosen_model.predict_proba([news_text])[0][1]) * 100
    probabilities = {}

    for model_name, model in bundle["models"].items():
        probabilities[model_name] = round(float(model.predict_proba([news_text])[0][1]) * 100, 2)

    prediction_label = "REAL" if chosen_proba >= 50 else "FAKE"
    confidence = chosen_proba if chosen_proba >= 50 else 100 - chosen_proba

    return {
        "prediction": prediction_label,
        "confidence": round(confidence, 2),
        "probabilities": probabilities,
        "champion_model": champion_name,
        "selected_model": chosen_model_name,
        "model_accuracy": bundle["scores"][chosen_model_name]["accuracy"],
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    artifacts = train_models(project_root)
    print(f"Training complete. Champion model: {artifacts['champion_model']}")
