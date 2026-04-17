import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from model.train_model import ensure_model_artifacts, load_model_bundle, predict_news


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database" / "streamlit_app.db"
METRICS_FILE = BASE_DIR / "model" / "metrics.json"
CONFUSION_MATRIX_FILE = BASE_DIR / "static" / "images" / "confusion_matrix.png"

DEFAULT_ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@fakenews.local")
DEFAULT_ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "Admin@123")
DEFAULT_ADMIN_NAME = os.environ.get("ADMIN_NAME", "System Admin")


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


CONNECTION = get_connection()


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, digest = stored_hash.split("$", 1)
    except ValueError:
        return False
    computed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000).hex()
    return hmac.compare_digest(computed, digest)


def init_database():
    CONNECTION.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """
    )
    CONNECTION.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            input_text TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            selected_model TEXT NOT NULL,
            champion_model TEXT NOT NULL,
            probabilities_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """
    )
    CONNECTION.commit()


def seed_admin():
    existing = CONNECTION.execute(
        "SELECT id FROM users WHERE email = ?",
        (DEFAULT_ADMIN_EMAIL.lower(),),
    ).fetchone()
    if existing:
        return

    CONNECTION.execute(
        """
        INSERT INTO users (full_name, email, password_hash, is_admin, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            DEFAULT_ADMIN_NAME,
            DEFAULT_ADMIN_EMAIL.lower(),
            hash_password(DEFAULT_ADMIN_PASSWORD),
            1,
            datetime.utcnow().isoformat(),
        ),
    )
    CONNECTION.commit()


def load_metrics():
    if METRICS_FILE.exists():
        return json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    return {}


def get_model_options():
    metrics = load_metrics()
    options = [("auto", "Auto")]
    for model_name in metrics.get("scores", {}).keys():
        options.append((model_name, format_model_name(model_name)))
    return options


def format_model_name(model_name: str) -> str:
    return model_name.replace("_", " ").title()


def create_user(full_name: str, email: str, password: str):
    if CONNECTION.execute("SELECT id FROM users WHERE email = ?", (email.lower(),)).fetchone():
        return False, "Email already exists."

    CONNECTION.execute(
        """
        INSERT INTO users (full_name, email, password_hash, is_admin, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (full_name.strip(), email.lower().strip(), hash_password(password), 0, datetime.utcnow().isoformat()),
    )
    CONNECTION.commit()
    return True, "Account created."


def authenticate_user(email: str, password: str):
    row = CONNECTION.execute(
        "SELECT * FROM users WHERE email = ?",
        (email.lower().strip(),),
    ).fetchone()
    if not row or not verify_password(password, row["password_hash"]):
        return None
    return dict(row)


def store_prediction(user_id: int, text: str, result: dict):
    CONNECTION.execute(
        """
        INSERT INTO predictions (
            user_id, input_text, prediction, confidence, selected_model,
            champion_model, probabilities_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            text,
            result["prediction"],
            result["confidence"],
            result["selected_model"],
            result["champion_model"],
            json.dumps(result["probabilities"]),
            datetime.utcnow().isoformat(),
        ),
    )
    CONNECTION.commit()


def get_user_history(user_id: int):
    rows = CONNECTION.execute(
        """
        SELECT * FROM predictions
        WHERE user_id = ?
        ORDER BY datetime(created_at) DESC
        """,
        (user_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_all_users():
    rows = CONNECTION.execute(
        "SELECT * FROM users ORDER BY datetime(created_at) DESC"
    ).fetchall()
    return [dict(row) for row in rows]


def get_all_predictions(limit: int | None = None):
    query = """
        SELECT predictions.*, users.full_name, users.email
        FROM predictions
        JOIN users ON users.id = predictions.user_id
        ORDER BY datetime(predictions.created_at) DESC
    """
    params = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (limit,)
    rows = CONNECTION.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def delete_user(user_id: int):
    CONNECTION.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,))
    CONNECTION.execute("DELETE FROM users WHERE id = ? AND is_admin = 0", (user_id,))
    CONNECTION.commit()


def login_view():
    st.title("Fake News Detection")
    login_tab, register_tab = st.tabs(["Login", "Create Account"])

    with login_tab:
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)
        if submitted:
            user = authenticate_user(email, password)
            if not user:
                st.error("Invalid email or password.")
            else:
                st.session_state.user = user
                st.session_state.page = "Admin Home" if user["is_admin"] else "Home"
                st.rerun()

    with register_tab:
        with st.form("register_form", clear_on_submit=True):
            full_name = st.text_input("Full Name")
            email = st.text_input("Email", key="register_email")
            password = st.text_input("Password", type="password", key="register_password")
            submitted = st.form_submit_button("Create Account", use_container_width=True)
        if submitted:
            if not full_name.strip() or not email.strip() or not password.strip():
                st.error("All fields are required.")
            else:
                ok, message = create_user(full_name, email, password)
                if ok:
                    st.success(message)
                else:
                    st.error(message)


def user_home(user: dict):
    metrics = load_metrics()
    history = get_user_history(user["id"])
    latest_prediction = history[0] if history else None

    st.title("Fake News Detection")
    col1, col2, col3, col4 = st.columns(4)
    champion_scores = metrics.get("scores", {}).get(metrics.get("champion_model", ""), {})
    col1.metric("Accuracy", f"{champion_scores.get('accuracy', 0) * 100:.2f}%")
    col2.metric("Champion", format_model_name(metrics.get("champion_model", "pending")))
    col3.metric("Dataset", metrics.get("dataset_size", 0))
    col4.metric("History", len(history))

    quick1, quick2, quick3 = st.columns(3)
    if quick1.button("Check News", use_container_width=True):
        st.session_state.page = "Detect News"
        st.rerun()
    if quick2.button("View History", use_container_width=True):
        st.session_state.page = "History"
        st.rerun()
    if quick3.button("View Metrics", use_container_width=True):
        st.session_state.page = "Metrics"
        st.rerun()

    if latest_prediction:
        st.subheader("Latest Prediction")
        st.write(latest_prediction["input_text"])
        tag_col1, tag_col2, tag_col3 = st.columns(3)
        tag_col1.metric("Prediction", latest_prediction["prediction"])
        tag_col2.metric("Confidence", f"{latest_prediction['confidence']:.2f}%")
        tag_col3.metric("Model", format_model_name(latest_prediction["selected_model"]))


def detect_news_page(user: dict):
    st.title("Detect News")
    model_options = get_model_options()
    option_map = {label: value for value, label in model_options}

    selected_label = st.selectbox("Model", list(option_map.keys()), index=0)
    news_text = st.text_area("News Text", height=260, placeholder="Paste news content here...")

    if st.button("Check News", type="primary", use_container_width=True):
        if len(news_text.strip()) < 30:
            st.error("Enter at least 30 characters.")
            return

        with st.spinner("Analyzing..."):
            bundle = load_model_bundle(BASE_DIR)
            result = predict_news(news_text.strip(), bundle, selected_model=option_map[selected_label])
            store_prediction(user["id"], news_text.strip(), result)

        st.success("Done")
        info1, info2, info3, info4 = st.columns(4)
        info1.metric("Prediction", result["prediction"])
        info2.metric("Confidence", f"{result['confidence']:.2f}%")
        info3.metric("Selected Model", format_model_name(result["selected_model"]))
        info4.metric("Accuracy", f"{result['model_accuracy'] * 100:.2f}%")

        probability_df = pd.DataFrame(
            {
                "Model": [format_model_name(name) for name in result["probabilities"].keys()],
                "Probability": list(result["probabilities"].values()),
            }
        ).set_index("Model")
        st.bar_chart(probability_df)


def history_page(user: dict):
    st.title("History")
    history = get_user_history(user["id"])
    if not history:
        st.info("No history found.")
        return

    for item in history:
        probabilities = json.loads(item["probabilities_json"])
        with st.container(border=True):
            st.write(item["input_text"])
            col1, col2, col3 = st.columns(3)
            col1.metric("Prediction", item["prediction"])
            col2.metric("Confidence", f"{item['confidence']:.2f}%")
            col3.metric("Model", format_model_name(item["selected_model"]))
            st.caption(
                " | ".join(
                    f"{format_model_name(name)}: {value:.2f}%"
                    for name, value in probabilities.items()
                )
            )


def metrics_page():
    st.title("Metrics")
    metrics = load_metrics()
    scores = metrics.get("scores", {})
    if not scores:
        st.warning("Metrics not available.")
        return

    data = []
    for model_name, score in scores.items():
        data.append(
            {
                "Model": format_model_name(model_name),
                "Accuracy": score["accuracy"],
                "Precision": score["precision"],
                "Recall": score["recall"],
                "F1 Score": score["f1_score"],
            }
        )
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

    if CONFUSION_MATRIX_FILE.exists():
        st.image(str(CONFUSION_MATRIX_FILE), caption="Confusion Matrix", use_container_width=True)


def admin_home():
    st.title("Admin Home")
    users = get_all_users()
    predictions = get_all_predictions(limit=10)
    metrics = load_metrics()
    champion_scores = metrics.get("scores", {}).get(metrics.get("champion_model", ""), {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Users", len(users))
    col2.metric("Predictions", len(get_all_predictions()))
    col3.metric("Accuracy", f"{champion_scores.get('accuracy', 0) * 100:.2f}%")
    col4.metric("Champion", format_model_name(metrics.get("champion_model", "pending")))

    st.subheader("Latest Predictions")
    for item in predictions:
        with st.container(border=True):
            st.write(item["input_text"])
            meta1, meta2, meta3 = st.columns(3)
            meta1.metric("Prediction", item["prediction"])
            meta2.metric("User", item["full_name"])
            meta3.metric("Model", format_model_name(item["selected_model"]))


def admin_users_page(current_user: dict):
    st.title("Users")
    users = get_all_users()
    for user in users:
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
            col1.write(user["full_name"])
            col2.write(user["email"])
            col3.write("Admin" if user["is_admin"] else "User")
            if user["is_admin"] or user["id"] == current_user["id"]:
                col4.write("Protected")
            else:
                if col4.button("Delete", key=f"delete_{user['id']}"):
                    delete_user(user["id"])
                    st.rerun()


def admin_predictions_page():
    st.title("Predictions")
    predictions = get_all_predictions()
    if not predictions:
        st.info("No predictions found.")
        return

    for item in predictions:
        probabilities = json.loads(item["probabilities_json"])
        with st.container(border=True):
            st.write(item["input_text"])
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Prediction", item["prediction"])
            col2.metric("Confidence", f"{item['confidence']:.2f}%")
            col3.metric("User", item["full_name"])
            col4.metric("Model", format_model_name(item["selected_model"]))
            st.caption(
                " | ".join(
                    f"{format_model_name(name)}: {value:.2f}%"
                    for name, value in probabilities.items()
                )
            )


def logout():
    st.session_state.user = None
    st.session_state.page = "Login"
    st.rerun()


def main():
    st.set_page_config(page_title="Fake News Detection", layout="wide")
    init_database()
    seed_admin()
    ensure_model_artifacts(BASE_DIR)

    if "user" not in st.session_state:
        st.session_state.user = None
    if "page" not in st.session_state:
        st.session_state.page = "Login"

    if not st.session_state.user:
        login_view()
        return

    user = st.session_state.user
    with st.sidebar:
        st.title("Fake News Detection")
        st.write(user["full_name"])
        st.write(user["email"])
        if user["is_admin"]:
            pages = ["Admin Home", "Users", "Predictions"]
        else:
            pages = ["Home", "Detect News", "History", "Metrics"]
        current_page = st.radio("Menu", pages, index=pages.index(st.session_state.page) if st.session_state.page in pages else 0)
        st.session_state.page = current_page
        st.button("Logout", on_click=logout, use_container_width=True)

    if user["is_admin"]:
        if st.session_state.page == "Admin Home":
            admin_home()
        elif st.session_state.page == "Users":
            admin_users_page(user)
        else:
            admin_predictions_page()
    else:
        if st.session_state.page == "Home":
            user_home(user)
        elif st.session_state.page == "Detect News":
            detect_news_page(user)
        elif st.session_state.page == "History":
            history_page(user)
        else:
            metrics_page()


if __name__ == "__main__":
    main()
