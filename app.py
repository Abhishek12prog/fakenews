import json
import os
from datetime import datetime
from functools import wraps
from pathlib import Path

from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

from model.train_model import ensure_model_artifacts, load_model_bundle, predict_news


BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "database"
DB_DIR.mkdir(exist_ok=True)
MODEL_DIR = BASE_DIR / "model"
METRICS_FILE = MODEL_DIR / "metrics.json"
CONFUSION_MATRIX_FILE = BASE_DIR / "static" / "images" / "confusion_matrix.png"


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "change-this-secret-key")
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_DIR / 'app.db'}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    predictions = db.relationship(
        "PredictionHistory",
        backref="user",
        lazy=True,
        cascade="all, delete-orphan",
    )

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    input_text = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    champion_model = db.Column(db.String(50), nullable=False, default="naive_bayes")
    selected_model = db.Column(db.String(50), nullable=False, default="naive_bayes")
    logistic_probability = db.Column(db.Float, nullable=False)
    naive_bayes_probability = db.Column(db.Float, nullable=False)
    linear_svm_probability = db.Column(db.Float, nullable=False, default=0.0)
    passive_aggressive_probability = db.Column(db.Float, nullable=False, default=0.0)
    sgd_classifier_probability = db.Column(db.Float, nullable=False, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)


def load_metrics() -> dict:
    if METRICS_FILE.exists():
        with METRICS_FILE.open("r", encoding="utf-8") as file:
            return json.load(file)
    return {}


def login_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login to continue.", "warning")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapped_view


def admin_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        user = get_current_user()
        if not user or not user.is_admin:
            flash("Admin access required.", "danger")
            return redirect(url_for("dashboard"))
        return view_func(*args, **kwargs)

    return wrapped_view


def get_current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    return db.session.get(User, user_id)


def seed_admin():
    admin_email = os.environ.get("ADMIN_EMAIL", "admin@fakenews.local")
    admin_password = os.environ.get("ADMIN_PASSWORD", "Admin@123")
    admin_name = os.environ.get("ADMIN_NAME", "System Admin")

    admin_user = User.query.filter_by(email=admin_email).first()
    if admin_user:
        return

    admin_user = User(full_name=admin_name, email=admin_email, is_admin=True)
    admin_user.set_password(admin_password)
    db.session.add(admin_user)
    db.session.commit()


def ensure_database_schema():
    inspector = db.inspect(db.engine)
    prediction_columns = {column["name"] for column in inspector.get_columns("prediction_history")} if inspector.has_table("prediction_history") else set()
    required_columns = {
        "champion_model": "ALTER TABLE prediction_history ADD COLUMN champion_model VARCHAR(50) NOT NULL DEFAULT 'naive_bayes'",
        "selected_model": "ALTER TABLE prediction_history ADD COLUMN selected_model VARCHAR(50) NOT NULL DEFAULT 'naive_bayes'",
        "linear_svm_probability": "ALTER TABLE prediction_history ADD COLUMN linear_svm_probability FLOAT NOT NULL DEFAULT 0.0",
        "passive_aggressive_probability": "ALTER TABLE prediction_history ADD COLUMN passive_aggressive_probability FLOAT NOT NULL DEFAULT 0.0",
        "sgd_classifier_probability": "ALTER TABLE prediction_history ADD COLUMN sgd_classifier_probability FLOAT NOT NULL DEFAULT 0.0",
    }

    for column_name, statement in required_columns.items():
        if column_name not in prediction_columns:
            db.session.execute(db.text(statement))

    db.session.commit()


@app.context_processor
def inject_globals():
    return {
        "current_user": get_current_user(),
        "app_metrics": load_metrics(),
    }


@app.route("/")
def index():
    user = get_current_user()
    if user:
        if user.is_admin:
            return redirect(url_for("admin_overview"))
        return redirect(url_for("home"))
    return redirect(url_for("login"))


@app.route("/home")
@login_required
def home():
    user = get_current_user()
    if user.is_admin:
        return redirect(url_for("admin_overview"))
    history_count = PredictionHistory.query.filter_by(user_id=user.id).count()
    latest_prediction = (
        PredictionHistory.query.filter_by(user_id=user.id)
        .order_by(PredictionHistory.created_at.desc())
        .first()
    )
    metrics = load_metrics()
    return render_template(
        "index.html",
        history_count=history_count,
        latest_prediction=latest_prediction,
        metrics=metrics,
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()

        if not full_name or not email or not password:
            flash("All fields are required.", "danger")
            return redirect(url_for("register"))

        if User.query.filter_by(email=email).first():
            flash("This email is already registered.", "warning")
            return redirect(url_for("register"))

        user = User(full_name=full_name, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash("Registration successful. Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()

        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password):
            flash("Invalid email or password.", "danger")
            return redirect(url_for("login"))

        session["user_id"] = user.id
        flash("Login successful.", "success")
        if user.is_admin:
            return redirect(url_for("admin_overview"))
        return redirect(url_for("home"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))


@app.route("/dashboard")
@login_required
def dashboard():
    user = get_current_user()
    if user.is_admin:
        return redirect(url_for("admin_overview"))
    metrics = load_metrics()
    return render_template("dashboard.html", metrics=metrics, model_options=build_model_options(metrics))


@app.route("/history")
@login_required
def history():
    user = get_current_user()
    history = (
        PredictionHistory.query.filter_by(user_id=user.id)
        .order_by(PredictionHistory.created_at.desc())
        .all()
    )
    return render_template("history.html", history=history)


@app.route("/metrics")
@login_required
def metrics_page():
    user = get_current_user()
    if user.is_admin:
        return redirect(url_for("admin_overview"))
    metrics = load_metrics()
    return render_template(
        "metrics.html",
        metrics=metrics,
        confusion_matrix_exists=CONFUSION_MATRIX_FILE.exists(),
    )


@app.route("/admin")
@login_required
@admin_required
def admin_overview():
    users = User.query.order_by(User.created_at.desc()).all()
    predictions = PredictionHistory.query.order_by(PredictionHistory.created_at.desc()).limit(10).all()
    metrics = load_metrics()
    return render_template(
        "admin_dashboard.html",
        users=users,
        predictions=predictions,
        metrics=metrics,
        confusion_matrix_exists=CONFUSION_MATRIX_FILE.exists(),
    )


@app.route("/admin/users")
@login_required
@admin_required
def admin_users():
    users = User.query.order_by(User.created_at.desc()).all()
    return render_template("admin_users.html", users=users)


@app.route("/admin/predictions")
@login_required
@admin_required
def admin_predictions():
    predictions = PredictionHistory.query.order_by(PredictionHistory.created_at.desc()).all()
    return render_template("admin_predictions.html", predictions=predictions)


@app.route("/admin/users/<int:user_id>/delete", methods=["POST"])
@login_required
@admin_required
def delete_user(user_id: int):
    user = db.session.get(User, user_id)
    current_user = get_current_user()
    if not user:
        flash("User not found.", "warning")
        return redirect(url_for("admin_users"))

    if user.id == current_user.id:
        flash("You cannot delete the currently logged-in admin.", "danger")
        return redirect(url_for("admin_users"))

    db.session.delete(user)
    db.session.commit()
    flash("User deleted successfully.", "success")
    return redirect(url_for("admin_users"))


def save_prediction_history(user_id: int, text: str, result: dict) -> None:
    history = PredictionHistory(
        user_id=user_id,
        input_text=text,
        prediction=result["prediction"],
        confidence=result["confidence"],
        champion_model=result["champion_model"],
        selected_model=result["selected_model"],
        logistic_probability=result["probabilities"]["logistic_regression"],
        naive_bayes_probability=result["probabilities"]["naive_bayes"],
        linear_svm_probability=result["probabilities"].get("linear_svm", 0.0),
        passive_aggressive_probability=result["probabilities"].get("passive_aggressive", 0.0),
        sgd_classifier_probability=result["probabilities"].get("sgd_classifier", 0.0),
    )
    db.session.add(history)
    db.session.commit()


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    ensure_model_artifacts(base_dir=BASE_DIR)
    payload = request.get_json(silent=True) or {}
    news_text = payload.get("news_text") if payload else request.form.get("news_text", "")
    selected_model = payload.get("selected_model", "auto") if payload else request.form.get("selected_model", "auto")
    news_text = (news_text or "").strip()
    selected_model = (selected_model or "auto").strip()

    if len(news_text) < 30:
        error_response = {"error": "Please enter at least 30 characters of news text."}
        if request.is_json:
            return jsonify(error_response), 400
        flash(error_response["error"], "danger")
        return redirect(url_for("dashboard"))

    model_bundle = load_model_bundle(base_dir=BASE_DIR)
    valid_models = set(model_bundle["models"].keys()) | {"auto"}
    if selected_model not in valid_models:
        error_response = {"error": "Invalid model selected."}
        if request.is_json:
            return jsonify(error_response), 400
        flash(error_response["error"], "danger")
        return redirect(url_for("dashboard"))

    result = predict_news(news_text, model_bundle, selected_model=selected_model)
    save_prediction_history(session["user_id"], news_text, result)

    response = {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "champion_model": result["champion_model"],
        "selected_model": result["selected_model"],
        "model_accuracy": result["model_accuracy"],
        "detected_at": datetime.utcnow().isoformat(),
    }

    if request.is_json:
        return jsonify(response)

    flash("Prediction generated successfully.", "success")
    return redirect(url_for("dashboard"))


@app.errorhandler(404)
def page_not_found(error):
    return render_template("error.html", error_code=404, message="Page not found."), 404


@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template("error.html", error_code=500, message="Something went wrong on the server."), 500


def bootstrap_application():
    ensure_model_artifacts(base_dir=BASE_DIR)
    with app.app_context():
        db.create_all()
        ensure_database_schema()
        seed_admin()


def build_model_options(metrics: dict):
    options = [{"value": "auto", "label": "Auto"}]
    for model_name in metrics.get("scores", {}).keys():
        options.append({"value": model_name, "label": model_name.replace("_", " ").title()})
    return options


bootstrap_application()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
