from __future__ import annotations

from datetime import datetime, timedelta
from functools import wraps
import importlib
import io
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, send_file, url_for
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from werkzeug.security import check_password_hash, generate_password_hash
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    Prophet = importlib.import_module("prophet").Prophet  # pyright: ignore[reportMissingImports]
    PROPHET_AVAILABLE = True
except Exception:
    Prophet = None
    PROPHET_AVAILABLE = False


app = Flask(__name__)
app.config["SECRET_KEY"] = "replace-me-in-production"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///pharmacy_pms.db?timeout=30"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(30), nullable=False, default="cashier")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sku = db.Column(db.String(80), unique=True, nullable=False)
    name = db.Column(db.String(255), nullable=False)
    category = db.Column(db.String(120), nullable=False)
    unit_price = db.Column(db.Float, nullable=False)
    quantity = db.Column(db.Integer, nullable=False, default=0)
    reorder_point = db.Column(db.Integer, nullable=False, default=10)
    expiration_date = db.Column(db.Date, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class InventoryMovement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey("product.id"), nullable=False)
    movement_type = db.Column(db.String(20), nullable=False)  # in, out, adjust
    quantity = db.Column(db.Integer, nullable=False)
    note = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    product = db.relationship("Product", backref="movements")


class Sale(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    invoice_no = db.Column(db.String(80), unique=True, nullable=False)
    sold_at = db.Column(db.DateTime, default=datetime.utcnow)
    cashier_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    total_amount = db.Column(db.Float, nullable=False, default=0)

    cashier = db.relationship("User", backref="sales")


class SaleItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sale_id = db.Column(db.Integer, db.ForeignKey("sale.id"), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey("product.id"), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    unit_price = db.Column(db.Float, nullable=False)
    subtotal = db.Column(db.Float, nullable=False)

    sale = db.relationship("Sale", backref="items")
    product = db.relationship("Product")


class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)
    action = db.Column(db.String(100), nullable=False)
    details = db.Column(db.String(500), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship("User")


class ImportedSaleRow(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    row_hash = db.Column(db.String(64), unique=True, nullable=False)
    source_file = db.Column(db.String(255), nullable=False)
    row_number = db.Column(db.Integer, nullable=False)
    sale_id = db.Column(db.Integer, db.ForeignKey("sale.id"), nullable=False)
    imported_at = db.Column(db.DateTime, default=datetime.utcnow)

    sale = db.relationship("Sale")


# Initialize database tables at app startup (works in gunicorn + Railway too)
with app.app_context():
    db.create_all()


@login_manager.user_loader
def load_user(user_id: str) -> User | None:
    return db.session.get(User, int(user_id))


def roles_required(*allowed_roles: str):
    def decorator(func_ref):
        @wraps(func_ref)
        def wrapper(*args, **kwargs):
            if not current_user.is_authenticated:
                return login_manager.unauthorized()
            if current_user.role not in allowed_roles:
                flash("Access denied for your role.", "danger")
                return redirect(url_for("dashboard"))
            return func_ref(*args, **kwargs)

        return wrapper

    return decorator


def initialize_database() -> None:
    db.create_all()
    # Keep backward compatibility for legacy plaintext users by migrating to hashes.
    legacy_users = User.query.all()
    updated = False
    for user in legacy_users:
        if not user.password.startswith("pbkdf2:") and not user.password.startswith("scrypt:"):
            user.password = generate_password_hash(user.password)
            updated = True
    if updated:
        db.session.commit()


def has_users() -> bool:
    return db.session.query(User.id).first() is not None


def generate_invoice_no() -> str:
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"INV-{stamp}"


def write_audit(action: str, details: str = "", user_id: int | None = None) -> None:
    uid = user_id
    if uid is None and current_user.is_authenticated:
        uid = current_user.id
    db.session.add(AuditLog(user_id=uid, action=action, details=details[:500]))


def verify_password(user: User, plaintext: str) -> bool:
    if not user or not plaintext:
        return False

    if check_password_hash(user.password, plaintext):
        return True

    # Backward compatibility for any old plaintext records. Auto-upgrade to hash on success.
    if user.password == plaintext:
        user.password = generate_password_hash(plaintext)
        db.session.commit()
        return True

    return False


def find_or_create_product(name: str, unit_price: float, qty_seed: int = 0) -> Product:
    normalized = name.strip()
    product = Product.query.filter(func.lower(Product.name) == normalized.lower()).first()
    if product:
        if unit_price > 0:
            product.unit_price = unit_price
        return product

    base_sku = "".join(ch for ch in normalized.upper() if ch.isalnum())[:10] or "ITEM"
    sku = base_sku
    suffix = 1
    while Product.query.filter_by(sku=sku).first():
        suffix += 1
        sku = f"{base_sku[:7]}{suffix:03d}"

    product = Product(
        sku=sku,
        name=normalized,
        category="General",
        unit_price=unit_price if unit_price > 0 else 1.0,
        quantity=max(0, qty_seed),
        reorder_point=max(5, int(round(max(0, qty_seed) * 0.2))) if qty_seed else 10,
    )
    db.session.add(product)
    db.session.flush()
    return product


def import_inventory_csv(csv_source: Any, overwrite_quantities: bool = True) -> tuple[int, int]:
    df = pd.read_csv(csv_source)
    required = {"ProductName", "quantity", "UnitPrice"}
    if not required.issubset(df.columns):
        raise ValueError("Inventory CSV must contain ProductName, quantity, UnitPrice columns")

    created = 0
    updated = 0
    for _, row in df.iterrows():
        name = str(row["ProductName"]).strip()
        if not name:
            continue
        qty = int(float(row["quantity"]))
        price = float(row["UnitPrice"])

        existing = Product.query.filter(func.lower(Product.name) == name.lower()).first()
        if existing:
            old_qty = existing.quantity
            existing.unit_price = price
            if overwrite_quantities:
                existing.quantity = qty
                delta = qty - old_qty
                if delta != 0:
                    db.session.add(
                        InventoryMovement(
                            product_id=existing.id,
                            movement_type="adjust",
                            quantity=abs(delta),
                            note=f"CSV import adjustment ({'+' if delta > 0 else '-'}{abs(delta)})",
                        )
                    )
            updated += 1
        else:
            product = find_or_create_product(name=name, unit_price=price, qty_seed=qty)
            db.session.add(
                InventoryMovement(
                    product_id=product.id,
                    movement_type="in",
                    quantity=qty,
                    note="Initial stock from inventory CSV",
                )
            )
            created += 1

    return created, updated


def import_sales_csv(csv_source: Any, cashier_id: int, source_name: str = "sales.csv") -> tuple[int, int, int]:
    df = pd.read_csv(csv_source)
    required = {"ProductName", "quantity", "UnitPrice", "Date", "Time"}
    if not required.issubset(df.columns):
        raise ValueError("Sales CSV must contain ProductName, quantity, UnitPrice, Date, Time columns")

    imported_sales = 0
    imported_items = 0
    skipped_duplicates = 0
    source_file = source_name or "sales.csv"

    for idx, row in df.iterrows():
        name = str(row["ProductName"]).strip()
        if not name:
            continue

        qty = int(float(row["quantity"]))
        unit_price = float(row["UnitPrice"])
        sold_date = str(row["Date"])
        sold_time = str(row["Time"])
        sold_at = datetime.strptime(f"{sold_date} {sold_time}", "%Y-%m-%d %H:%M:%S")

        dedup_material = f"{source_file}|{idx}|{sold_date}|{sold_time}|{name}|{qty}|{unit_price:.4f}"
        row_hash = hashlib.sha256(dedup_material.encode("utf-8")).hexdigest()
        if ImportedSaleRow.query.filter_by(row_hash=row_hash).first():
            skipped_duplicates += 1
            continue

        product = find_or_create_product(name=name, unit_price=unit_price)

        sale = Sale(
            invoice_no=f"IMP-{sold_at.strftime('%Y%m%d%H%M%S')}-{idx}",
            sold_at=sold_at,
            cashier_id=cashier_id,
            total_amount=max(0.0, qty * unit_price),
        )
        db.session.add(sale)
        db.session.flush()

        db.session.add(
            SaleItem(
                sale_id=sale.id,
                product_id=product.id,
                quantity=qty,
                unit_price=unit_price,
                subtotal=max(0.0, qty * unit_price),
            )
        )
        db.session.add(
            ImportedSaleRow(
                row_hash=row_hash,
                source_file=source_file,
                row_number=int(idx) + 2,
                sale_id=sale.id,
            )
        )
        imported_sales += 1
        imported_items += 1

    return imported_sales, imported_items, skipped_duplicates


def report_data() -> tuple[list[Any], list[Any]]:
    monthly_sales = (
        db.session.query(
            func.strftime("%Y-%m", Sale.sold_at).label("month"),
            func.coalesce(func.sum(Sale.total_amount), 0).label("total"),
        )
        .group_by("month")
        .order_by("month")
        .all()
    )

    top_products = (
        db.session.query(Product.name, func.coalesce(func.sum(SaleItem.quantity), 0).label("qty"))
        .join(SaleItem, Product.id == SaleItem.product_id)
        .group_by(Product.name)
        .order_by(func.sum(SaleItem.quantity).desc())
        .limit(10)
        .all()
    )

    return monthly_sales, top_products


def to_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = to_mape(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def sales_series(product_id: int | None = None) -> pd.Series:
    query = (
        db.session.query(Sale.sold_at, func.sum(SaleItem.quantity).label("qty"))
        .join(SaleItem, Sale.id == SaleItem.sale_id)
        .group_by(func.date(Sale.sold_at))
        .order_by(Sale.sold_at.asc())
    )

    if product_id:
        query = query.filter(SaleItem.product_id == product_id)

    rows = query.all()
    if not rows:
        return pd.Series(dtype="float64")

    df = pd.DataFrame(rows, columns=["date", "qty"])
    df["date"] = pd.to_datetime(df["date"])
    monthly = df.set_index("date").resample("MS")["qty"].sum().astype(float)
    monthly = monthly.asfreq("MS", fill_value=0.0)
    return monthly


def run_forecasting_models(series: pd.Series, horizon: int = 3) -> dict[str, Any]:
    if len(series) < 8:
        return {
            "error": "Not enough historical points. Add at least 8 months of sales history.",
            "scores": {},
            "best_model": None,
            "future_forecast": [],
        }

    split_index = max(1, int(len(series) * 0.7))
    train = series.iloc[:split_index]
    test = series.iloc[split_index:]

    model_scores: dict[str, dict[str, float]] = {}
    model_predictions: dict[str, np.ndarray] = {}

    # Holt-Winters
    try:
        hw_model = ExponentialSmoothing(train, trend="add", seasonal=None)
        hw_fit = hw_model.fit(optimized=True)
        hw_pred = hw_fit.forecast(len(test)).to_numpy()
        model_predictions["Holt-Winters"] = hw_pred
        model_scores["Holt-Winters"] = metrics(test.to_numpy(), hw_pred)
    except Exception:
        pass

    # SARIMA
    try:
        sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 12), enforce_stationarity=False)
        sarima_fit = sarima_model.fit(disp=False)
        sarima_pred = sarima_fit.forecast(steps=len(test)).to_numpy()
        model_predictions["SARIMA"] = sarima_pred
        model_scores["SARIMA"] = metrics(test.to_numpy(), sarima_pred)
    except Exception:
        pass

    # Prophet
    if PROPHET_AVAILABLE:
        try:
            prophet_train = train.reset_index()
            prophet_train.columns = ["ds", "y"]
            prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            prophet_model.fit(prophet_train)
            future = prophet_model.make_future_dataframe(periods=len(test), freq="MS")
            forecast = prophet_model.predict(future)
            prophet_pred = forecast["yhat"].tail(len(test)).to_numpy()
            model_predictions["Prophet"] = prophet_pred
            model_scores["Prophet"] = metrics(test.to_numpy(), prophet_pred)
        except Exception:
            pass

    if not model_scores:
        return {
            "error": "All forecasting models failed. Check the sales data quality.",
            "scores": {},
            "best_model": None,
            "future_forecast": [],
        }

    best_model = min(model_scores.items(), key=lambda item: item[1]["RMSE"])[0]

    # Refit best model on full data for horizon forecasts.
    future_values: np.ndarray
    if best_model == "Holt-Winters":
        best_fit = ExponentialSmoothing(series, trend="add", seasonal=None).fit(optimized=True)
        future_values = best_fit.forecast(horizon).to_numpy()
    elif best_model == "SARIMA":
        best_fit = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 0, 1, 12), enforce_stationarity=False).fit(disp=False)
        future_values = best_fit.forecast(steps=horizon).to_numpy()
    else:
        prophet_full = series.reset_index()
        prophet_full.columns = ["ds", "y"]
        p_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        p_model.fit(prophet_full)
        p_future = p_model.make_future_dataframe(periods=horizon, freq="MS")
        p_forecast = p_model.predict(p_future)
        future_values = p_forecast["yhat"].tail(horizon).to_numpy()

    start = series.index[-1] + pd.offsets.MonthBegin(1)
    future_index = pd.date_range(start=start, periods=horizon, freq="MS")

    return {
        "error": None,
        "scores": model_scores,
        "best_model": best_model,
        "future_forecast": [
            {"month": d.strftime("%Y-%m"), "predicted_demand": max(0, float(v))}
            for d, v in zip(future_index, future_values)
        ],
    }


@app.route("/")
def home():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return render_template("landing.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if not has_users():
        flash("No account found yet. Please sign up to create the first admin account.", "info")
        return redirect(url_for("signup"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        user = User.query.filter_by(username=username).first()
        if user and verify_password(user, password):
            login_user(user)
            write_audit("LOGIN_SUCCESS", f"User {username} logged in", user_id=user.id)
            db.session.commit()
            flash("Logged in successfully.", "success")
            return redirect(url_for("dashboard"))
        write_audit("LOGIN_FAILED", f"Failed login attempt for username: {username}")
        db.session.commit()
        flash("Invalid username or password.", "danger")
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    first_user = not has_users()
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()

        if not username or not password:
            flash("Username and password are required.", "danger")
        elif password != confirm_password:
            flash("Passwords do not match.", "danger")
        elif len(password) < 8:
            flash("Password must be at least 8 characters.", "danger")
        elif User.query.filter_by(username=username).first():
            flash("Username already exists.", "danger")
        else:
            assigned_role = "admin" if first_user else "cashier"
            new_user = User(
                username=username,
                password=generate_password_hash(password),
                role=assigned_role,
            )
            db.session.add(new_user)
            db.session.flush()
            write_audit(
                "SIGNUP",
                f"User {username} created via signup with role {assigned_role}",
                user_id=new_user.id,
            )
            db.session.commit()
            flash("Account created successfully. Please sign in.", "success")
            return redirect(url_for("login"))

    return render_template("signup.html", first_user=first_user)


@app.route("/logout")
@login_required
def logout():
    write_audit("LOGOUT", f"User {current_user.username} logged out", user_id=current_user.id)
    db.session.commit()
    logout_user()
    flash("You have logged out.", "info")
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    product_count = Product.query.count()
    category_count = db.session.query(func.count(func.distinct(Product.category))).scalar() or 0
    low_stock_count = Product.query.filter(Product.quantity <= Product.reorder_point).count()
    expiring_count = Product.query.filter(
        Product.expiration_date.isnot(None),
        Product.expiration_date <= (datetime.utcnow().date() + timedelta(days=60)),
    ).count()
    today_sales = (
        db.session.query(func.coalesce(func.sum(Sale.total_amount), 0.0))
        .filter(func.date(Sale.sold_at) == datetime.utcnow().date())
        .scalar()
    )
    transactions_count = Sale.query.count()
    items_sold = int(db.session.query(func.coalesce(func.sum(SaleItem.quantity), 0)).scalar() or 0)

    today = datetime.utcnow().date()
    last_30_start = today - timedelta(days=29)
    prev_30_start = today - timedelta(days=59)
    prev_30_end = today - timedelta(days=30)

    last_30_sales = (
        db.session.query(func.coalesce(func.sum(Sale.total_amount), 0.0))
        .filter(func.date(Sale.sold_at) >= last_30_start, func.date(Sale.sold_at) <= today)
        .scalar()
        or 0.0
    )
    prev_30_sales = (
        db.session.query(func.coalesce(func.sum(Sale.total_amount), 0.0))
        .filter(func.date(Sale.sold_at) >= prev_30_start, func.date(Sale.sold_at) <= prev_30_end)
        .scalar()
        or 0.0
    )
    if prev_30_sales > 0:
        sales_growth = ((last_30_sales - prev_30_sales) / prev_30_sales) * 100
    else:
        sales_growth = 0.0

    return render_template(
        "dashboard.html",
        product_count=product_count,
        category_count=category_count,
        low_stock_count=low_stock_count,
        expiring_count=expiring_count,
        today_sales=today_sales,
        transactions_count=transactions_count,
        items_sold=items_sold,
        sales_growth=sales_growth,
    )


@app.route("/inventory-dashboard")
@login_required
def inventory_dashboard():
    product_count = Product.query.count()
    category_count = db.session.query(func.count(func.distinct(Product.category))).scalar() or 0
    low_stock_count = Product.query.filter(Product.quantity <= Product.reorder_point).count()
    out_of_stock_count = Product.query.filter(Product.quantity <= 0).count()

    near_expiry_products = Product.query.filter(
        Product.expiration_date.isnot(None),
        Product.expiration_date <= (datetime.utcnow().date() + timedelta(days=60)),
    ).all()

    near_expiry_count = len(near_expiry_products)
    near_expiry_categories = len({p.category for p in near_expiry_products}) if near_expiry_products else 0

    top_products = (
        db.session.query(Product.name, func.coalesce(func.sum(SaleItem.quantity), 0).label("qty"))
        .outerjoin(SaleItem, Product.id == SaleItem.product_id)
        .group_by(Product.name)
        .order_by(func.coalesce(func.sum(SaleItem.quantity), 0).desc())
        .limit(12)
        .all()
    )

    max_qty = max([int(qty) for _, qty in top_products], default=1)
    top_product_bars = [
        {"name": name, "qty": int(qty), "height": max(12, int((int(qty) / max_qty) * 100))}
        for name, qty in top_products
    ]

    monthly_rows = (
        db.session.query(
            func.strftime("%Y-%m", Sale.sold_at).label("month_key"),
            func.coalesce(func.sum(SaleItem.quantity), 0).label("qty"),
        )
        .join(SaleItem, Sale.id == SaleItem.sale_id)
        .group_by("month_key")
        .all()
    )
    monthly_map = {row.month_key: int(row.qty) for row in monthly_rows}

    now = datetime.utcnow()
    months: list[dict[str, Any]] = []
    for offset in range(11, -1, -1):
        total_months = now.year * 12 + now.month - 1 - offset
        year = total_months // 12
        month = total_months % 12 + 1
        month_key = f"{year:04d}-{month:02d}"
        month_label = datetime(year, month, 1).strftime("%b")
        qty = monthly_map.get(month_key, 0)
        months.append({"key": month_key, "label": month_label, "qty": qty})

    max_month_qty = max([m["qty"] for m in months], default=1)
    for m in months:
        m["height"] = max(8, int((m["qty"] / max_month_qty) * 100)) if max_month_qty > 0 else 8

    return render_template(
        "inventory_dashboard.html",
        product_count=product_count,
        category_count=category_count,
        low_stock_count=low_stock_count,
        out_of_stock_count=out_of_stock_count,
        near_expiry_count=near_expiry_count,
        near_expiry_categories=near_expiry_categories,
        top_products=top_products,
        top_product_bars=top_product_bars,
        monthly_sales_bars=months,
    )


@app.route("/users", methods=["GET", "POST"])
@login_required
@roles_required("admin")
def users():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        role = request.form.get("role", "cashier").strip()
        if not username or not password:
            flash("Username and password are required.", "danger")
        elif User.query.filter_by(username=username).first():
            flash("Username already exists.", "danger")
        else:
            db.session.add(User(username=username, password=generate_password_hash(password), role=role))
            write_audit("USER_CREATED", f"Created user {username} with role {role}")
            db.session.commit()
            flash("User created.", "success")
            return redirect(url_for("users"))

    data = User.query.order_by(User.created_at.desc()).all()
    return render_template("users.html", users=data)


@app.route("/inventory", methods=["GET", "POST"])
@login_required
def inventory():
    if request.method == "POST":
        return redirect(url_for("inventory_new"))

    def infer_type(category_value: str) -> str:
        lowered = (category_value or "").lower()
        if "baby" in lowered:
            return "Mom&Baby"
        if "vitamin" in lowered:
            return "Supplement"
        if "care" in lowered or "hygiene" in lowered:
            return "Personal Care"
        return "Medicine"

    def stock_status(product_ref: Product) -> tuple[str, str]:
        if product_ref.quantity <= 0:
            return "Empty", "empty"
        if product_ref.quantity <= product_ref.reorder_point:
            return "Low", "low"
        return "Good", "good"

    keyword = request.args.get("q", "").strip()
    category_filter = request.args.get("category", "").strip()

    query = Product.query
    if keyword:
        query = query.filter((Product.name.ilike(f"%{keyword}%")) | (Product.sku.ilike(f"%{keyword}%")))
    if category_filter:
        query = query.filter(Product.category == category_filter)

    products = query.order_by(Product.name.asc()).all()
    categories = [row[0] for row in db.session.query(Product.category).distinct().order_by(Product.category.asc()).all() if row[0]]

    rows: list[dict[str, Any]] = []
    for p in products:
        item_type = infer_type(p.category)
        status_label, status_class = stock_status(p)
        strip_size = 10
        qty_piece = max(p.quantity, 0)
        qty_strip = qty_piece // strip_size
        price_piece = max(float(p.unit_price), 0.0)
        price_strip = price_piece * strip_size

        rows.append(
            {
                "id": p.id,
                "name": p.name,
                "code": p.sku,
                "type": item_type,
                "category": p.category,
                "size": p.reorder_point,
                "qty_piece": qty_piece,
                "qty_strip": qty_strip,
                "price_piece": price_piece,
                "price_strip": price_strip,
                "status_label": status_label,
                "status_class": status_class,
                "expiration": p.expiration_date,
            }
        )

    return render_template(
        "inventory.html",
        rows=rows,
        keyword=keyword,
        categories=categories,
        selected_category=category_filter,
    )


@app.route("/inventory/new", methods=["GET", "POST"])
@login_required
def inventory_new():
    if request.method == "POST":
        try:
            product_name = request.form.get("name", "").strip()
            code = request.form.get("sku", "").strip()
            category = request.form.get("category", "General").strip() or "General"
            qty_piece = int(request.form.get("quantity_piece", "0") or "0")
            qty_strip = int(request.form.get("quantity_strip", "0") or "0")
            unit_price_piece = float(request.form.get("unit_price_piece", "0") or "0")
            unit_price_strip = float(request.form.get("unit_price_strip", "0") or "0")
            strip_size = 10
            total_quantity = max(0, qty_piece) + (max(0, qty_strip) * strip_size)
            derived_price_piece = unit_price_piece if unit_price_piece > 0 else (unit_price_strip / strip_size if unit_price_strip > 0 else 0)

            if not product_name:
                raise ValueError("Product name is required")
            if not code:
                raise ValueError("Code is required")
            if derived_price_piece <= 0:
                raise ValueError("Unit price should be greater than 0")

            if Product.query.filter_by(sku=code).first():
                raise ValueError("Code already exists")

            reorder_raw = request.form.get("size", "10").strip()
            reorder_point = int(reorder_raw) if reorder_raw.isdigit() else 10

            expiration_value = request.form.get("expiration_date", "").strip()
            expiration_date = datetime.strptime(expiration_value, "%Y-%m-%d").date() if expiration_value else None

            product = Product(
                sku=code,
                name=product_name,
                category=category,
                unit_price=derived_price_piece,
                quantity=total_quantity,
                reorder_point=max(1, reorder_point),
                expiration_date=expiration_date,
            )
            db.session.add(product)
            db.session.commit()

            db.session.add(
                InventoryMovement(
                    product_id=product.id,
                    movement_type="in",
                    quantity=product.quantity,
                    note="Initial stock from Add Item form",
                )
            )
            write_audit("PRODUCT_CREATED", f"Added product {product.name} (qty={product.quantity})")
            db.session.commit()
            flash("Product added to inventory.", "success")
            return redirect(url_for("inventory"))
        except Exception as exc:
            db.session.rollback()
            flash(f"Unable to add product: {exc}", "danger")

    return render_template("inventory_new.html")


@app.route("/inventory/product/<int:product_id>")
@login_required
def inventory_product_info(product_id: int):
    product = db.session.get(Product, product_id)
    if not product:
        flash("Product not found.", "danger")
        return redirect(url_for("inventory"))

    strip_size = 10
    qty_piece = max(product.quantity, 0)
    qty_strip = qty_piece // strip_size
    unit_price_piece = max(float(product.unit_price), 0.0)
    unit_price_strip = unit_price_piece * strip_size

    return render_template(
        "inventory_product_info.html",
        product=product,
        qty_piece=qty_piece,
        qty_strip=qty_strip,
        strip_size=strip_size,
        unit_price_piece=unit_price_piece,
        unit_price_strip=unit_price_strip,
    )


@app.route("/inventory/delete/<int:product_id>", methods=["POST"])
@login_required
def inventory_delete(product_id: int):
    product = db.session.get(Product, product_id)
    if not product:
        flash("Product not found.", "danger")
        return redirect(url_for("inventory"))

    try:
        InventoryMovement.query.filter_by(product_id=product.id).delete(synchronize_session=False)
        SaleItem.query.filter_by(product_id=product.id).delete(synchronize_session=False)
        db.session.delete(product)
        write_audit("PRODUCT_DELETED", f"Deleted product {product.name} ({product.sku})")
        db.session.commit()
        flash("Product deleted.", "success")
    except Exception as exc:
        db.session.rollback()
        flash(f"Unable to delete product: {exc}", "danger")

    return redirect(url_for("inventory"))


@app.route("/inventory/adjust/<int:product_id>", methods=["POST"])
@login_required
def adjust_inventory(product_id: int):
    product = db.session.get(Product, product_id)
    if not product:
        flash("Product not found.", "danger")
        return redirect(url_for("inventory"))

    try:
        movement_type = request.form.get("movement_type", "adjust").strip()
        quantity = int(request.form.get("quantity", "0"))
        note = request.form.get("note", "").strip()
        if quantity <= 0:
            raise ValueError("Quantity should be greater than 0")

        if movement_type == "in":
            product.quantity += quantity
        else:
            if quantity > product.quantity:
                raise ValueError("Cannot deduct beyond available stock")
            product.quantity -= quantity

        db.session.add(
            InventoryMovement(
                product_id=product.id,
                movement_type=movement_type,
                quantity=quantity,
                note=note,
            )
        )
        write_audit(
            "INVENTORY_ADJUSTED",
            f"{product.name}: {movement_type} {quantity}. New qty={product.quantity}. Note={note}",
        )
        db.session.commit()
        flash("Inventory updated.", "success")
    except Exception as exc:
        db.session.rollback()
        flash(f"Inventory update failed: {exc}", "danger")

    return redirect(url_for("inventory"))


@app.route("/pos-terminal", methods=["GET", "POST"])
@login_required
def pos_terminal():
    products = Product.query.order_by(Product.name.asc()).all()

    if request.method == "POST":
        product_ids = request.form.getlist("product_id")
        quantities = request.form.getlist("quantity")

        if not product_ids:
            flash("Add at least one item to process a sale.", "danger")
            return redirect(url_for("pos_terminal"))

        sale = Sale(invoice_no=generate_invoice_no(), cashier_id=current_user.id, total_amount=0)
        total_amount = 0.0

        try:
            db.session.add(sale)
            db.session.flush()

            for idx, pid in enumerate(product_ids):
                qty = int(quantities[idx])
                if qty <= 0:
                    continue
                product = db.session.get(Product, int(pid))
                if not product:
                    raise ValueError("Selected product no longer exists")
                if product.quantity < qty:
                    raise ValueError(f"Insufficient stock for {product.name}")

                subtotal = product.unit_price * qty
                total_amount += subtotal
                product.quantity -= qty

                db.session.add(
                    SaleItem(
                        sale_id=sale.id,
                        product_id=product.id,
                        quantity=qty,
                        unit_price=product.unit_price,
                        subtotal=subtotal,
                    )
                )
                db.session.add(
                    InventoryMovement(
                        product_id=product.id,
                        movement_type="out",
                        quantity=qty,
                        note=f"Sold via {sale.invoice_no}",
                    )
                )

            if total_amount <= 0:
                raise ValueError("No valid sale items were entered")

            sale.total_amount = total_amount
            write_audit("SALE_COMPLETED", f"Invoice {sale.invoice_no} total PHP {total_amount:.2f}")
            db.session.commit()
            flash(f"Sale completed. Invoice: {sale.invoice_no}", "success")
            return redirect(url_for("pos_terminal"))
        except Exception as exc:
            db.session.rollback()
            flash(f"Sale failed: {exc}", "danger")

    return render_template("pos_terminal.html", products=products)


@app.route("/sales")
@login_required
def sales():
    sales_data = Sale.query.order_by(Sale.sold_at.desc()).limit(20).all()
    sales_rows = [
        {
            "transaction_id": f"TID-{sale.id:04d}",
            "series_code": f"{sale.id:05d}",
            "date": sale.sold_at.strftime("%m-%d-%Y"),
            "time": sale.sold_at.strftime("%I:%M:%S %p"),
            "details": {
                "invoice_no": sale.invoice_no,
                "cashier": sale.cashier.username if sale.cashier else "N/A",
                "date": sale.sold_at.strftime("%m-%d-%Y"),
                "time": sale.sold_at.strftime("%I:%M:%S %p"),
                "total_amount": float(sale.total_amount),
                "items": [
                    {
                        "name": item.product.name if item.product else "Unknown Item",
                        "quantity": int(item.quantity),
                        "unit_price": float(item.unit_price),
                        "subtotal": float(item.subtotal),
                    }
                    for item in sale.items
                ],
            },
        }
        for sale in sales_data
    ]
    return render_template("sales.html", sales_rows=sales_rows)


@app.route("/billing")
@login_required
def billing():
    sales_data = Sale.query.order_by(Sale.sold_at.desc()).limit(20).all()
    billing_rows = [
        {
            "transaction_id": f"TID-{sale.id:04d}",
            "mode_payment": "Cash",
            "total": sale.total_amount,
            "date": sale.sold_at.strftime("%m-%d-%Y"),
            "ref_number": f"{sale.id:04d}-{sale.sold_at.strftime('%H%M')}-{sale.sold_at.strftime('%m%d')}",
        }
        for sale in sales_data
    ]
    return render_template("billing.html", billing_rows=billing_rows)


@app.route("/reports")
@login_required
def reports():
    def infer_type(category_value: str) -> str:
        lowered = (category_value or "").lower()
        if "baby" in lowered:
            return "Mom&Baby"
        if "vitamin" in lowered:
            return "Supplement"
        if "care" in lowered or "hygiene" in lowered:
            return "Personal Care"
        return "Medicine"

    sales_rows_query = (
        db.session.query(
            Sale.sold_at,
            Product.name,
            Product.sku,
            Product.category,
            SaleItem.quantity,
            SaleItem.unit_price,
            SaleItem.subtotal,
        )
        .join(SaleItem, Sale.id == SaleItem.sale_id)
        .join(Product, Product.id == SaleItem.product_id)
        .order_by(Sale.sold_at.desc())
        .limit(250)
        .all()
    )

    sales_rows = [
        {
            "source": "Sales",
            "transaction_date": row.sold_at.strftime("%m/%d/%Y"),
            "product_name": row.name,
            "code": row.sku,
            "type": infer_type(row.category),
            "category": row.category,
            "quantity": int(row.quantity),
            "unit_price": float(row.unit_price),
            "total_sales": float(row.subtotal),
            "payment_method": "Cash",
        }
        for row in sales_rows_query
    ]

    inventory_rows_query = Product.query.order_by(Product.name.asc()).limit(250).all()
    inventory_rows = []
    for product in inventory_rows_query:
        total_sales = (
            db.session.query(func.coalesce(func.sum(SaleItem.subtotal), 0.0))
            .filter(SaleItem.product_id == product.id)
            .scalar()
            or 0.0
        )
        inventory_rows.append(
            {
                "source": "Inventory",
                "transaction_date": "-",
                "product_name": product.name,
                "code": product.sku,
                "quantity": product.quantity,
                "unit_price": float(product.unit_price),
                "total_sales": float(total_sales),
                "payment_method": "Cash",
                "type": infer_type(product.category),
                "category": product.category,
                "current_stock": product.quantity,
            }
        )

    report_rows = [
        {
            **row,
            "current_stock": "-",
        }
        for row in sales_rows
    ] + inventory_rows

    categories = [
        row[0]
        for row in db.session.query(Product.category).distinct().order_by(Product.category.asc()).all()
        if row[0]
    ]
    report_types = ["Weekly", "Monthly", "Custom"]

    return render_template(
        "reports.html",
        report_rows=report_rows,
        categories=categories,
        report_types=report_types,
    )


@app.route("/reports/pdf")
@login_required
def reports_pdf():
    monthly_sales, top_products = report_data()
    stream = io.BytesIO()
    pdf = canvas.Canvas(stream, pagesize=A4)
    width, height = A4

    y = height - 40
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(40, y, "Nica Xandra Drugstore - Sales Report")
    y -= 24
    pdf.setFont("Helvetica", 9)
    pdf.drawString(40, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    y -= 30
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(40, y, "Monthly Sales")
    y -= 16
    pdf.setFont("Helvetica", 10)
    for month, total in monthly_sales:
        if y < 60:
            pdf.showPage()
            y = height - 40
            pdf.setFont("Helvetica", 10)
        pdf.drawString(50, y, str(month))
        pdf.drawRightString(width - 40, y, f"PHP {float(total):.2f}")
        y -= 14

    y -= 18
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(40, y, "Top Products by Quantity Sold")
    y -= 16
    pdf.setFont("Helvetica", 10)
    for name, qty in top_products:
        if y < 60:
            pdf.showPage()
            y = height - 40
            pdf.setFont("Helvetica", 10)
        pdf.drawString(50, y, str(name)[:70])
        pdf.drawRightString(width - 40, y, str(int(qty)))
        y -= 14

    pdf.save()
    stream.seek(0)
    write_audit("REPORT_EXPORTED_PDF", "Sales report exported as PDF")
    db.session.commit()
    return send_file(stream, mimetype="application/pdf", as_attachment=True, download_name="sales-report.pdf")


@app.route("/reports/excel")
@login_required
def reports_excel():
    data = (
        db.session.query(
            Sale.sold_at,
            Product.name,
            Product.sku,
            Product.category,
            SaleItem.quantity,
            SaleItem.unit_price,
            SaleItem.subtotal,
        )
        .join(SaleItem, Sale.id == SaleItem.sale_id)
        .join(Product, Product.id == SaleItem.product_id)
        .order_by(Sale.sold_at.desc())
        .limit(1000)
        .all()
    )

    rows = [
        {
            "Transaction Date": row.sold_at.strftime("%Y-%m-%d"),
            "Product Name": row.name,
            "Code": row.sku,
            "Category": row.category,
            "Quantity": int(row.quantity),
            "Unit Price": float(row.unit_price),
            "Total Sales": float(row.subtotal),
            "Payment Method": "Cash",
        }
        for row in data
    ]

    csv_text = pd.DataFrame(rows).to_csv(index=False)
    stream = io.BytesIO(csv_text.encode("utf-8"))
    write_audit("REPORT_EXPORTED_EXCEL", "Sales report exported as CSV")
    db.session.commit()
    return send_file(stream, mimetype="text/csv", as_attachment=True, download_name="sales-report.csv")




@app.route("/receipt/<int:sale_id>")
@login_required
def receipt(sale_id: int):
    sale = db.session.get(Sale, sale_id)
    if not sale:
        flash("Receipt not found.", "danger")
        return redirect(url_for("sales"))
    return render_template("receipt.html", sale=sale)


@app.route("/audit-logs")
@login_required
@roles_required("admin")
def audit_logs():
    logs = AuditLog.query.order_by(AuditLog.created_at.desc()).limit(500).all()
    return render_template("audit_logs.html", logs=logs)


@app.route("/import-data", methods=["GET", "POST"])
@login_required
@roles_required("admin")
def import_data():
    default_inventory = str(Path(__file__).resolve().parents[1] / "LESSONS" / "REPORT" / "ritedose-stock-inventory.csv")
    default_sales = str(Path(__file__).resolve().parents[1] / "LESSONS" / "REPORT" / "ritedose-sales.csv")

    if request.method == "POST":
        import_type = request.form.get("import_type", "both")
        inventory_path = request.form.get("inventory_path", "").strip() or default_inventory
        sales_path = request.form.get("sales_path", "").strip() or default_sales
        overwrite_inventory = request.form.get("overwrite_inventory") == "on"
        reset_sales = request.form.get("reset_sales") == "on"
        inventory_file = request.files.get("inventory_file")
        sales_file = request.files.get("sales_file")

        try:
            created = 0
            updated = 0
            imported_sales = 0
            imported_items = 0
            skipped_duplicates = 0

            if import_type in {"inventory", "both"}:
                inventory_source: Any
                inventory_source_label: str
                if inventory_file and inventory_file.filename:
                    inventory_source = inventory_file
                    inventory_source_label = inventory_file.filename
                else:
                    inventory_source = inventory_path
                    inventory_source_label = inventory_path

                created, updated = import_inventory_csv(inventory_source, overwrite_quantities=overwrite_inventory)

            if import_type in {"sales", "both"}:
                if reset_sales:
                    SaleItem.query.delete()
                    Sale.query.delete()
                    ImportedSaleRow.query.delete()

                sales_source: Any
                sales_source_label: str
                if sales_file and sales_file.filename:
                    sales_source = sales_file
                    sales_source_label = sales_file.filename
                else:
                    sales_source = sales_path
                    sales_source_label = Path(sales_path).name

                imported_sales, imported_items, skipped_duplicates = import_sales_csv(
                    sales_source,
                    cashier_id=current_user.id,
                    source_name=sales_source_label,
                )

            write_audit(
                "CSV_IMPORT",
                (
                    f"type={import_type}; Inventory file={inventory_file.filename if inventory_file and inventory_file.filename else inventory_path}; "
                    f"Sales file={sales_file.filename if sales_file and sales_file.filename else sales_path}; "
                    f"products_created={created}; products_updated={updated}; "
                    f"sales_imported={imported_sales}; items_imported={imported_items}; "
                    f"sales_duplicates_skipped={skipped_duplicates}; reset_sales={reset_sales}"
                ),
            )
            db.session.commit()
            if import_type == "inventory":
                flash(f"Inventory import successful: {created} new products, {updated} updated products.", "success")
            elif import_type == "sales":
                flash(
                    (
                        f"Sales import successful: {imported_sales} sales rows loaded, "
                        f"{skipped_duplicates} duplicates skipped."
                    ),
                    "success",
                )
            else:
                flash(
                    (
                        f"Import successful: {created} new products, {updated} updated products, "
                        f"{imported_sales} sales rows loaded, {skipped_duplicates} duplicates skipped."
                    ),
                    "success",
                )
            return redirect(url_for("import_data"))
        except Exception as exc:
            db.session.rollback()
            flash(f"Import failed: {exc}", "danger")

    return render_template(
        "import_data.html",
        default_inventory=default_inventory,
        default_sales=default_sales,
    )


@app.route("/export-dataset/inventory")
@login_required
@roles_required("admin")
def export_dataset_inventory():
    rows = [
        {
            "ProductName": product.name,
            "quantity": int(product.quantity),
            "UnitPrice": float(product.unit_price),
            "SKU": product.sku,
            "Category": product.category,
            "ReorderPoint": int(product.reorder_point),
            "ExpiryDate": product.expiration_date.isoformat() if product.expiration_date else "",
        }
        for product in Product.query.order_by(Product.name.asc()).all()
    ]

    csv_text = pd.DataFrame(rows).to_csv(index=False)
    stream = io.BytesIO(csv_text.encode("utf-8"))
    write_audit("DATASET_EXPORTED", "Inventory dataset exported as CSV")
    db.session.commit()
    return send_file(stream, mimetype="text/csv", as_attachment=True, download_name="inventory-dataset.csv")


@app.route("/export-dataset/sales")
@login_required
@roles_required("admin")
def export_dataset_sales():
    records = (
        db.session.query(
            Sale.sold_at,
            Product.name,
            SaleItem.quantity,
            SaleItem.unit_price,
            Sale.invoice_no,
        )
        .join(SaleItem, Sale.id == SaleItem.sale_id)
        .join(Product, Product.id == SaleItem.product_id)
        .order_by(Sale.sold_at.asc())
        .all()
    )

    rows = [
        {
            "ProductName": row.name,
            "quantity": int(row.quantity),
            "UnitPrice": float(row.unit_price),
            "Date": row.sold_at.strftime("%Y-%m-%d"),
            "Time": row.sold_at.strftime("%H:%M:%S"),
            "InvoiceNo": row.invoice_no,
        }
        for row in records
    ]

    csv_text = pd.DataFrame(rows).to_csv(index=False)
    stream = io.BytesIO(csv_text.encode("utf-8"))
    write_audit("DATASET_EXPORTED", "Sales dataset exported as CSV")
    db.session.commit()
    return send_file(stream, mimetype="text/csv", as_attachment=True, download_name="sales-dataset.csv")


@app.route("/alerts")
@login_required
def alerts():
    today = datetime.utcnow().date()
    soon = today + timedelta(days=60)
    selected_tab = request.args.get("tab", "all")
    if selected_tab not in {"all", "out-of-stock", "low-stock", "near-expiry"}:
        selected_tab = "all"

    page = max(1, request.args.get("page", default=1, type=int) or 1)
    per_page = 40

    out_of_stock_query = Product.query.with_entities(
        Product.id, Product.name, Product.sku, Product.category, Product.quantity
    ).filter(Product.quantity <= 0).order_by(Product.name.asc())

    low_stock_query = Product.query.with_entities(
        Product.id, Product.name, Product.sku, Product.category, Product.quantity, Product.reorder_point
    ).filter((Product.quantity > 0) & (Product.quantity <= Product.reorder_point)).order_by(Product.quantity.asc(), Product.name.asc())

    expiring_query = Product.query.with_entities(
        Product.id, Product.name, Product.sku, Product.category, Product.expiration_date
    ).filter(
        Product.expiration_date.isnot(None),
        Product.expiration_date <= soon,
        Product.expiration_date >= today,
    ).order_by(Product.expiration_date.asc(), Product.name.asc())

    out_of_stock_count = out_of_stock_query.count()
    low_stock_count = low_stock_query.count()
    expiring_count = expiring_query.count()

    out_of_stock_items: list[dict[str, Any]] = []
    low_stock_items: list[dict[str, Any]] = []
    expiring_items: list[dict[str, Any]] = []
    all_alerts_feed: list[dict[str, Any]] = []
    total_pages = 1

    if selected_tab == "out-of-stock":
        total_pages = max(1, (out_of_stock_count + per_page - 1) // per_page)
        page = min(page, total_pages)
        rows = out_of_stock_query.offset((page - 1) * per_page).limit(per_page).all()
        out_of_stock_items = [
            {
                "id": row.id,
                "name": row.name,
                "sku": row.sku,
                "category": row.category,
                "quantity": row.quantity,
            }
            for row in rows
        ]

    elif selected_tab == "low-stock":
        total_pages = max(1, (low_stock_count + per_page - 1) // per_page)
        page = min(page, total_pages)
        rows = low_stock_query.offset((page - 1) * per_page).limit(per_page).all()
        low_stock_items = [
            {
                "id": row.id,
                "name": row.name,
                "sku": row.sku,
                "category": row.category,
                "quantity": row.quantity,
                "reorder_point": row.reorder_point,
                "gap": row.reorder_point - row.quantity,
                "percent_filled": max(0, int((row.quantity / max(1, row.reorder_point)) * 100)),
            }
            for row in rows
        ]

    elif selected_tab == "near-expiry":
        total_pages = max(1, (expiring_count + per_page - 1) // per_page)
        page = min(page, total_pages)
        rows = expiring_query.offset((page - 1) * per_page).limit(per_page).all()
        expiring_items = [
            {
                "id": row.id,
                "name": row.name,
                "sku": row.sku,
                "category": row.category,
                "expiration_date": row.expiration_date,
                "days_left": (row.expiration_date - today).days if row.expiration_date else None,
                "urgency": "critical" if row.expiration_date and (row.expiration_date - today).days <= 7 else "warning",
            }
            for row in rows
        ]

    else:
        for row in out_of_stock_query.all():
            all_alerts_feed.append(
                {
                    "type": "out-of-stock",
                    "severity": 0,
                    "id": row.id,
                    "name": row.name,
                    "sku": row.sku,
                    "category": row.category,
                    "title": "Out of stock",
                    "summary": f"{row.name} is currently unavailable.",
                    "meta": f"SKU {row.sku} | Qty {row.quantity}",
                }
            )

        for row in low_stock_query.all():
            gap = row.reorder_point - row.quantity
            all_alerts_feed.append(
                {
                    "type": "low-stock",
                    "severity": 1,
                    "id": row.id,
                    "name": row.name,
                    "sku": row.sku,
                    "category": row.category,
                    "title": "Low stock",
                    "summary": f"{row.name} is running low.",
                    "meta": f"SKU {row.sku} | Need {gap} more to reach reorder level",
                }
            )

        for row in expiring_query.all():
            days_left = (row.expiration_date - today).days if row.expiration_date else 0
            all_alerts_feed.append(
                {
                    "type": "near-expiry",
                    "severity": 2,
                    "id": row.id,
                    "name": row.name,
                    "sku": row.sku,
                    "category": row.category,
                    "title": "Near expiry",
                    "summary": f"{row.name} expires in {days_left} day{'s' if days_left != 1 else ''}.",
                    "meta": f"SKU {row.sku} | Expires {row.expiration_date}",
                }
            )

        all_alerts_feed.sort(key=lambda item: (item["severity"], item["name"].lower()))
        total_pages = max(1, (len(all_alerts_feed) + per_page - 1) // per_page)
        page = min(page, total_pages)
        start = (page - 1) * per_page
        end = start + per_page
        all_alerts_feed = all_alerts_feed[start:end]

    return render_template(
        "alerts.html",
        out_of_stock_count=out_of_stock_count,
        low_stock_count=low_stock_count,
        expiring_count=expiring_count,
        out_of_stock_items=out_of_stock_items,
        low_stock_items=low_stock_items,
        expiring_items=expiring_items,
        all_alerts_feed=all_alerts_feed,
        selected_tab=selected_tab,
        page=page,
        total_pages=total_pages,
        has_prev=page > 1,
        has_next=page < total_pages,
    )


@app.route("/forecast", methods=["GET", "POST"])
@login_required
def forecast():
    products = Product.query.order_by(Product.name.asc()).all()
    result = None
    selected_product_id = request.form.get("product_id") if request.method == "POST" else None

    if request.method == "POST":
        horizon = int(request.form.get("horizon", "3"))
        pid = int(selected_product_id) if selected_product_id and selected_product_id != "all" else None
        series = sales_series(product_id=pid)
        result = run_forecasting_models(series=series, horizon=horizon)

    return render_template("forecast.html", products=products, result=result, selected_product_id=selected_product_id)


if __name__ == "__main__":
    with app.app_context():
        initialize_database()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
