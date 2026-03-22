# Nica Xandra Drugstore - Pharmacy Record Management System

A web-based Pharmacy Record Management System (PMS) tailored for small/community pharmacies that need practical inventory control, billing, and demand forecasting.

## Implemented Modules

- Inventory Management
  - Product master records
  - Stock-in/stock-out adjustments
  - Reorder points
  - Expiration-date tracking
- Sales and Billing
  - Sales transaction processing
  - Auto-generated invoice numbers
  - Automatic stock deduction per sale
  - Printable receipt per transaction
- Demand Forecasting
  - 70:30 train-test split
  - Compares Holt-Winters, SARIMA, and Prophet
  - Evaluation using MAE, RMSE, MAPE
  - Automatically picks best model by RMSE
- Sales Reporting
  - Monthly sales totals
  - Top products by quantity sold
- User Management and Security
  - Login/logout
  - Role-aware access (admin, pharmacist, cashier)
  - Password hashing
  - Audit logs for key actions
- Notifications and Alerts
  - Low-stock alerts
  - Expiring products (within 60 days)
  - Forecast-based reorder recommendations
- CSV Import Wizard
  - Inventory CSV import (quantity snapshot overwrite option)
  - Sales CSV import for historical analytics and forecasting
  - Duplicate sales row protection (re-import safe)
- Sales Report Export
  - PDF report export (monthly sales + top products)

## Tech Stack

- Flask
- SQLite (default local database)
- SQLAlchemy ORM
- Pandas, NumPy, Statsmodels, Prophet
- HTML + CSS (responsive)

## Quick Start

1. Open terminal in this folder.
2. Create and activate a virtual environment.
3. Install dependencies:
   pip install -r requirements.txt
4. Start the app:
   python app.py
5. Open:
   http://127.0.0.1:5000
6. Create your first account via Sign Up (this first account becomes admin).

7. Import your mock datasets (admin only):
  - Go to Import CSV page
  - Use default paths or set custom file paths
  - Run import

## Notes for Capstone Evaluation

- Forecasting model evaluation follows the scope requirement with a 70:30 split and MAE/RMSE/MAPE.
- If Prophet is not installed successfully in a local machine, the app still runs and compares available models.
- For production deployment, replace plaintext passwords with proper hashing (e.g., Werkzeug security helpers).
