from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sklearn.datasets import load_wine, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score

app = FastAPI()

# Enable CORS so the HTML frontend can talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODEL 1: WINE CLASSIFICATION ---
wine = load_wine()
X_w = pd.DataFrame(wine.data, columns=wine.feature_names)
y_w = wine.target
X_w_train, X_w_test, y_w_train, y_w_test = train_test_split(X_w, y_w, test_size=0.2, random_state=42)
scaler_w = StandardScaler()
X_w_train_scaled = scaler_w.fit_transform(X_w_train)
X_w_test_scaled = scaler_w.transform(X_w_test)
clf_model = RandomForestClassifier(random_state=42).fit(X_w_train_scaled, y_w_train)
wine_accuracy = accuracy_score(y_w_test, clf_model.predict(X_w_test_scaled)) * 100

class WineInput(BaseModel):
    features: list[float]

@app.post("/predict/wine")
def predict_wine(data: WineInput):
    input_df = pd.DataFrame([data.features], columns=wine.feature_names)
    scaled = scaler_w.transform(input_df)
    pred_idx = clf_model.predict(scaled)[0]
    return {
        "prediction": wine.target_names[pred_idx].upper(),
        "accuracy": round(wine_accuracy, 2)
    }

# --- MODEL 2: CALIFORNIA HOUSING REGRESSION ---
housing = fetch_california_housing(as_frame=True)
X_h = housing.data
y_h = housing.target
X_h_train, X_h_test, y_h_train, y_h_test = train_test_split(X_h, y_h, test_size=0.2, random_state=42)
scaler_h = StandardScaler()
X_h_train_scaled = scaler_h.fit_transform(X_h_train)
X_h_test_scaled = scaler_h.transform(X_h_test)
reg_model = LinearRegression().fit(X_h_train_scaled, y_h_train)
housing_r2 = r2_score(y_h_test, reg_model.predict(X_h_test_scaled)) * 100

class HousingInput(BaseModel):
    features: list[float]

@app.post("/predict/housing")
def predict_housing(data: HousingInput):
    input_df = pd.DataFrame([data.features], columns=housing.feature_names)
    scaled = scaler_h.transform(input_df)
    pred_val = reg_model.predict(scaled)[0] * 100000
    return {
        "prediction": round(pred_val, 2),
        "r2_score": round(housing_r2, 2)
    }

@app.get("/metadata")
def get_metadata():
    return {
        "wine_features": wine.feature_names,
        "wine_defaults": [float(X_w[col].mean()) for col in wine.feature_names],
        "housing_features": housing.feature_names,
        "housing_defaults": [float(X_h[col].mean()) for col in housing.feature_names]
    }