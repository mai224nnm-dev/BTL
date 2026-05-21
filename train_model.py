import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE

os.makedirs('models', exist_ok=True)

def build_and_save_model():
    print("1. Đang tải và chuẩn bị gốc dữ liệu...")
    df = pd.read_excel('dữ-liệu-lớn.xlsx')
    
    X = df.drop('bankrupt', axis=1, errors='ignore')
    y = df['bankrupt']
    features = X.columns.tolist()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print("2. Tiền xử lý Dữ liệu (Scaling & SMOTE)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    
    print("3. Đang huấn luyện Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_res, y_train_res)
    
    print("4. Đang huấn luyện Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    rf_model.fit(X_train_res, y_train_res)
    
    print("5. Đang huấn luyện XGBoost...")
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train_res, y_train_res)
    
    print("6. Đánh giá Mô hình đa chiều...")
    metrics = {}
    models_dict = {
        'Logistic Regression': lr_model,
        'Random Forest': rf_model,
        'XGBoost': xgb_model
    }
    
    for name, model in models_dict.items():
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        metrics[name] = {
            'Accuracy': float(accuracy_score(y_test, y_pred)),
            'F1-Score': float(f1_score(y_test, y_pred)),
            'ROC-AUC': float(roc_auc_score(y_test, y_proba))
        }
        
    print("7. Trích xuất Feature Importances từ XGBoost...")
    importances = xgb_model.feature_importances_
    feat_importances = pd.Series(importances, index=features).sort_values(ascending=False).head(20).to_dict()
    
    print("8. Lưu trữ Kiến trúc hệ thống mới...")
    joblib.dump({
        'models': models_dict,
        'scaler': scaler,
        'features': features,
        'metrics': metrics,
        'importances': feat_importances
    }, 'models/enterprise_resources.pkl')
    
    print("✅ Đã huấn luyện siêu hệ thống thành công!")

if __name__ == "__main__":
    build_and_save_model()
