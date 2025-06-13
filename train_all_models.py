import pandas as pd
import numpy as np
import joblib
import sys
import os

# Импорты моделей и компонентов
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from dotenv import load_dotenv

# --- НАСТРОЙКИ ---
PARQUET_FILENAME = 'full_ai_dataset_2025-06-12_1543.parquet' # <--- УКАЖИТЕ ИМЯ ВАШЕГО ПОСЛЕДНЕГО ФАЙЛА
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(dotenv_path)

# --- 1. ЗАГРУЗКА И ОБЩАЯ ПОДГОТОВКА ДАННЫХ ---
try:
    df = pd.read_parquet(PARQUET_FILENAME)
    print(f"✅ Файл '{PARQUET_FILENAME}' успешно загружен. Всего сигналов: {len(df)}")
except FileNotFoundError:
    print(f"❌ Ошибка: Файл '{PARQUET_FILENAME}' не найден.")
    sys.exit(1)

df.dropna(subset=['max_profit_pct', 'max_drawdown_pct'], inplace=True)
print(f"Осталось сигналов после очистки: {len(df)}")

df['is_successful'] = ((df['did_hit_profit_target_5pct'] == 1) & (df['did_hit_stop_loss_3pct'] == 0)).astype(int)

y = df['is_successful']
columns_to_drop = [
    'signal_id', 'token_id', 'symbol', 'timestamp', 'enriched_id', 'created_at', 'updated_at',
    'timeseries_to_max_profit', 'timeseries_to_stop_loss',
    'is_successful', 'did_hit_profit_target_5pct', 'did_hit_stop_loss_3pct',
    'max_profit_pct', 'max_drawdown_pct', 'time_to_max_profit_minutes'
]
X = df.drop(columns=columns_to_drop, errors='ignore')

print("\nРаспределение сигналов:")
print(y.value_counts())

# --- 2. СОЗДАНИЕ КОНВЕЙЕРОВ ПРЕДОБРАБОТКИ ---

# Определяем типы колонок
numeric_features = X.select_dtypes(include=np.number).columns.to_list()
categorical_features = X.select_dtypes(include=['object']).columns.to_list()

# Конвейер для моделей, требующих One-Hot Encoding (RF, XGBoost)
# Он сначала заполнит пропуски, а потом закодирует категории
preprocessor_dummies = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

# Конвейер для LightGBM
# Он просто заполнит пропуски, так как LGBM сам обработает категории
# ВАЖНО: Мы будем преобразовывать типы ПЕРЕД передачей в конвейер
preprocessor_lgbm = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features)
    ],
    remainder='passthrough'
)


# --- 3. ОБУЧЕНИЕ И СОХРАНЕНИЕ МОДЕЛЕЙ ---

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# --- Модель 1: Random Forest ---
print("\n🌳 Обучение модели Random Forest...")
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor_dummies),
                              ('classifier', RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', n_jobs=-1))])
rf_pipeline.fit(X_train, y_train)
joblib.dump(rf_pipeline, 'rf_pipeline.joblib')
print("✅ Конвейер Random Forest обучен и сохранен.")

# --- Модель 2: XGBoost ---
print("\n🚀 Обучение модели XGBoost...")
class_counts = y.value_counts()
scale_pos_weight = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 1
xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor_dummies),
                               ('classifier', xgb.XGBClassifier(objective='binary:logistic', n_estimators=150, learning_rate=0.1, max_depth=5, scale_pos_weight=scale_pos_weight, eval_metric='logloss', random_state=42, n_jobs=-1))])
xgb_pipeline.fit(X_train, y_train)
joblib.dump(xgb_pipeline, 'xgb_pipeline.joblib')
print("✅ Конвейер XGBoost обучен и сохранен.")

# --- Модель 3: LightGBM ---
print("\n💡 Обучение модели LightGBM...")
# Преобразуем категории перед передачей в конвейер
X_train_lgbm = X_train.copy()
for col in categorical_features:
    X_train_lgbm[col] = X_train_lgbm[col].astype('category')

lgbm_pipeline = Pipeline(steps=[('preprocessor', preprocessor_lgbm),
                                ('classifier', lgb.LGBMClassifier(objective='binary', n_estimators=150, learning_rate=0.1, num_leaves=31, is_unbalance=True, random_state=42, n_jobs=-1))])
lgbm_pipeline.fit(X_train_lgbm, y_train) # Обучаем на данных с правильным типом
joblib.dump(lgbm_pipeline, 'lgbm_pipeline.joblib')
joblib.dump(categorical_features, 'lgbm_categorical_features.joblib') # Сохраняем список категорий для предсказания
print("✅ Конвейер LightGBM обучен и сохранен.")


print("\n🏁 Все модели в виде конвейеров успешно обучены и сохранены!")

