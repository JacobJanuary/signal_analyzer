import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
import joblib # <<< ДОБАВЬТЕ ЭТОТ ИМПОРТ
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# --- НАСТРОЙКИ ---
PARQUET_FILENAME = 'full_ai_dataset_2025-06-12_1543.parquet' # <--- УКАЖИТЕ ИМЯ ВАШЕГО ФАЙЛА

# --- 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ---
try:
    df = pd.read_parquet(PARQUET_FILENAME)
    print(f"✅ Файл '{PARQUET_FILENAME}' успешно загружен.")
except FileNotFoundError:
    print(f"❌ Ошибка: Файл '{PARQUET_FILENAME}' не найден.")
    exit()

df.dropna(subset=['max_profit_pct', 'max_drawdown_pct'], inplace=True)
df['is_successful'] = ((df['did_hit_profit_target_5pct'] == 1) & (df['did_hit_stop_loss_3pct'] == 0)).astype(int)

y = df['is_successful']
columns_to_drop = [
    'signal_id', 'symbol', 'timestamp', 'enriched_id', 'created_at', 'updated_at',
    'timeseries_to_max_profit', 'timeseries_to_stop_loss',
    'is_successful', 'did_hit_profit_target_5pct', 'did_hit_stop_loss_3pct',
    'max_profit_pct', 'max_drawdown_pct', 'time_to_max_profit_minutes'
]
X_raw = df.drop(columns=columns_to_drop, errors='ignore')

categorical_features = X_raw.select_dtypes(include=['object']).columns
for col in categorical_features:
    X_raw[col] = X_raw[col].astype('category')

numeric_features = X_raw.select_dtypes(include=np.number).columns
imputer = SimpleImputer(strategy='median')
X_raw[numeric_features] = imputer.fit_transform(X_raw[numeric_features])

X = X_raw
feature_names = X.columns.to_list() # Сохраняем имена всех признаков

# --- 2. ОБУЧЕНИЕ МОДЕЛИ ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n💡 Обучение модели LightGBM...")
model = lgb.LGBMClassifier(objective='binary', n_estimators=150, learning_rate=0.1, num_leaves=31, is_unbalance=True, random_state=42, n_jobs=-1)
model.fit(X_train, y_train, categorical_feature=[str(c) for c in categorical_features])
print("Модель обучена!")


# --- 3. ОЦЕНКА КАЧЕСТВА МОДЕЛИ (без изменений) ---
# ... (этот блок остается как есть для вывода отчета) ...
print("\n--- Оценка качества модели LightGBM на тестовых данных ---")
y_pred = model.predict(X_test)
print("\nОтчет по классификации:")
print(classification_report(y_test, y_pred, target_names=['Неуспешный', 'Успешный']))


# --- 4. ВАЖНОСТЬ ПРИЗНАКОВ (без изменений) ---
# ... (этот блок остается как есть для вывода важности) ...
print("\n--- Важность признаков по версии LightGBM ---")
feature_importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
print("ТОП-15 самых важных признаков:")
print(feature_importances.head(15))


# --- 5. СОХРАНЕНИЕ МОДЕЛИ И КОМПОНЕНТОВ ---
print("\n💾 Сохранение модели и компонентов...")
joblib.dump(model, 'lgbm_model.joblib')
joblib.dump(imputer, 'imputer.joblib')
joblib.dump(feature_names, 'feature_names.joblib')
joblib.dump(list(categorical_features), 'categorical_features.joblib')
print("✅ Модель и компоненты сохранены в файлы: lgbm_model.joblib, imputer.joblib, feature_names.joblib, categorical_features.joblib")
print("\n✅ Скрипт обучения завершен.")

# ... весь код до конца ...

# --- 5. СОХРАНЕНИЕ МОДЕЛИ И КОМПОНЕНТОВ ---
print("\n💾 Сохранение модели и компонентов...")
joblib.dump(model, 'lgbm_model.joblib')
joblib.dump(imputer, 'imputer.joblib')
joblib.dump(feature_names, 'feature_names.joblib')
joblib.dump(list(categorical_features), 'categorical_features.joblib')
joblib.dump(list(numeric_features), 'numeric_features.joblib') # <<< ДОБАВЬТЕ ЭТУ СТРОКУ
print("✅ Модель и компоненты сохранены.")
print("\n✅ Скрипт обучения завершен.")