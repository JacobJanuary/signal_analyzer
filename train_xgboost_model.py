import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb  # <<< ИЗМЕНЕНИЕ: импортируем xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# --- НАСТРОЙКИ ---
# Укажите имя вашего файла с данными
PARQUET_FILENAME = 'full_ai_dataset_2025-06-12_1543.parquet' # <--- ЗАМЕНИТЕ НА ИМЯ ВАШЕГО ФАЙЛА

# --- 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ---
try:
    df = pd.read_parquet(PARQUET_FILENAME)
    print(f"✅ Файл '{PARQUET_FILENAME}' успешно загружен. Всего сигналов для анализа: {len(df)}")
except FileNotFoundError:
    print(f"❌ Ошибка: Файл '{PARQUET_FILENAME}' не найден.")
    exit()

df.dropna(subset=['max_profit_pct', 'max_drawdown_pct'], inplace=True)
print(f"Осталось сигналов после очистки: {len(df)}")

df['is_successful'] = ((df['did_hit_profit_target_5pct'] == 1) & (df['did_hit_stop_loss_3pct'] == 0)).astype(int)

y = df['is_successful']
columns_to_drop = [
    'signal_id', 'symbol', 'timestamp', 'enriched_id', 'created_at', 'updated_at',
    'timeseries_to_max_profit', 'timeseries_to_stop_loss',
    'is_successful', 'did_hit_profit_target_5pct', 'did_hit_stop_loss_3pct',
    'max_profit_pct', 'max_drawdown_pct', 'time_to_max_profit_minutes'
]
X_raw = df.drop(columns=columns_to_drop, errors='ignore')
X = pd.get_dummies(X_raw, dummy_na=True)
feature_names = X.columns
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

class_counts = y.value_counts()
print("\nРаспределение сигналов:")
print(class_counts)


# --- 2. РАЗДЕЛЕНИЕ ДАННЫХ И ОБУЧЕНИЕ МОДЕЛИ ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# <<< ИЗМЕНЕНИЕ: Создание и обучение модели XGBoost >>>

print("\n🚀 Обучение модели XGBoost...")

# Рассчитываем вес для борьбы с дисбалансом классов
# Это скажет модели уделять больше внимания редким "успешным" сигналам
scale_pos_weight = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 1

model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=150,          # Количество деревьев
    learning_rate=0.1,         # Скорость обучения
    max_depth=5,               # Максимальная глубина каждого дерева
    scale_pos_weight=scale_pos_weight, # Важный параметр для дисбаланса!
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("Модель обучена!")


# --- 3. ОЦЕНКА КАЧЕСТВА МОДЕЛИ ---
print("\n--- Оценка качества модели XGBoost на тестовых данных ---")
y_pred = model.predict(X_test)

print("\nМатрица ошибок:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Неуспешный', 'Успешный'], yticklabels=['Неуспешный', 'Успешный'])
plt.title('Матрица ошибок (XGBoost)')
plt.ylabel('Истинный класс')
plt.xlabel('Предсказанный класс')
plt.savefig('xgboost_confusion_matrix.png') # Новое имя файла
print("🖼️  Матрица ошибок сохранена в 'xgboost_confusion_matrix.png'")
plt.clf()

print("\nОтчет по классификации:")
print(classification_report(y_test, y_pred, target_names=['Неуспешный', 'Успешный']))


# --- 4. ВАЖНОСТЬ ПРИЗНАКОВ ---
print("\n--- Важность признаков по версии XGBoost ---")
feature_importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

print("ТОП-15 самых важных признаков:")
print(feature_importances.head(15))

plt.figure(figsize=(12, 10))
sns.barplot(x=feature_importances.head(20), y=feature_importances.head(20).index)
plt.title('ТОП-20 самых важных признаков по версии XGBoost')
plt.xlabel('Важность')
plt.ylabel('Признаки')
plt.tight_layout()
plt.savefig('xgboost_feature_importance.png') # Новое имя файла
print("\n🖼️  График важности признаков сохранен в 'xgboost_feature_importance.png'")
print("\n✅ Анализ и обучение завершены.")