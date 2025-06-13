import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# --- НАСТРОЙКИ ---
# Укажите имя вашего ПОЛНОГО файла с данными
PARQUET_FILENAME = 'full_ai_dataset_2025-06-12_1543.parquet' # <--- ЗАМЕНИТЕ НА ИМЯ ВАШЕГО ПОСЛЕДНЕГО ФАЙЛА

# --- 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ---
try:
    df = pd.read_parquet(PARQUET_FILENAME)
    print(f"✅ Файл '{PARQUET_FILENAME}' успешно загружен. Всего сигналов для анализа: {len(df)}")
except FileNotFoundError:
    print(f"❌ Ошибка: Файл '{PARQUET_FILENAME}' не найден. Убедитесь, что он находится в той же папке.")
    exit()

# Удаляем строки, где нет данных о результате
df.dropna(subset=['max_profit_pct', 'max_drawdown_pct'], inplace=True)
print(f"Осталось сигналов после очистки: {len(df)}")

# Создание нашей целевой переменной 'y': Успешный сигнал = (профит >= 5%) И (просадка > -3%)
df['is_successful'] = ((df['did_hit_profit_target_5pct'] == 1) & (df['did_hit_stop_loss_3pct'] == 0)).astype(int)

# --- ИЗМЕНЕНИЕ: ГОТОВИМ ПРИЗНАКИ ДЛЯ МОДЕЛИ ---

# Определяем цель (то, что мы предсказываем)
y = df['is_successful']

# Определяем признаки (X), используя ВСЕ данные, кроме служебных и целевых
# Исключаем ID, временные метки, текстовые описания и сложные вложенные данные
columns_to_drop = [
    'signal_id', 'symbol', 'timestamp', 'enriched_id', 'created_at', 'updated_at',
    'timeseries_to_max_profit', 'timeseries_to_stop_loss', # сложные вложенные данные
    'is_successful', 'did_hit_profit_target_5pct', 'did_hit_stop_loss_3pct', # целевые и производные от них
    'max_profit_pct', 'max_drawdown_pct', 'time_to_max_profit_minutes' # сами результаты тоже исключаем из признаков
]
X_raw = df.drop(columns=columns_to_drop, errors='ignore')

# Преобразуем категориальные переменные (текст) в числовой формат (One-Hot Encoding)
X = pd.get_dummies(X_raw, dummy_na=True)

# Запоминаем имена всех признаков после преобразования
feature_names = X.columns

# Обработка пропущенных числовых значений
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

print(f"\nМодель будет обучаться на {X.shape[1]} признаках.")
print("Распределение сигналов:")
print(y.value_counts())


# --- 2. РАЗДЕЛЕНИЕ ДАННЫХ И ОБУЧЕНИЕ МОДЕЛИ ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n🌳 Обучение модели Random Forest на полном наборе данных...")
model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', n_jobs=-1)
model.fit(X_train, y_train)
print("Модель обучена!")


# --- 3. ОЦЕНКА КАЧЕСТВА МОДЕЛИ ---
print("\n--- Оценка качества модели на тестовых данных ---")
y_pred = model.predict(X_test)

print("\nМатрица ошибок:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Неуспешный', 'Успешный'], yticklabels=['Неуспешный', 'Успешный'])
plt.title('Матрица ошибок')
plt.ylabel('Истинный класс')
plt.xlabel('Предсказанный класс')
plt.savefig('full_model_confusion_matrix.png')
print("🖼️  Матрица ошибок сохранена в 'full_model_confusion_matrix.png'")
plt.clf()

print("\nОтчет по классификации:")
print(classification_report(y_test, y_pred, target_names=['Неуспешный', 'Успешный']))


# --- 4. ВАЖНОСТЬ ПРИЗНАКОВ ---
print("\n--- Важность признаков для принятия решения моделью ---")
feature_importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

print("ТОП-15 самых важных признаков:")
print(feature_importances.head(15))

# Визуализация важности признаков
plt.figure(figsize=(12, 10))
sns.barplot(x=feature_importances.head(20), y=feature_importances.head(20).index)
plt.title('ТОП-20 самых важных признаков по версии Random Forest')
plt.xlabel('Важность')
plt.ylabel('Признаки')
plt.tight_layout()
plt.savefig('full_model_feature_importance.png')
print("\n🖼️  График важности признаков сохранен в 'full_model_feature_importance.png'")
print("\n✅ Анализ и обучение завершены.")