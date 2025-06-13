import pandas as pd
import numpy as np
import joblib
import argparse
import sys
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import urllib.parse


def create_db_engine():
    """Создает движок SQLAlchemy с проверкой переменных окружения."""
    # Умная загрузка .env файла из директории, где находится сам скрипт
    dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
    else:
        # Эта строка не должна появиться, если .env лежит рядом со скриптом
        print(f"❌ ПРЕДУПРЕЖДЕНИЕ: не найден .env файл по пути {dotenv_path}")

    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST')
    db_name = os.getenv('DB_NAME')
    db_port = os.getenv('DB_PORT', '3306')

    if not all([db_user, db_password, db_host, db_name]):
        print("❌ Ошибка: Одна или несколько переменных для подключения к БД не найдены в .env файле.")
        print("   Пожалуйста, проверьте наличие DB_USER, DB_PASSWORD, DB_HOST, DB_NAME.")
        sys.exit(1)
    try:
        encoded_password = urllib.parse.quote_plus(db_password)
        db_url = f"mysql+pymysql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
        return create_engine(db_url)
    except Exception as e:
        print(f"❌ Ошибка создания движка SQLAlchemy: {e}", file=sys.stderr)
        sys.exit(1)


def get_single_signal_data(engine, signal_id: int):
    """Получает все данные для одного конкретного сигнала."""
    query = text("""
                 SELECT s.*,
                        e.id as enriched_id,
                        e.oi_usdt_average,
                        e.oi_usdt_current,
                        e.oi_usdt_yesterday,
                        e.oi_usdt_change_current_to_yesterday,
                        e.oi_usdt_change_current_to_average,
                        e.oi_source_usdt,
                        e.spot_volume_usdt_average,
                        e.spot_volume_usdt_current,
                        e.spot_volume_usdt_yesterday,
                        e.spot_volume_usdt_change_current_to_yesterday,
                        e.spot_volume_usdt_change_current_to_average,
                        e.spot_volume_source_usdt,
                        e.spot_volume_btc_average,
                        e.spot_volume_btc_current,
                        e.spot_volume_btc_yesterday,
                        e.spot_volume_btc_change_current_to_yesterday,
                        e.spot_volume_btc_change_current_to_average,
                        e.spot_volume_source_btc,
                        e.spot_price_usdt_average,
                        e.spot_price_usdt_current,
                        e.spot_price_usdt_yesterday,
                        e.spot_price_usdt_change_1h,
                        e.spot_price_usdt_change_24h,
                        e.spot_price_usdt_change_7d,
                        e.spot_price_usdt_change_30d,
                        e.spot_price_source_usdt,
                        e.cmc_price_min_1h,
                        e.cmc_price_max_1h,
                        e.cmc_price_min_24h,
                        e.cmc_price_max_24h,
                        e.cmc_price_min_7d,
                        e.cmc_price_max_7d,
                        e.cmc_price_min_30d,
                        e.cmc_price_max_30d,
                        e.cmc_percent_change_1d,
                        e.cmc_percent_change_7d,
                        e.cmc_percent_change_30d
                 FROM signals_10min s
                          LEFT JOIN signals_10min_enriched e ON s.id = e.signal_id
                 WHERE s.id = :signal_id;
                 """)
    df = pd.read_sql(query, engine, params={"signal_id": signal_id})
    if df.empty:
        return None
    df.rename(columns={'id': 'signal_id'}, inplace=True)
    return df


def predict_signal(signal_df: pd.DataFrame, model, imputer, model_features, categorical_features, numeric_features):
    """Готовит данные и делает предсказание для нового сигнала."""

    # 1. Отбираем ВСЕ признаки, которые были при обучении, и создаем явную копию
    # reindex гарантирует, что все нужные колонки будут на месте, даже если в LEFT JOIN пришли NULL
    X_raw = signal_df.reindex(columns=model_features).copy()

    # 2. Преобразуем категориальные колонки в тип 'category'
    for col in categorical_features:
        if col in X_raw.columns:
            X_raw[col] = X_raw[col].astype('category')

    # 3. Применяем сохраненный Imputer к ТОМУ ЖЕ СПИСКУ числовых колонок
    cols_to_impute = [col for col in numeric_features if col in X_raw.columns]
    X_raw[cols_to_impute] = imputer.transform(X_raw[cols_to_impute])

    # 4. Предсказание вероятности
    # Подаем на вход данные с правильным порядком колонок
    prediction_proba = model.predict_proba(X_raw[model_features])

    return prediction_proba[0][1]  # Возвращаем вероятность класса "1" (успешный)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Предсказание успешности крипто-сигнала.")
    parser.add_argument('--signal_id', type=int, required=True, help='ID сигнала из таблицы signals_10min для анализа.')
    args = parser.parse_args()

    print(f"Загрузка модели и компонентов...")
    try:
        model = joblib.load('lgbm_model.joblib')
        imputer = joblib.load('imputer.joblib')
        model_features = joblib.load('feature_names.joblib')
        categorical_features = joblib.load('categorical_features.joblib')
        numeric_features = joblib.load('numeric_features.joblib')
        print("✅ Модель успешно загружена.")
    except FileNotFoundError as e:
        print(
            f"❌ Ошибка: Файл компонента не найден: {e.filename}. \n   Пожалуйста, сначала запустите скрипт train_lgbm_model.py для их создания.")
        sys.exit(1)

    engine = create_db_engine()

    print(f"\nПолучение данных для сигнала ID: {args.signal_id}...")
    signal_df = get_single_signal_data(engine, args.signal_id)

    if signal_df is None:
        print(f"❌ Сигнал с ID {args.signal_id} не найден в базе данных.")
    else:
        print("Данные получены. Выполнение предсказания...")
        probability_of_success = predict_signal(signal_df, model, imputer, model_features, categorical_features,
                                                numeric_features)

        # Вывод результата
        print("\n" + "=" * 50)
        print(f"АНАЛИЗ СИГНАЛА #{args.signal_id} ({signal_df['symbol'].iloc[0]})")
        print("=" * 50)
        print(f"ВЕРОЯТНОСТЬ УСПЕХА: {probability_of_success:.2%}")

        if probability_of_success > 0.65:
            print("\nРЕКОМЕНДАЦИЯ: 🔥 ВХОДИТЬ (Высокая вероятность)")
        elif probability_of_success > 0.50:
            print("\nРЕКОМЕНДАЦИЯ: 🤔 РАССМОТРЕТЬ (Средняя вероятность)")
        else:
            print("\nРЕКОМЕНДАЦИЯ: ⛔️ ПРОПУСТИТЬ (Низкая вероятность)")
        print("=" * 50)