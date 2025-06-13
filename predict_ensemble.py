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
    dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
    else:
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
    # Этот запрос извлекает все поля из обеих таблиц для указанного ID
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ансамблевое предсказание успешности крипто-сигнала.")
    parser.add_argument('--signal_id', type=int, required=True, help='ID сигнала для анализа.')
    args = parser.parse_args()

    print("Загрузка конвейеров моделей...")
    try:
        rf_pipeline = joblib.load('rf_pipeline.joblib')
        xgb_pipeline = joblib.load('xgb_pipeline.joblib')
        lgbm_pipeline = joblib.load('lgbm_pipeline.joblib')
        print("✅ Все модели успешно загружены.")
    except FileNotFoundError as e:
        print(
            f"❌ Ошибка: Файл модели не найден: {e.filename}. \n   Пожалуйста, сначала запустите скрипт train_all_models.py для их создания.")
        sys.exit(1)

    engine = create_db_engine()
    print(f"\nПолучение данных для сигнала ID: {args.signal_id}...")
    signal_df = get_single_signal_data(engine, args.signal_id)

    if signal_df is None:
        print(f"❌ Сигнал с ID {args.signal_id} не найден.")
    else:
        print("Данные получены. Выполнение предсказаний...")

        # Данные для предсказания - это просто сырой DataFrame.
        # Вся предобработка теперь происходит внутри конвейеров.
        X_predict = signal_df

        # Получаем "голоса" от каждой модели
        rf_vote = rf_pipeline.predict(X_predict)[0]
        rf_proba = rf_pipeline.predict_proba(X_predict)[0][1]

        xgb_vote = xgb_pipeline.predict(X_predict)[0]
        xgb_proba = xgb_pipeline.predict_proba(X_predict)[0][1]

        # Для LightGBM мы должны вручную преобразовать категории, как при обучении
        X_predict_lgbm = X_predict.copy()
        categorical_features = X_predict_lgbm.select_dtypes(include=['object']).columns
        for col in categorical_features:
            X_predict_lgbm[col] = X_predict_lgbm[col].astype('category')

        lgbm_vote = lgbm_pipeline.predict(X_predict_lgbm)[0]
        lgbm_proba = lgbm_pipeline.predict_proba(X_predict_lgbm)[0][1]

        votes = [rf_vote, xgb_vote, lgbm_vote]
        successful_votes = sum(votes)
        avg_proba = np.mean([rf_proba, xgb_proba, lgbm_proba])

        # Вывод результата
        print("\n" + "=" * 60)
        print(f"РЕЗУЛЬТАТ АНСАМБЛЕВОГО АНАЛИЗА ДЛЯ СИГНАЛА #{args.signal_id} ({signal_df['symbol'].iloc[0]})")
        print("=" * 60)
        print(f"  - Мнение Random Forest: {'Успех' if rf_vote else 'Провал'} (Вероятность успеха: {rf_proba:.1%})")
        print(f"  - Мнение XGBoost:       {'Успех' if xgb_vote else 'Провал'} (Вероятность успеха: {xgb_proba:.1%})")
        print(f"  - Мнение LightGBM:      {'Успех' if lgbm_vote else 'Провал'} (Вероятность успеха: {lgbm_proba:.1%})")
        print("-" * 60)
        print(f"ИТОГОВАЯ СРЕДНЯЯ ВЕРОЯТНОСТЬ УСПЕХА: {avg_proba:.2%}")

        if successful_votes == 3:
            print("\nРЕКОМЕНДАЦИЯ: 🔥 ВХОДИТЬ (Все модели согласны, высокая уверенность)")
        elif successful_votes == 2:
            print("\nРЕКОМЕНДАЦИЯ: 🤔 РАССМОТРЕТЬ (Большинство моделей ЗА, средняя уверенность)")
        else:
            print("\nРЕКОМЕНДАЦИЯ: ⛔️ ПРОПУСТИТЬ (Большинство моделей ПРОТИВ, низкая уверенность)")
        print("=" * 60)

