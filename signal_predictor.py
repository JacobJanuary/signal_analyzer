#!/usr/bin/env python3
"""
Signal Predictor - Использование обученных моделей для предсказания новых сигналов
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime
import json
import os
from typing import Dict
from dotenv import load_dotenv
from urllib.parse import quote_plus
import joblib
import warnings

warnings.filterwarnings('ignore')

load_dotenv()


class SignalPredictor:
    """
    Предсказатель успешности сигналов на основе обученных моделей.
    """

    def __init__(self, models_dir: str = 'models'):
        """Инициализация предсказателя."""
        self.engine = self._create_db_connection()
        self.models_dir = models_dir
        self.models = self._load_models()
        self.scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))

        with open(os.path.join(models_dir, 'feature_names.json'), 'r') as f:
            self.feature_names = json.load(f)

        print(f"✅ Загружено {len(self.models)} моделей")

    def _create_db_connection(self):
        """Создает и проверяет подключение к базе данных."""
        # УЛУЧШЕНО: Код идентичен анализатору, в реальном проекте
        # его стоило бы вынести в общий модуль.
        try:
            user = os.getenv('DB_USER')
            password = quote_plus(os.getenv('DB_PASSWORD', ''))
            host = os.getenv('DB_HOST')
            port = os.getenv('DB_PORT', 3306)
            database = os.getenv('DB_NAME')

            if not all([user, password, host, database]):
                raise ValueError("Не все переменные окружения для подключения к БД заданы.")

            connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
            engine = create_engine(connection_string, pool_pre_ping=True)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return engine
        except Exception as e:
            print(f"❌ Ошибка подключения к БД: {e}")
            raise

    def _load_models(self) -> Dict:
        """Загружает обученные модели из указанной директории."""
        models = {}
        model_files = [
            'random_forest_model.pkl', 'xgboost_model.pkl',
            'lightgbm_model.pkl', 'neural_network_model.pkl'
        ]
        for model_file in model_files:
            model_path = os.path.join(self.models_dir, model_file)
            if os.path.exists(model_path):
                model_name = model_file.replace('_model.pkl', '')
                models[model_name] = joblib.load(model_path)
                print(f"  - Загружена модель: {model_name}")
        return models

    def get_recent_signals(self, hours: int = 24) -> pd.DataFrame:
        """Получает последние сигналы, используя безопасный параметризованный запрос."""

        # УЛУЧШЕНО: Использование параметризованного запроса для безопасности
        query = text("""
                     SELECT s.id as signal_id_from_db,
                            s.token_id,
                            s.symbol,
                            s.timestamp,
                            s.OI_contracts_binance_now,
                            s.OI_contracts_binance_prev,
                            s.OI_contracts_binance_change,
                            s.OI_contracts_bybit_now,
                            s.OI_contracts_bybit_prev,
                            s.OI_contracts_bybit_change,
                            s.funding_rate_binance_now,
                            s.funding_rate_binance_prev,
                            s.funding_rate_binance_change,
                            s.funding_rate_bybit_now,
                            s.funding_rate_bybit_prev,
                            s.funding_rate_bybit_change,
                            s.volume_usd,
                            s.price_usd_now,
                            s.price_usd_prev,
                            s.price_usd_change,
                            s.market_cap_usd,
                            e.*
                     FROM signals_10min s
                              LEFT JOIN signals_10min_enriched e ON s.id = e.signal_id
                     WHERE s.timestamp >= DATE_SUB(NOW(), INTERVAL :hours_ago HOUR)
                       AND s.OI_contracts_binance_change > 0
                     ORDER BY s.timestamp DESC
                     """)

        df = pd.read_sql(query, self.engine, params={"hours_ago": hours})
        df = df.loc[:, ~df.columns.duplicated()]

        if df.empty:
            return df

        return self._add_technical_features(df)

    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет технические признаки, необходимые для предсказания."""

        # ИСПРАВЛЕНО: Блок переименования перенесен в начало функции,
        # чтобы столбцы были доступны для последующих вычислений.
        rename_map = {
            'timestamp': 'signal_timestamp',
            'price_usd_now': 'signal_price',
            'price_usd_change': 'price_change_10min',
            'volume_usd': 'signal_volume_usd'
        }
        df = df.rename(columns=rename_map)

        # УЛУЧШЕНО: pd.to_datetime вызывается один раз для эффективности
        dt_series = pd.to_datetime(df['signal_timestamp'])
        df['hour'] = dt_series.dt.hour
        df['day_of_week'] = dt_series.dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        df['oi_binance_bybit_ratio'] = df['OI_contracts_binance_change'] / (df['OI_contracts_bybit_change'] + 0.001)
        df['funding_binance_bybit_ratio'] = df['funding_rate_binance_now'] / (df['funding_rate_bybit_now'] + 1e-5)

        df['price_momentum'] = df['spot_price_usdt_change_1h'] * df['spot_price_usdt_change_24h']

        df['volatility_1h'] = ((df['cmc_price_max_1h'] - df['cmc_price_min_1h']) / df['signal_price'] * 100).fillna(0)
        df['volatility_24h'] = ((df['cmc_price_max_24h'] - df['cmc_price_min_24h']) / df['signal_price'] * 100).fillna(
            0)

        df['signal_strength'] = (
                df['OI_contracts_binance_change'] * 0.4 +
                df['OI_contracts_bybit_change'] * 0.3 +
                df['spot_volume_usdt_change_current_to_average'] * 0.2 +
                df['price_change_10min'] * 0.1
        ).fillna(0)

        return df

    def predict_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Делает предсказания для новых сигналов, используя загруженные модели."""
        if df.empty:
            return df

        print(f"\n🔮 Предсказание для {len(df)} сигналов...")

        # УЛУЧШЕНО: Заполнение пропусков медианой, как в обучающем скрипте
        X = df[self.feature_names].fillna(df[self.feature_names].median())
        X_scaled = self.scaler.transform(X)

        for model_name, model in self.models.items():
            input_data = X_scaled if model_name == 'neural_network' else X
            df[f'pred_{model_name}'] = model.predict(input_data)
            df[f'proba_{model_name}'] = model.predict_proba(input_data)[:, 1]

        proba_cols = [f'proba_{name}' for name in self.models.keys()]
        # УЛУЧШЕНО: Исключаем слабую нейросеть из ансамбля для повышения качества
        if 'proba_neural_network' in proba_cols:
            proba_cols.remove('proba_neural_network')

        df['proba_ensemble'] = df[proba_cols].mean(axis=1)
        df['pred_ensemble'] = (df['proba_ensemble'] > 0.5).astype(int)

        pred_cols = [f'pred_{name}' for name in self.models.keys()]
        if 'pred_neural_network' in pred_cols:
            pred_cols.remove('pred_neural_network')

        df['models_agree_count'] = df[pred_cols].sum(axis=1)
        df['all_models_agree'] = (df['models_agree_count'] == len(pred_cols)).astype(int)

        return df

    def generate_trading_signals(self, df: pd.DataFrame, min_probability: float = 0.6,
                                 min_models_agree: int = 2) -> pd.DataFrame:
        """Генерирует и форматирует торговые сигналы на основе предсказаний."""
        if df.empty:
            return df

        strong_signals = df[
            (df['proba_ensemble'] >= min_probability) &
            (df['models_agree_count'] >= min_models_agree)
            ].copy()

        strong_signals = strong_signals.sort_values('proba_ensemble', ascending=False)

        strong_signals['recommendation'] = strong_signals['proba_ensemble'].apply(
            lambda x: 'STRONG BUY' if x >= 0.8 else 'BUY' if x >= 0.7 else 'CONSIDER'
        )

        strong_signals['suggested_stop_loss'] = strong_signals['signal_price'] * 0.97  # -3%
        strong_signals['suggested_take_profit'] = strong_signals['signal_price'] * 1.05  # +5%

        return strong_signals

    def display_signals(self, signals: pd.DataFrame, top_n: int = 10):
        """Отображает топ N сигналов в удобном для чтения формате."""
        print(f"\n📊 ТОП-{min(top_n, len(signals))} СИЛЬНЫХ СИГНАЛОВ:")
        print("=" * 120)

        if signals.empty:
            print("Нет сильных сигналов для отображения.")
            return

        for _, signal in signals.head(top_n).iterrows():
            print(f"\n{'=' * 60}")
            print(f"🪙  {signal['symbol']} | {signal['recommendation']}")
            print(f"📅  {signal['signal_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"💰  Цена в момент сигнала: ${signal['signal_price']:.4f}")
            print(
                f"📈  OI Change: Binance {signal.get('OI_contracts_binance_change', 'N/A'):.1f} | Bybit {signal.get('OI_contracts_bybit_change', 'N/A'):.1f}")
            print(f"🎯  Вероятность успеха: {signal['proba_ensemble']:.1%}")
            print(f"🤝  Согласие моделей: {int(signal['models_agree_count'])}/{len(self.models) - 1}")
            print("-" * 30)
            print(f"🛡️  Stop Loss (рекоменд.): ${signal['suggested_stop_loss']:.4f} (-3%)")
            print(f"🏆  Take Profit (рекоменд.): ${signal['suggested_take_profit']:.4f} (+5%)")

            print("\n   Детализация по моделям:")
            for model_name in self.models.keys():
                if model_name != 'neural_network':
                    print(f"   - {model_name:<15}: {signal[f'proba_{model_name}']:.1%}")

    def save_predictions(self, df: pd.DataFrame, strong_signals: pd.DataFrame):
        """Сохраняет все предсказания и отфильтрованные сильные сигналы."""
        output_dir = 'predictions'
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_filename = os.path.join(output_dir, f'predictions_all_{timestamp}.csv')
        strong_filename = os.path.join(output_dir, f'predictions_strong_{timestamp}.csv')

        df.to_csv(all_filename, index=False)
        print(f"\n💾 Все предсказания сохранены: {all_filename}")

        strong_signals.to_csv(strong_filename, index=False)
        print(f"💾 Сильные сигналы сохранены: {strong_filename}")


def main():
    """Основная функция для запуска предсказателя из командной строки."""
    import argparse

    parser = argparse.ArgumentParser(description='Предсказание успешности криптовалютных сигналов')
    parser.add_argument('--hours', type=int, default=24, help='Количество часов для анализа (по умолчанию 24)')
    parser.add_argument('--min-probability', type=float, default=0.6,
                        help='Минимальная вероятность для сигнала (по умолчанию 0.6)')
    parser.add_argument('--min-models', type=int, default=2,
                        help='Минимальное количество согласных моделей (по умолчанию 2)')
    parser.add_argument('--top', type=int, default=10, help='Количество топ сигналов для отображения (по умолчанию 10)')
    parser.add_argument('--save', action='store_true', help='Сохранить предсказания в файл')

    args = parser.parse_args()

    try:
        predictor = SignalPredictor()

        print(f"\n📡 Получение сигналов за последние {args.hours} часов...")
        recent_signals = predictor.get_recent_signals(hours=args.hours)

        if recent_signals.empty:
            print("❌ Не найдено новых сигналов для анализа.")
            return

        print(f"✅ Найдено {len(recent_signals)} новых сигналов")

        predictions = predictor.predict_signals(recent_signals)

        trading_signals = predictor.generate_trading_signals(
            predictions,
            min_probability=args.min_probability,
            min_models_agree=args.min_models
        )

        if not trading_signals.empty:
            predictor.display_signals(trading_signals, top_n=args.top)
        else:
            print("\n❌ Не найдено сигналов, удовлетворяющих критериям (p > "
                  f"{args.min_probability}, согласие >= {args.min_models}).")

        if args.save:
            predictor.save_predictions(predictions, trading_signals)

        print(f"\n📊 ИТОГОВАЯ СТАТИСТИКА:")
        print(f"Всего проанализировано: {len(predictions)}")
        print(f"Найдено сильных сигналов: {len(trading_signals)}")
        if not trading_signals.empty:
            print("Распределение сильных сигналов по токенам:")
            print(trading_signals['symbol'].value_counts().head(5))

    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()