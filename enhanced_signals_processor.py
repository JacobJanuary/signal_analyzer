#!/usr/bin/env python3
"""
Enhanced Signals Processor - Интеграция рыночного контекста в основной анализ
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dotenv import load_dotenv
from urllib.parse import quote_plus
import warnings

# ML библиотеки
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
import joblib
from tqdm import tqdm

# Импортируем анализатор рыночного контекста
from market_context_analyzer import MarketContextAnalyzer, integrate_market_context

load_dotenv()

AVOID_ZERO_DIV_SMALL = 1e-5
AVOID_ZERO_DIV_TINY = 1e-9


class EnhancedSignalsProcessor:
    """
    Расширенный процессор сигналов с интеграцией рыночного контекста BTC
    """

    def __init__(self):
        """Инициализация процессора"""
        self.engine = self._create_db_connection()
        self.market_analyzer = MarketContextAnalyzer()

        # Расширенные параметры с учетом рыночного контекста
        self.profit_targets = [3, 5, 10]
        self.stop_losses = [2, 3, 5]
        self.time_windows_hours = [6, 12, 24, 48]

        self.success_criteria = {
            'profit_targets': self.profit_targets,
            'stop_losses': self.stop_losses,
            'time_windows': self.time_windows_hours,
        }

        # ML модели
        self.models: Dict[str, Any] = {}
        self.scaler = StandardScaler()

        # Список базовых признаков
        self.base_features = [
            # Основные OI метрики
            'OI_contracts_binance_change', 'OI_contracts_bybit_change',
            'OI_contracts_binance_now', 'OI_contracts_bybit_now',

            # Funding rates
            'funding_rate_binance_now', 'funding_rate_bybit_now',
            'funding_rate_binance_change', 'funding_rate_bybit_change',

            # Изменения цены
            'price_change_10min', 'spot_price_usdt_change_1h',
            'spot_price_usdt_change_24h', 'spot_price_usdt_change_7d',

            # Объемы
            'signal_volume_usd', 'spot_volume_usdt_current',
            'spot_volume_usdt_change_current_to_average',
            'spot_volume_usdt_change_current_to_yesterday',

            # OI enriched
            'oi_usdt_current', 'oi_usdt_change_current_to_average',
            'oi_usdt_change_current_to_yesterday',

            # Технические признаки
            'hour', 'day_of_week', 'is_weekend',
            'oi_binance_bybit_ratio', 'funding_binance_bybit_ratio',
            'price_momentum', 'volatility_1h', 'volatility_24h',
            'signal_strength',

            # CMC метрики
            'cmc_percent_change_1d', 'cmc_percent_change_7d',
            'cmc_percent_change_30d'
        ]

        # Дополнительные признаки рыночного контекста
        self.market_context_features = [
            'btc_price_at_signal', 'btc_volume_at_signal', 'btc_volatility_at_signal',
            'btc_daily_change_at_signal', 'btc_funding_at_signal', 'btc_oi_at_signal',
            'btc_trend_7d', 'btc_trend_30d', 'price_change_vs_btc',
            'market_regime_encoded', 'market_trend_encoded', 'market_regime_strength'
        ]

    def _create_db_connection(self):
        """Создает и проверяет подключение к базе данных"""
        try:
            user = os.getenv('DB_USER')
            password = quote_plus(os.getenv('DB_PASSWORD', ''))
            host = os.getenv('DB_HOST')
            port = os.getenv('DB_PORT', 3306)
            database = os.getenv('DB_NAME')

            if not all([user, password, host, database]):
                raise ValueError("Не все переменные окружения для подключения к БД заданы")

            connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
            engine = create_engine(connection_string, pool_pre_ping=True)

            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("✅ Подключение к БД установлено")
            return engine
        except Exception as e:
            print(f"❌ Ошибка подключения к БД: {e}")
            raise

    def get_all_signals(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Получает все сигналы с параметрами из базы данных"""
        print(f"📊 Загрузка сигналов из базы данных... (Лимит: {limit or 'Нет'})")

        limit_clause = "LIMIT :limit_val" if limit else ""

        query = text(f"""
        SELECT 
            s.id as signal_id, s.token_id, s.symbol, s.timestamp as signal_timestamp,
            s.OI_contracts_binance_now, s.OI_contracts_binance_prev, s.OI_contracts_binance_change,
            s.OI_contracts_bybit_now, s.OI_contracts_bybit_prev, s.OI_contracts_bybit_change,
            s.funding_rate_binance_now, s.funding_rate_binance_prev, s.funding_rate_binance_change,
            s.funding_rate_bybit_now, s.funding_rate_bybit_prev, s.funding_rate_bybit_change,
            s.volume_usd as signal_volume_usd, s.price_usd_now as signal_price,
            s.price_usd_prev as signal_price_prev, s.price_usd_change as price_change_10min,
            s.market_cap_usd,
            e.oi_usdt_average, e.oi_usdt_current, e.oi_usdt_yesterday,
            e.oi_usdt_change_current_to_yesterday, e.oi_usdt_change_current_to_average,
            e.spot_volume_usdt_average, e.spot_volume_usdt_current, e.spot_volume_usdt_yesterday,
            e.spot_volume_usdt_change_current_to_yesterday, e.spot_volume_usdt_change_current_to_average,
            e.spot_price_usdt_average, e.spot_price_usdt_current, e.spot_price_usdt_yesterday,
            e.spot_price_usdt_change_1h, e.spot_price_usdt_change_24h, e.spot_price_usdt_change_7d, e.spot_price_usdt_change_30d,
            e.cmc_price_min_1h, e.cmc_price_max_1h, e.cmc_price_min_24h, e.cmc_price_max_24h,
            e.cmc_price_min_7d, e.cmc_price_max_7d, e.cmc_price_min_30d, e.cmc_price_max_30d,
            e.cmc_percent_change_1d, e.cmc_percent_change_7d, e.cmc_percent_change_30d
        FROM signals_10min s
        LEFT JOIN signals_10min_enriched e ON s.id = e.signal_id
        WHERE s.OI_contracts_binance_change > 0
        ORDER BY s.timestamp DESC
        {limit_clause}
        """)

        params = {"limit_val": limit} if limit else {}
        df = pd.read_sql(query, self.engine, params=params)
        print(f"✅ Загружено {len(df)} сигналов")

        if not df.empty:
            df = self._add_technical_features(df)

        return df

    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет технические признаки"""
        print("🔧 Добавление технических признаков...")

        df['signal_timestamp'] = pd.to_datetime(df['signal_timestamp'])
        df['hour'] = df['signal_timestamp'].dt.hour
        df['day_of_week'] = df['signal_timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        df['oi_binance_bybit_ratio'] = df['OI_contracts_binance_change'] / (
                df['OI_contracts_bybit_change'] + AVOID_ZERO_DIV_SMALL)
        df['funding_binance_bybit_ratio'] = df['funding_rate_binance_now'] / (
                df['funding_rate_bybit_now'] + AVOID_ZERO_DIV_TINY)

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

    def get_futures_data_for_signal(self, token_id: int, signal_time: datetime,
                                    hours_after: int = 48) -> pd.DataFrame:
        """Получает исторические данные фьючерсов для одного сигнала"""
        query = text("""
                     SELECT fd.timestamp,
                            fd.price_usd,
                            fd.volume_usd,
                            fd.open_interest_usd
                     FROM futures_data fd
                              JOIN futures_pairs fp ON fd.pair_id = fp.id
                     WHERE fp.token_id = :token_id
                       AND fd.timestamp >= :start_time
                       AND fd.timestamp <= :end_time
                     ORDER BY fd.timestamp
                     """)

        params = {
            "token_id": token_id,
            "start_time": signal_time,
            "end_time": signal_time + timedelta(hours=hours_after)
        }

        return pd.read_sql(query, self.engine, params=params)

    def calculate_signal_outcomes_batch(self, signals_df: pd.DataFrame,
                                        batch_size: int = 100) -> pd.DataFrame:
        """Рассчитывает результаты для всех сигналов батчами"""
        print(f"\n🔄 Расчет результатов для {len(signals_df)} сигналов...")
        all_results = []

        max_hours = max(self.time_windows_hours)

        for _, signal in tqdm(signals_df.iterrows(), total=len(signals_df)):
            try:
                futures_data = self.get_futures_data_for_signal(
                    signal['token_id'],
                    signal['signal_timestamp'],
                    hours_after=max_hours
                )

                if not futures_data.empty:
                    outcomes = self._calculate_multi_criteria_outcomes(signal, futures_data)
                    signal_dict = signal.to_dict()
                    signal_dict.update(outcomes)
                    all_results.append(signal_dict)

            except Exception as e:
                print(f"\n⚠️ Ошибка при обработке сигнала {signal['signal_id']}: {e}")
                continue

        print(f"\n✅ Обработано {len(all_results)} сигналов с результатами")
        return pd.DataFrame(all_results)

    def _calculate_multi_criteria_outcomes(self, signal: pd.Series,
                                           futures_data: pd.DataFrame) -> Dict[str, Any]:
        """Рассчитывает метрики и результаты для одного сигнала"""
        entry_price = signal['signal_price']
        outcomes: Dict[str, Any] = {}

        futures_data['timestamp'] = pd.to_datetime(futures_data['timestamp'])
        signal_time = pd.to_datetime(signal['signal_timestamp'])

        futures_data['time_diff_min'] = (futures_data['timestamp'] - signal_time).dt.total_seconds() / 60
        futures_data['pnl_pct'] = ((futures_data['price_usd'] - entry_price) / entry_price) * 100

        outcomes['max_profit_pct'] = futures_data['pnl_pct'].max()
        outcomes['max_drawdown_pct'] = futures_data['pnl_pct'].min()
        outcomes['final_pnl_pct'] = futures_data['pnl_pct'].iloc[-1] if not futures_data.empty else 0

        if pd.notna(outcomes['max_profit_pct']) and outcomes['max_profit_pct'] > 0:
            outcomes['time_to_max_profit_min'] = futures_data.loc[futures_data['pnl_pct'].idxmax(), 'time_diff_min']
        else:
            outcomes['time_to_max_profit_min'] = None

        if pd.notna(outcomes['max_drawdown_pct']) and outcomes['max_drawdown_pct'] < 0:
            outcomes['time_to_max_drawdown_min'] = futures_data.loc[futures_data['pnl_pct'].idxmin(), 'time_diff_min']
        else:
            outcomes['time_to_max_drawdown_min'] = None

        for profit_target in self.profit_targets:
            for stop_loss in self.stop_losses:
                for time_window in self.time_windows_hours:
                    success_key = f'success_{profit_target}p_{stop_loss}sl_{time_window}h'
                    window_data = futures_data[futures_data['time_diff_min'] <= time_window * 60]

                    if window_data.empty:
                        outcomes[success_key] = 0
                        continue

                    profit_times = window_data.index[window_data['pnl_pct'] >= profit_target]
                    stop_times = window_data.index[window_data['pnl_pct'] <= -stop_loss]

                    first_profit_time = profit_times.min() if not profit_times.empty else np.inf
                    first_stop_time = stop_times.min() if not stop_times.empty else np.inf

                    is_successful = first_profit_time < first_stop_time
                    outcomes[success_key] = int(is_successful)

                    if profit_target == 5 and stop_loss == 3 and time_window == 24:
                        outcomes['is_successful_main'] = int(is_successful)
                        outcomes['hit_profit_target'] = int(not profit_times.empty)
                        outcomes['hit_stop_loss'] = int(not stop_times.empty)

        if len(futures_data) > 1:
            outcomes['volatility_after'] = futures_data['pnl_pct'].std()
            outcomes['volume_trend'] = 1 if futures_data['volume_usd'].iloc[-1] > futures_data['volume_usd'].iloc[
                0] else -1

            oi_initial = futures_data['open_interest_usd'].iloc[0]
            if oi_initial > 0:
                outcomes['oi_change_after'] = ((futures_data['open_interest_usd'].iloc[
                                                    -1] - oi_initial) / oi_initial) * 100
            else:
                outcomes['oi_change_after'] = 0
        else:
            outcomes['volatility_after'] = 0
            outcomes['volume_trend'] = 0
            outcomes['oi_change_after'] = 0

        return outcomes

    def process_with_market_context(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """Обрабатывает сигналы с добавлением рыночного контекста"""
        print("\n🌍 Интеграция рыночного контекста...")

        # Обогащаем сигналы данными BTC
        enriched_signals = self.market_analyzer.enrich_signals_with_market_context(signals_df)

        # Анализируем производительность по режимам
        if 'is_successful_main' in enriched_signals.columns:
            regime_performance = self.market_analyzer.analyze_performance_by_regime(enriched_signals)

            # Сохраняем анализ
            os.makedirs('reports', exist_ok=True)
            with open('reports/market_regime_analysis.json', 'w') as f:
                json.dump(regime_performance, f, indent=2, default=str)

        return enriched_signals

    def prepare_ml_features_enhanced(self, df: pd.DataFrame,
                                     target_column: str = 'is_successful_main') -> Tuple[
        pd.DataFrame, pd.Series, List[str]]:
        """Подготавливает расширенный набор признаков включая рыночный контекст"""
        print("\n🤖 Подготовка расширенных признаков для ML...")

        # Объединяем базовые признаки и признаки рыночного контекста
        all_features = self.base_features + self.market_context_features

        # Фильтруем существующие колонки
        available_features = [col for col in all_features if col in df.columns]

        df_clean = df.dropna(subset=[target_column])

        X = df_clean[available_features].fillna(df_clean[available_features].median())
        y = df_clean[target_column]

        print(f"✅ Подготовлено {len(X)} образцов с {len(available_features)} признаками")
        print(
            f"   Включено {len([f for f in self.market_context_features if f in available_features])} признаков рыночного контекста")

        if not y.empty:
            class_distribution = y.value_counts(normalize=True)
            print(
                f"   Распределение классов: 0: {class_distribution.get(0, 0):.2%}, 1: {class_distribution.get(1, 0):.2%}")

        return X, y, available_features

    def generate_enhanced_report(self, df: pd.DataFrame, results: Dict) -> str:
        """Генерирует расширенный отчет с анализом рыночного контекста"""
        print("\n📄 Генерация расширенного отчета...")

        report_lines = [
            "=" * 80,
            "РАСШИРЕННЫЙ ОТЧЕТ С АНАЛИЗОМ РЫНОЧНОГО КОНТЕКСТА",
            f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80, "",
            "ОБРАБОТАННЫЕ ДАННЫЕ:",
            f"- Всего сигналов: {len(df)}",
            f"- Период: {df['signal_timestamp'].min()} - {df['signal_timestamp'].max()}",
            f"- Уникальных токенов: {df['symbol'].nunique()}", ""
        ]

        # Анализ по рыночным режимам
        if 'market_regime' in df.columns:
            report_lines.extend([
                "АНАЛИЗ ПО РЫНОЧНЫМ РЕЖИМАМ:",
            ])

            for regime in df['market_regime'].unique():
                regime_df = df[df['market_regime'] == regime]
                if 'is_successful_main' in regime_df.columns:
                    success_rate = regime_df['is_successful_main'].mean() * 100
                    report_lines.append(
                        f"- {regime}: {len(regime_df)} сигналов ({success_rate:.1f}% успеха)"
                    )

        # Корреляция с BTC
        if 'price_change_vs_btc' in df.columns and 'is_successful_main' in df.columns:
            successful = df[df['is_successful_main'] == 1]
            unsuccessful = df[df['is_successful_main'] == 0]

            report_lines.extend([
                "",
                "КОРРЕЛЯЦИЯ С BTC:",
                f"- Средн. изменение vs BTC (успешные): {successful['price_change_vs_btc'].mean():.2f}%",
                f"- Средн. изменение vs BTC (неуспешные): {unsuccessful['price_change_vs_btc'].mean():.2f}%"
            ])

        # ML результаты
        report_lines.extend([
            "",
            "РЕЗУЛЬТАТЫ ML МОДЕЛЕЙ С РЫНОЧНЫМ КОНТЕКСТОМ:",
        ])

        for model_name, metrics in results.items():
            report_lines.extend([
                f"\n{model_name.upper()}:",
                f"  - Accuracy: {metrics['accuracy']:.4f}",
                f"  - AUC: {metrics['auc']:.4f}",
                f"  - Precision: {metrics['report']['1']['precision']:.3f}",
                f"  - Recall: {metrics['report']['1']['recall']:.3f}"
            ])

        # Важность признаков рыночного контекста
        if 'random_forest' in results and 'feature_importance' in results['random_forest']:
            importance_df = results['random_forest']['feature_importance']
            market_features = [f for f in self.market_context_features if f in importance_df['feature'].values]

            if market_features:
                report_lines.extend([
                    "",
                    "ВАЖНОСТЬ ПРИЗНАКОВ РЫНОЧНОГО КОНТЕКСТА:",
                ])

                for feature in market_features[:5]:
                    importance = importance_df[importance_df['feature'] == feature]['importance'].values[0]
                    report_lines.append(f"  - {feature}: {importance:.4f}")

        os.makedirs('reports', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f'reports/enhanced_analysis_report_{timestamp}.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"\n📄 Расширенный отчет сохранен: {report_path}")
        return report_path


def main():
    """Основная функция для запуска расширенного анализа"""
    print("🚀 Запуск расширенного процессора сигналов с рыночным контекстом...")
    print("=" * 80)

    try:
        # Инициализация
        processor = EnhancedSignalsProcessor()

        # Параметры
        limit_signals = None  # None для всех сигналов

        # 1. Загружаем сигналы
        signals = processor.get_all_signals(limit=limit_signals)

        if signals.empty:
            print("❌ Не найдено сигналов для обработки")
            return

        # 2. Рассчитываем результаты
        signals_with_outcomes = processor.calculate_signal_outcomes_batch(signals, batch_size=100)

        # 3. Добавляем рыночный контекст
        enriched_signals = processor.process_with_market_context(signals_with_outcomes)

        # 4. Сохраняем обогащенные данные
        print("\n💾 Сохранение данных с рыночным контекстом...")
        enriched_signals.to_pickle('signals_with_market_context.pkl')
        enriched_signals.to_csv('signals_with_market_context.csv', index=False)

        # 5. Подготавливаем данные для ML
        X, y, feature_names = processor.prepare_ml_features_enhanced(enriched_signals)

        if X.empty or y.empty:
            print("❌ Недостаточно данных для обучения моделей")
            return

        # 6. Обучаем модели
        from analize_signals import FullSignalsProcessor
        base_processor = FullSignalsProcessor()
        ml_results = base_processor.train_ml_models(X, y)

        # 7. Сохраняем модели
        base_processor.save_models_and_data(feature_names, ml_results, output_dir='models_enhanced')

        # 8. Генерируем отчет
        report_path = processor.generate_enhanced_report(enriched_signals, ml_results)

        # 9. Итоговая статистика
        print("\n" + "=" * 60)
        print("📊 ИТОГОВАЯ СТАТИСТИКА:")
        print("=" * 60)
        print(f"✅ Обработка завершена успешно!")
        print(f"📁 Результаты сохранены:")
        print(f"   - Данные: signals_with_market_context.pkl/csv")
        print(f"   - Модели: models_enhanced/")
        print(f"   - Отчет: {report_path}")

        # Лучшая модель
        best_model_name, best_model_metrics = max(ml_results.items(), key=lambda item: item[1]['auc'])
        print(f"🏆 Лучшая ML модель: {best_model_name.upper()} (AUC: {best_model_metrics['auc']:.4f})")

        # Статистика по режимам
        if 'market_regime' in enriched_signals.columns:
            print("\n📈 Распределение сигналов по рыночным режимам:")
            regime_counts = enriched_signals['market_regime'].value_counts()
            for regime, count in regime_counts.items():
                print(f"   - {regime}: {count} ({count / len(enriched_signals) * 100:.1f}%)")

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()