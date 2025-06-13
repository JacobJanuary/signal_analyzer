#!/usr/bin/env python3
"""
Full Signals Processor - Обработка ВСЕХ сигналов с полным набором параметров и ML моделями
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

# УЛУЧШЕНО: Не рекомендуется глобально отключать все предупреждения.
# Если есть конкретные известные предупреждения, их можно игнорировать точечно.
# warnings.filterwarnings('ignore')

# Загружаем переменные окружения
load_dotenv()

# УЛУЧШЕНО: Определяем константы для "магических чисел"
AVOID_ZERO_DIV_SMALL = 1e-5
AVOID_ZERO_DIV_TINY = 1e-9


class FullSignalsProcessor:
    """
    Полный процессор сигналов с ML моделями.

    Анализирует исторические данные сигналов, рассчитывает их успешность
    по различным критериям и обучает ансамбль ML-моделей для прогнозирования
    успешности будущих сигналов.
    """

    def __init__(self):
        """Инициализация процессора."""
        self.engine = self._create_db_connection()

        # УЛУЧШЕНО: Параметры вынесены в атрибуты для легкого доступа и конфигурации
        self.profit_targets = [3, 5, 10]  # %
        self.stop_losses = [2, 3, 5]  # %
        self.time_windows_hours = [6, 12, 24, 48]  # Часов

        self.success_criteria = {
            'profit_targets': self.profit_targets,
            'stop_losses': self.stop_losses,
            'time_windows': self.time_windows_hours,
        }

        # ML модели
        self.models: Dict[str, Any] = {}
        self.scaler = StandardScaler()

    def _create_db_connection(self):
        """Создает и проверяет подключение к базе данных."""
        try:
            # ИСПРАВЛЕНО: убран дублирующийся импорт
            user = os.getenv('DB_USER')
            password = quote_plus(os.getenv('DB_PASSWORD', ''))
            host = os.getenv('DB_HOST')
            port = os.getenv('DB_PORT', 3306)
            database = os.getenv('DB_NAME')

            if not all([user, password, host, database]):
                raise ValueError(
                    "Не все переменные окружения для подключения к БД заданы (DB_USER, DB_PASSWORD, DB_HOST, DB_NAME).")

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
        """
        Получает все сигналы с параметрами из базы данных, используя безопасные запросы.
        """
        print(f"📊 Загрузка сигналов из базы данных... (Лимит: {limit or 'Нет'})")

        # ИСПРАВЛЕНО: Использование параметризованного запроса для безопасности
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
        """Добавляет в датафрейм производные технические признаки."""
        print("🔧 Добавление технических признаков...")

        # УЛУЧШЕНО: pd.to_datetime вызывается один раз
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

        # УЛУЧШЕНО: веса для силы сигнала можно вынести в константы/конфигурацию
        df['signal_strength'] = (
                df['OI_contracts_binance_change'] * 0.4 +
                df['OI_contracts_bybit_change'] * 0.3 +
                df['spot_volume_usdt_change_current_to_average'] * 0.2 +
                df['price_change_10min'] * 0.1
        ).fillna(0)

        return df

    def get_futures_data_for_signal(self, token_id: int, signal_time: datetime, hours_after: int = 48) -> pd.DataFrame:
        """Получает исторические данные фьючерсов для одного сигнала, используя безопасный запрос."""

        # ИСПРАВЛЕНО: Использование параметризованного запроса для полной безопасности
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

    def calculate_signal_outcomes_batch(self, signals_df: pd.DataFrame, batch_size: int = 100) -> pd.DataFrame:
        """Рассчитывает результаты для всех сигналов батчами."""
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
                # УЛУЧШЕНО: Логирование каждой ошибки для лучшей диагностики
                print(f"\n⚠️ Ошибка при обработке сигнала {signal['signal_id']} ({signal['symbol']}): {e}")
                continue

        print(f"\n✅ Обработано {len(all_results)} сигналов с результатами")
        return pd.DataFrame(all_results)

    def _calculate_multi_criteria_outcomes(self, signal: pd.Series, futures_data: pd.DataFrame) -> Dict[str, Any]:
        """Рассчитывает метрики и результаты для одного сигнала по разным критериям."""
        entry_price = signal['signal_price']
        outcomes: Dict[str, Any] = {}

        # ИСПРАВЛЕНО: Убедимся, что оба столбца имеют тип datetime
        futures_data['timestamp'] = pd.to_datetime(futures_data['timestamp'])
        signal_time = pd.to_datetime(signal['signal_timestamp'])

        # ИСПРАВЛЕНО: Корректный расчет разницы во времени
        futures_data['time_diff_min'] = (futures_data['timestamp'] - signal_time).dt.total_seconds() / 60
        futures_data['pnl_pct'] = ((futures_data['price_usd'] - entry_price) / entry_price) * 100

        outcomes['max_profit_pct'] = futures_data['pnl_pct'].max()
        outcomes['max_drawdown_pct'] = futures_data['pnl_pct'].min()
        outcomes['final_pnl_pct'] = futures_data['pnl_pct'].iloc[-1] if not futures_data.empty else 0

        # ИСПРАВЛЕНО: Добавлена проверка на наличие положительной прибыли/убытка перед поиском индекса
        if outcomes['max_profit_pct'] > 0:
            outcomes['time_to_max_profit_min'] = futures_data.loc[futures_data['pnl_pct'].idxmax(), 'time_diff_min']
        else:
            outcomes['time_to_max_profit_min'] = None

        if outcomes['max_drawdown_pct'] < 0:
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

                    # УЛУЧШЕНО: Задаем основную целевую переменную (можно сделать настраиваемой)
                    if profit_target == 5 and stop_loss == 3 and time_window == 24:
                        outcomes['is_successful_main'] = int(is_successful)
                        outcomes['hit_profit_target'] = int(not profit_times.empty)
                        outcomes['hit_stop_loss'] = int(not stop_times.empty)

        # ИСПРАВЛЕНО: Добавлены проверки на количество строк для надежности
        if len(futures_data) > 1:
            outcomes['volatility_after'] = futures_data['pnl_pct'].std()
            outcomes['volume_trend'] = 1 if futures_data['volume_usd'].iloc[-1] > futures_data['volume_usd'].iloc[
                0] else -1  # 1: increasing, -1: decreasing

            oi_initial = futures_data['open_interest_usd'].iloc[0]
            if oi_initial > 0:
                outcomes['oi_change_after'] = ((futures_data['open_interest_usd'].iloc[
                                                    -1] - oi_initial) / oi_initial) * 100
            else:
                outcomes['oi_change_after'] = 0
        else:
            outcomes['volatility_after'] = 0
            outcomes['volume_trend'] = 0  # neutral
            outcomes['oi_change_after'] = 0

        return outcomes

    def prepare_ml_features(self, df: pd.DataFrame, target_column: str = 'is_successful_main') -> Tuple[
        pd.DataFrame, pd.Series, List[str]]:
        """
        Подготавливает признаки (X) и целевую переменную (y) для ML моделей.
        """
        print("\n🤖 Подготовка признаков для ML...")

        feature_columns = [
            'OI_contracts_binance_change', 'OI_contracts_bybit_change', 'OI_contracts_binance_now',
            'OI_contracts_bybit_now',
            'funding_rate_binance_now', 'funding_rate_bybit_now', 'funding_rate_binance_change',
            'funding_rate_bybit_change',
            'price_change_10min', 'spot_price_usdt_change_1h', 'spot_price_usdt_change_24h',
            'spot_price_usdt_change_7d',
            'signal_volume_usd', 'spot_volume_usdt_current', 'spot_volume_usdt_change_current_to_average',
            'spot_volume_usdt_change_current_to_yesterday',
            'oi_usdt_current', 'oi_usdt_change_current_to_average', 'oi_usdt_change_current_to_yesterday',
            'hour', 'day_of_week', 'is_weekend', 'oi_binance_bybit_ratio', 'funding_binance_bybit_ratio',
            'price_momentum', 'volatility_1h', 'volatility_24h', 'signal_strength',
            'cmc_percent_change_1d', 'cmc_percent_change_7d', 'cmc_percent_change_30d'
        ]

        available_features = [col for col in feature_columns if col in df.columns]

        df_clean = df.dropna(subset=[target_column])

        # УЛУЧШЕНО: Заполнение NaN медианой может быть более робастным, чем нулем
        X = df_clean[available_features].fillna(df_clean[available_features].median())
        y = df_clean[target_column]

        print(f"✅ Подготовлено {len(X)} образцов с {len(available_features)} признаками")
        if not y.empty:
            class_distribution = y.value_counts(normalize=True)
            print(
                f"   Распределение классов: 0: {class_distribution.get(0, 0):.2%}, 1: {class_distribution.get(1, 0):.2%}")

        return X, y, available_features

    def train_ml_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Обучает, оценивает и возвращает 4 ML модели."""
        print("\n🎯 Обучение ML моделей...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        results = {}

        # --- 1. Random Forest ---
        print("\n1️⃣  Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=20, class_weight='balanced',
                                          random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        results['random_forest'] = self._evaluate_model(rf_model, X_test, y_test, X.columns)

        # --- 2. XGBoost ---
        print("2️⃣  XGBoost...")
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1

        # ИСПРАВЛЕНО: 'eval_metric' указывается в конструкторе, а не в .fit()
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'  # <-- Параметр теперь здесь
        )

        # УЛУЧШЕНО: Добавлен eval_set для мониторинга и возможного early stopping
        eval_set = [(X_test, y_test)]
        xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        results['xgboost'] = self._evaluate_model(xgb_model, X_test, y_test)

        # --- 3. LightGBM ---
        print("3️⃣  LightGBM...")
        lgb_model = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, class_weight='balanced',
                                       random_state=42, verbosity=-1)
        lgb_model.fit(X_train, y_train)
        results['lightgbm'] = self._evaluate_model(lgb_model, X_test, y_test)

        # --- 4. Neural Network ---
        print("4️⃣  Neural Network...")
        nn_model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', alpha=0.001,
                                 learning_rate='adaptive', max_iter=500, early_stopping=True, random_state=42)
        nn_model.fit(X_train_scaled, y_train)
        results['neural_network'] = self._evaluate_model(nn_model, X_test_scaled, y_test)

        self.models = {name: res['model'] for name, res in results.items()}
        self._print_training_results(results)
        return results

    def analyze_feature_performance(self, df: pd.DataFrame) -> List[str]:
        """
        Анализирует производительность сигналов в разрезе ключевых признаков,
        чтобы выявить наиболее и наименее успешные категории.
        """
        if 'is_successful_main' not in df.columns or df['is_successful_main'].isna().all():
            return []

        report_lines = ["", "=" * 80, "ДЕТАЛЬНЫЙ АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ ПРИЗНАКОВ", "=" * 80]

        # --- Анализ по дням недели ---
        day_map = {0: 'Понедельник', 1: 'Вторник', 2: 'Среда', 3: 'Четверг', 4: 'Пятница', 5: 'Суббота',
                   6: 'Воскресенье'}
        if 'day_of_week' in df.columns:
            report_lines.extend(["", "АНАЛИЗ ПО ДНЯМ НЕДЕЛИ:"])
            daily_stats = df.groupby('day_of_week')['is_successful_main'].agg(['mean', 'count']).rename(index=day_map)
            daily_stats = daily_stats.sort_values(by='mean', ascending=False)
            for day, stats in daily_stats.iterrows():
                report_lines.append(f"  - {day:<12}: {stats['mean']:.1%} успеха ({int(stats['count'])} сигналов)")

        # --- Анализ по волатильности (24ч) ---
        if 'volatility_24h' in df.columns:
            report_lines.extend(["", "АНАЛИЗ ПО 24Ч ВОЛАТИЛЬНОСТИ:"])
            vol_bins = [-np.inf, 3, 6, 10, 15, np.inf]
            vol_labels = ["Низкая (<3%)", "Средняя (3-6%)", "Высокая (6-10%)", "Очень высокая (10-15%)",
                          "Экстремальная (>15%)"]
            df['volatility_bin'] = pd.cut(df['volatility_24h'], bins=vol_bins, labels=vol_labels, right=False)
            vol_stats = df.groupby('volatility_bin')['is_successful_main'].agg(['mean', 'count'])
            vol_stats = vol_stats.sort_values(by='mean', ascending=False)
            for vol_range, stats in vol_stats.iterrows():
                report_lines.append(f"  - {vol_range:<25}: {stats['mean']:.1%} успеха ({int(stats['count'])} сигналов)")

        # --- Анализ по тренду за 7 дней ---
        if 'cmc_percent_change_7d' in df.columns:
            report_lines.extend(["", "АНАЛИЗ ПО 7Д ТРЕНДУ ЦЕНЫ:"])
            trend_bins = [-np.inf, -20, -5, 5, 20, np.inf]
            trend_labels = ["Сильное падение (<-20%)", "Падение (-20%...-5%)", "Боковик (-5%...+5%)",
                            "Рост (+5%...+20%)", "Сильный рост (>+20%)"]
            df['trend_bin_7d'] = pd.cut(df['cmc_percent_change_7d'], bins=trend_bins, labels=trend_labels, right=False)
            trend_stats = df.groupby('trend_bin_7d')['is_successful_main'].agg(['mean', 'count'])
            trend_stats = trend_stats.sort_values(by='mean', ascending=False)
            for trend_range, stats in trend_stats.iterrows():
                report_lines.append(
                    f"  - {trend_range:<25}: {stats['mean']:.1%} успеха ({int(stats['count'])} сигналов)")

        # --- Анализ по реакции цены за 10 минут ---
        if 'price_change_10min' in df.columns:
            report_lines.extend(["", "АНАЛИЗ ПО РЕАКЦИИ ЦЕНЫ (10 МИН):"])
            react_bins = [-np.inf, -1, -0.25, 0.25, 1, np.inf]
            react_labels = ["Падение (<-1%)", "Снижение (-1%...-0.25%)", "На месте (-0.25%...+0.25%)",
                            "Рост (+0.25%...+1%)", "Сильный рост (>+1%)"]
            df['react_bin_10m'] = pd.cut(df['price_change_10min'], bins=react_bins, labels=react_labels, right=False)
            react_stats = df.groupby('react_bin_10m')['is_successful_main'].agg(['mean', 'count'])
            react_stats = react_stats.sort_values(by='mean', ascending=False)
            for react_range, stats in react_stats.iterrows():
                report_lines.append(
                    f"  - {react_range:<28}: {stats['mean']:.1%} успеха ({int(stats['count'])} сигналов)")

        report_lines.append("=" * 80)
        return report_lines

    def _evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                        feature_names: Optional[pd.Index] = None) -> Dict:
        """Вспомогательная функция для оценки модели."""
        pred = model.predict(X_test)
        pred_proba = model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, pred, output_dict=True, zero_division=0)

        metrics = {
            'model': model,
            'accuracy': report['accuracy'],
            'auc': roc_auc_score(y_test, pred_proba),
            'report': report,
        }

        if feature_names is not None and hasattr(model, 'feature_importances_'):
            metrics['feature_importance'] = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

        return metrics

    def _print_training_results(self, results: Dict):
        """Выводит результаты обучения в консоль."""
        print("\n📊 Результаты обучения:")
        print("-" * 60)
        for model_name, metrics in results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  AUC: {metrics['auc']:.4f}")
            print(f"  Precision (class 1): {metrics['report']['1']['precision']:.3f}")
            print(f"  Recall (class 1): {metrics['report']['1']['recall']:.3f}")

    def save_models_and_data(self, feature_names: List[str], results: Dict, output_dir: str = 'models'):
        """Сохраняет обученные модели, скейлер и метаданные."""
        os.makedirs(output_dir, exist_ok=True)

        for model_name, model in self.models.items():
            joblib.dump(model, os.path.join(output_dir, f'{model_name}_model.pkl'))

        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.pkl'))

        with open(os.path.join(output_dir, 'feature_names.json'), 'w') as f:
            json.dump(feature_names, f)

        json_results = {
            model_name: {
                'accuracy': float(m['accuracy']), 'auc': float(m['auc']),
                'precision_1': float(m['report']['1']['precision']), 'recall_1': float(m['report']['1']['recall'])
            } for model_name, m in results.items()
        }
        with open(os.path.join(output_dir, 'training_results.json'), 'w') as f:
            json.dump(json_results, f, indent=2)

        if 'random_forest' in results and 'feature_importance' in results['random_forest']:
            results['random_forest']['feature_importance'].to_csv(
                os.path.join(output_dir, 'feature_importance.csv'), index=False
            )

        print(f"\n💾 Модели и данные сохранены в папке '{output_dir}/'")

    def generate_comprehensive_report(self, df: pd.DataFrame, results: Dict) -> str:
        """Генерирует и сохраняет подробный текстовый отчет."""
        print("\n📄 Генерация отчета...")
        # ИСПРАВЛЕНО: Полностью переписана логика генерации отчета для корректной работы
        report_lines = [
            "=" * 80,
            "ПОЛНЫЙ ОТЧЕТ ПО АНАЛИЗУ КРИПТОВАЛЮТНЫХ СИГНАЛОВ",
            f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80, "",
            "ОБРАБОТАННЫЕ ДАННЫЕ:",
            f"- Всего сигналов обработано: {len(df)}",
            f"- Период: {df['signal_timestamp'].min()} - {df['signal_timestamp'].max()}",
            f"- Уникальных токенов: {df['symbol'].nunique()}", "",
            "РЕЗУЛЬТАТЫ ПО РАЗНЫМ КРИТЕРИЯМ УСПЕХА:",
        ]

        for profit in self.profit_targets:
            for stop in self.stop_losses:
                for window in self.time_windows_hours:
                    col_name = f'success_{profit}p_{stop}sl_{window}h'
                    if col_name in df.columns:
                        success_rate = df[col_name].mean() * 100
                        count = df[col_name].sum()
                        report_lines.append(
                            f"- {profit}% п. / {stop}% у. / {window}ч: {success_rate:.1f}% успеха ({count} из {len(df)})"
                        )

        report_lines.extend(["", "ТОП-10 ТОКЕНОВ ПО КОЛИЧЕСТВУ СИГНАЛОВ:"])
        top_tokens = df['symbol'].value_counts().head(10)
        for i, (token, count) in enumerate(top_tokens.items(), 1):
            token_df = df[df['symbol'] == token]
            if 'is_successful_main' in token_df.columns and not token_df.empty:
                success_rate = token_df['is_successful_main'].mean() * 100
                report_lines.append(f"{i}. {token}: {count} сигналов ({success_rate:.1f}% успеха)")
            else:
                report_lines.append(f"{i}. {token}: {count} сигналов")

        report_lines.extend(["", "РЕЗУЛЬТАТЫ ML МОДЕЛЕЙ (Цель: 5%п/3%у/24ч):"])
        for model_name, metrics in results.items():
            report_lines.extend([
                f"\n{model_name.upper()}:",
                f"  - Accuracy: {metrics['accuracy']:.4f}", f"  - AUC: {metrics['auc']:.4f}",
                f"  - Precision: {metrics['report']['1']['precision']:.3f}",
                f"  - Recall: {metrics['report']['1']['recall']:.3f}"
            ])

        if 'random_forest' in results and 'feature_importance' in results['random_forest']:
            report_lines.extend(["", "ТОП-15 ВАЖНЫХ ПРИЗНАКОВ (Random Forest):"])
            top_features = results['random_forest']['feature_importance'].head(15)
            for idx, row in top_features.iterrows():
                report_lines.append(f"  {idx + 1}. {row['feature']}: {row['importance']:.4f}")

        if 'hour' in df.columns and 'is_successful_main' in df.columns:
            report_lines.extend(["", "ВРЕМЕННЫЕ ПАТТЕРНЫ:"])
            hourly_success = df.groupby('hour')['is_successful_main'].agg(['mean', 'count'])
            best_hours = hourly_success.sort_values('mean', ascending=False).head(5)
            report_lines.append("Лучшие часы для сигналов (UTC):")
            for hour, stats in best_hours.iterrows():
                report_lines.append(
                    f"  - {hour:02d}:00 - {stats['mean'] * 100:.1f}% успеха ({stats['count']} сигналов)")

        feature_analysis_lines = self.analyze_feature_performance(
            df.copy())  # .copy() чтобы избежать SettingWithCopyWarning
        report_lines.extend(feature_analysis_lines)

        os.makedirs('reports', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f'reports/full_analysis_report_{timestamp}.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"\n📄 Отчет сохранен: {report_path}")
        print('\n' + '\n'.join(report_lines[:20]) + '\n...')
        return report_path


def main():
    """Основная функция для запуска полного цикла обработки и анализа сигналов."""
    print("🚀 Запуск полного процессора сигналов...")
    print("=" * 80)

    try:
        processor = FullSignalsProcessor()

        limit_signals = 1000  # None для обработки всех, число для тестирования

        signals = processor.get_all_signals(limit=limit_signals)

        if signals.empty:
            print("❌ Не найдено сигналов для обработки. Завершение работы.")
            return

        signals_with_outcomes = processor.calculate_signal_outcomes_batch(signals, batch_size=100)

        if signals_with_outcomes.empty:
            print("❌ После расчета результатов не осталось данных для анализа. Завершение работы.")
            return

        print("\n💾 Сохранение полных данных...")
        signals_with_outcomes.to_pickle('all_signals_with_outcomes.pkl')
        signals_with_outcomes.to_csv('all_signals_with_outcomes.csv', index=False)

        X, y, feature_names = processor.prepare_ml_features(signals_with_outcomes)

        if X.empty or y.empty:
            print("❌ Недостаточно данных для обучения моделей. Завершение работы.")
            return

        ml_results = processor.train_ml_models(X, y)

        processor.save_models_and_data(feature_names, ml_results)

        report_path = processor.generate_comprehensive_report(signals_with_outcomes, ml_results)

        print("\n" + "=" * 60)
        print("📊 ИТОГОВАЯ СТАТИСТИКА:")
        print("=" * 60)
        best_model_name, best_model_metrics = max(ml_results.items(), key=lambda item: item[1]['auc'])
        print(f"✅ Обработка завершена успешно!")
        print(f"📁 Результаты сохранены:")
        print(f"   - Данные: all_signals_with_outcomes.pkl/csv")
        print(f"   - Модели: models/")
        print(f"   - Отчет: {report_path}")
        print(f"🏆 Лучшая ML модель: {best_model_name.upper()} (AUC: {best_model_metrics['auc']:.4f})")

    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА В ОСНОВНОМ ПРОЦЕССЕ: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()