#!/usr/bin/env python3
"""
Enhanced Full Signals Processor - Использует ВСЕ доступные данные из таблиц
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
from tqdm import tqdm

# ML библиотеки
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from xgboost.callback import EarlyStopping
from sklearn.neural_network import MLPClassifier
import joblib

load_dotenv()

AVOID_ZERO_DIV_SMALL = 1e-5
AVOID_ZERO_DIV_TINY = 1e-9


class EnhancedFullSignalsProcessor:
    """
    Полный процессор сигналов, использующий ВСЕ доступные данные
    """

    def __init__(self):
        self.engine = self._create_db_connection()

        # Параметры анализа
        self.profit_targets = [3, 5, 10]
        self.stop_losses = [2, 3, 5]
        self.time_windows_hours = [6, 12, 24, 48]

        # ML компоненты
        self.models: Dict[str, Any] = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}

        # Расширенный список признаков
        self.define_all_features()

    def _create_db_connection(self):
        """Создает подключение к базе данных"""
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

    def define_all_features(self):
        """Определяет все доступные признаки из БД"""

        # Базовые признаки из signals_10min
        self.base_signal_features = [
            'OI_contracts_binance_now', 'OI_contracts_binance_prev', 'OI_contracts_binance_change',
            'OI_contracts_bybit_now', 'OI_contracts_bybit_prev', 'OI_contracts_bybit_change',
            'funding_rate_binance_now', 'funding_rate_binance_prev', 'funding_rate_binance_change',
            'funding_rate_bybit_now', 'funding_rate_bybit_prev', 'funding_rate_bybit_change',
            'signal_volume_usd', 'signal_price', 'signal_price_prev', 'price_change_10min',
            'market_cap_usd'
        ]

        # Обогащенные признаки из signals_10min_enriched
        self.enriched_features = [
            # OI метрики
            'oi_usdt_average', 'oi_usdt_current', 'oi_usdt_yesterday',
            'oi_usdt_change_current_to_yesterday', 'oi_usdt_change_current_to_average',

            # Спотовые объемы в USDT
            'spot_volume_usdt_average', 'spot_volume_usdt_current', 'spot_volume_usdt_yesterday',
            'spot_volume_usdt_change_current_to_yesterday', 'spot_volume_usdt_change_current_to_average',

            # Спотовые объемы в BTC - НОВОЕ!
            'spot_volume_btc_average', 'spot_volume_btc_current', 'spot_volume_btc_yesterday',
            'spot_volume_btc_change_current_to_yesterday', 'spot_volume_btc_change_current_to_average',

            # Спотовые цены
            'spot_price_usdt_average', 'spot_price_usdt_current', 'spot_price_usdt_yesterday',
            'spot_price_usdt_change_1h', 'spot_price_usdt_change_24h',
            'spot_price_usdt_change_7d', 'spot_price_usdt_change_30d',

            # CMC данные
            'cmc_price_min_1h', 'cmc_price_max_1h', 'cmc_price_min_24h', 'cmc_price_max_24h',
            'cmc_price_min_7d', 'cmc_price_max_7d', 'cmc_price_min_30d', 'cmc_price_max_30d',
            'cmc_percent_change_1d', 'cmc_percent_change_7d', 'cmc_percent_change_30d'
        ]

        # Категориальные признаки источников - НОВОЕ!
        self.source_features = [
            'oi_source_usdt', 'spot_volume_source_usdt',
            'spot_volume_source_btc', 'spot_price_source_usdt'
        ]

        # Технические признаки (вычисляемые)
        self.technical_features = [
            'hour', 'day_of_week', 'is_weekend',
            'oi_binance_bybit_ratio', 'funding_binance_bybit_ratio',
            'price_momentum', 'volatility_1h', 'volatility_24h', 'signal_strength',
            # Новые технические признаки
            'volume_btc_usdt_ratio', 'oi_to_volume_ratio', 'funding_divergence',
            'price_range_1h', 'price_range_24h', 'volume_profile_score',
            'multi_source_confidence', 'exchange_dominance'
        ]

    def get_all_signals_complete(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Получает ВСЕ доступные данные о сигналах
        """
        max_hours_ago = max(self.time_windows_hours)
        print(f"📊 Загрузка ПОЛНЫХ данных сигналов из БД (старше {max_hours_ago} часов)... (Лимит: {limit or 'Нет'})")

        limit_clause = "LIMIT :limit_val" if limit else ""

        # Расширенный запрос со ВСЕМИ полями
        query = text(f"""
        SELECT 
            -- Базовые данные сигнала
            s.id as signal_id, 
            s.token_id, 
            s.symbol, 
            s.timestamp as signal_timestamp,

            -- OI данные по биржам
            s.OI_contracts_binance_now, 
            s.OI_contracts_binance_prev, 
            s.OI_contracts_binance_change,
            s.OI_contracts_bybit_now, 
            s.OI_contracts_bybit_prev, 
            s.OI_contracts_bybit_change,

            -- Funding rates
            s.funding_rate_binance_now, 
            s.funding_rate_binance_prev, 
            s.funding_rate_binance_change,
            s.funding_rate_bybit_now, 
            s.funding_rate_bybit_prev, 
            s.funding_rate_bybit_change,

            -- Объемы и цены
            s.volume_usd as signal_volume_usd, 
            s.price_usd_now as signal_price,
            s.price_usd_prev as signal_price_prev, 
            s.price_usd_change as price_change_10min,
            s.market_cap_usd,

            -- ВСЕ обогащенные данные
            e.oi_usdt_average, 
            e.oi_usdt_current, 
            e.oi_usdt_yesterday,
            e.oi_usdt_change_current_to_yesterday, 
            e.oi_usdt_change_current_to_average,
            e.oi_source_usdt,  -- НОВОЕ: источник OI

            -- Спотовые объемы USDT
            e.spot_volume_usdt_average, 
            e.spot_volume_usdt_current, 
            e.spot_volume_usdt_yesterday,
            e.spot_volume_usdt_change_current_to_yesterday, 
            e.spot_volume_usdt_change_current_to_average,
            e.spot_volume_source_usdt,  -- НОВОЕ: источник объема USDT

            -- Спотовые объемы BTC - НОВОЕ!
            e.spot_volume_btc_average, 
            e.spot_volume_btc_current, 
            e.spot_volume_btc_yesterday,
            e.spot_volume_btc_change_current_to_yesterday, 
            e.spot_volume_btc_change_current_to_average,
            e.spot_volume_source_btc,  -- НОВОЕ: источник объема BTC

            -- Спотовые цены
            e.spot_price_usdt_average, 
            e.spot_price_usdt_current, 
            e.spot_price_usdt_yesterday,
            e.spot_price_usdt_change_1h, 
            e.spot_price_usdt_change_24h, 
            e.spot_price_usdt_change_7d, 
            e.spot_price_usdt_change_30d,
            e.spot_price_source_usdt,  -- НОВОЕ: источник цены

            -- CMC метрики
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
        WHERE s.OI_contracts_binance_change > 0
        ORDER BY s.timestamp DESC
        {limit_clause}
        """)

        params = {"limit_val": limit} if limit else {}
        df = pd.read_sql(query, self.engine, params=params)
        print(f"✅ Загружено {len(df)} сигналов со всеми данными")

        if not df.empty:
            df = self._add_enhanced_technical_features(df)
            df = self._encode_categorical_features(df)

        return df

    def _add_enhanced_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет расширенный набор технических признаков"""
        print("🔧 Добавление расширенных технических признаков...")

        # Базовые временные признаки
        df['signal_timestamp'] = pd.to_datetime(df['signal_timestamp'])
        df['hour'] = df['signal_timestamp'].dt.hour
        df['day_of_week'] = df['signal_timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Существующие соотношения
        df['oi_binance_bybit_ratio'] = df['OI_contracts_binance_change'] / (
                df['OI_contracts_bybit_change'] + AVOID_ZERO_DIV_SMALL)
        df['funding_binance_bybit_ratio'] = df['funding_rate_binance_now'] / (
                df['funding_rate_bybit_now'] + AVOID_ZERO_DIV_TINY)

        # Моментум и волатильность
        df['price_momentum'] = df['spot_price_usdt_change_1h'] * df['spot_price_usdt_change_24h']
        df['volatility_1h'] = ((df['cmc_price_max_1h'] - df['cmc_price_min_1h']) / df['signal_price'] * 100).fillna(0)
        df['volatility_24h'] = ((df['cmc_price_max_24h'] - df['cmc_price_min_24h']) / df['signal_price'] * 100).fillna(
            0)

        # НОВЫЕ признаки

        # Соотношение объемов BTC/USDT
        df['volume_btc_usdt_ratio'] = (df['spot_volume_btc_current'] * df['signal_price']) / (
                df['spot_volume_usdt_current'] + AVOID_ZERO_DIV_SMALL)

        # Соотношение OI к объему
        df['oi_to_volume_ratio'] = df['oi_usdt_current'] / (
                df['spot_volume_usdt_current'] + AVOID_ZERO_DIV_SMALL)

        # Дивергенция funding rates между биржами
        df['funding_divergence'] = abs(
            df['funding_rate_binance_now'] - df['funding_rate_bybit_now']
        )

        # Ценовые диапазоны
        df['price_range_1h'] = (df['cmc_price_max_1h'] - df['cmc_price_min_1h']).fillna(0)
        df['price_range_24h'] = (df['cmc_price_max_24h'] - df['cmc_price_min_24h']).fillna(0)

        # Профиль объема (насколько текущий объем отклоняется от среднего)
        df['volume_profile_score'] = (
                abs(df['spot_volume_usdt_change_current_to_average']) * 0.5 +
                abs(df['spot_volume_btc_change_current_to_average']) * 0.5
        ).fillna(0)

        # Доминирование биржи (какая биржа имеет больший OI change)
        df['exchange_dominance'] = np.where(
            df['OI_contracts_binance_change'] > df['OI_contracts_bybit_change'],
            1,  # Binance доминирует
            -1  # Bybit доминирует
        )

        # Сила сигнала с учетом новых данных
        df['signal_strength'] = (
                df['OI_contracts_binance_change'] * 0.3 +
                df['OI_contracts_bybit_change'] * 0.3 +
                df['spot_volume_usdt_change_current_to_average'] * 0.2 +
                df['price_change_10min'] * 0.1 +
                df['volume_profile_score'] * 0.1
        ).fillna(0)

        return df

    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Кодирует категориальные признаки источников данных"""
        print("🏷️ Кодирование категориальных признаков...")

        # Маппинг для источников
        source_mapping = {
            'binance': 2,
            'bybit': 1,
            'coinmarketcap': 3,
            None: 0
        }

        # Кодируем источники
        for source_col in self.source_features:
            if source_col in df.columns:
                df[f'{source_col}_encoded'] = df[source_col].map(source_mapping).fillna(0)

        # Оценка надежности на основе источников
        # (CMC считается наиболее надежным для цен, биржи - для объемов)
        df['multi_source_confidence'] = 0

        if 'spot_price_source_usdt' in df.columns:
            df['multi_source_confidence'] += (df['spot_price_source_usdt'] == 'coinmarketcap').astype(int) * 0.5

        if 'spot_volume_source_usdt' in df.columns:
            df['multi_source_confidence'] += (df['spot_volume_source_usdt'].isin(['binance', 'bybit'])).astype(
                int) * 0.5

        return df

    def get_enhanced_futures_data(self, token_id: int, signal_time: datetime,
                                  hours_after: int = 48) -> pd.DataFrame:
        """
        Получает расширенные данные фьючерсов включая все поля
        """
        query = text("""
                     SELECT fd.timestamp,
                            fd.price_usd,
                            fd.volume_usd,
                            fd.volume_btc,              -- НОВОЕ
                            fd.open_interest_usd,
                            fd.open_interest_contracts, -- НОВОЕ
                            fd.funding_rate,
                            fd.market_cap_usd,
                            fd.btc_price                -- НОВОЕ: цена BTC на момент записи
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

    def _calculate_enhanced_outcomes(self, signal: pd.Series,
                                     futures_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Рассчитывает расширенный набор результатов с учетом всех данных
        """
        entry_price = signal['signal_price']
        outcomes: Dict[str, Any] = {}

        futures_data['timestamp'] = pd.to_datetime(futures_data['timestamp'])
        signal_time = pd.to_datetime(signal['signal_timestamp'])

        futures_data['time_diff_min'] = (futures_data['timestamp'] - signal_time).dt.total_seconds() / 60
        futures_data['pnl_pct'] = ((futures_data['price_usd'] - entry_price) / entry_price) * 100

        # Базовые метрики
        outcomes['max_profit_pct'] = futures_data['pnl_pct'].max()
        outcomes['max_drawdown_pct'] = futures_data['pnl_pct'].min()
        outcomes['final_pnl_pct'] = futures_data['pnl_pct'].iloc[-1] if not futures_data.empty else 0

        # Время до экстремумов
        if pd.notna(outcomes['max_profit_pct']) and outcomes['max_profit_pct'] > 0:
            outcomes['time_to_max_profit_min'] = futures_data.loc[futures_data['pnl_pct'].idxmax(), 'time_diff_min']
        else:
            outcomes['time_to_max_profit_min'] = None

        if pd.notna(outcomes['max_drawdown_pct']) and outcomes['max_drawdown_pct'] < 0:
            outcomes['time_to_max_drawdown_min'] = futures_data.loc[futures_data['pnl_pct'].idxmin(), 'time_diff_min']
        else:
            outcomes['time_to_max_drawdown_min'] = None

        # Расчет успешности по разным критериям
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

        # НОВЫЕ расширенные метрики
        if len(futures_data) > 1:
            # Волатильность
            outcomes['volatility_after'] = futures_data['pnl_pct'].std()

            # Тренды объемов с проверкой на None
            last_vol_usd = futures_data['volume_usd'].iloc[-1]
            first_vol_usd = futures_data['volume_usd'].iloc[0]
            if pd.notna(last_vol_usd) and pd.notna(first_vol_usd):
                outcomes['volume_usd_trend'] = 1 if last_vol_usd > first_vol_usd else -1
            else:
                outcomes['volume_usd_trend'] = 0

            if 'volume_btc' in futures_data.columns:
                last_vol_btc = futures_data['volume_btc'].iloc[-1]
                first_vol_btc = futures_data['volume_btc'].iloc[0]
                if pd.notna(last_vol_btc) and pd.notna(first_vol_btc):
                    outcomes['volume_btc_trend'] = 1 if last_vol_btc > first_vol_btc else -1
                else:
                    outcomes['volume_btc_trend'] = 0
                outcomes['avg_volume_btc_after'] = futures_data['volume_btc'].mean()

            # OI изменения
            oi_initial = futures_data['open_interest_usd'].iloc[0]
            if oi_initial > 0:
                outcomes['oi_change_after'] = ((futures_data['open_interest_usd'].iloc[
                                                    -1] - oi_initial) / oi_initial) * 100
            else:
                outcomes['oi_change_after'] = 0

            # OI в контрактах
            if 'open_interest_contracts' in futures_data.columns:
                oi_contracts_initial = futures_data['open_interest_contracts'].iloc[0]
                if oi_contracts_initial > 0:
                    outcomes['oi_contracts_change_after'] = (
                            (futures_data['open_interest_contracts'].iloc[-1] - oi_contracts_initial) /
                            oi_contracts_initial * 100
                    )
                else:
                    outcomes['oi_contracts_change_after'] = 0

            # Funding rate динамика
            if 'funding_rate' in futures_data.columns:
                outcomes['funding_rate_change_after'] = (
                        futures_data['funding_rate'].iloc[-1] - futures_data['funding_rate'].iloc[0]
                )
                outcomes['avg_funding_rate_after'] = futures_data['funding_rate'].mean()

            # Корреляция с BTC
            if 'btc_price' in futures_data.columns:
                last_btc_price = futures_data['btc_price'].iloc[-1]
                first_btc_price = futures_data['btc_price'].iloc[0]

                # Проверяем, что значения существуют и знаменатель не равен нулю
                if pd.notna(first_btc_price) and first_btc_price > 0 and pd.notna(last_btc_price):
                    btc_change = ((last_btc_price - first_btc_price) / first_btc_price) * 100
                    outcomes['performance_vs_btc'] = outcomes.get('final_pnl_pct', 0) - btc_change
                else:
                    outcomes['performance_vs_btc'] = outcomes.get('final_pnl_pct', 0)

                    # --- ИЗМЕНЕНИЕ: ДОБАВЛЕНО ЛОГИРОВАНИЕ ---
                    # Логируем, если цена некорректна (0 или NULL)
                    if pd.notna(first_btc_price) and first_btc_price == 0:
                        print(
                            f"\n⚠️  [Warning] Обнаружена НУЛЕВАЯ цена BTC для сигнала ID: {signal.get('signal_id')}, символ: {signal.get('symbol')}.")
                    elif pd.isna(first_btc_price):
                        print(
                            f"\n⚠️  [Warning] Обнаружена NULL цена BTC для сигнала ID: {signal.get('signal_id')}, символ: {signal.get('symbol')}.")
                    # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        return outcomes

    def calculate_signal_outcomes_enhanced(self, signals_df: pd.DataFrame,
                                           batch_size: int = 100) -> pd.DataFrame:
        """
        Рассчитывает расширенные результаты для всех сигналов
        """
        print(f"\n🔄 Расчет расширенных результатов для {len(signals_df)} сигналов...")
        all_results = []

        max_hours = max(self.time_windows_hours)

        for _, signal in tqdm(signals_df.iterrows(), total=len(signals_df)):
            try:
                futures_data = self.get_enhanced_futures_data(
                    signal['token_id'],
                    signal['signal_timestamp'],
                    hours_after=max_hours
                )

                if not futures_data.empty:
                    outcomes = self._calculate_enhanced_outcomes(signal, futures_data)
                    signal_dict = signal.to_dict()
                    signal_dict.update(outcomes)
                    all_results.append(signal_dict)

            except Exception as e:
                print(f"\n⚠️ Ошибка при обработке сигнала {signal['signal_id']}: {e}")
                continue

        print(f"\n✅ Обработано {len(all_results)} сигналов с расширенными результатами")
        return pd.DataFrame(all_results)

    def prepare_full_feature_set(self, df: pd.DataFrame,
                                 target_column: str = 'is_successful_main') -> Tuple[
        pd.DataFrame, pd.Series, List[str]]:
        """
        Подготавливает полный набор признаков для ML
        """
        print("\n🤖 Подготовка полного набора признаков для ML...")

        # Объединяем все типы признаков
        all_features = (
                self.base_signal_features +
                self.enriched_features +
                self.technical_features +
                [f'{col}_encoded' for col in self.source_features]  # Закодированные источники
        )

        # Добавляем новые outcome признаки если они есть
        outcome_features = [
            'volume_btc_trend', 'avg_volume_btc_after', 'oi_contracts_change_after',
            'funding_rate_change_after', 'avg_funding_rate_after', 'performance_vs_btc'
        ]

        # Фильтруем только существующие колонки
        available_features = [col for col in all_features if col in df.columns]

        # Добавляем outcome признаки если анализ уже был проведен
        for feat in outcome_features:
            if feat in df.columns:
                available_features.append(feat)

        df_clean = df.dropna(subset=[target_column])

        # Заполняем пропуски медианой для числовых признаков
        X = df_clean[available_features].fillna(df_clean[available_features].median())
        y = df_clean[target_column]

        print(f"✅ Подготовлено {len(X)} образцов с {len(available_features)} признаками")
        print(f"   Базовые признаки: {len([f for f in self.base_signal_features if f in available_features])}")
        print(f"   Обогащенные признаки: {len([f for f in self.enriched_features if f in available_features])}")
        print(f"   Технические признаки: {len([f for f in self.technical_features if f in available_features])}")
        print(f"   Признаки источников: {len([f for f in available_features if f.endswith('_encoded')])}")

        if not y.empty:
            class_distribution = y.value_counts(normalize=True)
            print(
                f"   Распределение классов: 0: {class_distribution.get(0, 0):.2%}, 1: {class_distribution.get(1, 0):.2%}")

        return X, y, available_features

    def train_enhanced_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Обучает модели на расширенном наборе признаков"""
        print("\n🎯 Обучение ML моделей на полном наборе данных...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Масштабирование
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        results = {}

        # 1. Random Forest с оптимизированными параметрами
        print("\n1️⃣  Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        results['random_forest'] = self._evaluate_model(rf_model, X_test, y_test, X.columns)

        # 2. XGBoost с настройкой под большой набор признаков
        print("2️⃣  XGBoost...")
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1

        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.6,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss',
            reg_alpha=0.1,
            reg_lambda=1.0
        )

        eval_set = [(X_test, y_test)]

        # --- ЭТО ВЕРНЫЙ СИНТАКСИС ДЛЯ ВАШЕЙ ВЕРСИИ XGBOOST 3.0.2 ---
        xgb_model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=[EarlyStopping(rounds=50, save_best=True)],
            verbose=False
        )
        # ---

        results['xgboost'] = self._evaluate_model(xgb_model, X_test, y_test)

        # 3. LightGBM - эффективен для больших наборов признаков
        print("3️⃣  LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=31,
            feature_fraction=0.7,
            bagging_fraction=0.8,
            bagging_freq=5,
            class_weight='balanced',
            random_state=42,
            verbosity=-1
        )
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        results['lightgbm'] = self._evaluate_model(lgb_model, X_test, y_test)

        # 4. Neural Network с большей архитектурой
        print("4️⃣  Neural Network...")
        nn_model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42
        )
        nn_model.fit(X_train_scaled, y_train)
        results['neural_network'] = self._evaluate_model(nn_model, X_test_scaled, y_test)

        # Сохраняем модели
        self.models = {name: res['model'] for name, res in results.items()}

        # Выводим результаты
        self._print_enhanced_results(results)

        return results

    def _evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                        feature_names: Optional[pd.Index] = None) -> Dict:
        """Оценка модели с расширенными метриками"""
        pred = model.predict(X_test)
        pred_proba = model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, pred, output_dict=True, zero_division=0)

        metrics = {
            'model': model,
            'accuracy': report['accuracy'],
            'auc': roc_auc_score(y_test, pred_proba),
            'precision_0': report['0']['precision'],
            'recall_0': report['0']['recall'],
            'precision_1': report['1']['precision'],
            'recall_1': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'report': report,
        }

        # Важность признаков для древесных моделей
        if feature_names is not None and hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            metrics['feature_importance'] = importance_df
            metrics['top_features'] = importance_df.head(20)

        return metrics

    def _print_enhanced_results(self, results: Dict):
        """Выводит расширенные результаты обучения"""
        print("\n📊 Результаты обучения на полном наборе данных:")
        print("=" * 80)

        for model_name, metrics in results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  AUC: {metrics['auc']:.4f}")
            print(f"  Class 0 - Precision: {metrics['precision_0']:.3f}, Recall: {metrics['recall_0']:.3f}")
            print(f"  Class 1 - Precision: {metrics['precision_1']:.3f}, Recall: {metrics['recall_1']:.3f}")
            print(f"  F1-Score (Class 1): {metrics['f1_score']:.3f}")

            # Топ признаки для моделей с feature importance
            if 'top_features' in metrics:
                print(f"\n  Топ-10 важных признаков:")
                for idx, row in metrics['top_features'].head(10).iterrows():
                    print(f"    {idx + 1}. {row['feature']}: {row['importance']:.4f}")

    def analyze_feature_groups_importance(self, results: Dict) -> pd.DataFrame:
        """Анализирует важность групп признаков"""
        print("\n📊 Анализ важности групп признаков...")

        if 'random_forest' not in results or 'feature_importance' not in results['random_forest']:
            return pd.DataFrame()

        importance_df = results['random_forest']['feature_importance'].copy()

        # Классифицируем признаки по группам
        def classify_feature(feature_name):
            if feature_name in self.base_signal_features:
                return 'Базовые сигналы'
            elif feature_name in self.enriched_features:
                return 'Обогащенные данные'
            elif feature_name in self.technical_features:
                return 'Технические индикаторы'
            elif feature_name.endswith('_encoded'):
                return 'Источники данных'
            elif 'btc' in feature_name.lower():
                return 'BTC контекст'
            elif 'after' in feature_name:
                return 'Outcome метрики'
            else:
                return 'Прочие'

        importance_df['feature_group'] = importance_df['feature'].apply(classify_feature)

        # Агрегируем по группам
        group_importance = importance_df.groupby('feature_group').agg({
            'importance': ['sum', 'mean', 'count']
        }).round(4)

        group_importance.columns = ['total_importance', 'avg_importance', 'feature_count']
        group_importance = group_importance.sort_values('total_importance', ascending=False)

        print("\nВажность групп признаков:")
        print(group_importance)

        return group_importance

    def generate_comprehensive_report(self, df: pd.DataFrame, results: Dict,
                                      group_importance: pd.DataFrame) -> str:
        """Генерирует подробный отчет с анализом всех данных"""
        print("\n📄 Генерация комплексного отчета...")

        report_lines = [
            "=" * 100,
            "КОМПЛЕКСНЫЙ ОТЧЕТ ПО АНАЛИЗУ КРИПТОВАЛЮТНЫХ СИГНАЛОВ",
            "С ИСПОЛЬЗОВАНИЕМ ПОЛНОГО НАБОРА ДАННЫХ",
            f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 100, "",

            "ОБРАБОТАННЫЕ ДАННЫЕ:",
            f"- Всего сигналов: {len(df)}",
            f"- Период: {df['signal_timestamp'].min()} - {df['signal_timestamp'].max()}",
            f"- Уникальных токенов: {df['symbol'].nunique()}",
            f"- Общее количество признаков: {len([col for col in df.columns if col not in ['signal_id', 'symbol', 'signal_timestamp']])}",
            "",

            "ИСТОЧНИКИ ДАННЫХ:",
        ]

        # Анализ источников данных
        for source_col in self.source_features:
            if source_col in df.columns:
                source_dist = df[source_col].value_counts()
                report_lines.append(f"\n{source_col}:")
                for source, count in source_dist.items():
                    report_lines.append(f"  - {source}: {count} ({count / len(df) * 100:.1f}%)")

        # Статистика по группам признаков
        if not group_importance.empty:
            report_lines.extend([
                "",
                "ВАЖНОСТЬ ГРУПП ПРИЗНАКОВ (Random Forest):",
            ])
            for group, row in group_importance.iterrows():
                report_lines.append(
                    f"- {group}: {row['total_importance']:.3f} "
                    f"(среднее: {row['avg_importance']:.4f}, признаков: {int(row['feature_count'])})"
                )

        # Результаты по разным критериям
        report_lines.extend([
            "",
            "РЕЗУЛЬТАТЫ ПО КРИТЕРИЯМ УСПЕХА:",
        ])

        for profit in self.profit_targets:
            for stop in self.stop_losses:
                for window in self.time_windows_hours:
                    col_name = f'success_{profit}p_{stop}sl_{window}h'
                    if col_name in df.columns:
                        success_rate = df[col_name].mean() * 100
                        count = df[col_name].sum()
                        report_lines.append(
                            f"- {profit}% прибыль / {stop}% стоп / {window}ч: "
                            f"{success_rate:.1f}% успеха ({count} из {len(df)})"
                        )

        # ML результаты
        report_lines.extend([
            "",
            "РЕЗУЛЬТАТЫ ML МОДЕЛЕЙ:",
        ])

        best_model = None
        best_auc = 0

        for model_name, metrics in results.items():
            report_lines.extend([
                f"\n{model_name.upper()}:",
                f"  - Accuracy: {metrics['accuracy']:.4f}",
                f"  - AUC: {metrics['auc']:.4f}",
                f"  - Precision (Class 1): {metrics['precision_1']:.3f}",
                f"  - Recall (Class 1): {metrics['recall_1']:.3f}",
                f"  - F1-Score: {metrics['f1_score']:.3f}"
            ])

            if metrics['auc'] > best_auc:
                best_auc = metrics['auc']
                best_model = model_name

        # Топ признаки
        if 'random_forest' in results and 'top_features' in results['random_forest']:
            report_lines.extend([
                "",
                "ТОП-20 ВАЖНЫХ ПРИЗНАКОВ:",
            ])
            for idx, row in results['random_forest']['top_features'].head(20).iterrows():
                report_lines.append(f"  {idx + 1}. {row['feature']}: {row['importance']:.4f}")

        # Статистика по токенам
        report_lines.extend([
            "",
            "ТОП-10 ТОКЕНОВ ПО КОЛИЧЕСТВУ СИГНАЛОВ:",
        ])
        top_tokens = df['symbol'].value_counts().head(10)
        for i, (token, count) in enumerate(top_tokens.items(), 1):
            token_df = df[df['symbol'] == token]
            if 'is_successful_main' in token_df.columns:
                success_rate = token_df['is_successful_main'].mean() * 100
                avg_oi_change = (token_df['OI_contracts_binance_change'] +
                                 token_df['OI_contracts_bybit_change']).mean()
                report_lines.append(
                    f"{i}. {token}: {count} сигналов "
                    f"({success_rate:.1f}% успеха, средний OI change: {avg_oi_change:.2f})"
                )

        # Итоговые рекомендации
        report_lines.extend([
            "",
            "=" * 100,
            "ИТОГОВЫЕ РЕКОМЕНДАЦИИ:",
            f"✅ Лучшая модель: {best_model.upper()} (AUC: {best_auc:.4f})",
            "✅ Наиболее важные группы признаков: Обогащенные данные и Технические индикаторы",
            "✅ Рекомендуется использовать множественные источники данных для повышения надежности",
            "=" * 100
        ])

        # Сохранение отчета
        os.makedirs('reports', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f'reports/comprehensive_analysis_report_{timestamp}.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"\n📄 Отчет сохранен: {report_path}")
        return report_path


def main():
    """Основная функция для запуска полного анализа с использованием всех данных"""
    print("🚀 Запуск полного анализа с использованием ВСЕХ доступных данных...")
    print("   включая интеллектуальную обработку NULL значений")
    print("=" * 100)

    try:
        # Импортируем null-aware функциональность
        from null_aware_processor import enhance_processor_with_null_awareness

        # Создаем процессор с поддержкой null-aware
        NullAwareProcessor = enhance_processor_with_null_awareness(EnhancedFullSignalsProcessor)
        processor = NullAwareProcessor()

        # Параметры
        limit_signals = None  # None для всех сигналов

        # 1. Загружаем ВСЕ данные
        print("\n" + "=" * 50)
        print("ШАГ 1: ЗАГРУЗКА ПОЛНЫХ ДАННЫХ")
        print("=" * 50)
        signals = processor.get_all_signals_complete(limit=limit_signals)

        if signals.empty:
            print("❌ Не найдено сигналов для обработки")
            return

        # 2. Рассчитываем расширенные результаты
        print("\n" + "=" * 50)
        print("ШАГ 2: РАСЧЕТ РАСШИРЕННЫХ РЕЗУЛЬТАТОВ")
        print("=" * 50)
        signals_with_outcomes = processor.calculate_signal_outcomes_enhanced(signals, batch_size=100)

        if signals_with_outcomes.empty:
            print("❌ Не удалось рассчитать результаты")
            return

        # 3. Добавляем рыночный контекст (если доступен скрипт)
        print("\n" + "=" * 50)
        print("ШАГ 3: ИНТЕГРАЦИЯ РЫНОЧНОГО КОНТЕКСТА")
        print("=" * 50)
        try:
            from market_context_analyzer import MarketContextAnalyzer
            market_analyzer = MarketContextAnalyzer()
            enriched_signals = market_analyzer.enrich_signals_with_market_context(signals_with_outcomes)

            # Анализ по режимам
            regime_performance = market_analyzer.analyze_performance_by_regime(enriched_signals)

            # Сохраняем анализ режимов
            with open('reports/market_regime_analysis_full.json', 'w') as f:
                json.dump(regime_performance, f, indent=2, default=str)

        except ImportError:
            print("⚠️  Модуль market_context_analyzer не найден, продолжаем без рыночного контекста")
            enriched_signals = signals_with_outcomes

        # 4. Сохраняем полные данные
        print("\n" + "=" * 50)
        print("ШАГ 4: СОХРАНЕНИЕ ДАННЫХ")
        print("=" * 50)
        print("💾 Сохранение полных обогащенных данных...")
        enriched_signals.to_pickle('full_signals_with_all_data.pkl')
        enriched_signals.to_csv('full_signals_with_all_data.csv', index=False)
        print(f"✅ Сохранено {len(enriched_signals)} сигналов с {len(enriched_signals.columns)} признаками")

        # 5. Подготовка данных для ML
        print("\n" + "=" * 50)
        print("ШАГ 5: ПОДГОТОВКА ДАННЫХ ДЛЯ ML")
        print("=" * 50)
        X, y, feature_names = processor.prepare_full_feature_set(enriched_signals)

        if X.empty or y.empty:
            print("❌ Недостаточно данных для обучения моделей")
            return

        # 6. Обучение моделей
        print("\n" + "=" * 50)
        print("ШАГ 6: ОБУЧЕНИЕ ML МОДЕЛЕЙ")
        print("=" * 50)
        ml_results = processor.train_enhanced_models(X, y)

        # 7. Анализ важности групп признаков
        print("\n" + "=" * 50)
        print("ШАГ 7: АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ")
        print("=" * 50)
        group_importance = processor.analyze_feature_groups_importance(ml_results)

        # 8. Сохранение моделей и метаданных
        print("\n" + "=" * 50)
        print("ШАГ 8: СОХРАНЕНИЕ МОДЕЛЕЙ")
        print("=" * 50)
        output_dir = 'models_full_data'
        os.makedirs(output_dir, exist_ok=True)

        # Сохраняем модели
        for model_name, model in processor.models.items():
            joblib.dump(model, os.path.join(output_dir, f'{model_name}_model.pkl'))

        # Сохраняем скейлер и метаданные
        joblib.dump(processor.scaler, os.path.join(output_dir, 'scaler.pkl'))

        with open(os.path.join(output_dir, 'feature_names.json'), 'w') as f:
            json.dump(feature_names, f)

        # Сохраняем результаты обучения
        json_results = {
            model_name: {
                'accuracy': float(m['accuracy']),
                'auc': float(m['auc']),
                'precision_0': float(m['precision_0']),
                'recall_0': float(m['recall_0']),
                'precision_1': float(m['precision_1']),
                'recall_1': float(m['recall_1']),
                'f1_score': float(m['f1_score'])
            } for model_name, m in ml_results.items()
        }

        with open(os.path.join(output_dir, 'training_results.json'), 'w') as f:
            json.dump(json_results, f, indent=2)

        # Сохраняем важность признаков
        if 'random_forest' in ml_results and 'feature_importance' in ml_results['random_forest']:
            ml_results['random_forest']['feature_importance'].to_csv(
                os.path.join(output_dir, 'feature_importance_full.csv'),
                index=False
            )

        if not group_importance.empty:
            group_importance.to_csv(
                os.path.join(output_dir, 'feature_group_importance.csv')
            )

        print(f"✅ Модели и метаданные сохранены в '{output_dir}/'")

        # 9. Генерация отчета
        print("\n" + "=" * 50)
        print("ШАГ 9: ГЕНЕРАЦИЯ ОТЧЕТА")
        print("=" * 50)
        report_path = processor.generate_comprehensive_report(
            enriched_signals, ml_results, group_importance
        )

        # 10. Итоговая статистика
        print("\n" + "=" * 100)
        print("📊 ИТОГОВАЯ СТАТИСТИКА:")
        print("=" * 100)
        print(f"✅ Анализ завершен успешно!")
        print(f"📁 Результаты:")
        print(f"   - Данные: full_signals_with_all_data.pkl/csv")
        print(f"   - Модели: {output_dir}/")
        print(f"   - Отчет: {report_path}")

        # Лучшая модель
        best_model_name, best_model_metrics = max(
            ml_results.items(),
            key=lambda item: item[1]['auc']
        )
        print(f"\n🏆 Лучшая модель: {best_model_name.upper()}")
        print(f"   - AUC: {best_model_metrics['auc']:.4f}")
        print(f"   - F1-Score: {best_model_metrics['f1_score']:.3f}")

        # Статистика по данным
        print(f"\n📊 Статистика по данным:")
        print(f"   - Обработано сигналов: {len(enriched_signals)}")
        print(f"   - Использовано признаков: {len(feature_names)}")
        print(f"   - Успешных сигналов: {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")

        # Топ признаки
        if 'random_forest' in ml_results and 'top_features' in ml_results['random_forest']:
            print("\n🎯 Топ-5 важных признаков:")
            for idx, row in ml_results['random_forest']['top_features'].head(5).iterrows():
                print(f"   {idx + 1}. {row['feature']}: {row['importance']:.4f}")

    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()