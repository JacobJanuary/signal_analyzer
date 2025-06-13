#!/usr/bin/env python3
"""
Market Context Analyzer - Анализ рыночного контекста через BTC для улучшения предсказаний
ИСПРАВЛЕННАЯ ВЕРСИЯ
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import requests
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dotenv import load_dotenv
from urllib.parse import quote_plus
import time
from tqdm import tqdm

load_dotenv()


class MarketContextAnalyzer:
    """
    Анализатор рыночного контекста через данные BTC.
    Получает исторические данные через CoinMarketCap API и данные из БД.
    """

    def __init__(self):
        self.engine = self._create_db_connection()
        self.cmc_api_key = os.getenv('COINMARKETCAP_API_KEY')

        if not self.cmc_api_key:
            print("⚠️  Не найден COINMARKETCAP_API_KEY в .env файле")
            print("   Получите бесплатный ключ на https://coinmarketcap.com/api/")

        self.cmc_headers = {
            'X-CMC_PRO_API_KEY': self.cmc_api_key,
            'Accept': 'application/json'
        }

        self.btc_market_data = None
        self.btc_futures_data = None

    def _create_db_connection(self):
        """Создает подключение к БД"""
        try:
            user = os.getenv('DB_USER')
            password = quote_plus(os.getenv('DB_PASSWORD', ''))
            host = os.getenv('DB_HOST')
            port = os.getenv('DB_PORT', 3306)
            database = os.getenv('DB_NAME')

            connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
            engine = create_engine(connection_string, pool_pre_ping=True)

            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            return engine
        except Exception as e:
            print(f"❌ Ошибка подключения к БД: {e}")
            raise

    def fetch_btc_historical_data_cmc(self, days: int = 30) -> pd.DataFrame:
        """
        Получает исторические данные BTC через CoinMarketCap API

        Примечание: Бесплатный план CMC не поддерживает исторические данные.
        Для полного функционала нужен платный план.
        """
        print(f"📊 Получение данных BTC из CoinMarketCap за {days} дней...")

        # Для бесплатного плана доступны только текущие данные
        # Используем endpoint quotes/latest
        url = 'https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest'

        params = {
            'symbol': 'BTC',
            'convert': 'USD'
        }

        try:
            response = requests.get(url, headers=self.cmc_headers, params=params)

            if response.status_code == 200:
                data = response.json()
                btc_data = data['data']['BTC'][0]

                # Извлекаем текущие данные
                current_data = {
                    'timestamp': datetime.now(),
                    'price': btc_data['quote']['USD']['price'],
                    'volume_24h': btc_data['quote']['USD']['volume_24h'],
                    'market_cap': btc_data['quote']['USD']['market_cap'],
                    'percent_change_1h': btc_data['quote']['USD']['percent_change_1h'],
                    'percent_change_24h': btc_data['quote']['USD']['percent_change_24h'],
                    'percent_change_7d': btc_data['quote']['USD']['percent_change_7d'],
                    'percent_change_30d': btc_data['quote']['USD']['percent_change_30d'],
                    'market_cap_dominance': btc_data['quote']['USD']['market_cap_dominance']
                }

                print("✅ Получены текущие данные BTC из CMC")
                print(f"   Цена: ${current_data['price']:,.2f}")
                print(f"   Изменение 24ч: {current_data['percent_change_24h']:.2f}%")

                return pd.DataFrame([current_data])

            else:
                print(f"❌ Ошибка API: {response.status_code}")
                return pd.DataFrame()

        except Exception as e:
            print(f"❌ Ошибка при запросе к CoinMarketCap: {e}")
            return pd.DataFrame()

    def fetch_btc_historical_data_db(self, days: int = 30) -> pd.DataFrame:
        """
        Получает исторические данные BTC из базы данных
        ИСПРАВЛЕНО: Убраны лишние пробелы в SQL функциях
        """
        print(f"📊 Получение исторических данных BTC из БД за {days} дней...")

        # Получаем token_id для BTC
        btc_token_query = text("""
            SELECT id FROM tokens 
            WHERE symbol = 'BTC' OR symbol = 'BTCUSDT'
            LIMIT 1
        """)

        btc_token_result = pd.read_sql(btc_token_query, self.engine)

        if btc_token_result.empty:
            print("❌ BTC не найден в базе данных")
            return pd.DataFrame()

        btc_token_id = btc_token_result.iloc[0]['id']

        # ИСПРАВЛЕНО: Убраны пробелы между функциями и скобками
        query = text("""
            SELECT
                DATE(fd.timestamp) as date,
                AVG(fd.price_usd) as avg_price,
                MIN(fd.price_usd) as min_price,
                MAX(fd.price_usd) as max_price,
                AVG(fd.volume_usd) as avg_volume,
                SUM(fd.volume_usd) as total_volume,
                AVG(fd.open_interest_usd) as avg_oi,
                AVG(fd.funding_rate) as avg_funding,

                -- Изменения
                (MAX(fd.price_usd) - MIN(fd.price_usd)) / MIN(fd.price_usd) * 100 as daily_volatility,

                -- Первая и последняя цена дня для расчета изменения
                SUBSTRING_INDEX(GROUP_CONCAT(fd.price_usd ORDER BY fd.timestamp), ',', 1) as open_price,
                SUBSTRING_INDEX(GROUP_CONCAT(fd.price_usd ORDER BY fd.timestamp DESC), ',', 1) as close_price

            FROM futures_data fd
            JOIN futures_pairs fp ON fd.pair_id = fp.id
            WHERE fp.token_id = :token_id
                AND fd.timestamp >= DATE_SUB(NOW(), INTERVAL :days DAY)
            GROUP BY DATE(fd.timestamp)
            ORDER BY date DESC
        """)

        params = {"token_id": int(btc_token_id), "days": days}

        try:
            df = pd.read_sql(query, self.engine, params=params)

            if not df.empty:
                # Конвертируем строковые значения в числа
                df['open_price'] = pd.to_numeric(df['open_price'])
                df['close_price'] = pd.to_numeric(df['close_price'])

                # Рассчитываем дневное изменение
                df['daily_change_pct'] = ((df['close_price'] - df['open_price']) / df['open_price'] * 100)

                # Рассчитываем изменения относительно предыдущего дня
                df = df.sort_values('date')
                df['price_change_vs_prev'] = df['avg_price'].pct_change() * 100
                df['volume_change_vs_prev'] = df['total_volume'].pct_change() * 100
                df['oi_change_vs_prev'] = df['avg_oi'].pct_change() * 100

                # Скользящие средние
                df['ma_7'] = df['avg_price'].rolling(window=7, min_periods=1).mean()
                df['ma_30'] = df['avg_price'].rolling(window=30, min_periods=1).mean()

                # Тренд
                df['trend_7d'] = np.where(df['avg_price'] > df['ma_7'], 1, -1)
                df['trend_30d'] = np.where(df['avg_price'] > df['ma_30'], 1, -1)

                print(f"✅ Загружено {len(df)} дней данных BTC из БД")

                # Сортируем обратно по убыванию даты
                df = df.sort_values('date', ascending=False)

            return df

        except Exception as e:
            print(f"❌ Ошибка при получении данных BTC из БД: {e}")
            return pd.DataFrame()

    def calculate_market_regime(self, btc_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Определяет текущий рыночный режим на основе данных BTC
        """
        if btc_data.empty:
            return {
                'regime': 'unknown',
                'strength': 0,
                'volatility': 'normal',
                'trend': 'neutral'
            }

        latest = btc_data.iloc[0]

        # Определяем режим волатильности
        avg_volatility = btc_data['daily_volatility'].mean()
        current_volatility = latest['daily_volatility']

        if current_volatility > avg_volatility * 1.5:
            volatility_regime = 'high'
        elif current_volatility < avg_volatility * 0.5:
            volatility_regime = 'low'
        else:
            volatility_regime = 'normal'

        # Определяем тренд
        week_change = 0
        if len(btc_data) >= 7:
            week_change = ((latest['avg_price'] - btc_data.iloc[6]['avg_price']) / btc_data.iloc[6]['avg_price'] * 100)

            if week_change > 10:
                trend = 'strong_bullish'
            elif week_change > 3:
                trend = 'bullish'
            elif week_change < -10:
                trend = 'strong_bearish'
            elif week_change < -3:
                trend = 'bearish'
            else:
                trend = 'neutral'
        else:
            trend = 'neutral'

        # Определяем общий режим
        if trend in ['strong_bullish', 'bullish'] and volatility_regime != 'high':
            regime = 'bull_market'
        elif trend in ['strong_bearish', 'bearish'] and volatility_regime != 'high':
            regime = 'bear_market'
        elif volatility_regime == 'high':
            regime = 'high_volatility'
        else:
            regime = 'ranging'

        # Сила режима (0-100)
        strength = min(abs(week_change) * 5, 100)

        return {
            'regime': regime,
            'strength': strength,
            'volatility': volatility_regime,
            'trend': trend,
            'week_change': week_change,
            'current_volatility': current_volatility,
            'avg_volatility': avg_volatility
        }

    def enrich_signals_with_market_context(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Обогащает сигналы данными о рыночном контексте
        """
        print("\n🔄 Обогащение сигналов рыночным контекстом...")

        # Получаем исторические данные BTC
        self.btc_market_data = self.fetch_btc_historical_data_db(days=30)

        if self.btc_market_data.empty:
            print("⚠️  Не удалось получить данные BTC, продолжаем без рыночного контекста")
            return signals_df

        # Рассчитываем текущий рыночный режим
        market_regime = self.calculate_market_regime(self.btc_market_data)

        print(f"\n📈 Текущий рыночный режим:")
        print(f"   Режим: {market_regime['regime']}")
        print(f"   Тренд: {market_regime['trend']}")
        print(f"   Волатильность: {market_regime['volatility']}")
        print(f"   Недельное изменение: {market_regime['week_change']:.2f}%")

        # Добавляем контекст к каждому сигналу
        enriched_signals = []

        for idx, signal in tqdm(signals_df.iterrows(), total=len(signals_df), desc="Обогащение сигналов"):
            signal_date = pd.to_datetime(signal['signal_timestamp']).date()

            # Находим данные BTC для даты сигнала
            btc_day_data = self.btc_market_data[self.btc_market_data['date'] == signal_date]

            if not btc_day_data.empty:
                btc_context = btc_day_data.iloc[0]

                # Добавляем контекст BTC
                signal['btc_price_at_signal'] = btc_context['avg_price']
                signal['btc_volume_at_signal'] = btc_context['total_volume']
                signal['btc_volatility_at_signal'] = btc_context['daily_volatility']
                signal['btc_daily_change_at_signal'] = btc_context['daily_change_pct']
                signal['btc_funding_at_signal'] = btc_context['avg_funding']
                signal['btc_oi_at_signal'] = btc_context['avg_oi']
                signal['btc_trend_7d'] = btc_context['trend_7d']
                signal['btc_trend_30d'] = btc_context['trend_30d']

                # Рассчитываем относительные метрики
                if 'market_cap_usd' in signal and pd.notna(signal['market_cap_usd']) and signal['market_cap_usd'] > 0:
                    signal['market_cap_to_btc_volume_ratio'] = signal['market_cap_usd'] / btc_context['total_volume']

                # Корреляция с BTC
                if pd.notna(signal.get('price_change_10min')):
                    signal['price_change_vs_btc'] = signal['price_change_10min'] - btc_context['daily_change_pct']
                else:
                    signal['price_change_vs_btc'] = 0

            else:
                # Используем последние известные данные
                if not self.btc_market_data.empty:
                    latest_btc = self.btc_market_data.iloc[0]
                    signal['btc_price_at_signal'] = latest_btc['avg_price']
                    signal['btc_volume_at_signal'] = latest_btc['total_volume']
                    signal['btc_volatility_at_signal'] = latest_btc['daily_volatility']
                    signal['btc_daily_change_at_signal'] = latest_btc['daily_change_pct']
                    signal['btc_funding_at_signal'] = latest_btc['avg_funding']
                    signal['btc_oi_at_signal'] = latest_btc['avg_oi']
                    signal['btc_trend_7d'] = latest_btc['trend_7d']
                    signal['btc_trend_30d'] = latest_btc['trend_30d']

            # Добавляем общий рыночный режим
            signal['market_regime'] = market_regime['regime']
            signal['market_regime_strength'] = market_regime['strength']
            signal['market_volatility_regime'] = market_regime['volatility']
            signal['market_trend'] = market_regime['trend']

            enriched_signals.append(signal)

        enriched_df = pd.DataFrame(enriched_signals)

        # Добавляем категориальные признаки для ML
        regime_mapping = {
            'bull_market': 3,
            'ranging': 2,
            'bear_market': 1,
            'high_volatility': 0,
            'unknown': -1
        }

        trend_mapping = {
            'strong_bullish': 2,
            'bullish': 1,
            'neutral': 0,
            'bearish': -1,
            'strong_bearish': -2
        }

        enriched_df['market_regime_encoded'] = enriched_df['market_regime'].map(regime_mapping).fillna(-1)
        enriched_df['market_trend_encoded'] = enriched_df['market_trend'].map(trend_mapping).fillna(0)

        print(f"✅ Обогащено {len(enriched_df)} сигналов рыночным контекстом")

        return enriched_df

    def analyze_performance_by_regime(self, signals_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализирует производительность сигналов в разных рыночных режимах
        """
        if 'market_regime' not in signals_df.columns or 'is_successful_main' not in signals_df.columns:
            return {}

        print("\n📊 Анализ производительности по рыночным режимам...")

        regime_stats = {}

        for regime in signals_df['market_regime'].unique():
            regime_signals = signals_df[signals_df['market_regime'] == regime]

            if len(regime_signals) > 0:
                success_rate = regime_signals['is_successful_main'].mean() * 100
                total_signals = len(regime_signals)

                # Средние показатели успешных сигналов в режиме
                successful = regime_signals[regime_signals['is_successful_main'] == 1]

                # ИСПРАВЛЕНО: Добавлены проверки на наличие колонок
                avg_max_profit = successful['max_profit_pct'].mean() if 'max_profit_pct' in successful.columns and len(successful) > 0 else 0

                # ИСПРАВЛЕНО: Обработка None значений в time_to_max_profit_min
                if 'time_to_max_profit_min' in successful.columns and len(successful) > 0:
                    time_values = successful['time_to_max_profit_min'].dropna()
                    avg_time_to_profit = time_values.mean() if len(time_values) > 0 else 0
                else:
                    avg_time_to_profit = 0

                regime_stats[regime] = {
                    'total_signals': total_signals,
                    'success_rate': success_rate,
                    'successful_count': len(successful),
                    'avg_max_profit': avg_max_profit,
                    'avg_time_to_profit': avg_time_to_profit,
                    'top_tokens': regime_signals['symbol'].value_counts().head(5).to_dict()
                }

                print(f"\n{regime.upper()}:")
                print(f"  Сигналов: {total_signals}")
                print(f"  Успешность: {success_rate:.1f}%")
                print(f"  Средняя макс. прибыль: {avg_max_profit:.2f}%")

        return regime_stats

    def get_recommended_parameters_by_regime(self, current_regime: str) -> Dict[str, Any]:
        """
        Возвращает рекомендуемые параметры торговли для текущего режима
        """
        regime_params = {
            'bull_market': {
                'min_oi_change': 2.5,  # Более агрессивный в бычьем рынке
                'profit_target': 7,
                'stop_loss': 4,
                'position_size_multiplier': 1.2
            },
            'bear_market': {
                'min_oi_change': 4.0,  # Более консервативный в медвежьем
                'profit_target': 4,
                'stop_loss': 2,
                'position_size_multiplier': 0.7
            },
            'high_volatility': {
                'min_oi_change': 5.0,  # Очень избирательный
                'profit_target': 10,
                'stop_loss': 5,
                'position_size_multiplier': 0.5
            },
            'ranging': {
                'min_oi_change': 3.0,
                'profit_target': 5,
                'stop_loss': 3,
                'position_size_multiplier': 1.0
            }
        }

        return regime_params.get(current_regime, regime_params['ranging'])


def integrate_market_context(processor, signals_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Интегрирует анализ рыночного контекста в основной процесс
    """
    # Создаем анализатор контекста
    context_analyzer = MarketContextAnalyzer()

    # Обогащаем сигналы рыночным контекстом
    enriched_signals = context_analyzer.enrich_signals_with_market_context(signals_df)

    # Анализируем производительность по режимам
    regime_performance = context_analyzer.analyze_performance_by_regime(enriched_signals)

    # Сохраняем анализ режимов
    with open('market_regime_analysis.json', 'w') as f:
        json.dump(regime_performance, f, indent=2, default=str)

    # Добавляем новые признаки в список для ML
    new_market_features = [
        'btc_price_at_signal', 'btc_volume_at_signal', 'btc_volatility_at_signal',
        'btc_daily_change_at_signal', 'btc_funding_at_signal', 'btc_oi_at_signal',
        'btc_trend_7d', 'btc_trend_30d', 'price_change_vs_btc',
        'market_regime_encoded', 'market_trend_encoded', 'market_regime_strength'
    ]

    print(f"\n✅ Добавлено {len(new_market_features)} новых признаков рыночного контекста")

    return enriched_signals, new_market_features


# Обновленная функция для использования в основном скрипте
def main_with_market_context():
    """
    Пример интеграции рыночного контекста в основной анализ
    """
    from analize_signals import FullSignalsProcessor

    print("🚀 Запуск анализа с рыночным контекстом...")

    try:
        # Основной процессор
        processor = FullSignalsProcessor()

        # Получаем сигналы
        signals = processor.get_all_signals(limit=1000)

        if signals.empty:
            print("❌ Не найдено сигналов")
            return

        # Рассчитываем базовые результаты
        signals_with_outcomes = processor.calculate_signal_outcomes_batch(signals)

        # ИНТЕГРИРУЕМ РЫНОЧНЫЙ КОНТЕКСТ
        enriched_signals, new_features = integrate_market_context(processor, signals_with_outcomes)

        # Обновляем список признаков для ML
        X, y, base_features = processor.prepare_ml_features(enriched_signals)

        # Добавляем новые признаки
        all_features = base_features + [f for f in new_features if f in enriched_signals.columns]
        X_enriched = enriched_signals[all_features].fillna(0)

        # Обучаем модели с расширенным набором признаков
        print(f"\n🎯 Обучение моделей с {len(all_features)} признаками (включая рыночный контекст)...")
        ml_results = processor.train_ml_models(X_enriched, y)

        # Сохраняем все
        processor.save_models_and_data(all_features, ml_results, output_dir='models_with_context')
        enriched_signals.to_pickle('signals_with_market_context.pkl')

        print("\n✅ Анализ с рыночным контекстом завершен!")

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main_with_market_context()