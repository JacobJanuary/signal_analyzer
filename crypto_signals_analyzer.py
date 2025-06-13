#!/usr/bin/env python3
"""
Crypto Signals Analyzer - Полный скрипт для анализа торговых сигналов
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore')

# Загружаем переменные окружения
load_dotenv()


class CryptoSignalsAnalyzer:
    """
    Основной класс для анализа криптовалютных сигналов
    """

    def __init__(self):
        """Инициализация анализатора"""
        # Подключение к БД
        self.engine = self._create_db_connection()

        # Параметры анализа
        self.signal_threshold = {
            'oi_change_min': 3.0,  # Минимальный рост OI для сигнала
            'funding_positive': True,  # Требовать положительный фандинг
            'hours_to_analyze': 24,  # Часов для анализа после сигнала
            'profit_target': 5.0,  # Целевая прибыль %
            'stop_loss': 3.0  # Стоп-лосс %
        }

    def _create_db_connection(self):
        """Создает подключение к базе данных"""
        try:
            engine = create_engine(
                f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
                f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT', 3306)}/{os.getenv('DB_NAME')}",
                pool_pre_ping=True
            )
            # Тестируем подключение
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("✅ Подключение к БД установлено")
            return engine
        except Exception as e:
            print(f"❌ Ошибка подключения к БД: {e}")
            raise

    def get_recent_signals(self, days: int = 30, limit: int = 1000) -> pd.DataFrame:
        """
        Получает последние сигналы за указанный период

        Args:
            days: Количество дней для анализа
            limit: Максимальное количество сигналов
        """
        query = f"""
        SELECT 
            s.id as signal_id,
            s.token_id,
            s.symbol,
            s.timestamp as signal_timestamp,
            s.OI_contracts_binance_change,
            s.OI_contracts_bybit_change,
            s.funding_rate_binance_now,
            s.funding_rate_bybit_now,
            s.volume_usd as signal_volume_usd,
            s.price_usd_now as signal_price,
            s.price_usd_change as price_change_10min,
            s.market_cap_usd,

            -- Данные из enriched таблицы
            e.oi_usdt_change_current_to_average,
            e.oi_usdt_change_current_to_yesterday,
            e.spot_volume_usdt_change_current_to_average,
            e.spot_volume_usdt_change_current_to_yesterday,
            e.spot_price_usdt_change_1h,
            e.spot_price_usdt_change_24h,
            e.spot_price_usdt_change_7d,
            e.spot_price_usdt_change_30d,
            e.cmc_price_min_24h,
            e.cmc_price_max_24h,
            e.cmc_price_min_7d,
            e.cmc_price_max_7d

        FROM signals_10min s
        LEFT JOIN signals_10min_enriched e ON s.id = e.signal_id
        WHERE s.timestamp >= DATE_SUB(NOW(), INTERVAL {days} DAY)
            AND s.OI_contracts_binance_change >= {self.signal_threshold['oi_change_min']}
        ORDER BY s.timestamp DESC
        LIMIT {limit}
        """

        print(f"📊 Загрузка сигналов за последние {days} дней...")
        df = pd.read_sql(query, self.engine)
        print(f"✅ Загружено {len(df)} сигналов")

        return df

    def calculate_signal_outcomes(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитывает результаты для каждого сигнала
        """
        print("\n🔄 Расчет результатов сигналов...")

        results = []
        total_signals = len(signals_df)

        for idx, signal in signals_df.iterrows():
            if idx % 50 == 0:
                print(f"  Обработано {idx}/{total_signals} сигналов...", end='\r')

            outcome = self._analyze_signal_outcome(
                signal['token_id'],
                signal['signal_timestamp'],
                signal['signal_price']
            )

            # Объединяем данные сигнала с результатом
            signal_dict = signal.to_dict()
            signal_dict.update(outcome)
            results.append(signal_dict)

        print(f"\n✅ Обработано {total_signals} сигналов")
        return pd.DataFrame(results)

    def _analyze_signal_outcome(self, token_id: int, signal_time: datetime,
                                entry_price: float) -> Dict:
        """
        Анализирует результат конкретного сигнала
        """
        # Получаем данные после сигнала
        query = f"""
        SELECT 
            fd.timestamp,
            fd.price_usd,
            fd.open_interest_usd,
            fd.volume_usd
        FROM futures_data fd
        JOIN futures_pairs fp ON fd.pair_id = fp.id
        WHERE fp.token_id = {token_id}
            AND fd.timestamp > '{signal_time}'
            AND fd.timestamp <= '{signal_time + timedelta(hours=self.signal_threshold['hours_to_analyze'])}'
        ORDER BY fd.timestamp
        LIMIT 500
        """

        try:
            price_data = pd.read_sql(query, self.engine)

            if price_data.empty or entry_price == 0:
                return self._empty_outcome()

            # Рассчитываем P&L
            price_data['pnl_pct'] = ((price_data['price_usd'] - entry_price) / entry_price) * 100

            # Ключевые метрики
            max_profit = price_data['pnl_pct'].max()
            max_drawdown = price_data['pnl_pct'].min()

            # Время до экстремумов
            time_to_max_profit = None
            time_to_max_drawdown = None

            if max_profit > 0:
                max_profit_idx = price_data['pnl_pct'].idxmax()
                time_to_max_profit = (price_data.loc[max_profit_idx, 'timestamp'] - signal_time).total_seconds() / 60

            if max_drawdown < 0:
                max_drawdown_idx = price_data['pnl_pct'].idxmin()
                time_to_max_drawdown = (price_data.loc[
                                            max_drawdown_idx, 'timestamp'] - signal_time).total_seconds() / 60

            # Проверка достижения целей
            hit_profit_target = (price_data['pnl_pct'] >= self.signal_threshold['profit_target']).any()
            hit_stop_loss = (price_data['pnl_pct'] <= -self.signal_threshold['stop_loss']).any()

            # Определяем успешность
            if hit_profit_target and hit_stop_loss:
                # Что было раньше?
                first_target_idx = price_data[price_data['pnl_pct'] >= self.signal_threshold['profit_target']].index[0]
                first_stop_idx = price_data[price_data['pnl_pct'] <= -self.signal_threshold['stop_loss']].index[0]
                is_successful = first_target_idx < first_stop_idx
            else:
                is_successful = hit_profit_target

            # Финальный результат
            final_pnl = price_data.iloc[-1]['pnl_pct'] if not price_data.empty else 0

            # Волатильность
            volatility = price_data['pnl_pct'].std()

            # Анализ объемов
            avg_volume = price_data['volume_usd'].mean()
            volume_trend = 'increasing' if price_data['volume_usd'].iloc[-1] > price_data['volume_usd'].iloc[
                0] else 'decreasing'

            return {
                'max_profit_pct': round(max_profit, 2),
                'max_drawdown_pct': round(max_drawdown, 2),
                'time_to_max_profit_min': round(time_to_max_profit, 0) if time_to_max_profit else None,
                'time_to_max_drawdown_min': round(time_to_max_drawdown, 0) if time_to_max_drawdown else None,
                'final_pnl_pct': round(final_pnl, 2),
                'is_successful': int(is_successful),
                'hit_profit_target': int(hit_profit_target),
                'hit_stop_loss': int(hit_stop_loss),
                'volatility': round(volatility, 2),
                'avg_volume_after': avg_volume,
                'volume_trend': volume_trend,
                'data_points': len(price_data)
            }

        except Exception as e:
            print(f"\n⚠️  Ошибка при анализе сигнала: {e}")
            return self._empty_outcome()

    def _empty_outcome(self) -> Dict:
        """Возвращает пустой результат"""
        return {
            'max_profit_pct': 0,
            'max_drawdown_pct': 0,
            'time_to_max_profit_min': None,
            'time_to_max_drawdown_min': None,
            'final_pnl_pct': 0,
            'is_successful': 0,
            'hit_profit_target': 0,
            'hit_stop_loss': 0,
            'volatility': 0,
            'avg_volume_after': 0,
            'volume_trend': 'unknown',
            'data_points': 0
        }

    def analyze_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Анализирует паттерны успешных и неуспешных сигналов
        """
        print("\n📈 Анализ паттернов...")

        # Разделяем на успешные и неуспешные
        successful = df[df['is_successful'] == 1]
        failed = df[df['is_successful'] == 0]

        analysis = {
            'general_stats': {
                'total_signals': len(df),
                'successful_signals': len(successful),
                'failed_signals': len(failed),
                'success_rate': len(successful) / len(df) * 100 if len(df) > 0 else 0,
                'avg_max_profit': successful['max_profit_pct'].mean() if len(successful) > 0 else 0,
                'avg_max_loss': failed['max_drawdown_pct'].mean() if len(failed) > 0 else 0,
                'avg_time_to_profit': successful['time_to_max_profit_min'].mean() if len(successful) > 0 else 0
            },
            'successful_patterns': self._analyze_group_patterns(successful),
            'failed_patterns': self._analyze_group_patterns(failed),
            'best_tokens': self._find_best_tokens(df),
            'best_time_slots': self._find_best_time_slots(df),
            'correlation_analysis': self._analyze_correlations(df)
        }

        return analysis

    def _analyze_group_patterns(self, group_df: pd.DataFrame) -> Dict:
        """Анализирует паттерны для группы сигналов"""
        if len(group_df) == 0:
            return {}

        return {
            'avg_oi_change_binance': group_df['OI_contracts_binance_change'].mean(),
            'avg_oi_change_bybit': group_df['OI_contracts_bybit_change'].mean(),
            'avg_funding_binance': group_df['funding_rate_binance_now'].mean(),
            'avg_funding_bybit': group_df['funding_rate_bybit_now'].mean(),
            'avg_volume_change': group_df[
                'spot_volume_usdt_change_current_to_average'].mean() if 'spot_volume_usdt_change_current_to_average' in group_df else 0,
            'avg_price_change_1h': group_df[
                'spot_price_usdt_change_1h'].mean() if 'spot_price_usdt_change_1h' in group_df else 0,
            'avg_price_change_24h': group_df[
                'spot_price_usdt_change_24h'].mean() if 'spot_price_usdt_change_24h' in group_df else 0,
            'count': len(group_df)
        }

    def _find_best_tokens(self, df: pd.DataFrame, top_n: int = 10) -> List[Dict]:
        """Находит наиболее успешные токены"""
        token_stats = df.groupby('symbol').agg({
            'is_successful': ['sum', 'count', 'mean'],
            'max_profit_pct': 'mean',
            'max_drawdown_pct': 'mean'
        }).round(2)

        token_stats.columns = ['successful_signals', 'total_signals', 'success_rate', 'avg_max_profit', 'avg_max_loss']
        token_stats['success_rate'] = token_stats['success_rate'] * 100

        # Фильтруем токены с минимум 5 сигналами
        token_stats = token_stats[token_stats['total_signals'] >= 5]

        # Сортируем по успешности
        token_stats = token_stats.sort_values('success_rate', ascending=False).head(top_n)

        return token_stats.reset_index().to_dict('records')

    def _find_best_time_slots(self, df: pd.DataFrame) -> Dict:
        """Находит лучшее время для торговли"""
        df['hour'] = pd.to_datetime(df['signal_timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['signal_timestamp']).dt.dayofweek

        hourly_stats = df.groupby('hour')['is_successful'].agg(['sum', 'count', 'mean'])
        hourly_stats['success_rate'] = hourly_stats['mean'] * 100

        best_hours = hourly_stats.sort_values('success_rate', ascending=False).head(5)
        worst_hours = hourly_stats.sort_values('success_rate', ascending=True).head(5)

        return {
            'best_hours': best_hours.index.tolist(),
            'worst_hours': worst_hours.index.tolist(),
            'hourly_success_rates': hourly_stats['success_rate'].to_dict()
        }

    def _analyze_correlations(self, df: pd.DataFrame) -> Dict:
        """Анализирует корреляции с успешностью"""
        numeric_columns = [
            'OI_contracts_binance_change', 'OI_contracts_bybit_change',
            'funding_rate_binance_now', 'funding_rate_bybit_now',
            'spot_volume_usdt_change_current_to_average',
            'spot_price_usdt_change_1h', 'spot_price_usdt_change_24h'
        ]

        correlations = {}
        for col in numeric_columns:
            if col in df.columns:
                corr = df[col].corr(df['is_successful'])
                if not np.isnan(corr):
                    correlations[col] = round(corr, 3)

        # Сортируем по абсолютному значению
        sorted_corr = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))

        return sorted_corr

    def generate_report(self, analysis: Dict, output_path: str = 'reports'):
        """
        Генерирует отчет с результатами анализа
        """
        os.makedirs(output_path, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Сохраняем JSON отчет
        with open(f'{output_path}/analysis_report_{timestamp}.json', 'w') as f:
            json.dump(analysis, f, indent=2, default=str)

        # Создаем текстовый отчет
        report_lines = [
            "=" * 80,
            "ОТЧЕТ ПО АНАЛИЗУ КРИПТОВАЛЮТНЫХ СИГНАЛОВ",
            f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
            "",
            "ОБЩАЯ СТАТИСТИКА:",
            f"- Всего сигналов: {analysis['general_stats']['total_signals']}",
            f"- Успешных: {analysis['general_stats']['successful_signals']}",
            f"- Неуспешных: {analysis['general_stats']['failed_signals']}",
            f"- Процент успеха: {analysis['general_stats']['success_rate']:.1f}%",
            f"- Средняя макс. прибыль: {analysis['general_stats']['avg_max_profit']:.2f}%",
            f"- Средний макс. убыток: {analysis['general_stats']['avg_max_loss']:.2f}%",
            f"- Среднее время до макс. прибыли: {analysis['general_stats']['avg_time_to_profit']:.0f} мин",
            "",
            "ПАТТЕРНЫ УСПЕШНЫХ СИГНАЛОВ:",
            f"- Средний рост OI (Binance): {analysis['successful_patterns'].get('avg_oi_change_binance', 0):.2f}%",
            f"- Средний рост OI (Bybit): {analysis['successful_patterns'].get('avg_oi_change_bybit', 0):.2f}%",
            f"- Средний funding (Binance): {analysis['successful_patterns'].get('avg_funding_binance', 0):.4f}%",
            f"- Среднее изм. объема: {analysis['successful_patterns'].get('avg_volume_change', 0):.2f}%",
            "",
            "ТОП-5 ТОКЕНОВ ПО УСПЕШНОСТИ:",
        ]

        for i, token in enumerate(analysis['best_tokens'][:5], 1):
            report_lines.append(
                f"{i}. {token['symbol']}: {token['success_rate']:.1f}% успеха "
                f"({token['successful_signals']}/{token['total_signals']} сигналов)"
            )

        report_lines.extend([
            "",
            "ЛУЧШЕЕ ВРЕМЯ ДЛЯ ТОРГОВЛИ (UTC):",
            f"Лучшие часы: {', '.join(map(str, analysis['best_time_slots']['best_hours']))}",
            f"Худшие часы: {', '.join(map(str, analysis['best_time_slots']['worst_hours']))}",
            "",
            "КОРРЕЛЯЦИИ С УСПЕШНОСТЬЮ:",
        ])

        for feature, corr in list(analysis['correlation_analysis'].items())[:5]:
            report_lines.append(f"- {feature}: {corr:.3f}")

        # Сохраняем текстовый отчет
        with open(f'{output_path}/analysis_report_{timestamp}.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"\n📄 Отчеты сохранены в папке '{output_path}/'")
        print('\n'.join(report_lines[:20]) + '\n...')

        return f'{output_path}/analysis_report_{timestamp}'

    def export_for_ml(self, df: pd.DataFrame, output_path: str = 'ml_data'):
        """
        Экспортирует данные для машинного обучения
        """
        os.makedirs(output_path, exist_ok=True)

        # Сохраняем полный датасет
        df.to_csv(f'{output_path}/signals_with_outcomes.csv', index=False)

        # Создаем датасет только с признаками для ML
        feature_columns = [
            'OI_contracts_binance_change', 'OI_contracts_bybit_change',
            'funding_rate_binance_now', 'funding_rate_bybit_now',
            'price_change_10min', 'signal_volume_usd',
            'oi_usdt_change_current_to_average', 'oi_usdt_change_current_to_yesterday',
            'spot_volume_usdt_change_current_to_average',
            'spot_price_usdt_change_1h', 'spot_price_usdt_change_24h',
            'spot_price_usdt_change_7d'
        ]

        # Фильтруем существующие колонки
        existing_features = [col for col in feature_columns if col in df.columns]

        ml_dataset = df[existing_features + ['is_successful']].dropna()
        ml_dataset.to_csv(f'{output_path}/ml_features.csv', index=False)

        print(f"\n💾 Данные для ML сохранены в папке '{output_path}/'")
        print(f"   - Полный датасет: {len(df)} записей")
        print(f"   - ML датасет: {len(ml_dataset)} записей")


def main():
    """Основная функция"""
    print("🚀 Запуск анализатора криптовалютных сигналов...")

    try:
        # Инициализация анализатора
        analyzer = CryptoSignalsAnalyzer()

        # Получаем сигналы за последние 30 дней
        signals = analyzer.get_recent_signals(days=30)

        if len(signals) == 0:
            print("⚠️  Не найдено сигналов для анализа")
            return

        # Рассчитываем результаты
        signals_with_outcomes = analyzer.calculate_signal_outcomes(signals)

        # Анализируем паттерны
        analysis = analyzer.analyze_patterns(signals_with_outcomes)

        # Генерируем отчет
        report_path = analyzer.generate_report(analysis)

        # Экспортируем для ML
        analyzer.export_for_ml(signals_with_outcomes)

        print("\n✅ Анализ завершен успешно!")
        print(f"📊 Проанализировано {len(signals)} сигналов")
        print(f"📈 Успешность: {analysis['general_stats']['success_rate']:.1f}%")

        # Сохраняем результаты для дальнейшего использования
        signals_with_outcomes.to_pickle('latest_analysis.pkl')
        print("\n💾 Результаты сохранены в 'latest_analysis.pkl'")

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()