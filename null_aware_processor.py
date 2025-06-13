#!/usr/bin/env python3
"""
Null-Aware Enhanced Processor - Интеллектуальная обработка NULL как информативных признаков
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class NullAwareFeatureEngineering:
    """
    Класс для интеллектуальной обработки NULL значений как информативных признаков
    """

    def __init__(self):
        # Группы признаков, где NULL имеет смысловое значение
        self.btc_volume_features = [
            'spot_volume_btc_average', 'spot_volume_btc_current',
            'spot_volume_btc_yesterday', 'spot_volume_btc_change_current_to_yesterday',
            'spot_volume_btc_change_current_to_average'
        ]

        self.spot_features = [
            'spot_volume_usdt_average', 'spot_volume_usdt_current',
            'spot_volume_usdt_yesterday', 'spot_volume_usdt_change_current_to_yesterday',
            'spot_volume_usdt_change_current_to_average', 'spot_price_usdt_average',
            'spot_price_usdt_current', 'spot_price_usdt_yesterday',
            'spot_price_usdt_change_1h', 'spot_price_usdt_change_24h',
            'spot_price_usdt_change_7d', 'spot_price_usdt_change_30d'
        ]

    def create_null_indicator_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создает индикаторные признаки для NULL значений
        """
        print("🔍 Создание индикаторных признаков для NULL значений...")

        # 1. Индикатор наличия BTC пары
        df['has_btc_pair'] = (~df['spot_volume_btc_current'].isna()).astype(int)

        # 2. Индикатор наличия спотовой торговли
        df['has_spot_trading'] = (~df['spot_volume_usdt_current'].isna()).astype(int)

        # 3. Тип торговли токена
        df['trading_type'] = 0  # По умолчанию
        df.loc[df['has_spot_trading'] == 1, 'trading_type'] = 1  # Только спот
        # ИСПРАВЛЕНО: 'volume_usd' заменено на 'signal_volume_usd'
        df.loc[(df['has_spot_trading'] == 1) & (~df['signal_volume_usd'].isna()), 'trading_type'] = 2  # Спот + фьючерсы
        # ИСПРАВЛЕНО: 'volume_usd' заменено на 'signal_volume_usd'
        df.loc[(df['has_spot_trading'] == 0) & (~df['signal_volume_usd'].isna()), 'trading_type'] = 3  # Только фьючерсы

        # 4. Категория токена по объему BTC
        df['btc_volume_category'] = 0  # Нет BTC пары

        # Для токенов с BTC парой категоризируем по объему
        btc_mask = df['has_btc_pair'] == 1
        if btc_mask.any():
            btc_volumes = df.loc[btc_mask, 'spot_volume_btc_current']
            # Квантили для категоризации
            q25 = btc_volumes.quantile(0.25)
            q50 = btc_volumes.quantile(0.50)
            q75 = btc_volumes.quantile(0.75)

            df.loc[btc_mask & (df['spot_volume_btc_current'] <= q25), 'btc_volume_category'] = 1  # Низкий
            df.loc[btc_mask & (df['spot_volume_btc_current'] > q25) & (
                    df['spot_volume_btc_current'] <= q50), 'btc_volume_category'] = 2  # Средний
            df.loc[btc_mask & (df['spot_volume_btc_current'] > q50) & (
                    df['spot_volume_btc_current'] <= q75), 'btc_volume_category'] = 3  # Высокий
            df.loc[btc_mask & (df['spot_volume_btc_current'] > q75), 'btc_volume_category'] = 4  # Очень высокий

        # 5. Полнота данных
        spot_cols = ['spot_volume_usdt_current', 'spot_price_usdt_current',
                     'spot_price_usdt_change_1h', 'spot_price_usdt_change_24h']
        df['spot_data_completeness'] = df[spot_cols].notna().sum(axis=1) / len(spot_cols)

        # 6. Источник данных агрегированный
        df['data_source_diversity'] = 0
        source_cols = ['oi_source_usdt', 'spot_volume_source_usdt',
                       'spot_volume_source_btc', 'spot_price_source_usdt']

        for col in source_cols:
            if col in df.columns:
                df['data_source_diversity'] += df[col].notna().astype(int)

        print(f"✅ Создано индикаторных признаков: 6")
        print(f"   - Токенов с BTC парой: {df['has_btc_pair'].sum()} ({df['has_btc_pair'].mean() * 100:.1f}%)")
        print(
            f"   - Токенов со спотовой торговлей: {df['has_spot_trading'].sum()} ({df['has_spot_trading'].mean() * 100:.1f}%)")

        return df

    def smart_fill_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Интеллектуальное заполнение NULL значений с сохранением их информативности
        """
        print("\n🧠 Интеллектуальное заполнение NULL значений...")

        # Для BTC volume features - заполняем специальным значением, если нет BTC пары
        for col in self.btc_volume_features:
            if col in df.columns:
                # Где есть BTC пара - заполняем медианой
                btc_median = df.loc[df['has_btc_pair'] == 1, col].median()
                df.loc[(df['has_btc_pair'] == 1) & df[col].isna(), col] = btc_median

                # Где нет BTC пары - заполняем -1 (специальное значение)
                df.loc[df['has_btc_pair'] == 0, col] = -1

        # Для спотовых features
        for col in self.spot_features:
            if col in df.columns:
                # Где есть спот - заполняем медианой
                spot_median = df.loc[df['has_spot_trading'] == 1, col].median()
                df.loc[(df['has_spot_trading'] == 1) & df[col].isna(), col] = spot_median

                # Где нет спота - заполняем -999 (специальное значение)
                df.loc[df['has_spot_trading'] == 0, col] = -999

        # Для процентных изменений где нет спота - логично поставить 0
        change_cols = [col for col in df.columns if 'change' in col and col in self.spot_features]
        for col in change_cols:
            df.loc[df['has_spot_trading'] == 0, col] = 0

        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создает признаки взаимодействия на основе типа торговли
        """
        print("\n🔄 Создание признаков взаимодействия...")

        # 1. Соотношение фьючерсного объема к спотовому (если есть спот)
        df['futures_to_spot_volume_ratio'] = np.where(
            (df['has_spot_trading'] == 1) & (df['spot_volume_usdt_current'] > 0),
            df['signal_volume_usd'] / df['spot_volume_usdt_current'],
            -1  # Специальное значение для отсутствующего спота
        )

        # 2. Премия фьючерсов к споту
        df['futures_spot_premium'] = np.where(
            (df['has_spot_trading'] == 1) & (df['spot_price_usdt_current'] > 0),
            (df['signal_price'] - df['spot_price_usdt_current']) / df['spot_price_usdt_current'] * 100,
            0
        )

        # 3. Относительная сила движения для токенов с BTC парой
        df['btc_relative_strength'] = np.where(
            df['has_btc_pair'] == 1,
            df['spot_volume_btc_change_current_to_average'] - df['spot_volume_usdt_change_current_to_average'],
            0
        )

        # 4. Ликвидность токена (комбинированная метрика)
        df['liquidity_score'] = (
                df['has_spot_trading'] * 0.5 +
                df['has_btc_pair'] * 0.3 +
                (df['data_source_diversity'] / 4) * 0.2
        )

        # 5. Риск токена (обратная ликвидности)
        df['token_risk_score'] = 1 - df['liquidity_score']

        return df

    def analyze_null_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Анализирует паттерны NULL значений и их связь с успешностью
        """
        print("\n📊 Анализ паттернов NULL значений...")

        analysis = {}

        if 'is_successful_main' in df.columns:
            # Успешность по типам торговли
            trading_type_success = df.groupby('trading_type')['is_successful_main'].agg(['mean', 'count'])
            trading_type_names = {
                0: 'Неизвестно',
                1: 'Только спот',
                2: 'Спот + Фьючерсы',
                3: 'Только фьючерсы'
            }

            analysis['trading_type_success'] = {}
            for idx, row in trading_type_success.iterrows():
                type_name = trading_type_names.get(idx, f'Тип {idx}')
                analysis['trading_type_success'][type_name] = {
                    'success_rate': row['mean'] * 100,
                    'count': row['count']
                }

            # Успешность для токенов с/без BTC пары
            analysis['btc_pair_impact'] = {
                'with_btc_pair': df[df['has_btc_pair'] == 1]['is_successful_main'].mean() * 100,
                'without_btc_pair': df[df['has_btc_pair'] == 0]['is_successful_main'].mean() * 100,
                'count_with_btc': (df['has_btc_pair'] == 1).sum(),
                'count_without_btc': (df['has_btc_pair'] == 0).sum()
            }

            # Успешность по категориям объема BTC
            btc_volume_success = df.groupby('btc_volume_category')['is_successful_main'].agg(['mean', 'count'])
            volume_category_names = {
                0: 'Нет BTC пары',
                1: 'Низкий объем BTC',
                2: 'Средний объем BTC',
                3: 'Высокий объем BTC',
                4: 'Очень высокий объем BTC'
            }

            analysis['btc_volume_category_success'] = {}
            for idx, row in btc_volume_success.iterrows():
                cat_name = volume_category_names.get(idx, f'Категория {idx}')
                analysis['btc_volume_category_success'][cat_name] = {
                    'success_rate': row['mean'] * 100,
                    'count': row['count']
                }

        # Статистика по NULL значениям
        null_stats = {}
        for col in self.btc_volume_features + self.spot_features:
            if col in df.columns:
                null_stats[col] = {
                    'null_count': df[col].isna().sum(),
                    'null_percentage': df[col].isna().mean() * 100
                }

        analysis['null_statistics'] = null_stats

        return analysis


def enhance_processor_with_null_awareness(processor_class):
    """
    Декоратор для добавления null-aware функциональности к существующему процессору
    """

    class NullAwareProcessor(processor_class):
        def __init__(self):
            super().__init__()
            self.null_handler = NullAwareFeatureEngineering()

            # Добавляем новые признаки к списку
            self.null_indicator_features = [
                'has_btc_pair', 'has_spot_trading', 'trading_type',
                'btc_volume_category', 'spot_data_completeness',
                'data_source_diversity', 'futures_to_spot_volume_ratio',
                'futures_spot_premium', 'btc_relative_strength',
                'liquidity_score', 'token_risk_score'
            ]

        def _add_enhanced_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
            """Переопределяем метод для добавления null-aware признаков"""
            # Сначала вызываем оригинальный метод
            df = super()._add_enhanced_technical_features(df)

            # Добавляем null-aware признаки
            df = self.null_handler.create_null_indicator_features(df)
            df = self.null_handler.smart_fill_nulls(df)
            df = self.null_handler.create_interaction_features(df)

            return df

        def prepare_full_feature_set(self, df: pd.DataFrame,
                                     target_column: str = 'is_successful_main') -> Tuple[
            pd.DataFrame, pd.Series, List[str]]:
            """Обновляем список признаков"""
            # Вызываем родительский метод
            X, y, features = super().prepare_full_feature_set(df, target_column)

            # Добавляем null-indicator признаки
            null_features = [f for f in self.null_indicator_features if f in df.columns]

            # Обновляем X с новыми признаками
            if null_features:
                X_enhanced = df.loc[X.index, features + null_features]

                print(f"\n✅ Добавлено NULL-aware признаков: {len(null_features)}")
                print(f"   Итого признаков: {len(features + null_features)}")

                return X_enhanced, y, features + null_features

            return X, y, features

        def generate_comprehensive_report(self, df: pd.DataFrame, results: Dict,
                                          group_importance: pd.DataFrame) -> str:
            """Дополняем отчет анализом NULL паттернов"""
            # Сначала генерируем базовый отчет
            report_path = super().generate_comprehensive_report(df, results, group_importance)

            # Анализируем NULL паттерны
            null_analysis = self.null_handler.analyze_null_patterns(df)

            # Дополняем отчет
            with open(report_path, 'a', encoding='utf-8') as f:
                f.write("\n\n" + "=" * 100 + "\n")
                f.write("АНАЛИЗ ПАТТЕРНОВ NULL ЗНАЧЕНИЙ:\n")
                f.write("=" * 100 + "\n")

                if 'trading_type_success' in null_analysis:
                    f.write("\nУСПЕШНОСТЬ ПО ТИПАМ ТОРГОВЛИ:\n")
                    for trading_type, stats in null_analysis['trading_type_success'].items():
                        f.write(f"- {trading_type}: {stats['success_rate']:.1f}% успеха ({stats['count']} сигналов)\n")

                if 'btc_pair_impact' in null_analysis:
                    f.write("\nВЛИЯНИЕ НАЛИЧИЯ BTC ПАРЫ:\n")
                    btc_impact = null_analysis['btc_pair_impact']
                    f.write(
                        f"- С BTC парой: {btc_impact['with_btc_pair']:.1f}% успеха ({btc_impact['count_with_btc']} сигналов)\n")
                    f.write(
                        f"- Без BTC пары: {btc_impact['without_btc_pair']:.1f}% успеха ({btc_impact['count_without_btc']} сигналов)\n")

                if 'btc_volume_category_success' in null_analysis:
                    f.write("\nУСПЕШНОСТЬ ПО КАТЕГОРИЯМ ОБЪЕМА BTC:\n")
                    for category, stats in null_analysis['btc_volume_category_success'].items():
                        f.write(f"- {category}: {stats['success_rate']:.1f}% успеха ({stats['count']} сигналов)\n")

                # Рекомендации на основе анализа
                f.write("\nРЕКОМЕНДАЦИИ НА ОСНОВЕ NULL АНАЛИЗА:\n")

                # Проверяем какой тип торговли наиболее успешен
                if 'trading_type_success' in null_analysis:
                    best_type = max(null_analysis['trading_type_success'].items(),
                                    key=lambda x: x[1]['success_rate'])
                    f.write(
                        f"✅ Наиболее успешный тип торговли: {best_type[0]} ({best_type[1]['success_rate']:.1f}% успеха)\n")

                # Проверяем влияние BTC пары
                if 'btc_pair_impact' in null_analysis:
                    btc_impact = null_analysis['btc_pair_impact']
                    if btc_impact['with_btc_pair'] > btc_impact['without_btc_pair']:
                        f.write("✅ Токены с BTC парой показывают лучшую производительность\n")
                    else:
                        f.write("⚠️  Наличие BTC пары не является определяющим фактором успеха\n")

            return report_path

    return NullAwareProcessor


# Пример использования с существующим процессором
if __name__ == "__main__":
    # Импортируем базовый процессор
    from enhanced_full_processor import EnhancedFullSignalsProcessor

    # Создаем null-aware версию
    NullAwareFullProcessor = enhance_processor_with_null_awareness(EnhancedFullSignalsProcessor)

    # Используем как обычно
    processor = NullAwareFullProcessor()

    # Дальше работаем как с обычным процессором
    # processor.get_all_signals_complete()
    # и т.д.