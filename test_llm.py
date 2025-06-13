import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List
import os
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()


class LLMPromptGenerator:
    """
    Генератор промптов для анализа криптовалютных сигналов с помощью LLM
    """

    def __init__(self, signals_data: pd.DataFrame = None):
        self.df = signals_data

    def generate_analysis_prompt(self, signal_data: Dict) -> str:
        """
        Генерирует промпт для анализа конкретного сигнала
        """
        prompt = f"""
Analyze the following cryptocurrency trading signal for {signal_data.get('symbol', 'UNKNOWN')}:

SIGNAL METRICS:
- Price: ${signal_data.get('signal_price', 0):.4f}
- 10-min price change: {signal_data.get('price_change_10min', 0):.2f}%
- Open Interest Change (Binance): {signal_data.get('OI_contracts_binance_change', 0):.2f}%
- Open Interest Change (Bybit): {signal_data.get('OI_contracts_bybit_change', 0):.2f}%
- Funding Rate (Binance): {signal_data.get('funding_rate_binance_now', 0):.4f}%
- Funding Rate (Bybit): {signal_data.get('funding_rate_bybit_now', 0):.4f}%

MARKET CONTEXT:
- OI change vs average: {signal_data.get('oi_usdt_change_current_to_average', 0):.2f}%
- Volume change vs average: {signal_data.get('spot_volume_usdt_change_current_to_average', 0):.2f}%
- 1h price change: {signal_data.get('spot_price_usdt_change_1h', 0):.2f}%
- 24h price change: {signal_data.get('spot_price_usdt_change_24h', 0):.2f}%
- 7d price change: {signal_data.get('spot_price_usdt_change_7d', 0):.2f}%

Based on these metrics, provide:
1. Signal strength assessment (1-10 scale)
2. Key bullish factors
3. Key risk factors
4. Recommended action (Strong Buy/Buy/Hold/Avoid)
5. Suggested stop-loss and take-profit levels
"""
        return prompt

    def generate_pattern_learning_prompt(self, successful_signals: List[Dict],
                                         failed_signals: List[Dict]) -> str:
        """
        Генерирует промпт для обучения на паттернах
        """
        # Преобразуем в DataFrame если это список словарей
        if isinstance(successful_signals, list):
            success_df = pd.DataFrame(successful_signals)
        else:
            success_df = successful_signals

        if isinstance(failed_signals, list):
            fail_df = pd.DataFrame(failed_signals)
        else:
            fail_df = failed_signals

        # Рассчитываем средние значения для успешных сигналов
        success_stats = {
            'avg_oi_change': success_df[
                'OI_contracts_binance_change'].mean() if 'OI_contracts_binance_change' in success_df else 0,
            'avg_volume_change': success_df.get('spot_volume_usdt_change_current_to_average', 0).mean(),
            'avg_funding': success_df[
                'funding_rate_binance_now'].mean() if 'funding_rate_binance_now' in success_df else 0,
            'avg_1h_change': success_df.get('spot_price_usdt_change_1h', 0).mean()
        }

        # Рассчитываем средние значения для неуспешных сигналов
        fail_stats = {
            'avg_oi_change': fail_df[
                'OI_contracts_binance_change'].mean() if 'OI_contracts_binance_change' in fail_df else 0,
            'avg_volume_change': fail_df.get('spot_volume_usdt_change_current_to_average', 0).mean(),
            'avg_funding': fail_df['funding_rate_binance_now'].mean() if 'funding_rate_binance_now' in fail_df else 0,
            'avg_1h_change': fail_df.get('spot_price_usdt_change_1h', 0).mean()
        }

        prompt = f"""
Learn from these historical trading patterns:

SUCCESSFUL SIGNALS (reached 5% profit without 3% drawdown):
- Average OI Change: {success_stats['avg_oi_change']:.2f}%
- Average Volume Change vs Average: {success_stats['avg_volume_change']:.2f}%
- Average Funding Rate: {success_stats['avg_funding']:.4f}%
- Average 1h Price Change: {success_stats['avg_1h_change']:.2f}%
- Total successful signals: {len(success_df)}

FAILED SIGNALS:
- Average OI Change: {fail_stats['avg_oi_change']:.2f}%
- Average Volume Change vs Average: {fail_stats['avg_volume_change']:.2f}%
- Average Funding Rate: {fail_stats['avg_funding']:.4f}%
- Average 1h Price Change: {fail_stats['avg_1h_change']:.2f}%
- Total failed signals: {len(fail_df)}

Based on these patterns, create a decision framework that:
1. Identifies the most important differentiating factors
2. Provides specific thresholds for each metric
3. Suggests a scoring system for signal evaluation
4. Recommends position sizing based on signal strength
"""
        return prompt

    def generate_market_regime_prompt(self, recent_data: pd.DataFrame) -> str:
        """
        Генерирует промпт для определения рыночного режима
        """
        # Анализ последних сигналов
        recent_success_rate = recent_data['is_successful'].mean() * 100 if 'is_successful' in recent_data else 0
        avg_volatility = recent_data.get('volatility_after_signal', 0).mean()

        # Топ токены
        top_tokens = []
        if 'symbol' in recent_data:
            top_tokens = recent_data['symbol'].value_counts().head(5).index.tolist()

        prompt = f"""
Analyze the current market regime based on recent trading signals:

RECENT PERFORMANCE (last {len(recent_data)} signals):
- Success rate: {recent_success_rate:.1f}%
- Average volatility after signals: {avg_volatility:.2f}%
- Trending tokens: {', '.join(top_tokens) if top_tokens else 'N/A'}

MARKET INDICATORS:
- Dominant funding rates: {'Positive' if recent_data.get('funding_rate_binance_now', pd.Series([0])).mean() > 0 else 'Negative'}
- Volume trends: {'Increasing' if recent_data.get('spot_volume_usdt_change_current_to_average', pd.Series([0])).mean() > 0 else 'Decreasing'}
- OI trends: {'Expanding' if recent_data.get('OI_contracts_binance_change', pd.Series([0])).mean() > 0 else 'Contracting'}

Provide:
1. Current market regime classification (Bullish/Bearish/Ranging/Volatile)
2. Recommended adjustments to signal thresholds
3. Tokens to focus on in current conditions
4. Risk management suggestions for this regime
"""
        return prompt


# Функция для тестирования
def test_llm_prompts():
    """
    Тестирует генерацию промптов для LLM
    """
    print("=== Тестирование генератора промптов для LLM ===\n")

    # Создаем тестовые данные
    test_signal = {
        'symbol': 'BTCUSDT',
        'signal_price': 45000.0,
        'price_change_10min': 2.5,
        'OI_contracts_binance_change': 5.2,
        'OI_contracts_bybit_change': 4.8,
        'funding_rate_binance_now': 0.0234,
        'funding_rate_bybit_now': 0.0256,
        'oi_usdt_change_current_to_average': 15.3,
        'spot_volume_usdt_change_current_to_average': 45.7,
        'spot_price_usdt_change_1h': 1.2,
        'spot_price_usdt_change_24h': 3.5,
        'spot_price_usdt_change_7d': -2.1
    }

    # Инициализируем генератор
    generator = LLMPromptGenerator()

    # Тест 1: Анализ одного сигнала
    print("1. Тест генерации промпта для анализа сигнала:")
    print("-" * 50)
    analysis_prompt = generator.generate_analysis_prompt(test_signal)
    print(analysis_prompt[:500] + "...\n")

    # Тест 2: Паттерны успешных/неуспешных сигналов
    print("2. Тест генерации промпта для обучения на паттернах:")
    print("-" * 50)

    # Создаем тестовые данные для успешных и неуспешных сигналов
    successful = [
        {
            'OI_contracts_binance_change': 6.5,
            'spot_volume_usdt_change_current_to_average': 80.0,
            'funding_rate_binance_now': 0.025,
            'spot_price_usdt_change_1h': 2.1
        },
        {
            'OI_contracts_binance_change': 5.8,
            'spot_volume_usdt_change_current_to_average': 65.0,
            'funding_rate_binance_now': 0.032,
            'spot_price_usdt_change_1h': 1.8
        }
    ]

    failed = [
        {
            'OI_contracts_binance_change': 2.1,
            'spot_volume_usdt_change_current_to_average': 15.0,
            'funding_rate_binance_now': 0.085,
            'spot_price_usdt_change_1h': -0.5
        },
        {
            'OI_contracts_binance_change': 1.5,
            'spot_volume_usdt_change_current_to_average': 10.0,
            'funding_rate_binance_now': 0.095,
            'spot_price_usdt_change_1h': -1.2
        }
    ]

    pattern_prompt = generator.generate_pattern_learning_prompt(successful, failed)
    print(pattern_prompt[:500] + "...\n")

    # Тест 3: Анализ рыночного режима
    print("3. Тест генерации промпта для анализа рыночного режима:")
    print("-" * 50)

    recent_data = pd.DataFrame({
        'is_successful': [1, 0, 1, 1, 0, 1, 0, 1],
        'volatility_after_signal': [2.5, 3.2, 1.8, 2.1, 4.5, 1.9, 3.8, 2.2],
        'symbol': ['BTCUSDT', 'ETHUSDT', 'BTCUSDT', 'SOLUSDT', 'ETHUSDT', 'BTCUSDT', 'ADAUSDT', 'SOLUSDT'],
        'funding_rate_binance_now': [0.025, 0.032, 0.028, 0.045, 0.055, 0.021, 0.038, 0.041],
        'spot_volume_usdt_change_current_to_average': [45, -12, 38, 65, -25, 52, 15, 48],
        'OI_contracts_binance_change': [3.2, -1.5, 4.8, 6.2, -2.8, 5.1, 1.2, 4.5]
    })

    regime_prompt = generator.generate_market_regime_prompt(recent_data)
    print(regime_prompt[:500] + "...\n")

    print("=== Все тесты пройдены успешно! ===")

    # Сохраняем примеры промптов
    output_dir = "llm_prompts"
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/signal_analysis_prompt.txt", "w") as f:
        f.write(analysis_prompt)

    with open(f"{output_dir}/pattern_learning_prompt.txt", "w") as f:
        f.write(pattern_prompt)

    with open(f"{output_dir}/market_regime_prompt.txt", "w") as f:
        f.write(regime_prompt)

    print(f"\nПромпты сохранены в директории '{output_dir}/'")

    # Создаем JSON с примерами для интеграции с API
    api_examples = {
        "signal_analysis": {
            "input": test_signal,
            "prompt": analysis_prompt,
            "expected_output_structure": {
                "signal_strength": "1-10",
                "bullish_factors": ["factor1", "factor2"],
                "risk_factors": ["risk1", "risk2"],
                "recommendation": "Strong Buy/Buy/Hold/Avoid",
                "stop_loss": "price",
                "take_profit": "price"
            }
        },
        "pattern_learning": {
            "input": {
                "successful_signals": successful,
                "failed_signals": failed
            },
            "prompt": pattern_prompt
        },
        "market_regime": {
            "input": recent_data.to_dict('records'),
            "prompt": regime_prompt
        }
    }

    with open(f"{output_dir}/api_integration_examples.json", "w") as f:
        json.dump(api_examples, f, indent=2)

    print(f"Примеры для API интеграции сохранены в '{output_dir}/api_integration_examples.json'")


if __name__ == "__main__":
    test_llm_prompts()