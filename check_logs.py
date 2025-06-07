#!/usr/bin/env python3
"""Check recent error logs."""
import json
from pathlib import Path
from datetime import datetime


def check_recent_logs():
    """Check recent error logs."""
    log_file = Path('crypto_signals_enrichment.log')

    if not log_file.exists():
        print("❌ Лог файл не найден")
        return

    print("📋 Последние ошибки из лога:")
    print("=" * 60)

    errors = []

    with open(log_file, 'r') as f:
        for line in f:
            try:
                log_entry = json.loads(line)
                if log_entry.get('level') == 'ERROR' and 'coinmarketcap' in log_entry.get('logger', ''):
                    errors.append(log_entry)
            except:
                pass

    # Показываем последние 5 ошибок
    for error in errors[-5:]:
        print(f"\n⚠️  {error.get('timestamp', 'N/A')}")
        print(f"   Message: {error.get('message', 'N/A')}")
        print(f"   URL: {error.get('url', 'N/A')}")

        # Показываем детали ответа если есть
        if 'response_json' in error:
            resp = error['response_json']
            if 'status' in resp:
                print(f"   API Error: {resp['status'].get('error_message', 'Unknown')}")
                print(f"   Error Code: {resp['status'].get('error_code', 'N/A')}")
        elif 'response_text' in error:
            print(f"   Response: {error['response_text']}")

        if 'status_code' in error:
            print(f"   HTTP Status: {error['status_code']}")


if __name__ == "__main__":
    check_recent_logs()