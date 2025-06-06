#!/usr/bin/env python3
"""Check project structure and create missing files."""
import os
from pathlib import Path

# Структура проекта
PROJECT_STRUCTURE = {
    'api_clients': ['__init__.py', 'base.py', 'binance_client.py', 'bybit_client.py'],
    'config': ['__init__.py', 'settings.py'],
    'database': ['__init__.py', 'connection.py', 'models.py'],
    'modules': ['__init__.py', 'oi_processor.py'],
    'tests': ['__init__.py', 'test_oi_processor.py'],
    'utils': ['__init__.py', 'logger.py', 'helpers.py'],
}

ROOT_FILES = ['.env.example', 'requirements.txt', 'main.py', 'pytest.ini', 'conftest.py']


def check_structure():
    """Check and report project structure."""
    print("🔍 Проверка структуры проекта...\n")

    missing_dirs = []
    missing_files = []

    # Проверяем директории и файлы
    for directory, files in PROJECT_STRUCTURE.items():
        dir_path = Path(directory)

        if not dir_path.exists():
            missing_dirs.append(directory)
            print(f"❌ Директория отсутствует: {directory}/")
        else:
            print(f"✅ Директория существует: {directory}/")

        for file in files:
            file_path = dir_path / file
            if not file_path.exists():
                missing_files.append(str(file_path))
                print(f"   ❌ Файл отсутствует: {file_path}")
            else:
                print(f"   ✅ Файл существует: {file_path}")

    # Проверяем корневые файлы
    print("\n📁 Корневые файлы:")
    for file in ROOT_FILES:
        if not Path(file).exists():
            missing_files.append(file)
            print(f"❌ Файл отсутствует: {file}")
        else:
            print(f"✅ Файл существует: {file}")

    # Итоги
    print("\n" + "=" * 50)
    if missing_dirs or missing_files:
        print("❗ Обнаружены отсутствующие элементы:")
        if missing_dirs:
            print(f"\nОтсутствующие директории: {len(missing_dirs)}")
            for d in missing_dirs:
                print(f"  - {d}/")
        if missing_files:
            print(f"\nОтсутствующие файлы: {len(missing_files)}")
            for f in missing_files:
                print(f"  - {f}")

        print("\n💡 Создайте отсутствующие файлы из предоставленного кода.")
    else:
        print("✅ Все файлы и директории на месте!")
        print("\n🚀 Готово к запуску:")
        print("   python main.py")
        print("\n🧪 Для запуска тестов:")
        print("   pytest tests/ -v")
        print("\n📊 Для запуска тестов с покрытием:")
        print("   pytest tests/ --cov=modules --cov-report=html")


if __name__ == "__main__":
    check_structure()