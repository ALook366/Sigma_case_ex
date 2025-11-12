"""
Скрипт для загрузки обученной модели из GitHub репозитория.
"""
import os
import shutil
from pathlib import Path
import requests
import zipfile
import tempfile

def download_model_from_github(
    repo_url: str = "https://github.com/tabant99-neo/Sigma_case",
    model_path: str = "my_trained_model_2",
    target_dir: Path = None
):
    """
    Загружает модель из GitHub репозитория.
    
    Args:
        repo_url: URL репозитория
        model_path: Путь к модели в репозитории
        target_dir: Целевая директория для сохранения модели
    """
    if target_dir is None:
        # Определяем путь относительно текущего файла (fastApiProject)
        current_file = Path(__file__).resolve()
        # Путь к модели: fastApiProject/Sigma_case/trained_models/current_model
        target_dir = current_file.parent / "Sigma_case" / "trained_models" / "current_model"
    
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Загрузка модели из {repo_url}/{model_path}")
    print(f"Целевая директория: {target_dir}")
    
    # Пробуем скачать через GitHub API
    try:
        # Формируем URL для скачивания архива
        archive_url = f"{repo_url}/archive/refs/heads/main.zip"
        
        print("Скачивание архива репозитория...")
        response = requests.get(archive_url, stream=True)
        response.raise_for_status()
        
        # Сохраняем во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_file_path = tmp_file.name
        
        # Распаковываем архив
        print("Распаковка архива...")
        with zipfile.ZipFile(tmp_file_path, 'r') as zip_ref:
            # Показываем структуру архива для отладки
            all_files = zip_ref.namelist()
            print(f"\nВсего файлов в архиве: {len(all_files)}")
            
            # Находим файлы модели в архиве
            model_files = [f for f in all_files if model_path in f]
            
            print(f"\nНайдено файлов с '{model_path}': {len(model_files)}")
            if model_files:
                print("Первые 10 найденных файлов:")
                for f in model_files[:10]:
                    print(f"  - {f}")
            
            if not model_files:
                print(f"\n⚠️  Модель не найдена в архиве по пути {model_path}")
                print("\nПроверяем структуру архива...")
                # Показываем первые уровни директорий
                top_dirs = set()
                for f in all_files[:50]:
                    parts = f.split('/')
                    if len(parts) > 1:
                        top_dirs.add(parts[0])
                print(f"Корневые директории в архиве: {sorted(top_dirs)}")
                print("\nПопробуйте загрузить модель вручную.")
                os.unlink(tmp_file_path)
                return False
            
            # Извлекаем файлы модели
            extracted_count = 0
            for file_path in model_files:
                # Пропускаем директории
                if file_path.endswith('/'):
                    continue
                
                # Получаем относительный путь
                # Формат может быть: Sigma_case-main/my_trained_model_2/file.txt
                # или: my_trained_model_2/file.txt
                parts = file_path.split('/')
                
                # Находим индекс папки модели
                try:
                    model_index = next(i for i, part in enumerate(parts) if model_path in part)
                    # Берем все части после папки модели
                    relative_path = '/'.join(parts[model_index + 1:])
                except StopIteration:
                    # Если не нашли, берем последнюю часть
                    relative_path = parts[-1]
                
                if not relative_path:
                    continue
                
                target_file = target_dir / relative_path
                target_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Извлекаем файл
                try:
                    with zip_ref.open(file_path) as source, open(target_file, 'wb') as target:
                        content = source.read()
                        target.write(content)
                    
                    extracted_count += 1
                    file_size = len(content)
                    print(f"✓ Извлечен: {relative_path} ({file_size} байт)")
                except Exception as e:
                    print(f"⚠️  Ошибка при извлечении {relative_path}: {e}")
            
            print(f"\nИзвлечено файлов: {extracted_count} из {len([f for f in model_files if not f.endswith('/')])}")
        
        # Удаляем временный файл
        os.unlink(tmp_file_path)
        
        # Проверяем, что файлы действительно были сохранены
        saved_files = list(target_dir.glob('*'))
        if saved_files:
            print(f"\n✅ Модель успешно загружена в {target_dir}")
            print(f"Сохранено файлов: {len(saved_files)}")
            print("Список файлов:")
            for f in saved_files:
                size = f.stat().st_size if f.is_file() else 0
                print(f"  - {f.name} ({size} байт)")
            return True
        else:
            print(f"\n⚠️  Файлы не были сохранены в {target_dir}")
            return False
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка при загрузке: {e}")
        print("\nАльтернативный способ:")
        print(f"1. Клонируйте репозиторий: git clone {repo_url}")
        print(f"2. Скопируйте файлы из {model_path} в {target_dir}")
        return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_model_manual_instructions():
    """Выводит инструкции для ручной загрузки модели."""
    current_file = Path(__file__).resolve()
    # Путь к модели: fastApiProject/Sigma_case/trained_models/current_model
    target_dir = current_file.parent / "Sigma_case" / "trained_models" / "current_model"
    
    print("\n" + "="*60)
    print("ИНСТРУКЦИЯ ПО РУЧНОЙ ЗАГРУЗКЕ МОДЕЛИ")
    print("="*60)
    print(f"\n1. Откройте репозиторий: https://github.com/tabant99-neo/Sigma_case")
    print(f"2. Перейдите в папку: my_trained_model_2")
    print(f"3. Скачайте все файлы из этой папки")
    print(f"4. Поместите их в директорию:")
    print(f"   {target_dir}")
    print(f"\nТребуемые файлы:")
    print("   - config.json")
    print("   - pytorch_model.bin (или model.bin, model.pt, model.pth)")
    print("   - tokenizer_config.json")
    print("   - vocab.txt (или vocab.json)")
    print("   - merges.txt (опционально, для BPE)")
    print("\n" + "="*60)


if __name__ == "__main__":
    import sys
    
    print("Загрузка обученной модели из GitHub...")
    print("-" * 60)
    
    success = download_model_from_github()
    
    if not success:
        download_model_manual_instructions()
        sys.exit(1)
    
    print("\n✅ Готово! Модель загружена и готова к использованию.")

