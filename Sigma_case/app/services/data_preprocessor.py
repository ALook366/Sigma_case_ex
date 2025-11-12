import pandas as pd
import re
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import io


def clean_html_from_csv(file_path: str, delimiter: str = None) -> str:
    """
    Очищает CSV файл от HTML тегов и приводит к нормальному CSV виду.
    Сохраняет структуру CSV, удаляя только HTML теги из значений.
    
    Args:
        file_path: Путь к исходному CSV файлу с HTML тегами
        delimiter: Разделитель CSV (; или ,). Если None, определяется автоматически.
        
    Returns:
        Строка с очищенным CSV содержимым
    """
    # Определяем разделитель если не указан
    if delimiter is None:
        delimiter = detect_delimiter(file_path)
    
    # Читаем файл с учетом возможной кодировки
    encodings = ['utf-8', 'cp1251', 'windows-1251', 'latin-1']
    lines = None
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue
    
    if lines is None:
        raise ValueError(f"Не удалось прочитать файл {file_path} с поддерживаемыми кодировками")
    
    def clean_html_from_text(text: str) -> str:
        """Удаляет HTML теги из текста, сохраняя содержимое."""
        if not text or '<' not in text and '>' not in text:
            return text
        
        # Используем BeautifulSoup для удаления HTML тегов
        soup = BeautifulSoup(text, 'html.parser')
        cleaned = soup.get_text()
        
        # Нормализуем пробелы (множественные пробелы заменяем на один)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()
    
    def clean_csv_line(line: str, delimiter: str = ';') -> str:
        """
        Очищает одну строку CSV от HTML тегов, сохраняя структуру CSV.
        Обрабатывает значения в кавычках и без кавычек.
        Поддерживает разные разделители (; или ,).
        """
        line = line.rstrip('\n\r')
        if not line.strip():
            return line
        
        # Разделяем строку на поля, учитывая кавычки
        fields = []
        current_field = ""
        in_quotes = False
        i = 0
        
        while i < len(line):
            char = line[i]
            
            if char == '"':
                if in_quotes and i + 1 < len(line) and line[i + 1] == '"':
                    # Экранированная кавычка ""
                    current_field += '""'
                    i += 2
                    continue
                else:
                    # Начало или конец значения в кавычках
                    in_quotes = not in_quotes
                    current_field += char
                    i += 1
            elif (char == delimiter or char == ',') and not in_quotes:
                # Разделитель полей (поддерживаем оба варианта)
                fields.append(current_field)
                current_field = ""
                i += 1
            else:
                current_field += char
                i += 1
        
        # Добавляем последнее поле
        if current_field or line.endswith(delimiter) or line.endswith(','):
            fields.append(current_field)
        
        # Очищаем каждое поле от HTML тегов
        cleaned_fields = []
        for field in fields:
            if field.startswith('"') and field.endswith('"'):
                # Значение в кавычках - очищаем содержимое от HTML
                inner = field[1:-1]
                # Обрабатываем экранированные кавычки
                inner = inner.replace('""', '"')
                inner_cleaned = clean_html_from_text(inner)
                # Восстанавливаем экранированные кавычки если нужно
                inner_cleaned = inner_cleaned.replace('"', '""')
                cleaned_fields.append(f'"{inner_cleaned}"')
            else:
                # Значение без кавычек - просто очищаем от HTML
                cleaned_fields.append(clean_html_from_text(field))
        
        return delimiter.join(cleaned_fields)
    
    # Обрабатываем каждую строку
    cleaned_lines = []
    for line in lines:
        cleaned_line = clean_csv_line(line, delimiter=delimiter)
        if cleaned_line.strip():  # Пропускаем полностью пустые строки
            cleaned_lines.append(cleaned_line)
    
    return '\n'.join(cleaned_lines)


def detect_delimiter(file_path: str, encodings: list = None) -> str:
    """
    Определяет разделитель CSV файла (; или ,).
    """
    if encodings is None:
        encodings = ['utf-8', 'cp1251', 'windows-1251', 'latin-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                first_line = f.readline()
                # Подсчитываем количество разделителей
                semicolon_count = first_line.count(';')
                comma_count = first_line.count(',')
                
                # Если точка с запятой встречается чаще, используем её
                if semicolon_count > comma_count:
                    return ';'
                elif comma_count > 0:
                    return ','
                else:
                    # По умолчанию пробуем точку с запятой
                    return ';'
        except UnicodeDecodeError:
            continue
    
    return ';'  # По умолчанию


def clean_column_name(name: str) -> str:
    """
    Очищает название колонки от HTML тегов и нормализует.
    Сохраняет специальные символы (например, №).
    """
    if not name or pd.isna(name):
        return ""
    
    name = str(name).strip()
    
    # Сохраняем символ № перед обработкой
    has_number_sign = '№' in name
    
    # Удаляем HTML теги из названия колонки
    if '<' in name and '>' in name:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(name, 'html.parser')
        name = soup.get_text()
        # Восстанавливаем символ № если он был потерян
        if has_number_sign and '№' not in name:
            # Пробуем найти где был символ № (обычно в начале)
            if 'вопроса' in name.lower() or 'номер' in name.lower():
                name = '№ ' + name.lstrip()
    
    # Нормализуем пробелы (но сохраняем одиночные пробелы)
    name = re.sub(r'\s+', ' ', name).strip()
    
    # Удаляем невидимые символы, но сохраняем видимые (включая №)
    # Символ № имеет код U+2116, проверяем его явно
    cleaned = []
    for char in name:
        if char.isprintable() or char == '№' or ord(char) == 8470:  # 8470 = U+2116 (№)
            cleaned.append(char)
        elif char in ['\n', '\t', '\r']:
            continue
    name = ''.join(cleaned).strip()
    
    return name


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Загружает CSV и возвращает DataFrame.
    Автоматически очищает файл от HTML тегов перед загрузкой.
    Определяет разделитель автоматически.
    """
    # Определяем разделитель
    delimiter = detect_delimiter(file_path)
    
    # Проверяем наличие HTML тегов в файле перед загрузкой
    encodings = ['utf-8', 'cp1251', 'windows-1251', 'latin-1']
    sample = ""
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                sample = f.read(10000)  # Читаем первые 10KB для проверки
            break
        except UnicodeDecodeError:
            continue
    
    has_html = '<' in sample and '>' in sample and any(tag in sample.lower() for tag in ['<div', '<span', '<p', '<br', '<td', '<th', '<tr', '<table'])
    
    # Определяем кодировку для чтения
    file_encoding = 'utf-8'
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read(100)
            file_encoding = encoding
            break
        except UnicodeDecodeError:
            continue
    
    if has_html:
        # Если обнаружены HTML теги, очищаем файл
        try:
            cleaned_csv = clean_html_from_csv(file_path, delimiter=delimiter)
            # Пробуем загрузить с определенным разделителем
            try:
                df = pd.read_csv(io.StringIO(cleaned_csv), sep=delimiter, on_bad_lines='skip', engine='python')
            except TypeError:
                # Для старых версий pandas
                df = pd.read_csv(io.StringIO(cleaned_csv), sep=delimiter, error_bad_lines=False, warn_bad_lines=False, engine='python')
        except Exception as clean_error:
            # Если очистка не помогла, пробуем загрузить как есть с обработкой ошибок
            try:
                try:
                    df = pd.read_csv(file_path, sep=delimiter, encoding=file_encoding, on_bad_lines='skip', engine='python')
                except TypeError:
                    df = pd.read_csv(file_path, sep=delimiter, encoding=file_encoding, error_bad_lines=False, warn_bad_lines=False, engine='python')
            except Exception as e:
                raise ValueError(f"Ошибка при загрузке и очистке CSV файла: {str(clean_error)}. Ошибка загрузки: {str(e)}")
    else:
        # Обычная загрузка CSV
        try:
            df = pd.read_csv(file_path, sep=delimiter, encoding=file_encoding)
        except Exception as e:
            # Если обычная загрузка не удалась, пробуем очистить от HTML
            try:
                cleaned_csv = clean_html_from_csv(file_path, delimiter=delimiter)
                try:
                    df = pd.read_csv(io.StringIO(cleaned_csv), sep=delimiter, on_bad_lines='skip', engine='python')
                except TypeError:
                    df = pd.read_csv(io.StringIO(cleaned_csv), sep=delimiter, error_bad_lines=False, warn_bad_lines=False, engine='python')
            except Exception as clean_error:
                raise ValueError(f"Ошибка при загрузке CSV файла: {str(e)}. Ошибка очистки: {str(clean_error)}")
    
    if df.empty:
        raise ValueError("CSV-файл пуст или не содержит данных после очистки.")
    
    # Очищаем названия колонок от HTML тегов
    original_columns = list(df.columns)
    df.columns = [clean_column_name(col) for col in df.columns]
    
    # Логируем изменения в названиях колонок
    import logging
    logger = logging.getLogger(__name__)
    for orig, cleaned in zip(original_columns, df.columns):
        if orig != cleaned:
            logger.info(f"Колонка переименована: '{orig}' -> '{cleaned}'")
    
    logger.info(f"Итоговые колонки после очистки: {list(df.columns)}")
    
    # Удаляем полностью пустые строки
    df = df.dropna(how='all')
    
    # Заполняем пропущенные значения в критических колонках пустыми строками
    # чтобы избежать ошибок при обработке
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('')
    
    return df


def prepare_features(df: pd.DataFrame, target_column: str):
    """Разделяет на признаки и целевую переменную, масштабирует X."""
    if target_column not in df.columns:
        raise ValueError(f"Целевая колонка '{target_column}' отсутствует в данных.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
