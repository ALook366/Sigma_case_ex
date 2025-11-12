"""
Сервис для обработки CSV файлов с оценкой ответов на основе AI модели.
Работает с новой структурой данных согласно ТЗ.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Callable
from datetime import datetime
from time import time
from Sigma_case.app.services.data_preprocessor import load_csv
from Sigma_case.app.services.answer_evaluator import get_evaluator
from Sigma_case.app.core.config import settings
import logging

logger = logging.getLogger(__name__)


def normalize_score_by_question(score: float, question_number: int) -> float:
    """
    Нормализует оценку в зависимости от номера вопроса.
    
    Диапазоны оценок:
    - Вопрос 1: от 0 до 1 балла
    - Вопрос 2: от 0 до 2 баллов
    - Вопрос 3: от 0 до 1 балла
    - Вопрос 4: от 0 до 2 баллов
    
    Args:
        score: Оценка от модели (0-100)
        question_number: Номер вопроса (1-4)
        
    Returns:
        Нормализованная оценка в диапазоне вопроса
    """
    # Определяем максимальный балл для вопроса
    max_scores = {1: 1, 2: 2, 3: 1, 4: 2}
    max_score = max_scores.get(question_number, 2)
    
    # Нормализуем оценку от 0-100 к диапазону вопроса
    normalized = (score / 100.0) * max_score
    
    # Округляем до целого числа
    return round(normalized)


def find_column_flexible(target_name: str, columns: List[str], keywords: List[List[str]]) -> Optional[str]:
    """
    Гибкий поиск колонки с приоритетами.
    
    Args:
        target_name: Целевое название колонки
        columns: Список доступных колонок
        keywords: Список списков ключевых слов (от более специфичных к менее)
        
    Returns:
        Найденное название колонки или None
    """
    # Сначала проверяем точное совпадение (с учетом регистра и пробелов)
    for col in columns:
        if col.strip() == target_name.strip():
            logger.info(f"Точное совпадение для '{target_name}': '{col}'")
            return col
    
    # Проверяем совпадение без учета регистра
    target_lower = target_name.lower().strip()
    for col in columns:
        if col.lower().strip() == target_lower:
            logger.info(f"Совпадение без учета регистра для '{target_name}': '{col}'")
            return col
    
    # Проверяем по ключевым словам с приоритетами
    for keyword_group in keywords:
        keywords_lower = [kw.lower().strip() for kw in keyword_group]
        for col in columns:
            col_lower = col.lower().strip()
            # Проверяем, содержит ли название колонки все ключевые слова из группы
            if all(kw in col_lower for kw in keywords_lower if kw):
                logger.info(f"Найдено по ключевым словам {keyword_group} для '{target_name}': '{col}'")
                return col
            # Или хотя бы одно ключевое слово
            elif any(kw in col_lower for kw in keywords_lower if kw):
                logger.info(f"Найдено по частичному совпадению {keyword_group} для '{target_name}': '{col}'")
                return col
    
    # Последняя попытка - поиск по любому ключевому слову
    all_keywords = [kw for group in keywords for kw in group if kw]
    for col in columns:
        col_lower = col.lower().strip()
        for kw in all_keywords:
            if kw.lower() in col_lower:
                logger.info(f"Найдено по любому ключевому слову '{kw}' для '{target_name}': '{col}'")
                return col
    
    return None


def evaluate_csv_answers(
    file_path: str,
    max_rows: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, float, float], None]] = None
) -> str:
    """
    Обрабатывает CSV файл, оценивает ответы с помощью AI модели и сохраняет результат.
    
    Структура входного CSV:
    - Id экзамена
    - Id вопроса
    - № вопроса (1-4)
    - Текст вопроса (может содержать HTML)
    - Картинка из вопроса (опционально)
    - Оценка экзаменатора
    - Транскрибация ответа
    - Ссылка на оригинальный файл записи
    
    Args:
        file_path: Путь к CSV файлу
        max_rows: Максимальное количество строк для обработки (None = все)
        progress_callback: Callback для обновления прогресса
        
    Returns:
        Путь к .csv файлу с результатами обработки
    """
    try:
        # Загружаем CSV файл (с автоматической очисткой от HTML тегов)
        logger.info(f"Загрузка CSV файла: {file_path}")
        try:
            df = load_csv(file_path)
        except Exception as e:
            logger.error(f"Ошибка при загрузке CSV файла: {e}", exc_info=True)
            raise ValueError(f"Не удалось загрузить CSV файл: {str(e)}")
        
        logger.info(f"Загружено строк: {len(df)}")
        logger.info(f"Колонки после загрузки ({len(df.columns)}): {list(df.columns)}")
        logger.info(f"Типы колонок: {df.dtypes.to_dict()}")
        
        # Выводим первые несколько значений каждой колонки для отладки
        if len(df) > 0:
            logger.info("Примеры данных из первых строк:")
            for col in df.columns[:5]:  # Первые 5 колонок
                sample = df[col].head(3).tolist()
                logger.info(f"  {col}: {sample}")
        
        # Ограничиваем количество строк если указано (до поиска колонок)
        if max_rows and max_rows > 0:
            df = df.head(max_rows)
            logger.info(f"Обработка ограничена {max_rows} строками")
        
        # Ищем необходимые колонки с гибким сопоставлением
        column_mapping = {}
        
        # Поиск колонки "№ вопроса" с приоритетами
        question_num_keywords = [
            ['№ вопроса'],  # Точное название
            ['номер', 'вопрос'],  # Оба слова
            ['№'],  # Только символ
            ['номер'],  # Только номер
            ['вопрос', 'question'],  # Вопрос
            ['num', 'number']  # Английские варианты
        ]
        question_num_col = find_column_flexible('№ вопроса', df.columns, question_num_keywords)
        if question_num_col:
            column_mapping['№ вопроса'] = question_num_col
            logger.info(f"✓ Найдена колонка '№ вопроса': '{question_num_col}'")
        else:
            logger.warning(f"✗ Колонка '№ вопроса' не найдена. Доступные: {list(df.columns)}")
        
        # Поиск колонки "Транскрибация ответа" с приоритетами
        transcription_keywords = [
            ['транскрибация', 'ответа'],  # Точное название
            ['транскрибация'],  # Только транскрибация
            ['ответ'],  # Только ответ
            ['answer', 'transcription'],  # Английские варианты
            ['ответ', 'answer']  # Любой ответ
        ]
        transcription_col = find_column_flexible('Транскрибация ответа', df.columns, transcription_keywords)
        if transcription_col:
            column_mapping['Транскрибация ответа'] = transcription_col
            logger.info(f"✓ Найдена колонка 'Транскрибация ответа': '{transcription_col}'")
        else:
            logger.warning(f"✗ Колонка 'Транскрибация ответа' не найдена. Доступные: {list(df.columns)}")
        
        # Поиск колонки "Оценка экзаменатора" с приоритетами
        examiner_score_keywords = [
            ['оценка', 'экзаменатора'],  # Точное название
            ['оценка', 'эксперта'],  # Альтернативное
            ['оценка'],  # Только оценка
            ['score'],  # Английский вариант
            ['балл', 'баллы'],  # Баллы
            ['оценка']  # Любая оценка
        ]
        examiner_score_col = find_column_flexible('Оценка экзаменатора', df.columns, examiner_score_keywords)
        if examiner_score_col:
            column_mapping['Оценка экзаменатора'] = examiner_score_col
            logger.info(f"✓ Найдена колонка 'Оценка экзаменатора': '{examiner_score_col}'")
        else:
            logger.warning(f"✗ Колонка 'Оценка экзаменатора' не найдена. Доступные: {list(df.columns)}")
        
        # Проверяем, все ли колонки найдены
        required_columns = ['№ вопроса', 'Транскрибация ответа', 'Оценка экзаменатора']
        missing_columns = [col for col in required_columns if col not in column_mapping]
        
        # Если колонки не найдены, пробуем использовать позиционный подход
        if missing_columns and len(df.columns) >= 3:
            logger.warning(f"Колонки не найдены по ключевым словам: {missing_columns}")
            logger.info("Пробуем использовать позиционный подход...")
            
            # Ожидаемый порядок колонок в файле:
            # Id экзамена;Id вопроса;№ вопроса;Текст вопроса;Картинка из вопроса;Оценка экзаменатора;Транскрибация ответа;Ссылка на оригинальный файл запис
            # Индексы: 0, 1, 2, 3, 4, 5, 6, 7
            
            # Пробуем найти по позиции
            if '№ вопроса' not in column_mapping and len(df.columns) > 2:
                # Обычно это 3-я колонка (индекс 2)
                potential_col = df.columns[2] if len(df.columns) > 2 else None
                if potential_col:
                    # Проверяем, что это похоже на номер вопроса
                    sample_values = df[potential_col].head(10).astype(str).tolist()
                    # Если значения похожи на номера (1, 2, 3, 4)
                    if any(str(v).strip() in ['1', '2', '3', '4'] for v in sample_values):
                        column_mapping['№ вопроса'] = potential_col
                        logger.info(f"Найдена колонка '№ вопроса' по позиции: '{potential_col}'")
            
            if 'Оценка экзаменатора' not in column_mapping and len(df.columns) > 5:
                # Обычно это 6-я колонка (индекс 5)
                potential_col = df.columns[5] if len(df.columns) > 5 else None
                if potential_col:
                    # Проверяем, что это похоже на оценку (числа 0-2)
                    try:
                        sample_values = pd.to_numeric(df[potential_col].head(10), errors='coerce')
                        if sample_values.notna().any() and (sample_values <= 2).any():
                            column_mapping['Оценка экзаменатора'] = potential_col
                            logger.info(f"Найдена колонка 'Оценка экзаменатора' по позиции: '{potential_col}'")
                    except:
                        pass
            
            if 'Транскрибация ответа' not in column_mapping and len(df.columns) > 6:
                # Обычно это 7-я колонка (индекс 6)
                potential_col = df.columns[6] if len(df.columns) > 6 else None
                if potential_col:
                    # Проверяем, что это похоже на текст (длинные строки)
                    sample_values = df[potential_col].head(10).astype(str)
                    avg_length = sample_values.str.len().mean()
                    if avg_length > 50:  # Транскрипция обычно длинная
                        column_mapping['Транскрибация ответа'] = potential_col
                        logger.info(f"Найдена колонка 'Транскрибация ответа' по позиции: '{potential_col}'")
        
        # Проверяем еще раз
        missing_columns = [col for col in required_columns if col not in column_mapping]
        if missing_columns:
            error_msg = (
                f"Не найдены необходимые колонки: {missing_columns}. "
                f"Доступные колонки: {list(df.columns)}. "
                f"Найденные колонки: {list(column_mapping.keys())}. "
                f"Проверьте формат файла и наличие всех необходимых колонок."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Сохраняем исходные названия колонок для последующего использования
        original_columns_mapping = {new_name: old_name for new_name, old_name in column_mapping.items()}
        
        # Переименовываем колонки для единообразия
        logger.info(f"Переименование колонок: {column_mapping}")
        try:
            df = df.rename(columns=column_mapping)
            logger.info(f"Колонки после переименования: {list(df.columns)}")
        except Exception as e:
            logger.error(f"Ошибка при переименовании колонок: {e}", exc_info=True)
            raise ValueError(f"Ошибка при переименовании колонок: {str(e)}. Маппинг: {column_mapping}")
        
        # Дополнительная проверка после переименования
        for req_col in required_columns:
            if req_col not in df.columns:
                # Пробуем найти колонку с похожим названием
                found = False
                for col in df.columns:
                    if req_col.lower() in col.lower() or col.lower() in req_col.lower():
                        logger.warning(f"Колонка '{req_col}' не найдена, но найдена похожая: '{col}'")
                        # Переименовываем найденную колонку
                        df = df.rename(columns={col: req_col})
                        found = True
                        logger.info(f"Колонка '{col}' переименована в '{req_col}'")
                        break
                
                if not found:
                    error_msg = (
                        f"Критическая ошибка: колонка '{req_col}' отсутствует после переименования. "
                        f"Текущие колонки: {list(df.columns)}. "
                        f"Исходный маппинг: {column_mapping}"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
        
        # Сохраняем исходный DataFrame для извлечения дополнительных колонок
        # Создаем копию с исходными названиями колонок (до переименования)
        # Нам нужно найти исходные колонки: Id экзамена, Текст вопроса, Картинка из вопроса
        # Эти колонки не были переименованы, поэтому ищем их в исходном df
        
        # Создаем обратный маппинг для поиска исходных колонок
        reverse_mapping = {v: k for k, v in column_mapping.items()}
        
        # Сохраняем ссылку на исходные колонки до переименования
        # Нам нужны: Id экзамена, Текст вопроса, Картинка из вопроса
        # Эти колонки не входят в column_mapping, поэтому они остались с исходными названиями
        
        # Проверяем наличие колонки с текстом вопроса (опционально)
        question_column = None
        question_text_keywords = [
            ['текст', 'вопроса'],
            ['текст'],
            ['вопрос', 'question'],
            ['question', 'text']
        ]
        for col in df.columns:
            if col not in ['№ вопроса', 'Транскрибация ответа', 'Оценка экзаменатора']:
                col_lower = col.lower()
                for keyword_group in question_text_keywords:
                    if all(kw in col_lower for kw in keyword_group):
                        question_column = col
                        logger.info(f"✓ Найдена колонка с текстом вопроса: '{question_column}'")
                        break
                if question_column:
                    break
        
        # Проверяем валидность данных после переименования
        if '№ вопроса' not in df.columns:
            error_msg = (
                f"Колонка '№ вопроса' не найдена после переименования. "
                f"Текущие колонки: {list(df.columns)}. "
                f"Маппинг: {column_mapping}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if 'Транскрибация ответа' not in df.columns:
            error_msg = (
                f"Колонка 'Транскрибация ответа' не найдена после переименования. "
                f"Текущие колонки: {list(df.columns)}. "
                f"Маппинг: {column_mapping}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if 'Оценка экзаменатора' not in df.columns:
            error_msg = (
                f"Колонка 'Оценка экзаменатора' не найдена после переименования. "
                f"Текущие колонки: {list(df.columns)}. "
                f"Маппинг: {column_mapping}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Обрабатываем номера вопросов - конвертируем в int, обрабатывая ошибки
        try:
            logger.info(f"Обработка колонки '№ вопроса'. Тип: {df['№ вопроса'].dtype}")
            logger.info(f"Примеры значений: {df['№ вопроса'].head(10).tolist()}")
            
            question_numbers = pd.to_numeric(df['№ вопроса'], errors='coerce')
            
            # Заполняем пропущенные значения
            if question_numbers.isna().any():
                na_count = question_numbers.isna().sum()
                logger.warning(f"Найдено {na_count} пропущенных значений в '№ вопроса'")
                mode_value = question_numbers.mode()
                if len(mode_value) > 0:
                    fill_value = int(mode_value[0])
                    logger.info(f"Заполнение пропущенных значений модой: {fill_value}")
                    question_numbers = question_numbers.fillna(fill_value)
                else:
                    logger.warning("Мода не найдена, используем значение по умолчанию: 1")
                    question_numbers = question_numbers.fillna(1)
            
            question_numbers = question_numbers.astype(int)
            logger.info(f"Номера вопросов обработаны. Диапазон: {question_numbers.min()}-{question_numbers.max()}")
            
        except Exception as e:
            logger.error(f"Ошибка при обработке номеров вопросов: {e}", exc_info=True)
            raise ValueError(f"Не удалось обработать колонку '№ вопроса': {str(e)}")
        
        # Проверяем диапазон номеров вопросов
        invalid_questions = question_numbers[~question_numbers.isin([1, 2, 3, 4])]
        if len(invalid_questions) > 0:
            invalid_unique = invalid_questions.unique()
            logger.warning(f"Найдены невалидные номера вопросов: {invalid_unique} (всего: {len(invalid_questions)})")
            # Заменяем невалидные номера на 1 (по умолчанию)
            question_numbers = question_numbers.replace(invalid_unique, 1)
            logger.info(f"Невалидные номера заменены на 1")
        
        # Обновляем DataFrame с обработанными номерами вопросов
        df['№ вопроса'] = question_numbers
        
        # Обновляем начальный прогресс перед загрузкой модели
        total_rows = len(df)
        if progress_callback:
            progress_callback(0, total_rows, 0.0, 0.0)
        
        # Получаем оценщик ответов
        logger.info("Получение оценщика ответов...")
        evaluator = get_evaluator()
        logger.info(f"Путь к модели: {evaluator.model_path}")
        logger.info(f"Модель загружена: {evaluator.is_model_loaded()}")
        
        if not evaluator.is_model_loaded():
            # Проверяем, существуют ли файлы модели
            model_path = evaluator.model_path
            config_files = list(model_path.glob("config*.json"))
            model_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("*.bin")) + list(model_path.glob("*.pt"))
            tokenizer_files = list(model_path.glob("tokenizer*.json")) + list(model_path.glob("vocab*.txt"))
            
            error_msg = (
                f"AI модель не загружена. "
                f"Путь: {model_path}, "
                f"Существует: {model_path.exists()}, "
                f"Config: {len(config_files)}, "
                f"Model: {len(model_files)}, "
                f"Tokenizer: {len(tokenizer_files)}. "
                f"Проверьте логи сервера."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Подготавливаем данные для оценки
        logger.info("Подготовка данных для оценки...")
        answers = df['Транскрибация ответа'].astype(str).tolist()
        questions = None
        if question_column:
            questions = df[question_column].astype(str).tolist()
            logger.info(f"Используется колонка с текстом вопроса: '{question_column}'")
        else:
            logger.info("Колонка с текстом вопроса не используется")
        
        question_numbers_list = question_numbers.tolist()
        
        # Обрабатываем оценки экзаменатора - конвертируем в float, обрабатывая ошибки
        try:
            logger.info(f"Обработка колонки 'Оценка экзаменатора'. Тип: {df['Оценка экзаменатора'].dtype}")
            logger.info(f"Примеры значений: {df['Оценка экзаменатора'].head(10).tolist()}")
            
            examiner_scores_series = pd.to_numeric(df['Оценка экзаменатора'], errors='coerce')
            
            # Заполняем пропущенные значения 0
            if examiner_scores_series.isna().any():
                na_count = examiner_scores_series.isna().sum()
                logger.warning(f"Найдено {na_count} пропущенных значений в 'Оценка экзаменатора'")
                examiner_scores_series = examiner_scores_series.fillna(0.0)
            
            examiner_scores = examiner_scores_series.astype(float).tolist()
            logger.info(f"Оценки экзаменатора обработаны. Диапазон: {min(examiner_scores)}-{max(examiner_scores)}")
            
        except Exception as e:
            logger.error(f"Ошибка при обработке оценок экзаменатора: {e}", exc_info=True)
            # Если не удалось обработать, используем нули
            examiner_scores = [0.0] * len(df)
            logger.warning("Используются нулевые оценки экзаменатора из-за ошибки обработки")
        
        total_rows = len(answers)
        start_time = time()
        
        # Оцениваем ответы с отслеживанием прогресса
        logger.info(f"Начало оценки ответов... Всего строк: {total_rows}")
        
        # Обновляем прогресс - начало обработки
        if progress_callback:
            progress_callback(0, total_rows, 0.0, 0.0)
        
        # Оцениваем по частям для отслеживания прогресса
        results = []
        batch_size = 10  # Обрабатываем по 10 ответов за раз
        
        for i in range(0, total_rows, batch_size):
            batch_answers = answers[i:i+batch_size]
            batch_questions = questions[i:i+batch_size] if questions else None
            
            try:
                batch_results = evaluator.evaluate_batch(
                    answers=batch_answers,
                    questions=batch_questions,
                    contexts=None
                )
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"Ошибка при оценке батча {i}-{i+batch_size}: {e}", exc_info=True)
                # В случае ошибки добавляем результаты с нулевой оценкой
                for j in range(len(batch_answers)):
                    results.append({
                        'score': 0.0,
                        'raw_score': 0.0,
                        'predicted_class': 0,
                        'confidence': 0.0,
                        'answer': batch_answers[j],
                        'status': 'error',
                        'error': str(e)
                    })
            
            # Обновляем прогресс
            processed = min(i + batch_size, total_rows)
            elapsed_time = time() - start_time
            
            if processed > 0 and elapsed_time > 0:
                progress_percent = (processed / total_rows) * 100
                avg_time_per_row = elapsed_time / processed
                remaining_rows = total_rows - processed
                estimated_remaining_time = avg_time_per_row * remaining_rows
                
                if progress_callback:
                    progress_callback(processed, total_rows, progress_percent, estimated_remaining_time)
                
                logger.info(f"Обработано: {processed}/{total_rows} ({progress_percent:.1f}%) | "
                          f"Осталось: ~{estimated_remaining_time:.1f} сек")
        
        # Нормализуем оценки по диапазонам вопросов
        logger.info("Нормализация оценок по диапазонам вопросов...")
        normalized_scores = []
        for i, result in enumerate(results):
            try:
                score = result.get('score', 0.0)
                question_num = question_numbers_list[i]
                normalized = normalize_score_by_question(score, question_num)
                normalized_scores.append(normalized)
            except Exception as e:
                logger.error(f"Ошибка при нормализации оценки для строки {i}: {e}")
                normalized_scores.append(0)

        
        # Рассчитываем MAE (Mean Absolute Error)
        try:
            model_scores = np.array(normalized_scores)
            examiner_scores_array = np.array(examiner_scores)
            
            # Фильтруем только валидные оценки (исключаем ошибки)
            valid_indices = [i for i, r in enumerate(results) if r.get('status') != 'error']
            if len(valid_indices) > 0:
                valid_model_scores = model_scores[valid_indices]
                valid_examiner_scores = examiner_scores_array[valid_indices]
                mae = np.mean(np.abs(valid_model_scores - valid_examiner_scores))
            else:
                mae = 0.0
                logger.warning("Нет валидных оценок для расчета MAE")
            
            logger.info(f"MAE (Mean Absolute Error): {mae:.4f}")
            print(f"\n{'='*60}")
            print(f"РЕЗУЛЬТАТЫ РАБОТЫ МОДЕЛИ")
            print(f"{'='*60}")
            print(f"Обработано строк: {total_rows}")
            print(f"Валидных оценок: {len(valid_indices) if len(valid_indices) > 0 else 0}")
            print(f"MAE (Mean Absolute Error): {mae:.4f}")
            if len(valid_indices) > 0:
                print(f"Средняя оценка модели: {np.mean(valid_model_scores):.2f}")
                print(f"Средняя оценка экзаменатора: {np.mean(valid_examiner_scores):.2f}")
            print(f"{'='*60}\n")
        except Exception as e:
            logger.error(f"Ошибка при расчете MAE: {e}", exc_info=True)
            mae = 0.0
        
        # Формируем результирующий DataFrame с необходимыми колонками
        logger.info("Формирование результирующего DataFrame...")
        logger.info(f"Доступные колонки в df: {list(df.columns)}")
        
        # Ищем необходимые колонки из исходного файла
        result_data = {}
        
        # 1. Id экзамена - ищем в df (не переименовывалась)
        id_exam_keywords = [
            ['id', 'экзамена'],
            ['id', 'exam'],
            ['экзамен', 'id'],
            ['id']
        ]
        id_exam_col = find_column_flexible('Id экзамена', df.columns, id_exam_keywords)
        if id_exam_col and id_exam_col in df.columns:
            result_data['Id экзамена'] = df[id_exam_col].astype(str).tolist()
            logger.info(f"✓ Добавлена колонка 'Id экзамена' из '{id_exam_col}'")
        else:
            # Если не найдено, пробуем первую колонку (обычно это Id экзамена)
            if len(df.columns) > 0:
                # Исключаем уже переименованные колонки
                available_cols = [col for col in df.columns if col not in ['№ вопроса', 'Транскрибация ответа', 'Оценка экзаменатора']]
                if available_cols:
                    result_data['Id экзамена'] = df[available_cols[0]].astype(str).tolist()
                    logger.info(f"✓ Добавлена колонка 'Id экзамена' из первой доступной колонки: '{available_cols[0]}'")
                else:
                    result_data['Id экзамена'] = [''] * len(df)
                    logger.warning(f"Колонка 'Id экзамена' не найдена, используется пустое значение")
            else:
                result_data['Id экзамена'] = [''] * len(df)
                logger.warning(f"Колонка 'Id экзамена' не найдена, используется пустое значение")
        
        # 2. № вопроса (уже обработан)
        result_data['№ вопроса'] = question_numbers.tolist()
        logger.info(f"✓ Добавлена колонка '№ вопроса'")
        
        # 3. Текст вопроса
        if question_column and question_column in df.columns:
            result_data['Текст вопроса'] = df[question_column].astype(str).tolist()
            logger.info(f"✓ Добавлена колонка 'Текст вопроса' из '{question_column}'")
        else:
            # Ищем колонку с текстом вопроса среди не переименованных
            question_text_keywords = [
                ['текст', 'вопроса'],
                ['текст', 'question'],
                ['текст'],
                ['question', 'text']
            ]
            question_text_col = find_column_flexible('Текст вопроса', df.columns, question_text_keywords)
            if question_text_col and question_text_col in df.columns:
                result_data['Текст вопроса'] = df[question_text_col].astype(str).tolist()
                logger.info(f"✓ Добавлена колонка 'Текст вопроса' из '{question_text_col}'")
            else:
                # Пробуем найти по позиции (обычно 4-я колонка, индекс 3)
                available_cols = [col for col in df.columns if col not in ['№ вопроса', 'Транскрибация ответа', 'Оценка экзаменатора', 'Id экзамена']]
                if available_cols:
                    result_data['Текст вопроса'] = df[available_cols[0]].astype(str).tolist()
                    logger.info(f"✓ Добавлена колонка 'Текст вопроса' из доступной колонки: '{available_cols[0]}'")
                else:
                    result_data['Текст вопроса'] = [''] * len(df)
                    logger.warning(f"Колонка 'Текст вопроса' не найдена, используется пустое значение")
        
        # 4. Картинка из вопроса
        image_keywords = [
            ['картинка', 'вопроса'],
            ['картинка', 'из'],
            ['картинка'],
            ['image', 'picture'],
            ['img', 'src']
        ]
        image_col = find_column_flexible('Картинка из вопроса', df.columns, image_keywords)
        if image_col and image_col in df.columns:
            result_data['Картинка из вопроса'] = df[image_col].astype(str).tolist()
            logger.info(f"✓ Добавлена колонка 'Картинка из вопроса' из '{image_col}'")
        else:
            # Пробуем найти по позиции среди не переименованных колонок
            available_cols = [col for col in df.columns if col not in ['№ вопроса', 'Транскрибация ответа', 'Оценка экзаменатора', 'Id экзамена']]
            if len(available_cols) > 1:
                # Обычно картинка идет после текста вопроса
                result_data['Картинка из вопроса'] = df[available_cols[1]].astype(str).tolist()
                logger.info(f"✓ Добавлена колонка 'Картинка из вопроса' по позиции: '{available_cols[1]}'")
            else:
                result_data['Картинка из вопроса'] = [''] * len(df)
                logger.warning(f"Колонка 'Картинка из вопроса' не найдена, используется пустое значение")

        # 5. Оценка за экзамен (нормализованная оценка модели)
        result_data['Оценка за экзамен'] = [int(score) for score in normalized_scores]
        logger.info(f"✓ Добавлена колонка 'Оценка за экзамен' (нормализованные оценки модели)")
        
        # 6. Транскрибация ответа (уже есть в df после переименования)
        if 'Транскрибация ответа' in df.columns:
            result_data['Транскрибация ответа'] = df['Транскрибация ответа'].astype(str).tolist()
            logger.info(f"✓ Добавлена колонка 'Транскрибация ответа'")
        else:
            result_data['Транскрибация ответа'] = [''] * len(df)
            logger.warning(f"Колонка 'Транскрибация ответа' не найдена, используется пустое значение")
        
        # Создаем результирующий DataFrame с правильным порядком колонок
        column_order = ['Id экзамена', '№ вопроса', 'Текст вопроса', 'Картинка из вопроса', 'Оценка за экзамен', 'Транскрибация ответа']
        
        # Проверяем, что все колонки имеют одинаковую длину
        lengths = {col: len(result_data[col]) for col in result_data.keys()}
        if len(set(lengths.values())) > 1:
            logger.warning(f"Разные длины колонок: {lengths}")
            min_length = min(lengths.values())
            # Обрезаем все колонки до минимальной длины
            for col in result_data:
                result_data[col] = result_data[col][:min_length]
            logger.info(f"Обрезано до минимальной длины: {min_length}")
        
        # Создаем DataFrame с правильным порядком колонок
        ordered_data = {col: result_data[col] for col in column_order if col in result_data}
        result_df = pd.DataFrame(ordered_data)
        
        logger.info(f"Результирующий DataFrame создан. Колонки: {list(result_df.columns)}")
        logger.info(f"Количество строк: {len(result_df)}")
        
        # Сохраняем результат в CSV файл
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"прогноз_{timestamp}.csv"
        result_path = settings.RESULTS_DIR / result_filename
        # Параллельно подготовим упрощенный файл
        # Ищем колонку Id вопроса в исходных данных
        id_question_keywords = [
            ['id', 'вопроса'],
            ['id', 'question'],
            ['вопрос', 'id'],
            ['id']
        ]
        id_question_col = find_column_flexible('Id вопроса', df.columns, id_question_keywords)
        if id_question_col and id_question_col in df.columns:
            simple_id_question = df[id_question_col].astype(str).tolist()
            logger.info(f"✓ Колонка 'Id вопроса' найдена: '{id_question_col}'")
        else:
            # если нет, оставим пустые
            simple_id_question = [''] * len(result_df)
            logger.warning("Колонка 'Id вопроса' не найдена, упрощенный файл будет без значений Id вопроса")
        # Собираем упрощенный датафрейм
        simple_df = pd.DataFrame({
            'Id экзамена': result_data.get('Id экзамена', [''] * len(result_df)),
            'Id вопроса': simple_id_question,
            'Оценка за экзамен': result_data.get('Оценка за экзамен', [])
        })
        simple_filename = f"прогноз_{timestamp}_short.csv"
        simple_path = settings.RESULTS_DIR / simple_filename
        
        try:
            # Определяем разделитель (используем тот же, что и в исходном файле)
            delimiter = ';'  # По умолчанию точка с запятой для русских файлов
            
            result_df.to_csv(result_path, index=False, encoding='utf-8-sig', sep=delimiter)
            # Сохраняем упрощенный файл
            simple_df.to_csv(simple_path, index=False, encoding='utf-8-sig', sep=delimiter)
            
            logger.info(f"Результаты сохранены в: {result_path}")
            logger.info(f"Колонки в результате: {list(result_df.columns)}")
            logger.info(f"Количество строк: {len(result_df)}")
            logger.info(f"Оценено ответов: {len(results)}")
            logger.info(f"MAE: {mae:.4f}")
            logger.info(f"Упрощенный результат сохранен в: {simple_path}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении результата: {e}", exc_info=True)
            raise
        
        return {"full": str(result_path), "simple": str(simple_path)}
        
    except ValueError as e:
        logger.error(f"Ошибка валидации при обработке CSV: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Ошибка при обработке CSV: {e}", exc_info=True)
        raise


def get_csv_info(file_path: str) -> Dict:
    """
    Получает информацию о CSV файле (колонки, количество строк).
    
    Args:
        file_path: Путь к CSV файлу
        
    Returns:
        Словарь с информацией о файле
    """
    try:
        df = load_csv(file_path)
        return {
            "columns": list(df.columns),
            "row_count": len(df),
            "sample_rows": df.head(3).to_dict('records') if len(df) > 0 else []
        }
    except Exception as e:
        logger.error(f"Ошибка при получении информации о CSV: {e}", exc_info=True)
        raise
