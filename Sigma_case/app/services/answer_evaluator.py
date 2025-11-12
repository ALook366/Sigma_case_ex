"""
Сервис для оценки ответов на основе обученной AI модели.
Поддерживает работу с моделями из transformers (BERT, RoBERTa и т.д.)
"""
import os
import torch
import shutil
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Union, Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModel,
    AutoConfig,
    BertForSequenceClassification,
    BertModel,
    RobertaForSequenceClassification,
    RobertaModel
)
import numpy as np
from Sigma_case.app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class AnswerEvaluator:
    """Класс для оценки ответов с использованием обученной AI модели."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Инициализация оценщика ответов.
        
        Args:
            model_path: Путь к директории с обученной моделью.
                      Если None, используется путь из настроек.
        """
        self.model_path = Path(model_path) if model_path else self._find_model_path()
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
    
    def _find_model_path(self) -> Path:
        """Находит путь к обученной модели."""
        # Проверяем несколько возможных путей
        possible_paths = [
            settings.AI_MODEL_DIR,
            getattr(settings, 'AI_MODEL_DIR_ALT', None),
            getattr(settings, 'AI_MODEL_DIR_ALT2', None),
            Path(__file__).parent.parent.parent / "trained_models" / "current_model",
            Path(__file__).parent.parent.parent.parent / "trained_models" / "current_model",
        ]
        
        # Убираем None значения
        possible_paths = [p for p in possible_paths if p is not None]
        
        logger.info("Поиск модели в следующих путях:")
        for path in possible_paths:
            exists = path.exists()
            has_files = any(path.glob("*")) if exists else False
            logger.info(f"  - {path}: exists={exists}, has_files={has_files}")
            if exists and has_files:
                logger.info(f"✅ Найдена модель в: {path}")
                return path
        
        # Если модель не найдена, создаем директорию для будущей модели
        model_dir = settings.AI_MODEL_DIR
        model_dir.mkdir(parents=True, exist_ok=True)
        logger.warning(f"⚠️ Модель не найдена ни в одном из путей, используется: {model_dir}")
        return model_dir
    
    def _load_model(self):
        """Загружает модель и токенизатор."""
        try:
            # Проверяем наличие файлов модели
            config_files = list(self.model_path.glob("config*.json"))
            model_files = (
                list(self.model_path.glob("*.bin")) + 
                list(self.model_path.glob("*.pt")) + 
                list(self.model_path.glob("*.pth")) +
                list(self.model_path.glob("*.safetensors"))
            )
            
            # Проверяем наличие токенизатора
            tokenizer_files = (
                list(self.model_path.glob("tokenizer*.json")) +
                list(self.model_path.glob("vocab*.txt")) +
                list(self.model_path.glob("vocab*.json"))
            )
            
            logger.info(f"Проверка модели в {self.model_path}")
            logger.info(f"Найдены файлы конфигурации: {[f.name for f in config_files]}")
            logger.info(f"Найдены файлы модели: {[f.name for f in model_files]}")
            logger.info(f"Найдены файлы токенизатора: {[f.name for f in tokenizer_files]}")
            
            logger.info(f"Проверка файлов модели:")
            logger.info(f"  - config_files: {len(config_files)} файлов")
            logger.info(f"  - model_files: {len(model_files)} файлов")
            logger.info(f"  - tokenizer_files: {len(tokenizer_files)} файлов")
            
            if (config_files or model_files) and tokenizer_files:
                logger.info(f"✅ Все необходимые файлы найдены. Загрузка модели из {self.model_path}")
                
                # Если файлы имеют нестандартные имена, переименовываем их напрямую
                model_path_to_use = str(self.model_path)
                renamed_files = []  # Список переименованных файлов для восстановления
                
                try:
                    # Проверяем, нужны ли переименования
                    standard_config = self.model_path / "config.json"
                    standard_model = self.model_path / "model.safetensors"
                    
                    # Ищем safetensors файл
                    safetensors_files = list(self.model_path.glob("*.safetensors"))
                    
                    # Переименовываем config файл если нужно
                    if not standard_config.exists() and config_files:
                        config_file = config_files[0]
                        if config_file.name != "config.json":
                            logger.info(f"Переименование {config_file.name} -> config.json")
                            config_file.rename(standard_config)
                            renamed_files.append((standard_config, config_file))
                    
                    # Переименовываем model файл если нужно
                    if safetensors_files and not standard_model.exists():
                        model_file = safetensors_files[0]
                        if model_file.name != "model.safetensors":
                            logger.info(f"Переименование {model_file.name} -> model.safetensors")
                            model_file.rename(standard_model)
                            renamed_files.append((standard_model, model_file))
                    
                    # Проверяем config.json на наличие model_type и исправляем если нужно
                    if standard_config.exists():
                        try:
                            with open(standard_config, 'r', encoding='utf-8') as f:
                                config_data = json.load(f)
                            
                            config_modified = False
                            
                            if 'model_type' not in config_data:
                                logger.warning("В config.json отсутствует поле 'model_type'")
                                # Пробуем определить тип модели по другим признакам
                                if 'architectures' in config_data and config_data['architectures']:
                                    arch = config_data['architectures'][0].lower()
                                    logger.info(f"Найдена архитектура: {config_data['architectures'][0]}")
                                    
                                    # Определяем model_type по архитектуре
                                    if 'bert' in arch:
                                        config_data['model_type'] = 'bert'
                                        config_modified = True
                                    elif 'roberta' in arch:
                                        config_data['model_type'] = 'roberta'
                                        config_modified = True
                                    elif 'distilbert' in arch:
                                        config_data['model_type'] = 'distilbert'
                                        config_modified = True
                                    elif 'gpt' in arch or 'llama' in arch:
                                        config_data['model_type'] = 'llama' if 'llama' in arch else 'gpt2'
                                        config_modified = True
                                    elif 't5' in arch:
                                        config_data['model_type'] = 't5'
                                        config_modified = True
                                    
                                    if config_modified:
                                        logger.info(f"Добавлен model_type: {config_data['model_type']}")
                                
                                # Если не удалось определить по архитектуре, пробуем по другим полям
                                if not config_modified:
                                    # Проверяем наличие типичных полей BERT
                                    if 'hidden_size' in config_data and 'num_attention_heads' in config_data:
                                        # Похоже на BERT-подобную модель
                                        config_data['model_type'] = 'bert'
                                        config_modified = True
                                        logger.info("Определен model_type: bert (по структуре конфигурации)")
                                
                                # Сохраняем исправленный config
                                if config_modified:
                                    with open(standard_config, 'w', encoding='utf-8') as f:
                                        json.dump(config_data, f, indent=2, ensure_ascii=False)
                                    logger.info("Config.json обновлен")
                                else:
                                    logger.warning("Не удалось определить тип модели автоматически. Попробуем загрузить без model_type")
                        except Exception as e:
                            logger.warning(f"Ошибка при чтении/исправлении config.json: {e}")
                except Exception as e:
                    logger.warning(f"Ошибка при переименовании файлов: {e}")
                    # Восстанавливаем оригинальные имена если были переименованы
                    for new_path, old_path in renamed_files:
                        try:
                            if new_path.exists():
                                new_path.rename(old_path)
                        except:
                            pass
                
                # Пробуем загрузить как модель для классификации последовательностей
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_path_to_use,
                        local_files_only=True,
                        trust_remote_code=True
                    )
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        model_path_to_use,
                        local_files_only=True,
                        trust_remote_code=True
                    )
                    logger.info("Модель загружена как SequenceClassification")
                except Exception as e1:
                    logger.warning(f"Не удалось загрузить как SequenceClassification: {e1}")
                    # Пробуем загрузить как базовую модель
                    try:
                        if self.tokenizer is None:
                            self.tokenizer = AutoTokenizer.from_pretrained(
                                model_path_to_use,
                                local_files_only=True,
                                trust_remote_code=True
                            )
                        self.model = AutoModel.from_pretrained(
                            model_path_to_use,
                            local_files_only=True,
                            trust_remote_code=True
                        )
                        logger.info("Модель загружена как AutoModel")
                    except Exception as e2:
                        logger.warning(f"Не удалось загрузить как AutoModel: {e2}")
                        # Пробуем загрузить через AutoConfig и определить тип вручную
                        try:
                            if self.tokenizer is None:
                                self.tokenizer = AutoTokenizer.from_pretrained(
                                    model_path_to_use,
                                    local_files_only=True,
                                    trust_remote_code=True
                                )
                            
                            # Загружаем конфигурацию
                            config = AutoConfig.from_pretrained(
                                model_path_to_use,
                                local_files_only=True,
                                trust_remote_code=True
                            )
                            
                            # Пробуем определить тип модели и загрузить соответствующую архитектуру
                            model_type = getattr(config, 'model_type', None)
                            
                            if model_type == 'bert':
                                # Пробуем загрузить как BERT
                                try:
                                    self.model = BertForSequenceClassification.from_pretrained(
                                        model_path_to_use,
                                        config=config,
                                        local_files_only=True,
                                        trust_remote_code=True
                                    )
                                    logger.info("Модель загружена как BertForSequenceClassification")
                                except:
                                    self.model = BertModel.from_pretrained(
                                        model_path_to_use,
                                        config=config,
                                        local_files_only=True,
                                        trust_remote_code=True
                                    )
                                    logger.info("Модель загружена как BertModel")
                            elif model_type == 'roberta':
                                try:
                                    self.model = RobertaForSequenceClassification.from_pretrained(
                                        model_path_to_use,
                                        config=config,
                                        local_files_only=True,
                                        trust_remote_code=True
                                    )
                                    logger.info("Модель загружена как RobertaForSequenceClassification")
                                except:
                                    self.model = RobertaModel.from_pretrained(
                                        model_path_to_use,
                                        config=config,
                                        local_files_only=True,
                                        trust_remote_code=True
                                    )
                                    logger.info("Модель загружена как RobertaModel")
                            else:
                                # Последняя попытка - загрузить через AutoModel с игнорированием ошибок
                                logger.warning(f"Неизвестный тип модели: {model_type}, пробуем AutoModel с игнорированием ошибок")
                                raise e2
                                
                        except Exception as e3:
                            logger.error(f"Все попытки загрузки модели не удались: {e3}")
                            import traceback
                            logger.error(traceback.format_exc())
                            model_type_info = "неизвестен"
                            try:
                                config = AutoConfig.from_pretrained(
                                    model_path_to_use,
                                    local_files_only=True,
                                    trust_remote_code=True
                                )
                                model_type_info = getattr(config, 'model_type', 'неизвестен')
                            except:
                                pass
                            raise ValueError(
                                f"Не удалось загрузить модель. "
                                f"Тип модели: {model_type_info}, "
                                f"Ошибка: {str(e3)}"
                            )
                
                # Если модель успешно загружена, оставляем переименованные файлы
                # Иначе восстанавливаем оригинальные имена
                if self.model is None or self.tokenizer is None:
                    # Восстанавливаем оригинальные имена
                    for new_path, old_path in renamed_files:
                        try:
                            if new_path.exists():
                                new_path.rename(old_path)
                                logger.info(f"Восстановлено имя: {new_path.name} -> {old_path.name}")
                        except Exception as e:
                            logger.warning(f"Не удалось восстановить имя {new_path}: {e}")
                else:
                    logger.info("Файлы модели переименованы успешно, оставляем стандартные имена")
                
                if self.model is not None and self.tokenizer is not None:
                    self.model.to(self.device)
                    self.model.eval()
                    logger.info("Модель успешно загружена и готова к использованию")
            else:
                logger.warning(f"Файлы модели не найдены в {self.model_path}. "
                              f"Config: {len(config_files)}, Model: {len(model_files)}, Tokenizer: {len(tokenizer_files)}")
                self.model = None
                self.tokenizer = None
                
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.model = None
            self.tokenizer = None
    
    def evaluate_answer(
        self, 
        answer: str, 
        question: Optional[str] = None,
        context: Optional[str] = None
    ) -> Dict[str, Union[float, str]]:
        """
        Оценивает ответ на основе обученной модели.
        
        Args:
            answer: Текст ответа для оценки
            question: Вопрос (опционально)
            context: Контекст (опционально)
            
        Returns:
            Словарь с оценкой и метаданными
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError(
                "Модель не загружена. Убедитесь, что обученная модель находится в "
                f"{self.model_path}"
            )
        
        try:
            # Формируем входной текст
            if question and context:
                input_text = f"Вопрос: {question}\nКонтекст: {context}\nОтвет: {answer}"
            elif question:
                input_text = f"Вопрос: {question}\nОтвет: {answer}"
            else:
                input_text = answer
            
            # Токенизация
            inputs = self.tokenizer(
                input_text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Предсказание
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Получаем логиты или скоры
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif hasattr(outputs, 'last_hidden_state'):
                    # Если это базовая модель, используем средний пулинг
                    logits = outputs.last_hidden_state.mean(dim=1)
                else:
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                
                # Применяем softmax для получения вероятностей
                probs = torch.nn.functional.softmax(logits, dim=-1)
                scores = probs.cpu().numpy()[0]
            
            # Определяем оценку (предполагаем, что последний класс - это оценка)
            # Если модель бинарная (2 класса), используем вероятность положительного класса
            if len(scores) == 2:
                score = float(scores[1])  # Вероятность положительного класса
                predicted_class = 1 if score > 0.5 else 0
            else:
                # Для многоклассовой классификации используем максимальную вероятность
                predicted_class = int(np.argmax(scores))
                score = float(scores[predicted_class])
            
            # Нормализуем оценку в диапазон 0-100
            normalized_score = score * 100
            
            return {
                "score": round(normalized_score, 2),
                "raw_score": round(score, 4),
                "predicted_class": int(predicted_class),
                "confidence": round(score, 4),
                "answer": answer,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Ошибка при оценке ответа: {e}")
            return {
                "score": 0.0,
                "error": str(e),
                "answer": answer,
                "status": "error"
            }
    
    def evaluate_batch(
        self, 
        answers: List[str],
        questions: Optional[List[str]] = None,
        contexts: Optional[List[str]] = None
    ) -> List[Dict[str, Union[float, str]]]:
        """
        Оценивает список ответов.
        
        Args:
            answers: Список ответов для оценки
            questions: Список вопросов (опционально)
            contexts: Список контекстов (опционально)
            
        Returns:
            Список словарей с оценками
        """
        results = []
        for i, answer in enumerate(answers):
            question = questions[i] if questions and i < len(questions) else None
            context = contexts[i] if contexts and i < len(contexts) else None
            result = self.evaluate_answer(answer, question, context)
            results.append(result)
        return results
    
    def is_model_loaded(self) -> bool:
        """Проверяет, загружена ли модель."""
        return self.model is not None and self.tokenizer is not None


# Глобальный экземпляр оценщика (singleton)
_evaluator_instance: Optional[AnswerEvaluator] = None

def get_evaluator(model_path: Optional[str] = None) -> AnswerEvaluator:
    """Получает глобальный экземпляр оценщика ответов."""
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = AnswerEvaluator(model_path)
    return _evaluator_instance

