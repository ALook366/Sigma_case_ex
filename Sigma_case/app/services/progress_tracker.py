"""
Трекер прогресса для обработки CSV файлов.
"""
from typing import Dict, Optional
from datetime import datetime
import uuid

# Глобальное хранилище прогресса (в продакшене лучше использовать Redis)
_progress_store: Dict[str, Dict] = {}


def create_progress(task_id: Optional[str] = None) -> str:
    """Создает новую задачу отслеживания прогресса."""
    if task_id is None:
        task_id = str(uuid.uuid4())
    
    _progress_store[task_id] = {
        "processed": 0,
        "total": 0,
        "progress_percent": 0.0,
        "estimated_remaining_time": 0.0,
        "status": "processing",
        "message": "Инициализация...",
        "start_time": datetime.now().isoformat(),
        "error": None
    }
    return task_id


def update_progress(
    task_id: str,
    processed: int,
    total: int,
    progress_percent: float,
    estimated_remaining_time: float,
    message: Optional[str] = None
):
    """Обновляет прогресс задачи."""
    if task_id in _progress_store:
        # Обновляем total если он был 0 или если передан новый
        current_total = _progress_store[task_id].get("total", 0)
        if total > 0:
            current_total = total
        
        # Формируем сообщение
        if message is None:
            if processed == 0:
                message = f"Начало обработки... Всего строк: {current_total}"
            else:
                message = f"Обработано {processed} из {current_total} строк"
        
        _progress_store[task_id].update({
            "processed": processed,
            "total": current_total,
            "progress_percent": progress_percent,
            "estimated_remaining_time": estimated_remaining_time,
            "message": message,
            "status": "processing"
        })


def complete_progress(task_id: str, message: str = "Готово!"):
    """Отмечает задачу как завершенную."""
    if task_id in _progress_store:
        _progress_store[task_id].update({
            "status": "completed",
            "message": message,
            "progress_percent": 100.0,
            "estimated_remaining_time": 0.0
        })


def error_progress(task_id: str, error_message: str):
    """Отмечает задачу как завершенную с ошибкой."""
    if task_id in _progress_store:
        _progress_store[task_id].update({
            "status": "error",
            "error": error_message,
            "message": f"Ошибка: {error_message}"
        })


def get_progress(task_id: str) -> Optional[Dict]:
    """Получает текущий прогресс задачи."""
    return _progress_store.get(task_id)


def delete_progress(task_id: str):
    """Удаляет задачу из хранилища."""
    _progress_store.pop(task_id, None)

