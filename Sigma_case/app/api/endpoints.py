from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Form, Query
from fastapi.responses import FileResponse
from Sigma_case.app.core.config import settings
from Sigma_case.app.models.schemas import (
    AnswerEvaluationRequest,
    BatchAnswerEvaluationRequest,
    AnswerEvaluationResponse,
    BatchAnswerEvaluationResponse,
    AnswerEvaluationResult
)
from Sigma_case.app.services.answer_evaluator import get_evaluator
from Sigma_case.app.services.csv_answer_evaluator import evaluate_csv_answers, get_csv_info
from Sigma_case.app.services.progress_tracker import (
    create_progress, update_progress, complete_progress, 
    error_progress, get_progress, delete_progress
)

import shutil
from pathlib import Path
from urllib.parse import quote
from typing import Optional

router = APIRouter()

def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    """Сохраняет загруженный файл во временную папку uploads/"""
    file_path = settings.UPLOAD_DIR / upload_file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return file_path

@router.post("/evaluate-answer", response_model=AnswerEvaluationResponse)
async def evaluate_answer_endpoint(request: AnswerEvaluationRequest):
    """
    Эндпоинт для оценки одного ответа на основе обученной AI модели.
    
    Принимает:
    - answer: текст ответа для оценки
    - question: вопрос (опционально)
    - context: контекст (опционально)
    
    Возвращает оценку ответа от 0 до 100.
    """
    try:
        evaluator = get_evaluator()
        
        if not evaluator.is_model_loaded():
            raise HTTPException(
                status_code=503,
                detail=f"AI модель не загружена. Убедитесь, что обученная модель находится в {evaluator.model_path}"
            )
        
        result = evaluator.evaluate_answer(
            answer=request.answer,
            question=request.question,
            context=request.context
        )
        
        evaluation_result = AnswerEvaluationResult(**result)
        
        return AnswerEvaluationResponse(
            result=evaluation_result,
            message="Ответ успешно оценен"
        )
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при оценке ответа: {str(e)}")

@router.post("/evaluate-answers-batch", response_model=BatchAnswerEvaluationResponse)
async def evaluate_answers_batch_endpoint(request: BatchAnswerEvaluationRequest):
    """
    Эндпоинт для оценки нескольких ответов на основе обученной AI модели.
    
    Принимает:
    - answers: список ответов для оценки
    - questions: список вопросов (опционально, должен соответствовать answers)
    - contexts: список контекстов (опционально, должен соответствовать answers)
    
    Возвращает список оценок для каждого ответа.
    """
    try:
        evaluator = get_evaluator()
        
        if not evaluator.is_model_loaded():
            raise HTTPException(
                status_code=503,
                detail=f"AI модель не загружена. Убедитесь, что обученная модель находится в {evaluator.model_path}"
            )
        
        results = evaluator.evaluate_batch(
            answers=request.answers,
            questions=request.questions,
            contexts=request.contexts
        )
        
        evaluation_results = [AnswerEvaluationResult(**r) for r in results]
        
        return BatchAnswerEvaluationResponse(
            results=evaluation_results,
            message=f"Успешно оценено {len(results)} ответов",
            total_evaluated=len(results)
        )
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при оценке ответов: {str(e)}")

@router.post("/evaluate-csv")
async def evaluate_csv_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    max_rows: Optional[int] = Form(None),
    task_id: Optional[str] = None
):
    """
    Эндпоинт для обработки CSV файла с оценкой ответов на основе AI модели.
    
    Принимает CSV файл со структурой:
    - Id экзамена
    - Id вопроса
    - № вопроса (1-4)
    - Текст вопроса (может содержать HTML)
    - Картинка из вопроса (опционально)
    - Оценка экзаменатора
    - Транскрибация ответа
    - Ссылка на оригинальный файл записи
    
    Возвращает .csv файл с результатами обработки, содержащий:
    - Id экзамена
    - № вопроса
    - Текст вопроса
    - Картинка из вопроса
    - Оценка за экзамен (оценка модели)
    - Транскрибация ответа
    """
    task_id_local = None
    try:
        file_path = save_upload_file_tmp(file)
        
        # Создаем задачу отслеживания прогресса
        task_id_local = create_progress(task_id)
        
        # Callback для обновления прогресса
        def progress_callback(processed: int, total: int, progress_percent: float, estimated_time: float):
            update_progress(task_id_local, processed, total, progress_percent, estimated_time)
        
        # Обрабатываем CSV и оцениваем ответы
        result_path = evaluate_csv_answers(
            file_path=str(file_path),
            max_rows=max_rows,
            progress_callback=progress_callback
        )
        
        # Отмечаем как завершенную
        complete_progress(task_id_local, "Обработка завершена")
        
        # Планируем удаление прогресса через 5 минут
        background_tasks.add_task(delete_progress_after_delay, task_id_local, delay=300)
        
        # Поддержка как старого возвращаемого значения (str), так и нового (dict)
        full_path = None
        simple_path = None
        if isinstance(result_path, dict):
            full_path = result_path.get("full")
            simple_path = result_path.get("simple")
        else:
            full_path = str(result_path)

        headers = {"X-Task-Id": task_id_local}
        # Процентное кодирование имен файлов, чтобы соответствовать latin-1 в HTTP заголовках
        full_name_safe = quote(Path(full_path).name)
        headers["X-Full-Result"] = full_name_safe
        if simple_path:
            simple_name_safe = quote(Path(simple_path).name)
            headers["X-Simple-Result"] = simple_name_safe

        return FileResponse(
            path=full_path,
            filename=Path(full_path).name,
            media_type="text/csv",
            headers=headers
        )
    except ValueError as e:
        if task_id_local:
            error_progress(task_id_local, str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        if task_id_local:
            error_progress(task_id_local, str(e))
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке CSV: {str(e)}")


async def delete_progress_after_delay(task_id: str, delay: int = 300):
    """Удаляет прогресс после задержки."""
    import asyncio
    await asyncio.sleep(delay)
    delete_progress(task_id)


@router.get("/progress/{task_id}")
async def get_progress_endpoint(task_id: str):
    """Получает текущий прогресс обработки."""
    progress = get_progress(task_id)
    if progress is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return progress

@router.post("/csv-info")
async def csv_info_endpoint(file: UploadFile = File(...)):
    """Получает информацию о CSV файле (колонки, количество строк)."""
    try:
        file_path = save_upload_file_tmp(file)
        info = get_csv_info(str(file_path))
        return info
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/model-status")
async def model_status_endpoint():
    """Проверяет статус загрузки AI модели."""
    try:
        evaluator = get_evaluator()
        is_loaded = evaluator.is_model_loaded()
        
        return {
            "model_loaded": is_loaded,
            "model_path": str(evaluator.model_path),
            "model_exists": evaluator.model_path.exists(),
            "device": str(evaluator.device)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при проверке статуса модели: {str(e)}")

@router.get("/download-result")
async def download_result(name: str = Query(..., description="Имя файла из директории результатов (percent-encoded)")):
    """Безопасная выдача файла результата по имени из RESULTS_DIR.
    Принимает percent-encoded имя, декодирует, нормализует и отдает CSV.
    """
    try:
        from urllib.parse import unquote
        safe_name = Path(unquote(name)).name  # защита от traversal
        file_path = settings.RESULTS_DIR / safe_name
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="Файл не найден")
        return FileResponse(path=str(file_path), filename=safe_name, media_type="text/csv")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при скачивании файла: {str(e)}")
