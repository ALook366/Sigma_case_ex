from pydantic import BaseModel
from typing import List, Optional, Union

class AnswerEvaluationRequest(BaseModel):
    """Запрос на оценку ответа."""
    answer: str
    question: Optional[str] = None
    context: Optional[str] = None

class BatchAnswerEvaluationRequest(BaseModel):
    """Запрос на оценку нескольких ответов."""
    answers: List[str]
    questions: Optional[List[str]] = None
    contexts: Optional[List[str]] = None

class AnswerEvaluationResult(BaseModel):
    """Результат оценки ответа."""
    score: float
    raw_score: Optional[float] = None
    predicted_class: Optional[int] = None
    confidence: Optional[float] = None
    answer: str
    status: str
    error: Optional[str] = None

class AnswerEvaluationResponse(BaseModel):
    """Ответ с результатом оценки."""
    result: AnswerEvaluationResult
    message: str

class BatchAnswerEvaluationResponse(BaseModel):
    """Ответ с результатами оценки нескольких ответов."""
    results: List[AnswerEvaluationResult]
    message: str
    total_evaluated: int
