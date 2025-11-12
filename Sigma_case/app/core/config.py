import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # Sigma_case/app/core -> Sigma_case/app -> Sigma_case

class Settings:
    PROJECT_NAME: str = "ML Trainer & Predictor"
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    MODELS_DIR: Path = BASE_DIR / "models"
    RESULTS_DIR: Path = BASE_DIR / "results"
    # Путь к AI модели: Sigma_case/trained_models/current_model
    AI_MODEL_DIR: Path = BASE_DIR / "trained_models" / "current_model"
    # Альтернативные пути
    AI_MODEL_DIR_ALT: Path = BASE_DIR.parent / "trained_models" / "current_model"
    AI_MODEL_DIR_ALT2: Path = BASE_DIR.parent.parent / "trained_models" / "current_model"

    def __init__(self):
        for path in [self.UPLOAD_DIR, self.MODELS_DIR, self.RESULTS_DIR]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Создаем директорию для AI модели если её нет
        if not self.AI_MODEL_DIR.exists():
            self.AI_MODEL_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()
