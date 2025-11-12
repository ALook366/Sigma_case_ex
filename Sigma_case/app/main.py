from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from Sigma_case.app.api import endpoints

app = FastAPI(title="Model Trainer & Predictor")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Пути ---
BASE_DIR = Path(__file__).resolve().parent  # /.../app
templates_dir = BASE_DIR / "templates"
static_dir = BASE_DIR / "static"

# --- Проверка наличия директорий ---
if not templates_dir.exists():
    print(f"❌ Не найден каталог шаблонов: {templates_dir}")
else:
    print(f"✅ Каталог шаблонов: {templates_dir}")
    
if not static_dir.exists():
    print(f"❌ Не найден каталог статики: {static_dir}")
else:
    print(f"✅ Каталог статики: {static_dir}")

# --- Подключаем статику и шаблоны ---
try:
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    print("✅ Статика подключена")
except Exception as e:
    print(f"❌ Ошибка подключения статики: {e}")

try:
    templates = Jinja2Templates(directory=templates_dir)
    print("✅ Шаблоны подключены")
except Exception as e:
    print(f"❌ Ошибка подключения шаблонов: {e}")
    raise

# --- Маршрут на главную страницу ---
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    try:
        print("Запрос главной страницы")
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        print(f"Ошибка при загрузке шаблона: {e}")
        import traceback
        traceback.print_exc()
        raise

# Подключаем маршруты API (после статики и главной страницы)
app.include_router(endpoints.router, prefix="/api")
