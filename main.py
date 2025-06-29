import os
import re
import openai
import requests
import subprocess
import secrets
from fastapi import FastAPI, Request, HTTPException, Depends, Form, BackgroundTasks # <<< ИЗМЕНЕНИЕ
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# --- Общие настройки ---
app = FastAPI()
# ... (весь твой код до логики Битрикс24 без изменений) ...
# --- Общие настройки ---
app = FastAPI()
security = HTTPBasic()
templates = Jinja2Templates(directory="templates")

# Пути к файлам настроек
MODELS_LIST_FILE = "models_list.txt"
CURRENT_MODEL_FILE = "current_model.txt"
PROMPT_FILE = "prompt.txt"

# Получаем ключи и учетные данные для админки из переменных окружения
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
B24_WEBHOOK_URL_FOR_UPDATE = os.getenv("B24_WEBHOOK_URL_FOR_UPDATE")
B24_SECRET_TOKEN = os.getenv("B24_SECRET_TOKEN")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")

# Инициализируем клиент OpenAI
client = openai.OpenAI(api_key=OPENAI_API_KEY)


# Функция для проверки пароля
def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# Главная страница админки. ЗАМЕНЯЕТ СТАРЫЙ @app.get("/")
@app.get("/", response_class=HTMLResponse)
async def read_admin_ui(request: Request, username: str = Depends(get_current_username)):
    return templates.TemplateResponse("index.html", {"request": request})

# API для получения статуса
@app.get("/api/status")
async def get_status(username: str = Depends(get_current_username)):
    try:
        result = subprocess.run(["systemctl", "is-active", "bitrix-gpt.service"], capture_output=True, text=True)
        status = result.stdout.strip()
    except FileNotFoundError:
        status = "failed"
    return {"status": status}

# API для получения логов
@app.get("/api/logs")
async def get_logs(username: str = Depends(get_current_username)):
    try:
        result = subprocess.run(["journalctl", "-u", "bitrix-gpt.service", "--since", "5 minutes ago", "--no-pager"], capture_output=True, text=True)
        logs = result.stdout
    except FileNotFoundError:
        logs = "Не удалось загрузить логи."
    return {"logs": logs}

# API для получения текущих настроек
@app.get("/api/settings")
async def get_settings(username: str = Depends(get_current_username)):
    try:
        with open(MODELS_LIST_FILE, "r") as f: models_list = [line.strip() for line in f]
    except FileNotFoundError: models_list = ["gpt-4o"] # Значение по-умолчанию, если файла нет
    try:
        with open(CURRENT_MODEL_FILE, "r") as f: current_model = f.read().strip()
    except FileNotFoundError: current_model = models_list[0]
    try:
        with open(PROMPT_FILE, "r") as f: prompt = f.read().strip()
    except FileNotFoundError: prompt = "Промпт по умолчанию"
    return {"models_list": models_list, "current_model": current_model, "prompt": prompt}

# API для сохранения новых настроек
@app.post("/api/settings")
async def save_settings(
    username: str = Depends(get_current_username),
    model: str = Form(...),
    prompt: str = Form(...)
):
    with open(CURRENT_MODEL_FILE, "w") as f: f.write(model)
    with open(PROMPT_FILE, "w") as f: f.write(prompt)
    subprocess.run(["systemctl", "restart", "bitrix-gpt.service"])
    return {"status": "ok"}


# --- Тестовый чат в админке ---

class ChatRequest(BaseModel):
    user_message: str

@app.post("/api/chat")
async def handle_chat(
    chat_request: ChatRequest,
    username: str = Depends(get_current_username)
):
    try:
        with open(PROMPT_FILE, "r") as f: system_prompt = f.read().strip()
        with open(CURRENT_MODEL_FILE, "r") as f: model_name = f.read().strip()
    except FileNotFoundError:
        return JSONResponse(status_code=500, content={"ai_response": "Ошибка: Файлы настроек (prompt.txt или current_model.txt) не найдены."})
    full_input_prompt = f"{system_prompt}\n\nПроанализируй следующии характиристики клиента и подготовь ответ для клиента: \n\n{chat_request.user_message}"
    try:
        response = client.responses.create(
            model=model_name,
            input=full_input_prompt,
            tools=[{"type": "web_search_preview"}],
        )
        ai_response_text = response.output_text
    except Exception as e:
        ai_response_text = f"Ошибка при обращении к OpenAI: {str(e)}"
    return {"ai_response": ai_response_text}


# --- Логика для Битрикс24 (ПЕРЕДЕЛАНА НА АСИНХРОННУЮ) ---

def get_lead_data_from_b24(lead_id):
    # ... эта функция без изменений ...
    print(f"DEBUG: Получение данных лида {lead_id} из Битрикс24...")
    if not B24_WEBHOOK_URL_FOR_UPDATE: return None
    try:
        response = requests.post(f"{B24_WEBHOOK_URL_FOR_UPDATE}/crm.lead.get", json={"ID": lead_id})
        response.raise_for_status()
        print(f"DEBUG: Данные лида {lead_id} успешно получены.")
        return response.json().get("result")
    except Exception as e:
        print(f"ERROR: Ошибка при получении данных лида {lead_id}: {e}")
        return None

def update_b24_lead(lead_id, comment_text):
    # ... эта функция без изменений ...
    print(f"DEBUG: Обновление лида {lead_id} в Битрикс24...")
    if not B24_WEBHOOK_URL_FOR_UPDATE: return
    params = {"fields": {"ENTITY_ID": lead_id, "ENTITY_TYPE": "lead", "COMMENT": f"{comment_text}"}}
    try:
        requests.post(f"{B24_WEBHOOK_URL_FOR_UPDATE}/crm.timeline.comment.add", json=params)
        print(f"DEBUG: Лид {lead_id} успешно обновлен.")
    except Exception as e:
        print(f"ERROR: Ошибка при обновлении лида в Битрикс24: {e}")

# <<< НОВАЯ ФУНКЦИЯ ДЛЯ РАБОТЫ В ФОНЕ >>>
def process_lead_in_background(lead_id: str):
    """Эта функция будет выполняться в фоне, уже после того, как мы ответили Битриксу."""
    print(f"BACKGROUND: Начало фоновой обработки лида {lead_id}.")
    
    # 1. Получаем данные лида (старая логика)
    lead_data = get_lead_data_from_b24(lead_id)
    if not lead_data: 
        print(f"BACKGROUND ERROR: Не удалось получить данные лида {lead_id}, прерываем.")
        return # Просто выходим, если не смогли получить данные
    
    task_text = lead_data.get("COMMENTS", "Текст ТЗ не найден.")
    
    # 2. Читаем настройки (старая логика)
    try:
        with open(PROMPT_FILE, "r") as f: system_prompt = f.read().strip()
        with open(CURRENT_MODEL_FILE, "r") as f: model_name = f.read().strip()
    except FileNotFoundError:
        error_message = "Ошибка: Файлы настроек (prompt.txt или current_model.txt) не найдены!"
        print(f"BACKGROUND ERROR: {error_message}")
        update_b24_lead(lead_id, error_message) # Отправляем ошибку в Битрикс
        return

    full_input_prompt = f"{system_prompt}\n\nПроанализируй следующии характиристики клиента и подготовь ответ для клиента: \n\n{task_text}"
    print(f"BACKGROUND: Запрос к ИИ для лида {lead_id} сформирован. Отправка...")

    # 3. Обращаемся к OpenAI (старая логика)
    try:
        response = client.responses.create(
            model=model_name,
            input=full_input_prompt,
            tools=[{"type": "web_search_preview"}],
        )
        ai_response_text = response.output_text
        print(f"BACKGROUND: Ответ от ИИ для лида {lead_id} получен.")
    except Exception as e:
        ai_response_text = f"Ошибка при обращении к OpenAI с web_search: {str(e)}"
        print(f"BACKGROUND ERROR: {ai_response_text}")

    # 4. Обновляем лид в Битриксе (старая логика)
    update_b24_lead(lead_id, ai_response_text)
    print(f"BACKGROUND: Фоновая обработка лида {lead_id} завершена успешно.")


# <<< ИЗМЕНЕННАЯ ФУНКЦИЯ-ХУК >>>
@app.post("/b24-hook-a8xZk7pQeR1fG3hJkL")
async def b24_hook(req: Request, background_tasks: BackgroundTasks):
    print("DEBUG: Запрос от Битрикс24 получен.")
    try:
        form_data = await req.form()
        document_id_str = form_data.get("document_id[2]")
        if not document_id_str: raise ValueError("document_id не найден в форме")
        match = re.search(r'\d+', document_id_str)
        if not match: raise ValueError(f"Не удалось извлечь ID из {document_id_str}")
        lead_id = match.group(0)
        print(f"DEBUG: Извлечен ID лида: {lead_id}")
    except Exception as e:
        print(f"ERROR: Ошибка парсинга формы от Битрикс: {e}")
        raise HTTPException(status_code=400, detail=f"Bad form data: {e}")

    # Добавляем долгую задачу в фон
    background_tasks.add_task(process_lead_in_background, lead_id)
    
    # И СРАЗУ ЖЕ отвечаем Битриксу, что все хорошо
    print(f"DEBUG: Задача для лида {lead_id} добавлена в фон. Мгновенно отвечаем Битрикс24.")
    return {"status": "ok, task accepted"}
