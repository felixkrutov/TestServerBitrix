import os
import re
import requests
import subprocess
import secrets
from fastapi import FastAPI, Request, HTTPException, Depends, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# --- Импорты для API ---
import openai
import google.generativeai as genai 
from googleapiclient.discovery import build # Для Google Custom Search

# --- Общие настройки ---
app = FastAPI()
security = HTTPBasic()
templates = Jinja2Templates(directory="templates")

# Пути к файлам настроек
MODELS_LIST_FILE = "models_list.txt"
CURRENT_MODEL_FILE = "current_model.txt"
PROMPT_FILE = "prompt.txt"

# --- Получение всех ключей из переменных окружения ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Для Google Custom Search
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID") # Для Google Custom Search

B24_WEBHOOK_URL_FOR_UPDATE = os.getenv("B24_WEBHOOK_URL_FOR_UPDATE")
B24_SECRET_TOKEN = os.getenv("B24_SECRET_TOKEN")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")

# --- Инициализация клиентов API ---
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("WARNING: GEMINI_API_KEY не найден. Функционал Gemini будет недоступен.")

# --- Инструмент для поиска в интернете (для Gemini) ---
def search_internet(query: str):
    """Ищет в интернете с помощью Google Custom Search API по заданному запросу."""
    print(f"TOOL USE: Выполняется поиск в интернете по запросу: '{query}'")
    if not GOOGLE_API_KEY or not SEARCH_ENGINE_ID:
        return "Ошибка: GOOGLE_API_KEY или SEARCH_ENGINE_ID не настроены в переменных окружения."
    
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        res = service.cse().list(q=query, cx=SEARCH_ENGINE_ID, num=3).execute()
        items = res.get('items', [])
        if not items:
            return "Поиск не дал результатов."
        snippets = []
        for item in items:
            snippets.append(f"Заголовок: {item.get('title', '')}\nФрагмент: {item.get('snippet', '')}\nИсточник: {item.get('link', '')}")
        return "\n\n".join(snippets)
    except Exception as e:
        print(f"TOOL ERROR: Ошибка при поиске Google: {e}")
        return f"Ошибка при выполнении поиска: {e}"

# Описание инструмента для Gemini
gemini_tools = [
    {
        "function_declarations": [
            {
                "name": "search_internet",
                "description": "Ищет в интернете актуальную информацию, новости, факты по заданному запросу, если в текущем контексте нет ответа.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "query": {"type_": "STRING", "description": "Поисковый запрос на русском или английском языке"}
                    },
                    "required": ["query"]
                }
            }
        ]
    }
]

# --- Универсальная функция для вызова ИИ (ИСПРАВЛЕННАЯ ВЕРСИЯ) ---
def get_ai_response(model_name: str, full_input_prompt: str):
    """
    Получает ответ от ИИ, автоматически выбирая API (OpenAI или Gemini)
    и обрабатывая инструменты (поиск в интернете).
    """
    
    # --- Вариант 1: OpenAI ---
    if model_name.startswith("gpt-") or model_name.startswith("o4-"):
        print(f"INFO: Используется OpenAI API для модели {model_name}")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY не установлен.")
        response = openai_client.responses.create(
            model=model_name,
            input=full_input_prompt,
            tools=[{"type": "web_search_preview"}]
        )
        return response.output_text

    # --- Вариант 2: Gemini ---
    elif model_name.startswith("gemini-"):
        print(f"INFO: Используется Gemini API для модели {model_name}")
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY не установлен.")
            
        model = genai.GenerativeModel(model_name, tools=gemini_tools)
        chat = model.start_chat()
        response = chat.send_message(full_input_prompt)
        
        try:
            function_call = response.candidates[0].content.parts[0].function_call
        except (IndexError, AttributeError):
            return response.text # Если нет вызова функции, возвращаем текст

        if function_call.name == "search_internet":
            query = function_call.args['query']
            search_result = search_internet(query=query)
            response_after_search = chat.send_message(
                genai.types.Part(
                    function_response=genai.types.FunctionResponse(
                        name='search_internet',
                        response={'result': search_result},
                    ),
                ),
            )
            return response_after_search.text
        else:
            return f"Ошибка: Gemini запросил неизвестный инструмент '{function_call.name}'"

    # --- Вариант 3: Неизвестный провайдер ---
    else:
        raise ValueError(f"Ошибка: Неизвестный провайдер для модели '{model_name}'.")

# --- Админка, чат и логика Битрикс24 ---
def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password", headers={"WWW-Authenticate": "Basic"})
    return credentials.username

@app.get("/", response_class=HTMLResponse)
async def read_admin_ui(request: Request, username: str = Depends(get_current_username)):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/status")
async def get_status(username: str = Depends(get_current_username)):
    try:
        result = subprocess.run(["systemctl", "is-active", "bitrix-gpt.service"], capture_output=True, text=True)
        status = result.stdout.strip()
    except FileNotFoundError:
        status = "failed"
    return {"status": status}

@app.get("/api/logs")
async def get_logs(username: str = Depends(get_current_username)):
    try:
        result = subprocess.run(["journalctl", "-u", "bitrix-gpt.service", "--since", "5 minutes ago", "--no-pager"], capture_output=True, text=True)
        logs = result.stdout
    except FileNotFoundError:
        logs = "Не удалось загрузить логи."
    return {"logs": logs}

@app.get("/api/settings")
async def get_settings(username: str = Depends(get_current_username)):
    default_models = ["o4-mini-2025-04-16", "gemini-2.5-pro"]
    try:
        with open(MODELS_LIST_FILE, "r") as f: models_list = [line.strip() for line in f]
    except FileNotFoundError: models_list = default_models
    try:
        with open(CURRENT_MODEL_FILE, "r") as f: current_model = f.read().strip()
    except FileNotFoundError: current_model = models_list[0] if models_list else default_models[0]
    try:
        with open(PROMPT_FILE, "r") as f: prompt = f.read().strip()
    except FileNotFoundError: prompt = "Промпт по умолчанию"
    return {"models_list": models_list, "current_model": current_model, "prompt": prompt}

@app.post("/api/settings")
async def save_settings(username: str = Depends(get_current_username), model: str = Form(...), prompt: str = Form(...)):
    with open(CURRENT_MODEL_FILE, "w") as f: f.write(model)
    with open(PROMPT_FILE, "w") as f: f.write(prompt)
    subprocess.run(["systemctl", "restart", "bitrix-gpt.service"])
    return {"status": "ok"}

class ChatRequest(BaseModel):
    user_message: str

@app.post("/api/chat")
async def handle_chat(chat_request: ChatRequest, username: str = Depends(get_current_username)):
    try:
        with open(PROMPT_FILE, "r") as f: system_prompt = f.read().strip()
        with open(CURRENT_MODEL_FILE, "r") as f: model_name = f.read().strip()
    except FileNotFoundError:
        return JSONResponse(status_code=500, content={"ai_response": "Ошибка: Файлы настроек не найдены."})
    full_input_prompt = f"{system_prompt}\n\nПроанализируй следующии характиристики клиента и подготовь ответ для клиента: \n\n{chat_request.user_message}"
    try:
        ai_response_text = get_ai_response(model_name, full_input_prompt)
    except Exception as e:
        ai_response_text = f"Ошибка при обращении к ИИ ({model_name}): {str(e)}"
    return {"ai_response": ai_response_text}

def get_lead_data_from_b24(lead_id):
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
    print(f"DEBUG: Обновление лида {lead_id} в Битрикс24...")
    if not B24_WEBHOOK_URL_FOR_UPDATE: return
    params = {"fields": {"ENTITY_ID": lead_id, "ENTITY_TYPE": "lead", "COMMENT": f"{comment_text}"}}
    try:
        requests.post(f"{B24_WEBHOOK_URL_FOR_UPDATE}/crm.timeline.comment.add", json=params)
        print(f"DEBUG: Лид {lead_id} успешно обновлен.")
    except Exception as e:
        print(f"ERROR: Ошибка при обновлении лида в Битрикс24: {e}")

def process_lead_in_background(lead_id: str):
    print(f"BACKGROUND: Начало фоновой обработки лида {lead_id}.")
    lead_data = get_lead_data_from_b24(lead_id)
    if not lead_data: 
        print(f"BACKGROUND ERROR: Не удалось получить данные лида {lead_id}, прерываем.")
        return 
    task_text = lead_data.get("COMMENTS", "Текст ТЗ не найден.")
    try:
        with open(PROMPT_FILE, "r") as f: system_prompt = f.read().strip()
        with open(CURRENT_MODEL_FILE, "r") as f: model_name = f.read().strip()
    except FileNotFoundError:
        error_message = "Ошибка: Файлы настроек (prompt.txt или current_model.txt) не найдены!"
        print(f"BACKGROUND ERROR: {error_message}")
        update_b24_lead(lead_id, error_message)
        return
    full_input_prompt = f"{system_prompt}\n\nПроанализируй следующии характиристики клиента и подготовь ответ для клиента: \n\n{task_text}"
    print(f"BACKGROUND: Запрос к ИИ для лида {lead_id} сформирован. Отправка...")
    try:
        ai_response_text = get_ai_response(model_name, full_input_prompt)
        print(f"BACKGROUND: Ответ от ИИ для лида {lead_id} получен.")
    except Exception as e:
        ai_response_text = f"Ошибка при обращении к ИИ ({model_name}): {str(e)}"
        print(f"BACKGROUND ERROR: {ai_response_text}")
    update_b24_lead(lead_id, ai_response_text)
    print(f"BACKGROUND: Фоновая обработка лида {lead_id} завершена успешно.")

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
    background_tasks.add_task(process_lead_in_background, lead_id)
    print(f"DEBUG: Задача для лида {lead_id} добавлена в фон. Мгновенно отвечаем Битрикс24.")
    return {"status": "ok, task accepted"}
