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
import json

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
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

B24_WEBHOOK_URL_FOR_UPDATE = os.getenv("B24_WEBHOOK_URL_FOR_UPDATE")
B24_SECRET_TOKEN = os.getenv("B24_SECRET_TOKEN")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")

# --- Инициализация клиентов API ---
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("WARNING: GEMINI_API_KEY не найден.")

# --- КОНСТАНТЫ ДЛЯ РАСЧЕТОВ (перенесены из промпта в код) ---
PIR_INDEX_2025_Q2 = 6.53
K_s = 1.15
K_t = 1.10
K_r = 1.10
K_total = K_s * K_t * K_r
EBITDA_MARGIN = 0.15
BUILD_COST_BASE = 110_000
BUILD_COST_LOW = 95_000
BUILD_COST_HIGH = 120_000
EQUIP_SHARE_DEF = 0.95
OTHER_SHARE = 0.10
# Данные СБЦ (a, b в тыс. руб. 2001 г.)
SBC_DATA = {
    "мукомольно-крупяная": {"a": 5147.2, "b": 2.4},
    "кондитерская": {"a": 6280.0, "b": 18.5},
    # Добавь сюда другие категории, когда они понадобятся
    "по умолчанию": {"a": 5147.2, "b": 2.4} # Для случаев, когда категория не найдена
}

# --- НОВЫЕ ИНСТРУМЕНТЫ-КАЛЬКУЛЯТОРЫ ---
def calculate_throughput_and_revenue(area_sqm: float, category: str):
    """Рассчитывает годовую производительность (Q) и годовую выручку (Revenue)."""
    print(f"TOOL USE: calculate_throughput_and_revenue(area_sqm={area_sqm}, category='{category}')")
    q_per_year = area_sqm / 0.485  # т/год
    # Здесь должны быть реальные данные по цене, пока используем заглушку
    avg_price_per_kg = 100.0 # руб/кг
    revenue = q_per_year * 1000 * avg_price_per_kg
    ebitda = EBITDA_MARGIN * revenue
    return json.dumps({
        "q_per_year": q_per_year,
        "revenue": revenue,
        "ebitda": ebitda
    })

def calculate_capex(area_sqm: float, ebitda: float):
    """Рассчитывает CAPEX и срок окупаемости (Payback)."""
    print(f"TOOL USE: calculate_capex(area_sqm={area_sqm}, ebitda={ebitda})")
    # Логика выбора стоимости строительства
    temp_capex = BUILD_COST_BASE * area_sqm * (1 + EQUIP_SHARE_DEF + OTHER_SHARE)
    temp_payback = temp_capex / ebitda if ebitda > 0 else float('inf')
    
    build_cost = BUILD_COST_BASE
    if temp_payback < 1.2:
        build_cost = BUILD_COST_HIGH
    elif temp_payback > 2.5:
        build_cost = BUILD_COST_LOW

    capex_build = build_cost * area_sqm
    capex_equip = EQUIP_SHARE_DEF * capex_build
    capex_other = OTHER_SHARE * capex_build
    capex_total = capex_build + capex_equip + capex_other
    payback_years = capex_total / ebitda if ebitda > 0 else float('inf')

    return json.dumps({
        "capex_total": capex_total,
        "payback_years": payback_years
    })

def calculate_design_cost(q_per_year: float, capex_total: float, category: str):
    """Рассчитывает стоимость проектирования (П+РД)."""
    print(f"TOOL USE: calculate_design_cost(q_per_year={q_per_year}, capex_total={capex_total}, category='{category}')")
    q_per_day = q_per_year / 365
    
    # Выбор коэффициентов a и b
    sbc = SBC_DATA.get(category, SBC_DATA["по умолчанию"])
    a, b = sbc["a"], sbc["b"]

    c0 = (a + b * q_per_day) * 1000 # в рублях 2001 г.
    c1 = c0 * PIR_INDEX_2025_Q2
    c_final = c1 * K_total
    
    design_cost_mln = c_final / 1_000_000
    design_share_percent = (c_final / capex_total) * 100 if capex_total > 0 else 0
    justification_mln = 0.10 * design_cost_mln

    return json.dumps({
        "design_cost_mln": design_cost_mln,
        "design_share_percent": design_share_percent,
        "justification_mln": justification_mln
    })

# --- ИНСТРУМЕНТ ДЛЯ ПОИСКА В ИНТЕРНЕТЕ ---
def search_internet(query: str):
    # ... (код этой функции без изменений) ...
    print(f"TOOL USE: Выполняется поиск в интернете по запросу: '{query}'")
    if not GOOGLE_API_KEY or not SEARCH_ENGINE_ID:
        return "Ошибка: GOOGLE_API_KEY или SEARCH_ENGINE_ID не настроены в переменных окружения."
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        res = service.cse().list(q=query, cx=SEARCH_ENGINE_ID, num=3).execute()
        items = res.get('items', [])
        if not items: return "Поиск не дал результатов."
        snippets = [f"Заголовок: {item.get('title', '')}\nФрагмент: {item.get('snippet', '')}\nИсточник: {item.get('link', '')}" for item in items]
        return "\n\n".join(snippets)
    except Exception as e:
        print(f"TOOL ERROR: Ошибка при поиске Google: {e}")
        return f"Ошибка при выполнении поиска: {e}"


# --- ОПИСАНИЕ ВСЕХ ИНСТРУМЕНТОВ ДЛЯ GEMINI ---
gemini_tools_declarations = [
    {"name": "calculate_throughput_and_revenue", "description": "Рассчитывает годовую производительность и выручку на основе площади и категории.", "parameters": {"type_": "OBJECT", "properties": {"area_sqm": {"type_": "NUMBER"}, "category": {"type_": "STRING"}}, "required": ["area_sqm", "category"]}},
    {"name": "calculate_capex", "description": "Рассчитывает общий CAPEX и срок окупаемости на основе площади и EBITDA.", "parameters": {"type_": "OBJECT", "properties": {"area_sqm": {"type_": "NUMBER"}, "ebitda": {"type_": "NUMBER"}}, "required": ["area_sqm", "ebitda"]}},
    {"name": "calculate_design_cost", "description": "Рассчитывает стоимость проектирования на основе годовой производительности, CAPEX и категории.", "parameters": {"type_": "OBJECT", "properties": {"q_per_year": {"type_": "NUMBER"}, "capex_total": {"type_": "NUMBER"}, "category": {"type_": "STRING"}}, "required": ["q_per_year", "capex_total", "category"]}},
    {"name": "search_internet", "description": "Ищет в интернете актуальную информацию.", "parameters": {"type_": "OBJECT", "properties": {"query": {"type_": "STRING"}}, "required": ["query"]}}
]
gemini_tools = [{"function_declarations": gemini_tools_declarations}]
# Словарь для вызова функций по имени
available_tools = {
    "calculate_throughput_and_revenue": calculate_throughput_and_revenue,
    "calculate_capex": calculate_capex,
    "calculate_design_cost": calculate_design_cost,
    "search_internet": search_internet,
}

# --- Универсальная функция для вызова ИИ (с циклом для инструментов) ---
def get_ai_response(model_name: str, full_input_prompt: str):
    # --- Вариант 1: OpenAI ---
    if model_name.startswith("gpt-") or model_name.startswith("o4-"):
        # ... (код для OpenAI без изменений) ...
        print(f"INFO: Используется OpenAI API для модели {model_name}")
        if not OPENAI_API_KEY: raise ValueError("OPENAI_API_KEY не установлен.")
        response = openai_client.responses.create(model=model_name, input=full_input_prompt, tools=[{"type": "web_search_preview"}])
        return response.output_text

    # --- Вариант 2: Gemini с инструментами ---
    elif model_name.startswith("gemini-"):
        print(f"INFO: Используется Gemini API для модели {model_name}")
        if not GEMINI_API_KEY: raise ValueError("GEMINI_API_KEY не установлен.")
        
        model = genai.GenerativeModel(model_name, tools=gemini_tools)
        chat = model.start_chat()
        response = chat.send_message(full_input_prompt)
        
        while True:
            try:
                function_call = response.candidates[0].content.parts[0].function_call
                if not hasattr(function_call, 'name'):
                    # Если вызов есть, но он "пустой", выходим из цикла
                    break
                
                function_name = function_call.name
                if function_name in available_tools:
                    function_to_call = available_tools[function_name]
                    function_args = function_call.args
                    
                    # Преобразуем аргументы в нужный формат
                    args_for_call = {key: value for key, value in function_args.items()}
                    
                    # Вызываем нужную функцию
                    function_response = function_to_call(**args_for_call)
                    
                    # Отправляем результат обратно в Gemini
                    response = chat.send_message(
                        genai.types.Part(
                            function_response=genai.types.FunctionResponse(
                                name=function_name,
                                response={'result': function_response},
                            ),
                        ),
                    )
                else:
                    return f"Ошибка: ИИ попытался вызвать неизвестный инструмент: {function_name}"
            
            except (IndexError, AttributeError, ValueError):
                # Если вызова функции нет, значит, это финальный ответ
                break
        
        return response.text

    # --- Вариант 3: Неизвестный провайдер ---
    else:
        raise ValueError(f"Ошибка: Неизвестный провайдер для модели '{model_name}'.")

# --- Админка, чат и логика Битрикс24 ---
# (Этот код использует универсальную функцию get_ai_response, поэтому он остается без изменений)

def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    # ... без изменений ...
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
    # ... без изменений ...
    try:
        result = subprocess.run(["systemctl", "is-active", "bitrix-gpt.service"], capture_output=True, text=True)
        status = result.stdout.strip()
    except FileNotFoundError:
        status = "failed"
    return {"status": status}


@app.get("/api/logs")
async def get_logs(username: str = Depends(get_current_username)):
    # ... без изменений ...
    try:
        result = subprocess.run(["journalctl", "-u", "bitrix-gpt.service", "--since", "5 minutes ago", "--no-pager"], capture_output=True, text=True)
        logs = result.stdout
    except FileNotFoundError:
        logs = "Не удалось загрузить логи."
    return {"logs": logs}


@app.get("/api/settings")
async def get_settings(username: str = Depends(get_current_username)):
    # ... без изменений ...
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
    # ... без изменений ...
    with open(CURRENT_MODEL_FILE, "w") as f: f.write(model)
    with open(PROMPT_FILE, "w") as f: f.write(prompt)
    subprocess.run(["systemctl", "restart", "bitrix-gpt.service"])
    return {"status": "ok"}


class ChatRequest(BaseModel):
    user_message: str

@app.post("/api/chat")
async def handle_chat(chat_request: ChatRequest, username: str = Depends(get_current_username)):
    # ... без изменений ...
    try:
        with open(PROMPT_FILE, "r") as f: system_prompt = f.read().strip()
        with open(CURRENT_MODEL_FILE, "r") as f: model_name = f.read().strip()
    except FileNotFoundError:
        return JSONResponse(status_code=500, content={"ai_response": "Ошибка: Файлы настроек не найдены."})
    full_input_prompt = f"{system_prompt}\n\n{chat_request.user_message}" # Убираем лишнюю инструкцию, так как ИИ теперь сам будет решать, что делать
    try:
        ai_response_text = get_ai_response(model_name, full_input_prompt)
    except Exception as e:
        ai_response_text = f"Ошибка при обращении к ИИ ({model_name}): {str(e)}"
    return {"ai_response": ai_response_text}

def get_lead_data_from_b24(lead_id):
    # ... без изменений ...
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
    # ... без изменений ...
    print(f"DEBUG: Обновление лида {lead_id} в Битрикс24...")
    if not B24_WEBHOOK_URL_FOR_UPDATE: return
    params = {"fields": {"ENTITY_ID": lead_id, "ENTITY_TYPE": "lead", "COMMENT": f"{comment_text}"}}
    try:
        requests.post(f"{B24_WEBHOOK_URL_FOR_UPDATE}/crm.timeline.comment.add", json=params)
        print(f"DEBUG: Лид {lead_id} успешно обновлен.")
    except Exception as e:
        print(f"ERROR: Ошибка при обновлении лида в Битрикс24: {e}")

def process_lead_in_background(lead_id: str):
    # ... без изменений ...
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
    full_input_prompt = f"{system_prompt}\n\n{task_text}"
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
    # ... без изменений ...
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
