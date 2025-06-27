import os
import re
import openai
import requests
import subprocess
import secrets
from fastapi import FastAPI, Request, HTTPException, Depends, Form
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates

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


# --- Админка для Бати ---

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
        result = subprocess.run(["journalctl", "-u", "bitrix-gpt.service", "-n", "50", "--no-pager"], capture_output=True, text=True)
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


# --- Логика для Битрикс24 (старая часть, без изменений) ---

def get_lead_data_from_b24(lead_id):
    if not B24_WEBHOOK_URL_FOR_UPDATE: return None
    try:
        response = requests.post(f"{B24_WEBHOOK_URL_FOR_UPDATE}/crm.lead.get", json={"ID": lead_id})
        response.raise_for_status()
        return response.json().get("result")
    except Exception as e:
        print(f"Ошибка при получении данных лида {lead_id}: {e}")
        return None

def update_b24_lead(lead_id, comment_text):
    if not B24_WEBHOOK_URL_FOR_UPDATE: return
    params = {"fields": {"ENTITY_ID": lead_id, "ENTITY_TYPE": "lead", "COMMENT": f"Ответ от цифрового сотрудника Николая:\n\n{comment_text}"}}
    try:
        requests.post(f"{B24_WEBHOOK_URL_FOR_UPDATE}/crm.timeline.comment.add", json=params)
    except Exception as e:
        print(f"Ошибка при обновлении лида в Битрикс24: {e}")

@app.post("/b24-hook-a8xZk7pQeR1fG3hJkL")
async def b24_hook(req: Request):
    try:
        form_data = await req.form()
        document_id_str = form_data.get("document_id[2]")
        if not document_id_str: raise ValueError("document_id не найден в форме")
        match = re.search(r'\d+', document_id_str)
        if not match: raise ValueError(f"Не удалось извлечь ID из {document_id_str}")
        lead_id = match.group(0)
    except Exception as e:
        print(f"Ошибка парсинга формы от Битрикс: {e}")
        raise HTTPException(status_code=400, detail=f"Bad form data: {e}")

    lead_data = get_lead_data_from_b24(lead_id)
    if not lead_data: raise HTTPException(status_code=500, detail="Не удалось получить данные лида")
    
    task_text = lead_data.get("COMMENTS", "Текст ТЗ не найден.")

    # Читаем промпт и модель из файлов
    try:
        with open(PROMPT_FILE, "r") as f: system_prompt = f.read().strip()
        with open(CURRENT_MODEL_FILE, "r") as f: model_name = f.read().strip()
    except FileNotFoundError:
        print("Ошибка: Файлы настроек prompt.txt или current_model.txt не найдены!")
        # Можно отправить ошибку в Битрикс или просто выйти
        return {"error": "config files not found"}

    user_prompt = f"Проанализируй следующии характиеристики клиента и подготовь ответ для клиента: \n\n{task_text}"

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            max_tokens=2500, temperature=0
        )
        ai_response_text = response.choices[0].message.content
    except Exception as e:
        ai_response_text = f"Ошибка при обращении к OpenAI: {str(e)}"

    update_b24_lead(lead_id, ai_response_text)
    return {"status": "ok"}
