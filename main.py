import os
import re
import shutil
import requests
import subprocess
import secrets
from fastapi import FastAPI, Request, HTTPException, Depends, Form, BackgroundTasks, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# --- Импорты для API ---
import openai
import google.generativeai as genai 

# --- НОВЫЕ ИМПОРТЫ ДЛЯ RAG (БАЗЫ ЗНАНИЙ) ---
import chromadb
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    PandasExcelLoader,
)
from langchain_openai import OpenAIEmbeddings # Используем эмбеддинги от OpenAI, они качественные

# --- Общие настройки ---
app = FastAPI()
security = HTTPBasic()
templates = Jinja2Templates(directory="templates")

# --- Пути к файлам и папкам ---
MODELS_LIST_FILE = "models_list.txt"
CURRENT_MODEL_FILE = "current_model.txt"
PROMPT_FILE = "prompt.txt"
DOCS_DIR = "documents"
DB_DIR = "chroma_db"

# --- Получение всех ключей из переменных окружения ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
B24_WEBHOOK_URL_FOR_UPDATE = os.getenv("B24_WEBHOOK_URL_FOR_UPDATE")
B24_SECRET_TOKEN = os.getenv("B24_SECRET_TOKEN")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")

# --- Инициализация клиентов API и RAG ---
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("WARNING: GEMINI_API_KEY не найден.")

# --- RAG: База Знаний ---
# Создаем папки, если их нет
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# Инициализируем модель для эмбеддингов и векторную базу
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Словарь для выбора загрузчика в зависимости от типа файла
LOADER_MAPPING = {
    ".txt": (TextLoader, {"encoding": "utf-8"}),
    ".md": (TextLoader, {"encoding": "utf-8"}),
    ".pdf": (PyPDFLoader, {}),
    ".docx": (Docx2txtLoader, {}),
    ".xlsx": (PandasExcelLoader, {}),
}

def load_and_process_document(file_path: str):
    """Загружает и обрабатывает один документ, добавляя его в базу."""
    ext = "." + file_path.rsplit(".", 1)[-1].lower()
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        vectorstore.add_documents(documents=docs)
        vectorstore.persist()
        print(f"INFO: Документ {file_path} успешно обработан и добавлен в базу.")
    else:
        print(f"WARNING: Неподдерживаемый формат файла: {file_path}")

@app.on_event("startup")
def on_startup():
    """При старте сервера проверяем все документы в папке."""
    # Эта функция может быть доработана для проверки уже обработанных файлов
    print("INFO: Сервер запущен. Проверка базы знаний...")

@app.post("/api/upload-document")
async def upload_document(file: UploadFile = File(...), username: str = Depends(get_current_username)):
    file_path = os.path.join(DOCS_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    load_and_process_document(file_path)
    return JSONResponse(content={"message": f"Файл '{file.filename}' успешно загружен и обработан."}, status_code=200)

@app.get("/api/documents")
async def get_documents(username: str = Depends(get_current_username)):
    files = [f for f in os.listdir(DOCS_DIR) if os.path.isfile(os.path.join(DOCS_DIR, f))]
    return JSONResponse(content={"documents": files})

@app.delete("/api/documents/{filename}")
async def delete_document(filename: str, username: str = Depends(get_current_username)):
    file_path = os.path.join(DOCS_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        # ПРИМЕЧАНИЕ: Удаление из ChromaDB - сложный процесс.
        # Пока мы просто удаляем файл. Для полной очистки нужно пересоздавать базу.
        # Для простоты, мы пока оставим векторы в базе.
        print(f"INFO: Файл {filename} удален. Для полной очистки базы ее нужно пересоздать.")
        return JSONResponse(content={"message": f"Файл '{filename}' удален."}, status_code=200)
    else:
        raise HTTPException(status_code=404, detail="Файл не найден")

# --- Универсальная функция для вызова ИИ (теперь с RAG) ---
def get_ai_response(model_name: str, full_input_prompt: str, user_query: str):
    """
    Получает ответ от ИИ, обогащая промпт данными из векторной базы.
    """
    print("INFO: Поиск релевантных документов в базе знаний...")
    # Ищем 3 самых релевантных документа
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(user_query)
    
    context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Создаем новый промпт с контекстом
    rag_prompt = f"""
CONTEXT:
{context}
---
PROMPT:
{full_input_prompt}

Используя предоставленный CONTEXT, ответь на PROMPT. Если в контексте нет ответа, сообщи, что информация не найдена в базе знаний.
"""
    
    # --- Вариант 1: OpenAI ---
    if model_name.startswith("gpt-") or model_name.startswith("o4-"):
        print(f"INFO: Используется OpenAI API для модели {model_name}")
        if not OPENAI_API_KEY: raise ValueError("OPENAI_API_KEY не установлен.")
        # Убираем web_search, так как теперь приоритет у нашей базы
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": rag_prompt}]
        )
        return response.choices[0].message.content

    # --- Вариант 2: Gemini ---
    elif model_name.startswith("gemini-"):
        print(f"INFO: Используется Gemini API для модели {model_name}")
        if not GEMINI_API_KEY: raise ValueError("GEMINI_API_KEY не установлен.")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(rag_prompt)
        return response.text

    # --- Вариант 3: Неизвестный провайдер ---
    else:
        raise ValueError(f"Ошибка: Неизвестный провайдер для модели '{model_name}'.")

# --- Админка, чат и логика Битрикс24 ---
# (Этот код использует универсальную функцию get_ai_response, поэтому он остается без изменений)
def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (correct_username and correct_password): raise HTTPException(status_code=401, detail="Incorrect username or password", headers={"WWW-Authenticate": "Basic"})
    return credentials.username
@app.get("/", response_class=HTMLResponse)
async def read_admin_ui(request: Request, username: str = Depends(get_current_username)): return templates.TemplateResponse("index.html", {"request": request})
@app.get("/api/status")
async def get_status(username: str = Depends(get_current_username)):
    try: result = subprocess.run(["systemctl", "is-active", "bitrix-gpt.service"], capture_output=True, text=True); status = result.stdout.strip()
    except FileNotFoundError: status = "failed"
    return {"status": status}
@app.get("/api/logs")
async def get_logs(username: str = Depends(get_current_username)):
    try: result = subprocess.run(["journalctl", "-u", "bitrix-gpt.service", "--since", "5 minutes ago", "--no-pager"], capture_output=True, text=True); logs = result.stdout
    except FileNotFoundError: logs = "Не удалось загрузить логи."
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
class ChatRequest(BaseModel): user_message: str
@app.post("/api/chat")
async def handle_chat(chat_request: ChatRequest, username: str = Depends(get_current_username)):
    try:
        with open(PROMPT_FILE, "r") as f: system_prompt = f.read().strip()
        with open(CURRENT_MODEL_FILE, "r") as f: model_name = f.read().strip()
    except FileNotFoundError: return JSONResponse(status_code=500, content={"ai_response": "Ошибка: Файлы настроек не найдены."})
    full_input_prompt = f"{system_prompt}\n\nClient request:\n{chat_request.user_message}"
    try:
        # Передаем оригинальный запрос пользователя для поиска по базе
        ai_response_text = get_ai_response(model_name, full_input_prompt, chat_request.user_message)
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
    except Exception as e: print(f"ERROR: Ошибка при обновлении лида в Битрикс24: {e}")
def process_lead_in_background(lead_id: str):
    print(f"BACKGROUND: Начало фоновой обработки лида {lead_id}.")
    lead_data = get_lead_data_from_b24(lead_id)
    if not lead_data: print(f"BACKGROUND ERROR: Не удалось получить данные лида {lead_id}, прерываем."); return 
    task_text = lead_data.get("COMMENTS", "Текст ТЗ не найден.")
    try:
        with open(PROMPT_FILE, "r") as f: system_prompt = f.read().strip()
        with open(CURRENT_MODEL_FILE, "r") as f: model_name = f.read().strip()
    except FileNotFoundError:
        error_message = "Ошибка: Файлы настроек не найдены!"
        print(f"BACKGROUND ERROR: {error_message}")
        update_b24_lead(lead_id, error_message)
        return
    full_input_prompt = f"{system_prompt}\n\nClient request:\n{task_text}"
    print(f"BACKGROUND: Запрос к ИИ для лида {lead_id} сформирован. Отправка...")
    try:
        # Передаем оригинальный запрос для поиска по базе
        ai_response_text = get_ai_response(model_name, full_input_prompt, task_text)
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
