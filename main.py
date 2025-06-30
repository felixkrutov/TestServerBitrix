import os
import re
import shutil
import requests
import subprocess
import secrets
import sqlite3
from datetime import datetime
from passlib.context import CryptContext

from fastapi import FastAPI, Request, HTTPException, Depends, Form, BackgroundTasks, UploadFile, File, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
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
    UnstructuredExcelLoader,
)
from langchain_openai import OpenAIEmbeddings

# --- Общие настройки FastAPI ---
app = FastAPI()
security = HTTPBasic()
templates = Jinja2Templates(directory="templates")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Pydantic Модели (ПЕРЕМЕЩЕНЫ В НАЧАЛО) ---
class ChatRequest(BaseModel):
    user_message: str

class UserCreate(BaseModel):
    username: str
    password: str

class UserUpdatePassword(BaseModel):
    user_id: int
    new_password: str
# --- КОНЕЦ Pydantic Моделей ---

# --- Пути к файлам и папкам ---
MODELS_LIST_FILE = "models_list.txt"
CURRENT_MODEL_FILE = "current_model.txt"
PROMPT_NIKOLAI_FILE = "prompt.txt"
PROMPT_MOSSAASSISTANT_FILE = "prompt_mossaassistant.txt"
USE_RAG_FILE = "use_rag.txt"
DOCS_DIR = "documents"
DB_DIR = "chroma_db"
USERS_DB_FILE = "users.db"

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
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

LOADER_MAPPING = {
    ".txt": (TextLoader, {"encoding": "utf-8"}),
    ".md": (TextLoader, {"encoding": "utf-8"}),
    ".pdf": (PyPDFLoader, {}),
    ".docx": (Docx2txtLoader, {}),
    ".xlsx": (UnstructuredExcelLoader, {"mode": "single"}),
}

def load_and_process_document(file_path: str):
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

# --- Функции для загрузки/сохранения состояния RAG ---
def load_rag_setting():
    """Загружает состояние использования RAG из файла."""
    try:
        with open(USE_RAG_FILE, "r") as f:
            return f.read().strip().lower() == "true"
    except FileNotFoundError:
        return True # По умолчанию RAG включен, если файл не найден

def save_rag_setting(value: bool):
    """Сохраняет состояние использования RAG в файл."""
    with open(USE_RAG_FILE, "w") as f:
        f.write(str(value).lower())

# Инициализация состояния RAG при запуске
USE_RAG = load_rag_setting()

# --- Функции для работы с базой данных пользователей ---
def get_db_connection():
    conn = sqlite3.connect(USERS_DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def create_users_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(username: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    user = cursor.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return user

def create_user(username: str, password: str):
    hashed_password = get_password_hash(password)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, hashed_password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def delete_user(user_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()

def update_user_password(user_id: int, new_password: str):
    hashed_password = get_password_hash(new_password)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET hashed_password = ? WHERE id = ?", (hashed_password, user_id))
    conn.commit()
    conn.close()

# --- Инициализация базы данных пользователей при запуске ---
@app.on_event("startup")
def on_startup():
    print("INFO: Сервер запущен. Проверка базы знаний...")
    create_users_table()

# --- Аутентификация для админ-панели "Николай" ---
def get_current_admin_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# --- Универсальная функция для вызова ИИ (теперь с RAG) ---
def get_ai_response(model_name: str, system_prompt_content: str, user_query: str):
    global USE_RAG

    context = ""
    if USE_RAG:
        print("INFO: Поиск релевантных документов в базе знаний...")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.get_relevant_documents(user_query)
        
        print(f"DEBUG: Найдено {len(relevant_docs)} релевантных документов.")
        for i, doc in enumerate(relevant_docs):
            print(f"DEBUG: Документ {i+1} (источник: {doc.metadata.get('source', 'N/A')}):")
            print(f"DEBUG: {doc.page_content[:200]}...")
        
        context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
    else:
        print("INFO: Использование базы знаний отключено.")

    if context:
        final_prompt = f"CONTEXT:\n{context}\n---\nPROMPT:\n{system_prompt_content}\n\nClient request:\n{user_query}\n\nИспользуя предоставленный CONTEXT, ответь на PROMPT. Если в контексте нет ответа, сообщи, что информация не найдена в базе знаний."
    else:
        final_prompt = f"{system_prompt_content}\n\nClient request:\n{user_query}"

    if model_name.startswith("gpt-") or model_name.startswith("o4-"):
        print(f"INFO: Используется OpenAI API для модели {model_name}")
        if not OPENAI_API_KEY: raise ValueError("OPENAI_API_KEY не установлен.")
        response = openai_client.chat.completions.create(model=model_name, messages=[{"role": "system", "content": final_prompt}])
        return response.choices[0].message.content
    elif model_name.startswith("gemini-"):
        print(f"INFO: Используется Gemini API для модели {model_name}")
        if not GEMINI_API_KEY: raise ValueError("GEMINI_API_KEY не установлен.")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(final_prompt)
        return response.text
    else:
        raise ValueError(f"Ошибка: Неизвестный провайдер для модели '{model_name}'.")

# --- Админ-панель "Николай" (http://your_ip/) ---
@app.get("/", response_class=HTMLResponse)
async def read_admin_ui(request: Request, username: str = Depends(get_current_admin_username)):
    default_models = ["o4-mini-2025-04-16", "gemini-2.5-pro"]
    try:
        with open(MODELS_LIST_FILE, "r") as f: models_list = [line.strip() for line in f]
    except FileNotFoundError: models_list = default_models
    try:
        with open(CURRENT_MODEL_FILE, "r") as f: current_model = f.read().strip()
    except FileNotFoundError: current_model = models_list[0] if models_list else default_models[0]
    try:
        with open(PROMPT_NIKOLAI_FILE, "r") as f: prompt = f.read().strip()
    except FileNotFoundError: prompt = "Промпт по умолчанию для Николая"
    
    uploaded_files = [f for f in os.listdir(DOCS_DIR) if os.path.isfile(os.path.join(DOCS_DIR, f))]

    try: result = subprocess.run(["journalctl", "-u", "bitrix-gpt.service", "--since", "5 minutes ago", "--no-pager"], capture_output=True, text=True); logs = result.stdout
    except FileNotFoundError: logs = "Не удалось загрузить логи."

    use_rag_setting = load_rag_setting()

    conn = get_db_connection()
    users_list = conn.execute("SELECT id, username FROM users").fetchall()
    conn.close()
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "current_model": current_model,
        "models": models_list,
        "system_prompt": prompt,
        "logs": logs,
        "uploaded_files": uploaded_files,
        "use_rag": use_rag_setting,
        "users_list": [{"id": user["id"], "username": user["username"]} for user in users_list]
    })

@app.get("/api/status")
async def get_status(username: str = Depends(get_current_admin_username)):
    try: result = subprocess.run(["systemctl", "is-active", "bitrix-gpt.service"], capture_output=True, text=True); status = result.stdout.strip()
    except FileNotFoundError: status = "failed"
    return {"status": status}

@app.get("/api/logs")
async def get_logs(username: str = Depends(get_current_admin_username)):
    try: result = subprocess.run(["journalctl", "-u", "bitrix-gpt.service", "--since", "5 minutes ago", "--no-pager"], capture_output=True, text=True); logs = result.stdout
    except FileNotFoundError: logs = "Не удалось загрузить логи."
    return {"logs": logs}

@app.get("/api/settings")
async def get_settings(username: str = Depends(get_current_admin_username)):
    default_models = ["o4-mini-2025-04-16", "gemini-2.5-pro"]
    try:
        with open(MODELS_LIST_FILE, "r") as f: models_list = [line.strip() for line in f]
    except FileNotFoundError: models_list = default_models
    try:
        with open(CURRENT_MODEL_FILE, "r") as f: current_model = f.read().strip()
    except FileNotFoundError: current_model = models_list[0] if models_list else default_models[0]
    try:
        with open(PROMPT_NIKOLAI_FILE, "r") as f: prompt = f.read().strip()
    except FileNotFoundError: prompt = "Промпт по умолчанию для Николая"
    
    use_rag_setting = load_rag_setting()

    return {
        "models_list": models_list,
        "current_model": current_model,
        "prompt": prompt,
        "use_rag": use_rag_setting
    }

@app.post("/api/settings")
async def save_settings(
    username: str = Depends(get_current_admin_username),
    model: str = Form(...),
    prompt: str = Form(...),
    use_rag: bool = Form(False)
):
    global USE_RAG
    
    with open(CURRENT_MODEL_FILE, "w") as f: f.write(model)
    with open(PROMPT_NIKOLAI_FILE, "w") as f: f.write(prompt)
    save_rag_setting(USE_RAG)

    subprocess.run(["systemctl", "restart", "bitrix-gpt.service"])
    
    return {"status": "ok"}

@app.post("/api/chat")
async def handle_chat(chat_request: ChatRequest, username: str = Depends(get_current_admin_username)):
    try:
        with open(PROMPT_NIKOLAI_FILE, "r") as f: system_prompt = f.read().strip()
        with open(CURRENT_MODEL_FILE, "r") as f: model_name = f.read().strip()
    except FileNotFoundError: return JSONResponse(status_code=500, content={"ai_response": "Ошибка: Файлы настроек не найдены."})
    
    try:
        ai_response_text = get_ai_response(model_name, system_prompt, chat_request.user_message)
    except Exception as e:
        ai_response_text = f"Ошибка при обращении к ИИ ({model_name}): {str(e)}"
    return {"ai_response": ai_response_text}

# --- Управление пользователями (новая вкладка в админке) ---
@app.get("/api/users", response_class=JSONResponse)
async def get_users(username: str = Depends(get_current_admin_username)):
    conn = get_db_connection()
    users = conn.execute("SELECT id, username FROM users").fetchall()
    conn.close()
    return {"users": [{"id": user["id"], "username": user["username"]} for user in users]}

@app.post("/api/users", response_class=JSONResponse)
async def add_user(user_data: UserCreate, username: str = Depends(get_current_admin_username)):
    if not user_data.username or not user_data.password:
        raise HTTPException(status_code=400, detail="Логин и пароль не могут быть пустыми.")
    if create_user(user_data.username, user_data.password):
        return {"message": f"Пользователь '{user_data.username}' успешно создан."}
    else:
        raise HTTPException(status_code=400, detail=f"Пользователь '{user_data.username}' уже существует.")

@app.put("/api/users/{user_id}", response_class=JSONResponse)
async def update_password(user_id: int, user_data: UserUpdatePassword, username: str = Depends(get_current_admin_username)):
    if not user_data.new_password:
        raise HTTPException(status_code=400, detail="Новый пароль не может быть пустым.")
    update_user_password(user_id, user_data.new_password)
    return {"message": f"Пароль пользователя ID {user_id} успешно обновлен."}

@app.delete("/api/users/{user_id}", response_class=JSONResponse)
async def remove_user(user_id: int, username: str = Depends(get_current_admin_username)):
    delete_user(user_id)
    return {"message": f"Пользователь ID {user_id} успешно удален."}

# --- Логика Битрикс24 (без изменений) ---
# ... (этот блок без изменений) ...

# --- Мосса Ассистент (для коллег) ---
def create_session_token(data: dict):
    return secrets.token_urlsafe(32)

def get_current_mossa_user(request: Request):
    session_token = request.cookies.get("mossa_session")
    if not session_token:
        raise HTTPException(status_code=302, detail="Не авторизован", headers={"Location": "/mossaassistant/login"})
    return session_token

@app.post("/mossaassistant/login")
async def login_for_access_token(response: Response, request: Request, username: str = Form(...), password: str = Form(...)): # <-- ИЗМЕНЕНО
    user = get_user(username)
    if not user or not verify_password(password, user["hashed_password"]):
        # Не выбрасываем ошибку, а возвращаем на страницу логина с сообщением
        return templates.TemplateResponse("login_mossaassistant.html", {
            "request": request,
            "error": "Неверный логин или пароль"
        })
    
    session_token = create_session_token({"sub": user["username"]})
    response.set_cookie(key="mossa_session", value=session_token, httponly=True, max_age=3600*24)
    return RedirectResponse(url="/mossaassistant/chat", status_code=302)

@app.get("/mossaassistant", response_class=RedirectResponse)
async def redirect_to_mossaassistant_chat():
    return RedirectResponse(url="/mossaassistant/chat", status_code=302)

@app.get("/mossaassistant/login", response_class=HTMLResponse)
async def mossaassistant_login_page(request: Request):
    return templates.TemplateResponse("login_mossaassistant.html", {"request": request, "error": None})

@app.get("/mossaassistant/chat", response_class=HTMLResponse)
async def mossaassistant_chat_page(request: Request, user_session: str = Depends(get_current_mossa_user)):
    return templates.TemplateResponse("chat_mossaassistant.html", {"request": request})

@app.post("/mossaassistant/api/chat", response_class=JSONResponse)
async def mossaassistant_handle_chat(chat_request: ChatRequest, user_session: str = Depends(get_current_mossa_user)):
    try:
        with open(PROMPT_MOSSAASSISTANT_FILE, "r") as f: system_prompt = f.read().strip()
        with open(CURRENT_MODEL_FILE, "r") as f: model_name = f.read().strip()
    except FileNotFoundError: return JSONResponse(status_code=500, content={"ai_response": "Ошибка: Файлы настроек не найдены."})
    
    try:
        ai_response_text = get_ai_response(model_name, system_prompt, chat_request.user_message)
    except Exception as e:
        ai_response_text = f"Ошибка при обращении к ИИ ({model_name}): {str(e)}"
    return {"ai_response": ai_response_text}

@app.post("/mossaassistant/logout", response_class=RedirectResponse)
async def mossaassistant_logout(response: Response):
    response.delete_cookie(key="mossa_session")
    return RedirectResponse(url="/mossaassistant/login", status_code=302)
