import os
import re
import shutil
import requests
import subprocess
import secrets
import sqlite3
from datetime import datetime, timedelta

from passlib.context import CryptContext

from fastapi import FastAPI, Request, HTTPException, Depends, Form, BackgroundTasks, UploadFile, File, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import openai
import google.generativeai as genai

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

app = FastAPI()
security = HTTPBasic()
templates = Jinja2Templates(directory="templates")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class ChatRequest(BaseModel):
    user_message: str

class MossaChatRequest(BaseModel):
    user_message: str
    chat_id: int | None = None

class ChatRenameRequest(BaseModel):
    new_title: str

class UserCreate(BaseModel):
    username: str
    password: str

class ChangelogEntry(BaseModel):
    content: str

class ThemeUpdateRequest(BaseModel):
    theme: str


MODELS_LIST_FILE = "models_list.txt"
CURRENT_MODEL_FILE = "current_model.txt"
PROMPT_NIKOLAI_FILE = "prompt.txt"
PROMPT_MOSSAASSISTANT_FILE = "prompt_mossaassistant.txt"
CURRENT_MODEL_MOSSA_FILE = "current_model_mossa.txt"
USE_RAG_NIKOLAI_FILE = "use_rag_nikolai.txt"
USE_RAG_MOSSA_FILE = "use_rag_mossa.txt"
DOCS_DIR = "documents"
DB_DIR = "chroma_db"
USERS_DB_FILE = "users.db"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
B24_WEBHOOK_URL_FOR_UPDATE = os.getenv("B24_WEBHOOK_URL_FOR_UPDATE")
B24_SECRET_TOKEN = os.getenv("B24_SECRET_TOKEN")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("WARNING: GEMINI_API_KEY не найден.")

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

def load_rag_setting(file_path: str):
    try:
        with open(file_path, "r") as f:
            return f.read().strip().lower() == "true"
    except FileNotFoundError:
        return True

def save_rag_setting(file_path: str, value: bool):
    with open(file_path, "w") as f:
        f.write(str(value).lower())

def get_db_connection():
    conn = sqlite3.connect(USERS_DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def initialize_database():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL
        )
    """)
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN theme TEXT DEFAULT 'dark'")
        print("INFO: Колонка 'theme' успешно добавлена в таблицу 'users'.")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("INFO: Колонка 'theme' уже существует в таблице 'users'.")
        else:
            raise
            
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            expires_at DATETIME NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chats (id) ON DELETE CASCADE
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS changelog_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
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
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return user

def create_user(username: str, password: str):
    hashed_password = get_password_hash(password)
    conn = get_db_connection()
    try:
        conn.execute("INSERT INTO users (username, hashed_password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def delete_user(user_id: int):
    conn = get_db_connection()
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()

@app.on_event("startup")
def on_startup():
    print("INFO: Сервер запущен. Инициализация базы данных...")
    initialize_database()

def get_current_admin_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password", headers={"WWW-Authenticate": "Basic"})
    return credentials.username

def get_ai_response(model_name: str, system_prompt_content: str, user_query: str, use_rag_for_this_request: bool, chat_history: list = None):
    if not user_query or not user_query.strip():
        return "Пожалуйста, задайте ваш вопрос."

    context = ""
    if use_rag_for_this_request:
        print("INFO: Поиск релевантных документов в базе знаний...")
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            relevant_docs = retriever.get_relevant_documents(user_query)
            if relevant_docs:
                context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
                print(f"INFO: Найдено {len(relevant_docs)} релевантных документов.")
            else:
                print("INFO: Релевантные документы не найдены.")
        except Exception as e:
            print(f"ERROR: Ошибка при поиске в базе знаний: {e}")
    else:
        print("INFO: Использование базы знаний для этого запроса отключено.")

    messages = []
    final_system_prompt = system_prompt_content
    if context:
        final_system_prompt += f"\n\nИспользуй следующий контекст из базы знаний для ответа:\n<context>\n{context}\n</context>"

    if final_system_prompt and final_system_prompt.strip():
        messages.append({"role": "system", "content": final_system_prompt})

    if chat_history:
        for msg in chat_history:
            if msg.get("content"):
                role = "model" if msg["role"] == "ai" else msg["role"]
                messages.append({"role": role, "content": msg["content"]})

    messages.append({"role": "user", "content": user_query})

    if model_name.startswith("gpt-") or model_name.startswith("o4-"):
        print(f"INFO: Используется OpenAI API для модели {model_name}")
        if not OPENAI_API_KEY: raise ValueError("OPENAI_API_KEY не установлен.")
        
        openai_messages = []
        for msg in messages:
            role = "assistant" if msg["role"] == "model" else msg["role"]
            openai_messages.append({"role": role, "content": msg["content"]})

        response = openai_client.chat.completions.create(model=model_name, messages=openai_messages)
        return response.choices[0].message.content

    elif model_name.startswith("gemini-"):
        print(f"INFO: Используется Gemini API для модели {model_name}")
        if not GEMINI_API_KEY: raise ValueError("GEMINI_API_KEY не установлен.")

        system_instruction = None
        gemini_history = []

        if messages and messages[0]["role"] == "system":
            system_instruction = messages[0]["content"]
            conversation_messages = messages[1:]
        else:
            conversation_messages = messages
        
        for msg in conversation_messages:
             gemini_history.append({"role": msg["role"], "parts": [msg["content"]]})

        model = genai.GenerativeModel(model_name, system_instruction=system_instruction)
        last_message = gemini_history.pop()["parts"][0]
        
        chat_session = model.start_chat(history=gemini_history)
        response = chat_session.send_message(last_message)
        return response.text
    else:
        raise ValueError(f"Ошибка: Неизвестный провайдер для модели '{model_name}'.")


@app.get("/", response_class=HTMLResponse)
async def read_admin_ui(request: Request, username: str = Depends(get_current_admin_username)):
    default_models = ["o4-mini-2025-04-16", "gemini-2.5-pro"]
    try:
        with open(MODELS_LIST_FILE, "r") as f: models_list = [line.strip() for line in f]
    except FileNotFoundError: models_list = default_models
    
    try:
        with open(CURRENT_MODEL_FILE, "r") as f: current_model_nikolai = f.read().strip()
    except FileNotFoundError: current_model_nikolai = models_list[0] if models_list else default_models[0]
    try:
        with open(PROMPT_NIKOLAI_FILE, "r") as f: prompt_nikolai = f.read().strip()
    except FileNotFoundError: prompt_nikolai = "Промпт по умолчанию для Николая"
    
    try:
        with open(CURRENT_MODEL_MOSSA_FILE, "r") as f: current_model_mossa = f.read().strip()
    except FileNotFoundError: current_model_mossa = models_list[0] if models_list else default_models[0]
    try:
        with open(PROMPT_MOSSAASSISTANT_FILE, "r") as f: prompt_mossa = f.read().strip()
    except FileNotFoundError: prompt_mossa = "Промпт по умолчанию для Мосса Ассистента"

    uploaded_files = [f for f in os.listdir(DOCS_DIR) if os.path.isfile(os.path.join(DOCS_DIR, f))]
    try: result = subprocess.run(["journalctl", "-u", "bitrix-gpt.service", "--since", "5 minutes ago", "--no-pager"], capture_output=True, text=True); logs = result.stdout
    except FileNotFoundError: logs = "Не удалось загрузить логи."
    
    use_rag_nikolai = load_rag_setting(USE_RAG_NIKOLAI_FILE)
    use_rag_mossa = load_rag_setting(USE_RAG_MOSSA_FILE)
    
    conn = get_db_connection()
    users_list = conn.execute("SELECT id, username FROM users").fetchall()
    changelog_entries = conn.execute("SELECT * FROM changelog_entries ORDER BY created_at DESC").fetchall()
    conn.close()
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "models": models_list,
        "current_model_nikolai": current_model_nikolai,
        "system_prompt_nikolai": prompt_nikolai,
        "current_model_mossa": current_model_mossa,
        "system_prompt_mossa": prompt_mossa,
        "logs": logs,
        "uploaded_files": uploaded_files,
        "use_rag_nikolai": use_rag_nikolai,
        "use_rag_mossa": use_rag_mossa,
        "changelog_entries": changelog_entries,
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

@app.post("/api/settings/nikolai")
async def save_nikolai_settings(username: str = Depends(get_current_admin_username), model: str = Form(...), prompt: str = Form(...), use_rag_nikolai: bool = Form(False)):
    with open(CURRENT_MODEL_FILE, "w") as f: f.write(model)
    with open(PROMPT_NIKOLAI_FILE, "w") as f: f.write(prompt)
    save_rag_setting(USE_RAG_NIKOLAI_FILE, use_rag_nikolai)
    subprocess.run(["systemctl", "restart", "bitrix-gpt.service"])
    return {"status": "ok", "message": "Настройки для 'Николая' сохранены."}

@app.post("/api/settings/mossa")
async def save_mossa_settings(username: str = Depends(get_current_admin_username), model: str = Form(...), prompt: str = Form(...), use_rag_mossa: bool = Form(False)):
    with open(CURRENT_MODEL_MOSSA_FILE, "w") as f: f.write(model)
    with open(PROMPT_MOSSAASSISTANT_FILE, "w") as f: f.write(prompt)
    save_rag_setting(USE_RAG_MOSSA_FILE, use_rag_mossa)
    subprocess.run(["systemctl", "restart", "bitrix-gpt.service"])
    return {"status": "ok", "message": "Настройки для 'Мосса Ассистента' сохранены."}

@app.get("/api/changelog", response_class=JSONResponse)
async def get_changelog(username: str = Depends(get_current_admin_username)):
    conn = get_db_connection()
    entries = conn.execute("SELECT * FROM changelog_entries ORDER BY created_at DESC").fetchall()
    conn.close()
    return {"entries": [{"id": e["id"], "content": e["content"], "created_at": e["created_at"]} for e in entries]}

@app.post("/api/changelog", response_class=JSONResponse)
async def add_changelog_entry(entry: ChangelogEntry, username: str = Depends(get_current_admin_username)):
    conn = get_db_connection()
    conn.execute("INSERT INTO changelog_entries (content) VALUES (?)", (entry.content,))
    conn.commit()
    conn.close()
    return {"status": "ok", "message": "Запись добавлена."}

@app.put("/api/changelog/{entry_id}", response_class=JSONResponse)
async def update_changelog_entry(entry_id: int, entry: ChangelogEntry, username: str = Depends(get_current_admin_username)):
    conn = get_db_connection()
    conn.execute("UPDATE changelog_entries SET content = ? WHERE id = ?", (entry.content, entry_id))
    conn.commit()
    conn.close()
    return {"status": "ok", "message": "Запись обновлена."}

@app.delete("/api/changelog/{entry_id}", response_class=JSONResponse)
async def delete_changelog_entry(entry_id: int, username: str = Depends(get_current_admin_username)):
    conn = get_db_connection()
    conn.execute("DELETE FROM changelog_entries WHERE id = ?", (entry_id,))
    conn.commit()
    conn.close()
    return {"status": "ok", "message": "Запись удалена."}

@app.post("/api/chat")
async def handle_chat(chat_request: ChatRequest, username: str = Depends(get_current_admin_username)):
    try:
        with open(PROMPT_NIKOLAI_FILE, "r") as f: system_prompt = f.read().strip()
        with open(CURRENT_MODEL_FILE, "r") as f: model_name = f.read().strip()
    except FileNotFoundError: return JSONResponse(status_code=500, content={"ai_response": "Ошибка: Файлы настроек не найдены."})
    
    use_rag_nikolai = load_rag_setting(USE_RAG_NIKOLAI_FILE)
    try:
        ai_response_text = get_ai_response(model_name, system_prompt, chat_request.user_message, use_rag_for_this_request=use_rag_nikolai)
    except Exception as e:
        print(f"ERROR: Ошибка при вызове get_ai_response: {e}")
        return JSONResponse(status_code=500, content={"ai_response": f"Критическая ошибка при обращении к ИИ: {e}"})
    return {"ai_response": ai_response_text}

@app.post("/api/upload-document")
async def upload_document(file: UploadFile = File(...), username: str = Depends(get_current_admin_username)):
    file_path = os.path.join(DOCS_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    load_and_process_document(file_path)
    return JSONResponse(content={"message": f"Файл '{file.filename}' успешно загружен и обработан."}, status_code=200)

@app.get("/api/documents")
async def get_documents(username: str = Depends(get_current_admin_username)):
    files = [f for f in os.listdir(DOCS_DIR) if os.path.isfile(os.path.join(DOCS_DIR, f))]
    return JSONResponse(content={"documents": files})

@app.delete("/api/documents/{filename}")
async def delete_document(filename: str, username: str = Depends(get_current_admin_username)):
    file_path = os.path.join(DOCS_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"INFO: Файл {filename} удален. Для полной очистки базы ее нужно пересоздать.")
        return JSONResponse(content={"message": f"Файл '{filename}' удален."}, status_code=200)
    else:
        raise HTTPException(status_code=404, detail="Файл не найден")

@app.post("/api/users", response_class=JSONResponse)
async def add_user(user_data: UserCreate, username: str = Depends(get_current_admin_username)):
    if not user_data.username or not user_data.password:
        raise HTTPException(status_code=400, detail="Логин и пароль не могут быть пустыми.")
    if create_user(user_data.username, user_data.password):
        return {"message": f"Пользователь '{user_data.username}' успешно создан."}
    else:
        raise HTTPException(status_code=400, detail=f"Пользователь '{user_data.username}' уже существует.")

@app.delete("/api/users/{user_id}", response_class=JSONResponse)
async def remove_user(user_id: int, username: str = Depends(get_current_admin_username)):
    delete_user(user_id)
    return {"message": f"Пользователь ID {user_id} успешно удален."}

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
        with open(PROMPT_NIKOLAI_FILE, "r") as f: system_prompt = f.read().strip()
        with open(CURRENT_MODEL_FILE, "r") as f: model_name = f.read().strip()
    except FileNotFoundError:
        error_message = "Ошибка: Файлы настроек не найдены!"
        print(f"BACKGROUND ERROR: {error_message}")
        update_b24_lead(lead_id, error_message)
        return
    
    use_rag_nikolai = load_rag_setting(USE_RAG_NIKOLAI_FILE)
    print(f"BACKGROUND: Запрос к ИИ для лида {lead_id} сформирован. Отправка...")
    try:
        ai_response_text = get_ai_response(model_name, system_prompt, task_text, use_rag_for_this_request=use_rag_nikolai)
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

async def get_current_mossa_user(request: Request):
    session_token = request.cookies.get("mossa_session")
    login_url = "/mossaassistant/login"

    if not session_token:
        raise HTTPException(status_code=307, detail="Не авторизован", headers={"Location": login_url})

    conn = get_db_connection()
    session = conn.execute(
        "SELECT user_id, expires_at FROM sessions WHERE session_token = ?", (session_token,)
    ).fetchone()

    if not session or datetime.fromisoformat(session["expires_at"]) < datetime.now():
        conn.close()
        response = RedirectResponse(url=login_url, status_code=307)
        response.delete_cookie("mossa_session")
        raise HTTPException(status_code=307, detail="Сессия недействительна или истекла", headers=response.headers)

    user = conn.execute("SELECT * FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    conn.close()
    
    if not user:
        response = RedirectResponse(url=login_url, status_code=307)
        response.delete_cookie("mossa_session")
        raise HTTPException(status_code=307, detail="Пользователь не найден", headers=response.headers)
    
    return user

def generate_chat_title(user_message: str) -> str:
    try:
        title_prompt = f"Создай очень короткое, лаконичное название (3-5 слов) для чата, который начинается с этого сообщения от пользователя: '{user_message}'. Ответь только названием, без кавычек и лишних слов."
        model_name = "o4-mini-2025-04-16" 
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": title_prompt}]
        )
        new_title = response.choices[0].message.content.strip().strip('"')
        print(f"INFO: Сгенерировано название для чата: '{new_title}'")
        return new_title
    except Exception as e:
        print(f"ERROR: Не удалось сгенерировать название для чата: {e}")
        return "Новый чат"

@app.post("/mossaassistant/login")
async def login_for_access_token(request: Request, username: str = Form(...), password: str = Form(...)):
    user = get_user(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return templates.TemplateResponse("login_mossaassistant.html", {"request": request, "error": "Неверный логин или пароль"})
    
    session_token = secrets.token_urlsafe(32)
    expires_at = datetime.now() + timedelta(days=7)
    
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO sessions (session_token, user_id, expires_at) VALUES (?, ?, ?)",
        (session_token, user["id"], expires_at.isoformat())
    )
    conn.commit()
    conn.close()
    
    response = RedirectResponse(url="/mossaassistant/chat", status_code=303)
    response.set_cookie(key="mossa_session", value=session_token, httponly=True, max_age=60*60*24*7, samesite="lax")
    return response

@app.post("/mossaassistant/logout", status_code=307)
async def mossaassistant_logout(request: Request, response: Response):
    session_token = request.cookies.get("mossa_session")
    if session_token:
        conn = get_db_connection()
        conn.execute("DELETE FROM sessions WHERE session_token = ?", (session_token,))
        conn.commit()
        conn.close()
    
    redirect_response = RedirectResponse(url="/mossaassistant/login", status_code=303)
    redirect_response.delete_cookie(key="mossa_session")
    return redirect_response

@app.get("/mossaassistant", response_class=RedirectResponse)
async def redirect_to_mossaassistant_chat(user: dict = Depends(get_current_mossa_user)):
    return RedirectResponse(url="/mossaassistant/chat", status_code=303)

@app.get("/mossaassistant/login", response_class=HTMLResponse)
async def mossaassistant_login_page(request: Request):
    return templates.TemplateResponse("login_mossaassistant.html", {"request": request, "error": None})

@app.get("/mossaassistant/chat", response_class=HTMLResponse)
async def mossaassistant_chat_page(request: Request, user: dict = Depends(get_current_mossa_user)):
    return templates.TemplateResponse("chat_mossaassistant.html", {"request": request, "user": user})

@app.get("/mossaassistant/api/chats", response_class=JSONResponse)
async def get_user_chats(user: dict = Depends(get_current_mossa_user)):
    conn = get_db_connection()
    chats = conn.execute(
        "SELECT id, title, created_at FROM chats WHERE user_id = ? ORDER BY created_at DESC",
        (user["id"],)
    ).fetchall()
    conn.close()
    return {"chats": [{"id": c["id"], "title": c["title"]} for c in chats]}

@app.get("/mossaassistant/api/chats/{chat_id}/messages", response_class=JSONResponse)
async def get_chat_messages(chat_id: int, user: dict = Depends(get_current_mossa_user)):
    conn = get_db_connection()
    chat_owner = conn.execute("SELECT user_id FROM chats WHERE id = ?", (chat_id,)).fetchone()
    if not chat_owner or chat_owner["user_id"] != user["id"]:
        conn.close()
        raise HTTPException(status_code=404, detail="Чат не найден")
    
    messages = conn.execute(
        "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY timestamp ASC",
        (chat_id,)
    ).fetchall()
    conn.close()
    return {"messages": [{"role": m["role"], "content": m["content"]} for m in messages]}

@app.post("/mossaassistant/api/chat", response_class=JSONResponse)
async def mossaassistant_handle_chat(
    chat_request: MossaChatRequest,
    user: dict = Depends(get_current_mossa_user)
):
    try:
        with open(PROMPT_MOSSAASSISTANT_FILE, "r") as f: system_prompt = f.read().strip()
        with open(CURRENT_MODEL_MOSSA_FILE, "r") as f: model_name = f.read().strip()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Ошибка: Файлы настроек не найдены.")

    use_rag_mossa = load_rag_setting(USE_RAG_MOSSA_FILE)
    conn = get_db_connection()
    current_chat_id = chat_request.chat_id
    is_new_chat = False
    new_title = None
    chat_history = []

    if current_chat_id:
        chat_owner = conn.execute("SELECT user_id FROM chats WHERE id = ?", (current_chat_id,)).fetchone()
        if not chat_owner or chat_owner["user_id"] != user["id"]:
            conn.close()
            raise HTTPException(status_code=404, detail="Чат не найден")
        
        history_rows = conn.execute(
            "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY timestamp ASC",
            (current_chat_id,)
        ).fetchall()
        chat_history = [{"role": row["role"], "content": row["content"]} for row in history_rows]
    
    try:
        ai_response_text = get_ai_response(model_name, system_prompt, chat_request.user_message, use_rag_for_this_request=use_rag_mossa, chat_history=chat_history)
        
        if chat_request.user_message and chat_request.user_message.strip():
            if not current_chat_id:
                is_new_chat = True
                new_title = generate_chat_title(chat_request.user_message)
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO chats (user_id, title) VALUES (?, ?)",
                    (user["id"], new_title)
                )
                conn.commit()
                current_chat_id = cursor.lastrowid

            conn.execute(
                "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
                (current_chat_id, 'user', chat_request.user_message)
            )
            conn.execute(
                "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
                (current_chat_id, 'ai', ai_response_text)
            )
            conn.commit()

    except Exception as e:
        conn.close()
        print(f"ERROR: Ошибка при обработке чата: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при обращении к ИИ: {e}")
    finally:
        conn.close()

    return {"ai_response": ai_response_text, "chat_id": current_chat_id, "is_new_chat": is_new_chat, "new_title": new_title}

@app.put("/mossaassistant/api/chats/{chat_id}", response_class=JSONResponse)
async def rename_chat(
    chat_id: int,
    rename_request: ChatRenameRequest,
    user: dict = Depends(get_current_mossa_user)
):
    conn = get_db_connection()
    chat_owner = conn.execute("SELECT user_id FROM chats WHERE id = ?", (chat_id,)).fetchone()
    if not chat_owner or chat_owner["user_id"] != user["id"]:
        conn.close()
        raise HTTPException(status_code=404, detail="Чат не найден")
    
    conn.execute("UPDATE chats SET title = ? WHERE id = ?", (rename_request.new_title, chat_id))
    conn.commit()
    conn.close()
    return {"message": "Чат успешно переименован"}

@app.delete("/mossaassistant/api/chats/{chat_id}", response_class=JSONResponse)
async def delete_chat(chat_id: int, user: dict = Depends(get_current_mossa_user)):
    conn = get_db_connection()
    chat_owner = conn.execute("SELECT user_id FROM chats WHERE id = ?", (chat_id,)).fetchone()
    if not chat_owner or chat_owner["user_id"] != user["id"]:
        conn.close()
        raise HTTPException(status_code=404, detail="Чат не найден")
    
    conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()
    conn.close()
    return {"message": "Чат успешно удален"}

@app.put("/mossaassistant/api/user/theme", response_class=JSONResponse)
async def update_user_theme(
    theme_request: ThemeUpdateRequest,
    user: dict = Depends(get_current_mossa_user)
):
    if theme_request.theme not in ["light", "dark"]:
        raise HTTPException(status_code=400, detail="Недопустимое значение темы")
    
    conn = get_db_connection()
    conn.execute(
        "UPDATE users SET theme = ? WHERE id = ?",
        (theme_request.theme, user["id"])
    )
    conn.commit()
    conn.close()
    return {"status": "ok", "message": "Тема успешно обновлена"}
