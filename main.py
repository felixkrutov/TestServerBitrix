import os
import re
import openai
import requests
from fastapi import FastAPI, Request, HTTPException

# --- Настройки ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
B24_WEBHOOK_URL_FOR_UPDATE = os.getenv("B24_WEBHOOK_URL_FOR_UPDATE")

# --- Инициализация ---
app = FastAPI()
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# --- НОВАЯ ФУНКЦИЯ для получения данных о лиде ---
def get_lead_data_from_b24(lead_id):
    """Делает запрос к API Битрикса, чтобы получить данные лида."""
    if not B24_WEBHOOK_URL_FOR_UPDATE:
        return None
    
    method = "crm.lead.get"
    url = f"{B24_WEBHOOK_URL_FOR_UPDATE}/{method}"
    params = {"ID": lead_id}
    
    try:
        response = requests.post(url, json=params)
        response.raise_for_status()
        return response.json().get("result")
    except Exception as e:
        print(f"Ошибка при получении данных лида {lead_id}: {e}")
        return None

# --- ИСПРАВЛЕННАЯ ГЛАВНАЯ ЛОГИКА ---
@app.post("/b24-hook")
async def b24_hook(req: Request):
    # 1. Читаем данные из формы, а не JSON
    try:
        form_data = await req.form()
        # Извлекаем ID лида. Битрикс присылает его в формате 'lead_123'
        document_id_str = form_data.get("document_id[2]")
        if not document_id_str:
             raise ValueError("document_id не найден в форме")
        
        # Используем регулярное выражение, чтобы точно вытащить число
        match = re.search(r'\d+', document_id_str)
        if not match:
            raise ValueError(f"Не удалось извлечь ID из {document_id_str}")
            
        lead_id = match.group(0)

    except Exception as e:
        print(f"Ошибка парсинга формы от Битрикс: {e}")
        raise HTTPException(status_code=400, detail=f"Bad form data: {e}")

    # 2. Получаем полные данные лида, включая комментарий
    lead_data = get_lead_data_from_b24(lead_id)
    if not lead_data:
        raise HTTPException(status_code=500, detail="Не удалось получить данные лида из Битрикс24")
    
    task_text = lead_data.get("COMMENTS", "Текст ТЗ не найден в комментарии лида.")

    # 3. Готовим и отправляем запрос к OpenRouter (без изменений)
    system_prompt = "Ты — ИИ-помощник для инжиниринговой компании. Твоя задача — проанализировать техническое задание клиента и подготовить краткое коммерческое предложение. Отвечай структурированно, вежливо и по делу."
    user_prompt = f"Проанализируй следующее ТЗ и подготовь ответ для клиента: \n\n{task_text}"

    try:
        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            extra_headers={
                "HTTP-Referer": "https://github.com/felixkrutov/TestServerBitrix",
                "X-Title": "Bitrix GPT Test",
            }
        )
        ai_response_text = response.choices[0].message.content
    except Exception as e:
        ai_response_text = f"Ошибка при обращении к OpenRouter: {str(e)}"

    # 4. Отправляем ответ обратно в Битрикс24 (без изменений)
    update_b24_lead(lead_id, ai_response_text)

    return {"status": "ok"}


def update_b24_lead(lead_id, comment_text):
    """Функция для добавления комментария к лиду в Битрикс24."""
    method = "crm.timeline.comment.add"
    params = {
        "fields": {
            "ENTITY_ID": lead_id,
            "ENTITY_TYPE": "lead",
            "COMMENT": f"Ответ от ИИ-помощника (Gemini):\n\n{comment_text}"
        }
    }
    try:
        requests.post(f"{B24_WEBHOOK_URL_FOR_UPDATE}/{method}", json=params)
    except Exception as e:
        print(f"Ошибка при обновлении лида в Битрикс24: {e}")

@app.get("/")
def read_root():
    return {"status": "Server is running on OpenRouter, v2"}
