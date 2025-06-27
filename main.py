import os
import re
import openai
import requests
from fastapi import FastAPI, Request, HTTPException

# --- Настройки ---
# Возвращаемся к ключу от OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
B24_WEBHOOK_URL_FOR_UPDATE = os.getenv("B24_WEBHOOK_URL_FOR_UPDATE")

# --- Инициализация ---
app = FastAPI()
# Инициализируем клиент OpenAI
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- Функция для получения данных о лиде ---
def get_lead_data_from_b24(lead_id):
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

# --- Главная логика ---
@app.post("/b24-hook")
async def b24_hook(req: Request):
    try:
        form_data = await req.form()
        document_id_str = form_data.get("document_id[2]")
        if not document_id_str:
             raise ValueError("document_id не найден в форме")
        
        match = re.search(r'\d+', document_id_str)
        if not match:
            raise ValueError(f"Не удалось извлечь ID из {document_id_str}")
            
        lead_id = match.group(0)

    except Exception as e:
        print(f"Ошибка парсинга формы от Битрикс: {e}")
        raise HTTPException(status_code=400, detail=f"Bad form data: {e}")

    lead_data = get_lead_data_from_b24(lead_id)
    if not lead_data:
        raise HTTPException(status_code=500, detail="Не удалось получить данные лида из Битрикс24")
    
    task_text = lead_data.get("COMMENTS", "Текст ТЗ не найден в комментарии лида.")

    system_prompt = "изучить запрос клиента, посмотреть отрасль и площадь объекта (возьми среднюю площадь), рассчитать примерную мощность этого предприятия и с помощью ГОСУДАРСТВЕННЫЙ СМЕТНЫЙ НОРМАТИВ «СПРАВОЧНИК БАЗОВЫХ ЦЕН НА ПРОЕКТНЫЕ РАБОТЫ В СТРОИТЕЛЬСТВЕ «ПРЕДПРИЯТИЯ АГРОПРОМЫШЛЕННОГО КОМПЛЕКСА, ТОРГОВЛИ И ОБЩЕСТВЕННОГО ПИТАНИЯ» СБЦП 81 - 2001 - 11 рассчитать стоимость проектирования по стадиям ОТР-П-Р"
    "\n"
    "Temperature 0"
    user_prompt = f"Проанализируй следующии характиеристики клиента и подготовь ответ для клиента: \n\n{task_text}"

    try:
        # ИСПОЛЬЗУЕМ МОДЕЛЬ OpenAI (быстрая и недорогая)
        response = client.chat.completions.create(
            model="o3-mini-2025-01-31",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=5000
        )
        ai_response_text = response.choices[0].message.content
    except Exception as e:
        ai_response_text = f"Ошибка при обращении к OpenAI: {str(e)}"

    update_b24_lead(lead_id, ai_response_text)

    return {"status": "ok"}


def update_b24_lead(lead_id, comment_text):
    method = "crm.timeline.comment.add"
    params = {
        "fields": {
            "ENTITY_ID": lead_id,
            "ENTITY_TYPE": "lead",
            "COMMENT": f"Ответ от цифрового сотрудника Николая:\n\n{comment_text}"
        }
    }
    try:
        requests.post(f"{B24_WEBHOOK_URL_FOR_UPDATE}/{method}", json=params)
    except Exception as e:
        print(f"Ошибка при обновлении лида в Битрикс24: {e}")

@app.get("/")
def read_root():
    return {"status": "Production-ready server is running"}
