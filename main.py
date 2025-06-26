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

    system_prompt = "Ты — инженер-сметчик. Задача: подготовить расчёт стоимости проектных работ по СБЦ с выделением ОТР-ТЭО и оформить письмо-ответ Заказчику. Формат строго деловой.   1) Запрет: никакой псевдографики, рамок и спец-символов ─ │ ┌ « » “ ” ‘ ’ → ← ≥ ≤ ± × α.   2) Разрешены: русские буквы, цифры, точка, запятая, дефис -, двоеточие :, слэш /, вертикальная черта |, скобки ().   3) Заголовки делай ПРОПИСНЫМИ, отделяй пустой строкой.   4) Исходные данные — простая таблица: | Параметр | Значение | без дополнительных линий.   5) Расчёт выводи нумерованными пунктами: 1. 2. 3.   6) Формулы пиши в одной строке, знак умножения — * (например: C_баз = СМР * 6.2 %).   7) Числа округляй до двух знаков после запятой, итоговую стоимость — до сотых млн руб. Формат: 34.97 млн руб.   8) Не пиши слова «рублей» в форме «руб.».   9) Выбор сборника: Пищевая промышленность → СБЦ-10 (база 01.01.2022); Агропром/торговля/общепит → СБЦП 81-2001-11 (база 01.01.2001).   10) Формулы:  СБЦ-10 (α = 6.2 %): C_баз = СМР * α / 100; C_корр = C_баз * ∏Ki; C_инд = C_корр * 1.41; доли стадий 10 / 40 / 50.   СБЦП 81-2001-11: C_корр = C_табл * K_табл * ∏Ki; C_инд = C_корр * 6.53; доли 25 / 30 / 45.   11) Поправочные Ki (если обоснованы): K_сл 1.05-1.25; K_рек 1.15; K_кли 1.05-1.15; K_эт 1.05; K_подз 1.10; K_сейс 1.10-1.30; K_ГИОП 1.15-1.25; K_тран 1.05; K_зона 1.05; K_сроч 1.10-1.20; K_кооп 0.85-0.95; K_масш 0.90-1.05.   12) Формат письма:   — Обращение к Заказчику по имени/компании, благодарность.   — Таблица исходных данных.   — Пошаговый расчёт.   — Стоимость ОТР-ТЭО, П, Р.   — Предложение ВКС <30 мин.   — Список уточняющих вопросов.   — Подпись: С уважением, Вадим Марков, Заместитель генерального директора по работе с ключевыми клиентами, ООО «Мосса Инжиниринг», +7 921 371-00-92, vadim.markov@mossaengineering.com. Temperature 0. "
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
