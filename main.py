"""
FastAPI-вебхук для Bitrix24:
1. Получает ID лида из входящей формы.
2. Запрашивает подробности лида через REST-webhook.
3. Передаёт комментарий (ТЗ) в OpenAI o3-mini.
4. Добавляет ответ в таймлайн лида.

Зависимости:
    pip install fastapi uvicorn openai==1.14.0 python-dotenv requests
"""

import os
import re
import logging
import requests
from fastapi import FastAPI, Request, HTTPException
from openai import OpenAI                         # SDK ≥ 1.14
from dotenv import load_dotenv                   # удобно в dev

# ──────────────────────────── настройки ────────────────────────────
load_dotenv()  # подтянуть .env, если есть
OPENAI_API_KEY            = os.getenv("OPENAI_API_KEY")
B24_WEBHOOK_URL_FOR_UPDATE = os.getenv("B24_WEBHOOK_URL_FOR_UPDATE")

if not OPENAI_API_KEY or not B24_WEBHOOK_URL_FOR_UPDATE:
    raise RuntimeError("Проверьте переменные окружения OPENAI_API_KEY и B24_WEBHOOK_URL_FOR_UPDATE")

# инициируем OpenAI-клиент
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ──────────────────────────── FastAPI ──────────────────────────────
app = FastAPI(title="B24-GPT webhook")

# ──────────────────────────── utils ────────────────────────────────
def get_lead_data_b24(lead_id: str) -> dict | None:
    """Вернуть объект лида по ID или None"""
    method = "crm.lead.get"
    try:
        r = requests.post(f"{B24_WEBHOOK_URL_FOR_UPDATE}/{method}", json={"ID": lead_id}, timeout=15)
        r.raise_for_status()
        return r.json().get("result")
    except Exception as exc:
        logging.exception("Ошибка при получении лида %s: %s", lead_id, exc)
        return None


def update_b24_lead_timeline(lead_id: str, text: str) -> None:
    """Добавить комментарий в таймлайн"""
    method = "crm.timeline.comment.add"
    payload = {
        "fields": {
            "ENTITY_ID": lead_id,
            "ENTITY_TYPE": "lead",
            "COMMENT": f"Ответ от цифрового сотрудника Боба:\n\n{text}"
        }
    }
    try:
        requests.post(f"{B24_WEBHOOK_URL_FOR_UPDATE}/{method}", json=payload, timeout=15)
    except Exception as exc:
        logging.exception("Ошибка при обновлении таймлайна лида %s: %s", lead_id, exc)


# ──────────────────────────── GPT-логика ───────────────────────────
SYSTEM_PROMPT = (
    "Ты должен на основании проведённого анализа подготовить расчёт по СБЦ на услуги проектирования "
    "с выделением предварительного этапа ОТР-ТЭО (СБЦ подбери по отрасли). "
    "В конце предложи провести ВКС <30 мин для уточнения задания и корректировки предложения. "
    "Начни ответ с обращения к Заказчику по имени (компания), поблагодари за обращение. "
    "В конце задай доп-вопросы для лучшего понимания и поставь подпись:\n"
    "С уважением, Вадим Марков\nЗаместитель генерального директора по работе с ключевыми клиентами "
    "ООО «Мосса Инжиниринг»\n+7 921 371-00-92\nvadim.markov@mossaengineering.com"
)

# ──────────────────────────── вебхук ───────────────────────────────
@app.post("/b24-hook")
async def b24_hook(request: Request):
    """Основной энд-пойнт, вызываемый Bitrix24"""
    # 1. Извлекаем ID лида из формы
    try:
        form = await request.form()
        raw_doc_id = form.get("document_id[2]")
        if not raw_doc_id:
            raise ValueError("document_id[2] отсутствует")
        lead_id_match = re.search(r"\d+", raw_doc_id)
        lead_id = lead_id_match.group(0) if lead_id_match else None
        if not lead_id:
            raise ValueError(f"Не удалось извлечь ID из {raw_doc_id}")
    except Exception as exc:
        logging.exception("Парсинг формы B24: %s", exc)
        raise HTTPException(status_code=400, detail=f"Bad form data: {exc}")

    # 2. Получаем данные лида
    lead = get_lead_data_b24(lead_id)
    if not lead:
        raise HTTPException(status_code=502, detail="Cannot fetch lead from B24")

    task_text = lead.get("COMMENTS") or "Текст ТЗ отсутствует."

    # 3. Формируем промпт и обращаемся к OpenAI
    user_prompt = (
        "Проанализируй следующие характеристики клиента и подготовь ответ:\n\n"
        f"{task_text}"
    )

    try:
        completion = openai_client.chat.completions.create(
            model="o3-mini-2025-01-31",
            temperature=0,
            max_tokens=2000,                    # ≤ context window
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt}
            ]
        )
        answer_text = completion.choices[0].message.content
        logging.info("Tokens used: %s", completion.usage.total_tokens)
    except Exception as exc:
        logging.exception("OpenAI error: %s", exc)
        answer_text = f"Ошибка при обращении к OpenAI: {exc}"

    # 4. Пишем ответ в Bitrix24
    update_b24_lead_timeline(lead_id, answer_text)
    return {"status": "ok"}


@app.get("/")
def health():
    return {"status": "running"}
