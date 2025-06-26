"""
FastAPI-webhook для Bitrix24  ➜  OpenAI  ➜  Bitrix24-таймлайн.

Зависимости: см. requirements.txt
Запуск локально:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
Render (Procfile):
    web: uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import os, re, logging, requests
from fastapi import FastAPI, Request, HTTPException
from openai import AsyncOpenAI

# ───────────────────── попытка подгрузить .env (опционально) ───────────────────
try:
    from dotenv import load_dotenv  # если python-dotenv не установлен, просто пропустим
    load_dotenv()
except ModuleNotFoundError:
    logging.warning("python-dotenv не установлен – пропускаю загрузку .env")

# ─────────────────────────── переменные окружения ─────────────────────────────
OPENAI_API_KEY            = os.getenv("OPENAI_API_KEY")
B24_WEBHOOK_URL_FOR_UPDATE = os.getenv("B24_WEBHOOK_URL_FOR_UPDATE")

if not OPENAI_API_KEY or not B24_WEBHOOK_URL_FOR_UPDATE:
    raise RuntimeError("OPENAI_API_KEY или B24_WEBHOOK_URL_FOR_UPDATE не заданы!")

# ────────────────────────── инициализация клиентов ────────────────────────────
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)  # асинхронный SDK ≥ 1.14
app = FastAPI(title="B24-GPT webhook")

# ══════════════════════ вспомогательные функции ═══════════════════════════════

def _b24(method: str, payload: dict) -> dict:
    """Унифицированный POST-запрос в Bitrix24 REST-вебхук"""
    url = f"{B24_WEBHOOK_URL_FOR_UPDATE}/{method}"
    r = requests.post(url, json=payload, timeout=15)
    r.raise_for_status()
    return r.json()

def get_lead(lead_id: str) -> dict | None:
    try:
        return _b24("crm.lead.get", {"ID": lead_id}).get("result")
    except Exception as exc:
        logging.exception("Ошибка B24 crm.lead.get: %s", exc)
        return None

def add_timeline_comment(lead_id: str, text: str) -> None:
    payload = {
        "fields": {
            "ENTITY_ID": lead_id,
            "ENTITY_TYPE": "lead",
            "COMMENT": f"Ответ от цифрового сотрудника Боба:\n\n{text}"
        }
    }
    try:
        _b24("crm.timeline.comment.add", payload)
    except Exception as exc:
        logging.exception("Ошибка B24 crm.timeline.comment.add: %s", exc)

# ════════════════════════ шаблоны промптов ════════════════════════════════════

SYSTEM_PROMPT = (
    "Ты должен на основании проведённого анализа подготовить расчёт по СБЦ на услуги "
    "проектирования с выделением предварительного этапа ОТР-ТЭО (СБЦ подбери по отрасли). "
    "В конце предложи провести ВКС <30 мин. Начни ответ с обращения к Заказчику, "
    "поблагодари за обращение. В конце задай доп-вопросы и подпись:\n"
    "С уважением, Вадим Марков\n"
    "Заместитель генерального директора по работе с ключевыми клиентами\n"
    "ООО «Мосса Инжиниринг»\n+7 921 371-00-92\nvadim.markov@mossaengineering.com"
)

# ══════════════════════ HTTP-энд-пойнты ═══════════════════════════════════════

@app.post("/b24-hook")
async def b24_hook(req: Request):
    """
    Вебхук «новый лид»: Bitrix24 шлёт multipart/form-data,
    вытаскиваем document_id[2] → ID лида.
    """
    try:
        form = await req.form()
        raw_id = form.get("document_id[2]")
        if not raw_id:
            raise ValueError("document_id[2] отсутствует")

        lead_id = re.search(r"\d+", raw_id).group(0)  # first numbers
    except Exception as exc:
        raise HTTPException(400, f"Bad form data: {exc}") from exc

    lead = get_lead(lead_id)
    if not lead:
        raise HTTPException(502, "Не удалось получить лид из Bitrix24")

    task_text = lead.get("COMMENTS") or "Текст ТЗ отсутствует."

    user_prompt = (
        "Проанализируй характеристики клиента и подготовь письмо-ответ:\n\n"
        f"{task_text}"
    )

    # --- обращаемся к OpenAI ---
    try:
        resp = await openai_client.chat.completions.create(
            model="o3-mini-2025-01-31",
            temperature=0,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt}
            ]
        )
        answer = resp.choices[0].message.content
        logging.info("OpenAI tokens used: %s", resp.usage.total_tokens)
    except Exception as exc:
        logging.exception("OpenAI error")
        answer = f"Ошибка при обращении к OpenAI: {exc}"

    add_timeline_comment(lead_id, answer)
    return {"status": "ok", "lead_id": lead_id}

@app.get("/")
def health():
    return {"status": "running"}
