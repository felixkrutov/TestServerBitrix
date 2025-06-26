import os
import openai
import requests
from fastapi import FastAPI, Request, HTTPException

# --- Настройки ---
# Эти значения мы позже добавим в настройках Render
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
B24_SECRET_TOKEN = os.getenv("B24_SECRET_TOKEN")
B24_WEBHOOK_URL_FOR_UPDATE = os.getenv("B24_WEBHOOK_URL_FOR_UPDATE")

# --- Инициализация ---
app = FastAPI()
openai.api_key = OPENAI_API_KEY

# --- Главная логика ---
@app.post("/b24-hook")
async def b24_hook(req: Request):
    try:
        data = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Bad request")

    # 1. Проверяем, что запрос пришел от Битрикс24
    if data.get("auth", {}).get("application_token") != B24_SECRET_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid token")

    # 2. Извлекаем данные из запроса Битрикса
    try:
        lead_id = data["data"]["FIELDS"]["ID"]
        # Используем COMMENTS как место для ТЗ. В реальном проекте может быть другое поле.
        task_text = data["data"]["FIELDS"].get("COMMENTS", "Нет текста задачи.")
    except KeyError:
        raise HTTPException(status_code=422, detail="Invalid data structure from Bitrix24")

    # 3. Готовим запрос к OpenAI
    system_prompt = "Ты — ИИ-помощник для инжиниринговой компании. Твоя задача — проанализировать техническое задание клиента и подготовить краткое коммерческое предложение. Отвечай структурированно, вежливо и по делу."
    user_prompt = f"Проанализируй следующее ТЗ и подготовь ответ для клиента: \n\n{task_text}"

    # 4. Отправляем запрос к OpenAI
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini", # Используем более дешевую и быструю модель для теста
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500
        )
        ai_response_text = response.choices[0].message.content
    except Exception as e:
        ai_response_text = f"Ошибка при обращении к OpenAI: {str(e)}"

    # 5. Отправляем ответ обратно в Битрикс24 в виде комментария к Лиду
    if B24_WEBHOOK_URL_FOR_UPDATE:
        update_b24_lead(lead_id, ai_response_text)

    return {"status": "ok", "ai_response": ai_response_text}

def update_b24_lead(lead_id, comment_text):
    """Функция для добавления комментария к лиду в Битрикс24."""
    method = "crm.timeline.comment.add"
    params = {
        "fields": {
            "ENTITY_ID": lead_id,
            "ENTITY_TYPE": "lead",
            "COMMENT": f"Ответ от ИИ-помощника:\n\n{comment_text}"
        }
    }
    try:
        # URL веб-хука для обновления должен быть полным
        requests.post(f"{B24_WEBHOOK_URL_FOR_UPDATE}/{method}", json=params)
    except Exception as e:
        # В реальной системе здесь нужно логирование ошибки
        print(f"Error updating Bitrix24: {e}")

@app.get("/")
def read_root():
    return {"status": "Server is running"}
