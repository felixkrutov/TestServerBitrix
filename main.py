import os
import openai
import requests
from fastapi import FastAPI, Request, HTTPException

# --- Настройки ---
# Эти значения мы позже добавим в настройках Render
# КЛЮЧ ТЕПЕРЬ БУДЕТ ОТ OPENROUTER
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
B24_SECRET_TOKEN = os.getenv("B24_SECRET_TOKEN")
B24_WEBHOOK_URL_FOR_UPDATE = os.getenv("B24_WEBHOOK_URL_FOR_UPDATE")

# --- Инициализация ---
app = FastAPI()

# УКАЗЫВАЕМ, ЧТО РАБОТАЕМ ЧЕРЕЗ OPENROUTER
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# --- Главная логика ---
@app.post("/b24-hook")
async def b24_hook(req: Request):
    try:
        data = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Bad request")

    if data.get("auth", {}).get("application_token") != B24_SECRET_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid token")

    try:
        lead_id = data["data"]["FIELDS"]["ID"]
        task_text = data["data"]["FIELDS"].get("COMMENTS", "Нет текста задачи.")
    except KeyError:
        raise HTTPException(status_code=422, detail="Invalid data structure from Bitrix24")

    system_prompt = "Ты — ИИ-помощник для инжиниринговой компании. Твоя задача — проанализировать техническое задание клиента и подготовить краткое коммерческое предложение. Отвечай структурированно, вежливо и по делу."
    user_prompt = f"Проанализируй следующее ТЗ и подготовь ответ для клиента: \n\n{task_text}"

    try:
        # ИСПОЛЬЗУЕМ НОВУЮ МОДЕЛЬ ОТ GOOGLE
        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-exp:free", # Указываем модель Gemini. Бесплатной "flash" может не быть, "pro" надежнее для старта
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            # Обязательные заголовки для OpenRouter
            extra_headers={
                "HTTP-Referer": "https://github.com/felixkrutov/TestServerBitrix", # Можно указать твой GitHub
                "X-Title": "Bitrix GPT Test", # Название твоего проекта
            }
        )
        ai_response_text = response.choices[0].message.content
    except Exception as e:
        ai_response_text = f"Ошибка при обращении к OpenRouter: {str(e)}"

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
            "COMMENT": f"Ответ от ИИ-помощника (Gemini):\n\n{comment_text}"
        }
    }
    try:
        requests.post(f"{B24_WEBHOOK_URL_FOR_UPDATE}/{method}", json=params)
    except Exception as e:
        print(f"Error updating Bitrix24: {e}")

@app.get("/")
def read_root():
    return {"status": "Server is running on OpenRouter"}
