"""
Toda - LangGraph Agent with DeepSeek + FastAPI
Полная версия: файлы, поиск, CMD, клавиши, погода, заметки, анализ файлов, мульти-чаты
"""

import json
import os
import subprocess
import platform
import traceback
import re
from typing import Annotated, Literal, TypedDict, List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, messages_from_dict, messages_to_dict

import opendeep as genai

# --------------------------
# 1. РАБОТА С ИСТОРИЕЙ В ФАЙЛЕ
# --------------------------
HISTORY_FILE = Path("conversation_history.json")


def load_history(thread_id: str) -> List[BaseMessage]:
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            all_histories = json.load(f)
        history_dicts = all_histories.get(thread_id, [])
        return messages_from_dict(history_dicts)
    return []


def save_history(thread_id: str, messages: List[BaseMessage]) -> None:
    all_histories = {}
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            all_histories = json.load(f)
    all_histories[thread_id] = messages_to_dict(messages)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(all_histories, f, ensure_ascii=False, indent=2)


# --------------------------
# 2. НАСТРОЙКИ DEEPSEEK
# --------------------------
DEEPSEEK_TOKEN = "9stnuavbRnmWSpQwm9+YxM0EzrHER6vdq36hCIxzSzg9eWmrSGvyK+yYpjdNZh5+"
genai.configure(api_key=DEEPSEEK_TOKEN)
MODEL_NAME = "deepseek-chat"


def ask_llm(prompt: str) -> str:
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt, stream=False)
        return response.text
    except Exception as e:
        print(f"Ошибка DeepSeek: {e}")
        return "Извините, произошла ошибка."


# --------------------------
# 3. ИНСТРУМЕНТЫ
# --------------------------
def save_note(content: str) -> str:
    with open("notes.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {content}\n")
    return f"Заметка сохранена: {content}"


def get_weather(city: str) -> str:
    import requests
    url = f"https://wttr.in/{city}?format=%C+%t&lang=ru"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return f"Погода в {city}: {resp.text.strip()}"
        return f"Не удалось получить погоду для {city}"
    except Exception as e:
        return f"Ошибка: {str(e)}"


def read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            if len(content) > 10000:
                content = content[:10000] + "\n... (файл слишком большой, показаны первые 10000 символов)"
            return f"```\n{content}\n```"
    except FileNotFoundError:
        return f"❌ Файл '{path}' не найден."
    except Exception as e:
        return f"Ошибка чтения {path}: {e}"


def write_file(path: str, content: str) -> str:
    try:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"✅ Файл {path} записан."
    except Exception as e:
        return f"❌ Ошибка: {e}"


def append_to_file(path: str, content: str) -> str:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(content + "\n")
        return f"✅ Добавлено в {path}."
    except Exception as e:
        return f"❌ Ошибка: {e}"


def create_file(name: str, content: str = "") -> str:
    if os.path.exists(name):
        return f"❌ Файл '{name}' уже существует."
    try:
        with open(name, "w", encoding="utf-8") as f:
            f.write(content)
        return f"✅ Файл '{name}' создан."
    except Exception as e:
        return f"❌ Ошибка: {e}"


def list_files(path: str = ".") -> str:
    try:
        items = os.listdir(path)
        result = f"📁 {path}:\n"
        for item in sorted(items):
            full = os.path.join(path, item)
            if os.path.isdir(full):
                result += f"📂 {item}/\n"
            else:
                size = os.path.getsize(full)
                result += f"📄 {item} ({size} байт)\n"
        return result
    except Exception as e:
        return f"❌ Ошибка: {e}"


def web_search(query: str) -> str:
    import requests
    url = "https://api.duckduckgo.com/"
    params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        data = resp.json()
        if data.get("AbstractText"):
            return data["AbstractText"]
        if data.get("RelatedTopics"):
            for topic in data["RelatedTopics"][:3]:
                if isinstance(topic, dict) and "Text" in topic:
                    return topic["Text"][:400]
        return "Ничего не найдено."
    except Exception as e:
        return f"Ошибка поиска: {e}"


def run_cmd(command: str) -> str:
    try:
        shell = platform.system() == "Windows"
        result = subprocess.run(command, shell=shell, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            output = result.stdout.strip()
            if len(output) > 3000:
                output = output[:3000] + "\n... (вывод обрезан)"
            return f"```\n{output or '(пусто)'}\n```"
        else:
            return f"Ошибка (код {result.returncode}):\n```\n{result.stderr.strip()}\n```"
    except subprocess.TimeoutExpired:
        return "⏰ Команда выполнялась более 30 секунд."
    except Exception as e:
        return f"Ошибка: {e}"


def press_keys(keys: str) -> str:
    try:
        import pyautogui
        pyautogui.FAILSAFE = True
        combos = keys.split('+')
        if len(combos) > 1:
            with pyautogui.hold(combos[0].lower()):
                for k in combos[1:]:
                    pyautogui.press(k.lower())
        else:
            pyautogui.press(keys.lower())
        return f"✅ Клавиша(и) '{keys}' нажата."
    except ImportError:
        return "❌ Требуется установка: pip install pyautogui"
    except Exception as e:
        return f"❌ Ошибка: {e}"


# --------------------------
# 4. АГЕНТ LANGGRAPH
# --------------------------
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    json_command: Optional[Dict[str, Any]]


def parse_json_command(state: AgentState):
    last_msg = state["messages"][-1].content

    # Если сообщение содержит прикреплённый файл — не пытаемся парсить JSON
    if "[Файл:" in last_msg and "```" in last_msg:
        return {"json_command": {"action": "none"}}

    prompt = f"""
Ты — строгий преобразователь запросов в JSON. Твой ответ должен быть только валидным JSON-объектом.

- Запомнить текст: {{"action": "note", "text": "..."}}
- Погода: {{"action": "weather", "city": "..."}}
- Напомнить: {{"action": "remind", "text": "..."}}
- Создать файл: {{"action": "create_file", "name": "...", "content": "..."}}
- Прочитать файл: {{"action": "read_file", "path": "..."}}
- Записать файл: {{"action": "write_file", "path": "...", "content": "..."}}
- Добавить в файл: {{"action": "append_file", "path": "...", "content": "..."}}
- Список файлов: {{"action": "list_files", "path": "..."}}
- Поиск в интернете: {{"action": "search", "query": "..."}}
- Выполнить команду CMD: {{"action": "cmd", "command": "..."}}
- Нажать клавиши: {{"action": "keys", "keys": "..."}}

Если запрос не соответствует ни одному действию, верни {{"action": "none"}}.

Запрос пользователя: "{last_msg[:500]}"
Ответ — только JSON:
"""
    resp = ask_llm(prompt)
    resp = resp.strip()
    if resp.startswith("```json"):
        resp = resp[7:]
    if resp.startswith("```"):
        resp = resp[3:]
    if resp.endswith("```"):
        resp = resp[:-3]
    resp = resp.strip()
    try:
        cmd = json.loads(resp)
    except:
        cmd = {"action": "none"}
    return {"json_command": cmd}


def execute_tool(state: AgentState):
    cmd = state.get("json_command", {})
    action = cmd.get("action")
    if not action or action == "none":
        return {"messages": []}

    if action == "note":
        result = save_note(cmd.get("text", ""))
    elif action == "weather":
        result = get_weather(cmd.get("city", "Москва"))
    elif action == "remind":
        result = f"🔔 Напоминание: {cmd.get('text')}"
    elif action == "create_file":
        result = create_file(cmd.get("name", ""), cmd.get("content", ""))
    elif action == "read_file":
        result = read_file(cmd.get("path", ""))
    elif action == "write_file":
        result = write_file(cmd.get("path", ""), cmd.get("content", ""))
    elif action == "append_file":
        result = append_to_file(cmd.get("path", ""), cmd.get("content", ""))
    elif action == "list_files":
        result = list_files(cmd.get("path", "."))
    elif action == "search":
        result = web_search(cmd.get("query", ""))
    elif action == "cmd":
        result = run_cmd(cmd.get("command", ""))
    elif action == "keys":
        result = press_keys(cmd.get("keys", ""))
    else:
        result = f"Неизвестное действие: {action}"

    return {"messages": [AIMessage(content=result)]}


def chat_llm(state: AgentState):
    history = state["messages"]

    # Проверяем, есть ли в последнем сообщении прикреплённый файл
    if history and isinstance(history[-1], HumanMessage):
        content = history[-1].content
        if "[Файл:" in content and "```" in content:
            file_match = re.search(r'```\n(.*?)\n```', content, re.DOTALL)
            user_request = re.sub(r'\[Файл:.*?```.*?```\n*', '', content, flags=re.DOTALL).strip()
            if not user_request:
                user_request = "Проанализируй содержимое этого файла"

            if file_match:
                file_content = file_match.group(1)
                analysis_prompt = f"""
Пользователь прикрепил файл. Проанализируй его содержимое и дай полезный ответ.

Содержимое файла:
{file_content[:8000]}

Задача пользователя: {user_request}

Требования:
- Выдели ключевую информацию, структуру, смысл.
- Дай рекомендации по улучшению, если это инструкция.
- Не копируй файл без комментариев.
- Отвечай полезно и по делу.
"""
                history[-1].content = analysis_prompt

    prompt = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in history
    )
    sys = "Ты ассистент Toda. Отвечай кратко, дружелюбно и по делу. Если нужно — анализируй, давай рекомендации."
    response = ask_llm(sys + "\n\n" + prompt)
    return {"messages": [AIMessage(content=response)]}


# Построение графа
workflow = StateGraph(AgentState)
workflow.add_node("parse_json_command", parse_json_command)
workflow.add_node("execute_tool", execute_tool)
workflow.add_node("chat_llm", chat_llm)

workflow.set_entry_point("parse_json_command")


def route(state):
    if state.get("json_command", {}).get("action") != "none":
        return "execute_tool"
    else:
        return "chat_llm"


workflow.add_conditional_edges("parse_json_command", route)
workflow.add_edge("execute_tool", END)
workflow.add_edge("chat_llm", END)

graph = workflow.compile()

# --------------------------
# 5. FASTAPI СЕРВЕР
# --------------------------
app = FastAPI(title="Toda Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"
    file: Optional[Dict[str, str]] = None


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        history = load_history(request.thread_id)

        user_message = request.message

        if request.file:
            file_context = f"\n\n[Файл: {request.file.get('name')}]\n```\n{request.file.get('content', '')[:50000]}\n```\n\n"
            user_message = file_context + user_message

        history.append(HumanMessage(content=user_message))

        initial_state = {"messages": history, "json_command": None}
        final_state = graph.invoke(initial_state)

        new_messages = final_state["messages"]
        assistant_message = None
        for msg in reversed(new_messages):
            if isinstance(msg, AIMessage):
                assistant_message = msg
                break

        answer = assistant_message.content if assistant_message else "Извините, не удалось получить ответ."

        save_history(request.thread_id, new_messages)

        return ChatResponse(response=answer)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Эндпоинт для скачивания файлов (для главной страницы)
@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = Path("static/download") / filename
    if file_path.exists():
        return FileResponse(file_path, filename=filename)
    raise HTTPException(status_code=404, detail="File not found")


# Раздача статики
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    print("🚀 Toda запущен на http://127.0.0.1:8000")
    print("📁 Главная страница: http://127.0.0.1:8000")
    print("💬 Чат: http://127.0.0.1:8000/chat.html")
    uvicorn.run(app, host="127.0.0.1", port=8000)