import os
import json
import time
from datetime import datetime, timezone, timedelta

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END, START
from typing import TypedDict

VOCABULARY_PATH = "files/vocabulary.json"
WORDS_PER_SESSION = 5
KST = timezone(timedelta(hours=9))


def pick_words() -> list:
    """날짜+세션 기반으로 인덱스 계산 - 상태 저장 없이 항상 동일한 단어 보장"""
    with open(VOCABULARY_PATH, "r", encoding="utf-8") as f:
        words = json.load(f)

    now = datetime.now(KST)

    # 하루 3세션: 오전 7시=0, 오후 12시=1, 오후 7시=2
    if now.hour < 12:
        session = 0
    elif now.hour < 19:
        session = 1
    else:
        session = 2

    # 2024-01-01 기준 경과 일수
    base = datetime(2024, 1, 1, tzinfo=KST)
    day_number = (now - base).days

    total_sessions = day_number * 3 + session
    start_index = (total_sessions * WORDS_PER_SESSION) % len(words)

    picked = [words[(start_index + i) % len(words)] for i in range(WORDS_PER_SESSION)]
    print(f"[{now.strftime('%Y-%m-%d %H:%M KST')}] 세션 {session} | 인덱스 {start_index} | 단어: {picked}")
    return picked


def ask_groq(picked: list) -> str:
    """LangGraph로 단어 뜻 + 예문 생성"""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ["GROQ_API_KEY"],
    )

    class VocabState(TypedDict):
        words: list
        result: str

    def generate_node(state: VocabState) -> VocabState:
        words_list = "\n".join(f"- {w}" for w in state["words"])
        prompt = f"""You are an English vocabulary teacher for Korean learners.

STRICT RULES:
- 발음기호, 예문(영어 문장)을 제외한 모든 텍스트는 반드시 한국어로만 작성하세요.
- 한국어, 영어 외 다른 언어(일본어, 중국어, 베트남어 등)는 절대 사용하지 마세요.
- 형식을 절대 변경하지 마세요. 아래 형식을 정확히 지키세요.

단어 목록:
{words_list}

아래 형식으로 단어마다 빈 줄을 넣어 작성하세요:

1️⃣ [영어단어] [발음기호]
뜻: [한국어 뜻] ([품사: 명사/동사/형용사/부사 중 하나])
예문: [자연스러운 영어 예문]
해석: [예문의 한국어 번역]

2️⃣ ...

형식 외의 추가 설명이나 주석은 절대 작성하지 마세요."""

        response = llm.invoke([HumanMessage(content=prompt)])
        return {**state, "result": response.content}

    graph = StateGraph(VocabState)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", END)
    app = graph.compile()

    result = app.invoke({"words": picked, "result": ""})
    return result["result"]


def send_telegram(result: str):
    """텔레그램으로 전송"""
    import requests

    now = datetime.now(KST)
    today = now.strftime("%Y년 %m월 %d일")
    hour = now.hour
    session = "🌅 아침" if hour < 12 else ("☀️ 점심" if hour < 19 else "🌙 저녁")

    message = (
        f"📚 <b>오늘의 영단어</b>\n"
        f"🗓 {today} | {session}\n"
        f"{'━' * 25}\n\n"
        f"{result}"
    )

    bot_token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    for i in range(0, len(message), 4000):
        chunk = message[i:i+4000]
        resp = requests.post(url, json={
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": "HTML",
        })
        if resp.ok:
            print("텔레그램 전송 성공")
        else:
            print(f"텔레그램 전송 오류: {resp.text}")
        time.sleep(0.5)


if __name__ == "__main__":
    picked = pick_words()
    result = ask_groq(picked)
    send_telegram(result)
