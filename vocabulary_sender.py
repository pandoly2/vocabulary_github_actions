import os
import json
import random
import time
from datetime import datetime, timezone, timedelta

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END, START
from typing import TypedDict

VOCABULARY_PATH = "files/vocabulary.json"
DIFFICULT_PATH = "files/difficult_words.json"
SENT_LOG_PATH = "files/sent_log.json"
WORDS_PER_SESSION = 5
MAX_REVIEW_INJECT = 2
MAX_INTERVAL = 30
KST = timezone(timedelta(hours=9))


def _load_pinned() -> list:
    try:
        with open(DIFFICULT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else list(data.keys())
    except Exception:
        return []


def pick_words() -> list:
    """날짜+세션 기반으로 인덱스 계산 + 복습 단어 주입."""
    with open(VOCABULARY_PATH, "r", encoding="utf-8") as f:
        words = json.load(f)

    now = datetime.now(KST)

    if now.hour < 12:
        session = 0
    elif now.hour < 19:
        session = 1
    else:
        session = 2

    base = datetime(2024, 1, 1, tzinfo=KST)
    day_number = (now - base).days
    total_sessions = day_number * 3 + session
    start_index = (total_sessions * WORDS_PER_SESSION) % len(words)
    regular = [words[(start_index + i) % len(words)] for i in range(WORDS_PER_SESSION)]

    # 복습 단어 주입: next_review <= today인 단어
    today = now.date().isoformat()
    difficult = _load_difficult()
    due = [w for w, d in difficult.items() if d.get("next_review", "9999") <= today][:MAX_REVIEW_INJECT]

    # 복습 단어를 앞에 배치 (regular에서 중복 제거 후 뒤 채움)
    picked = due + [w for w in regular if w.lower() not in [d.lower() for d in due]]
    picked = picked[:WORDS_PER_SESSION]

    # 핀 단어 중 랜덤으로 1~2개 주입
    pinned = _load_pinned()
    inject_count = min(MAX_REVIEW_INJECT, len(pinned))
    injected = random.sample(pinned, inject_count) if inject_count > 0 else []
    picked = injected + [w for w in regular if w.lower() not in [p.lower() for p in injected]]
    picked = picked[:WORDS_PER_SESSION]

    print(f"[{now.strftime('%Y-%m-%d %H:%M KST')}] 세션 {session} | 정규: {regular} | 핀주입: {injected} | 최종: {picked}")

    # 전송 기록 저장
    try:
        log = json.load(open(SENT_LOG_PATH, encoding="utf-8")) if os.path.exists(SENT_LOG_PATH) else {}
        today_key = now.strftime("%Y-%m-%d")
        if today_key not in log:
            log[today_key] = {}
        log[today_key][str(session)] = {"words": picked, "sent_at": now.strftime("%H:%M")}
        keys = sorted(log.keys())
        for old_key in keys[:-7]:
            del log[old_key]
        json.dump(log, open(SENT_LOG_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  [sent_log] 기록 실패: {e}")

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

=== ABSOLUTE RULES (NEVER VIOLATE) ===
1. 발음기호(IPA)는 모든 단어에 반드시 포함. 빠뜨리는 것은 허용되지 않음.
2. 뜻(뜻:)과 해석(해석:)은 오직 한국어(한글)만 사용. 한자(漢字), 아랍 문자, 프랑스어, 일본어, 베트남어 등 어떤 외국어도 절대 사용 금지.
3. 예문(예문:)은 오직 영어만 사용.
4. 아래 형식을 한 글자도 바꾸지 말고 정확히 지킬 것.
5. 추가 설명, 주석, 인사말 일절 금지.

=== 올바른 출력 예시 ===
1️⃣ example /ɪɡˈzæmpəl/
뜻: 예시, 사례 (명사)
예문: This is a good example of clear writing.
해석: 이것은 명확한 글쓰기의 좋은 예시입니다.

=== 틀린 예시 (절대 하지 말 것) ===
❌ 해석에 한자 포함: "나는健康을 위해..." → 반드시 "나는 건강을 위해..."
❌ 발음기호 없음: "condiment" 만 쓰고 발음기호 생략 → 반드시 "/ˈkɒndɪmənt/" 포함
❌ 해석에 외국어: "매우 interessant했다" → 반드시 "매우 흥미로웠다"
❌ 뜻에 한자/아랍문자: "야생의, وحشی한, 狂野한" → 반드시 "야생의, 거친"

단어 목록:
{words_list}

위 단어들을 아래 형식으로 작성하세요 (단어마다 빈 줄 추가):

1️⃣ [영어단어] [IPA 발음기호]
뜻: [한국어 뜻만] ([품사: 명사/동사/형용사/부사 중 하나])
예문: [자연스러운 영어 예문]
해석: [예문의 순수 한국어 번역만]

2️⃣ ..."""

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
