# Vocabulary GitHub Actions

GitHub Actions를 활용한 영어 단어 자동 학습 알림 시스템

## 개요

네이버 단어장에서 추출한 단어를 하루 3회 Telegram으로 전송합니다.
Groq LLM(LangGraph)이 단어의 뜻, 발음기호, 예문, 해석을 자동 생성합니다.

## 구조

```
├── .github/workflows/
│   └── vocabulary.yml     # GitHub Actions 스케줄 설정
├── files/
│   └── vocabulary.json    # 단어장 (426개)
├── vocabulary_sender.py   # 메인 실행 스크립트
```

## 동작 방식

- 날짜 + 세션(아침/점심/저녁) 기반으로 인덱스를 계산해 매번 5개 단어 선택
- 서버 없이 상태 저장 불필요 → 같은 시간대 실행 시 항상 동일한 단어 보장
- 426개 단어를 순서대로 소진 후 처음부터 반복

## 전송 스케줄 (KST)

| 시간 | UTC |
|------|-----|
| 오전 7시 | 22:00 (전날) |
| 오후 12시 | 03:00 |
| 오후 7시 | 10:00 |

## 설정

GitHub 레포 → Settings → Secrets and variables → Actions에 아래 3개 등록

| Secret | 설명 |
|--------|------|
| `GROQ_API_KEY` | Groq API 키 |
| `TELEGRAM_BOT_TOKEN` | Telegram 봇 토큰 |
| `TELEGRAM_CHAT_ID` | Telegram 채팅 ID |

## 수동 실행

GitHub Actions 탭 → Daily Vocabulary → Run workflow
