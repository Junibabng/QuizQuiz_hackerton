# QuizQuiz Hackerton

Early FastAPI prototype for the "upload/topic → notes → quizzes → FSRS" learning pipeline.

## Getting started

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## API surface

- `POST /v1/learn/prepare` – Generate placeholder notes and quiz items for a document or topic.
- `GET /v1/review/due` – Retrieve cards that are currently due for review for a specific `user_id`.
- `POST /v1/review/grade` – Submit a grade (1–4) for a reviewed card to update its next due date.

The current implementation stores state in memory and provides deterministic stub content suitable for further iteration.
