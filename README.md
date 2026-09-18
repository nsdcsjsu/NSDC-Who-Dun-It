# Data Detective — RAG Chatbot Table Demo

Two-screen setup for a pop-up table:
- **Screen 1** (`/`) — the chat interface students type questions into.
- **Screen 2** (`/engine`) — the live "how the AI is searching" visualization, synced to Screen 1 in real time.

The chatbot only knows what's written in the 13 case documents in `case_docs/`.
It is never told who the culprit is — that answer lives only in `solution_key.txt`,
which is NOT part of this project and should never be ingested.

## 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Set up your accounts

**Azure OpenAI:** you need a resource with two deployments:
- a chat model deployment (e.g. gpt-4o-mini, gpt-4o, or gpt-35-turbo)
- an embedding model deployment (e.g. text-embedding-3-small)

**Qdrant Cloud:** create a free cluster at https://cloud.qdrant.io — grab its
URL and API key from the cluster dashboard.

## 3. Configure environment variables

```bash
cp .env.example .env
```

Fill in `.env` with your real Azure OpenAI endpoint/key/deployment names and
your Qdrant Cloud URL/API key.

## 4. Ingest the case documents into Qdrant

Run this once (and again any time you edit files in `case_docs/`):

```bash
python ingest.py
```

This chunks all 13 `.txt` files, embeds them via Azure OpenAI, and uploads
them to your Qdrant collection.

## 5. Run the server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 6. Open the two screens

- Chat screen (front-facing, Screen 1): `http://localhost:8000/`
- Engine room (behind-the-scenes, Screen 2): `http://localhost:8000/engine`

Open each in its own browser window/tab, full-screened on its own monitor.
As soon as someone asks a question on Screen 1, Screen 2 will animate the
document scan, show similarity scores for the top matches, and display the
exact text sent to the model — synced live via Server-Sent Events, no manual
refreshing needed.

## Notes

- `TOP_K` in `app.py` controls how many chunks get retrieved per question
  (default 5). Lower it for faster/tighter answers, raise it for more context.
- The system prompt in `app.py` (`SYSTEM_PROMPT`) is what keeps the bot from
  ever naming a culprit — it's instructed to answer only from retrieved text
  and to explicitly refuse to speculate about guilt.
- If you add/edit documents in `case_docs/`, re-run `python ingest.py` to
  refresh the Qdrant collection (it recreates the collection each time).
- Keep `solution_key.txt` and `storyline.txt` OUT of `case_docs/` — only the
  13 numbered evidence documents should ever be ingested.
