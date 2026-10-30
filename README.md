# Prime Chatbot – n8n to FastAPI

This is a FastAPI rewrite of the uploaded n8n workflow.

## Matches your n8n setup

| n8n node | FastAPI replacement |
| --- | --- |
| When chat message received | `POST /chat` |
| HTTP Request tool | Pinecone Assistant API at `/assistant/chat/primechatbot` |
| Pinecone `gpt-4o` | `PINECONE_MODEL=gpt-4o` |
| Google Gemini Chat Model | Google GenAI (`gemini-flash-lite-latest`) |
| Agent system instruction | Gemini only uses Pinecone's answer, no outside knowledge |
| Simple Memory | Client sends and receives last 6 chat turns in `history` |

No documents, vector indexes, or Pinecone Assistant settings are modified.

## 1. Local startup

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# On Windows PowerShell: Copy-Item .env.example .env
# Edit .env with your NEW Pinecone API key and Gemini API key
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs to test `/chat` in Swagger UI.

Example request:

```json
{
  "message": "What services do you provide?",
  "history": []
}
```

Also accepts `chatInput` instead of `message` to ease migration.

Example response (illustrative only):

```json
{
  "answer": "Example answer based on your Pinecone documents.",
  "output": "Example answer based on your Pinecone documents.",
  "session_id": "generated-uuid",
  "history": [
    {"role": "user", "content": "What services do you provide?"},
    {"role": "assistant", "content": "Example answer based on your Pinecone documents."}
  ],
  "sources": [],
  "used_gemini": true
}
```

For the next question, send `history` exactly as returned, optionally with `session_id`. This simulates the n8n Memory node safely on serverless Vercel. **This is not durable server-side memory.** If you want memory shared across devices, connect a database such as Supabase or Redis later.

## 2. Deploy to Vercel

1. Push this project to a new GitHub repository, excluding `.env`.
2. Import the repository into Vercel; use the FastAPI preset / default auto detection.
3. In Vercel → Project → Settings → Environment Variables, add:
   - `PINECONE_API_KEY` (a newly rotated key)
   - `PINECONE_ASSISTANT_URL` (value in `.env.example`)
   - `PINECONE_MODEL` = `gpt-4o`
   - `GEMINI_API_KEY`
   - `GEMINI_MODEL` = `gemini-flash-lite-latest`
   - `USE_GEMINI` = `true`
   - `CORS_ORIGINS` = exact website URL, e.g., `https://example.com`
4. Deploy. Check `https://your-project.vercel.app/health` and `/docs`.
5. Your frontend POST URL is `https://your-project.vercel.app/chat`.

Do not put API secrets in the frontend. If you only need the Pinecone answer (no second Gemini rewrite), set `USE_GEMINI=false`; this also reduces latency/cost.

## 3. Frontend integration with conversation memory

```js
const API_URL = 'https://your-project.vercel.app/chat';
let history = JSON.parse(localStorage.getItem('prime_chat_history') || '[]');
let sessionId = localStorage.getItem('prime_chat_session_id') || null;

async function sendMessage(userText) {
  const response = await fetch(API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: userText, history, session_id: sessionId })
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'Chat failed');
  history = data.history;
  sessionId = data.session_id;
  localStorage.setItem('prime_chat_history', JSON.stringify(history));
  localStorage.setItem('prime_chat_session_id', sessionId);
  return data.answer;
}
```

## Security and production notes

- **Rotate the Pinecone API key**: the n8n JSON includes a full plain-text key. No key is included in this project.
- Restrict `CORS_ORIGINS` to your website; CORS is not authentication. Public chatbot APIs need rate limiting and abuse protection before broad exposure.
- Use a private Git repository if the surrounding project is confidential; `.env` is gitignored.
- Pinecone Assistant currently uses GPT-4o for its grounded answer, and Gemini rewrites it; this is **two model calls per turn**, just like your n8n flow. Disable Gemini if you want only the RAG answer.
- Gemini failures gracefully return the already-grounded Pinecone answer (`used_gemini:false`). Pinecone failures return an HTTP error instead of inventing a response.
- Model availability/latency and Pinecone costs depend on your accounts; this project has not been tested with your live API credentials.
