# Jamaru -- immersive chatbot

Dark, cinematic, hover-reveal chat UI for Jamaru, wired to a real Groq/Gemini
LLM backend and a hosted MCP tool server.

## Files

- `web_chatbot_immersive.py` -- the Flask app/entry point.
- `web_chatbot_langchain.py` -- shared connection/LLM logic (MCP session
  handling with auto-reconnect, Groq primary + Gemini fallback, tool
  binding, PDF export). Required by `web_chatbot_immersive.py` -- don't
  remove it.
- `static/jamaru_immersive_preview.html` -- the frontend.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env with your real GROQ_API_KEY, GEMINI_API_KEY, and MCP_SERVER_URL
```

## Run

```bash
python web_chatbot_immersive.py
```

Then open `http://localhost:5070` (or whatever `IMMERSIVE_PORT` is set to).

## Notes

- This chatbot connects to an **already-hosted** MCP server via
  `MCP_SERVER_URL` -- it does not run `server.py` itself. Point it at your
  deployed MCP server's `/mcp` endpoint.
- Chat history is stored locally in `immersive_chat_history.jsonl`
  (git-ignored) -- each conversation is keyed by a session cookie.
- If the MCP connection drops (e.g. a free-tier host spinning down from
  inactivity), it reconnects automatically in the background -- no
  manual restart needed.
