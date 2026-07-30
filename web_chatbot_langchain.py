#!/usr/bin/env python3
"""
web_chatbot_langchain.py
---------------------------------------------------------
A drop-in variant of web_chatbot.py that adds automatic LLM-provider
fallback using LangChain: Groq (llama-3.3-70b-versatile) is the PRIMARY
model, exactly as before. If Groq's account-level rate limit is hit
(groq.RateLimitError, the real exception the Groq SDK raises on a 429),
this automatically retries the same request against Gemini
(gemini-2.5-flash) instead of failing the turn.

This file is a NEW, separate file on purpose -- web_chatbot.py is left
completely untouched, so the already-working app can keep running (or be
reverted to instantly) if anything here needs more testing. It reuses
web_chatbot.py's two legacy/unused HTML templates (HTML_PAGE, WIDGET_HTML
for the /classic and /widget routes) via import rather than duplicating
~1800 lines of markup; the main "/" route (the actual demo page) does not
depend on that import at all.

Everything else -- the system prompt, the 12 real MCP tools, the
show_chart / search_the_web special tools, turn-aware history trimming,
persistent chat_history.jsonl, chat_log.txt / token_usage.txt logging,
and all Flask routes -- is IDENTICAL in behavior to web_chatbot.py. Only
the function that actually talks to the LLM (call_groq -> call_llm) was
replaced.

Setup:
    pip install langchain langchain-groq langchain-google-genai
    # GROQ_API_KEY must already be in .env (same as web_chatbot.py)
    # Add one more line to .env:
    #   GOOGLE_API_KEY=your-gemini-key-here
    # (free key: https://aistudio.google.com/apikey)

Run:
    python web_chatbot_langchain.py
Then open http://127.0.0.1:5000 in your browser -- same as before.
---------------------------------------------------------
"""

import asyncio
import datetime
import json
import logging
import os
import re
import sys
import threading
import uuid
from typing import Optional

import groq
import httpx
from dotenv import load_dotenv
from fpdf import FPDF
from flask import Flask, jsonify, make_response, render_template_string, request
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

load_dotenv()

# If set, connect to an MCP server already running elsewhere over HTTP (its
# own separately hosted deployment, started with MCP_TRANSPORT=streamable-http
# -- see server.py) instead of starting server.py as a private subprocess.
# Left unset (the default), everything behaves EXACTLY as before: this file
# spawns server.py itself over stdio, same as it always has. Setting this
# is what makes "host the MCP once, point any chatbot at it" possible,
# without touching the local, already-working subprocess path at all.
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL")

# If the hosted MCP server has authentication enabled (its own MCP_AUTH_TOKEN
# env var is set), this chatbot needs to send the same token back as a
# bearer header on every request, or the server will reject it with 401.
# Left unset, no Authorization header is sent at all -- matching the
# server's own "no MCP_AUTH_TOKEN set = open, no auth required" behavior.
MCP_AUTH_TOKEN = os.environ.get("MCP_AUTH_TOKEN")

# ---------------------------------------------------------
# Logging -- identical setup/paths to web_chatbot.py, so both apps can
# share the same log files if you ever run them side by side for
# comparison (each write is just an appended line either way).
# ---------------------------------------------------------
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_log.txt")
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)
logger = logging.getLogger("multi-api-mcp-langchain")

# ---------------------------------------------------------
# Provider config -- Groq stays the PRIMARY, exactly as before. Gemini is
# the FALLBACK, only ever used if Groq raises a real rate-limit error.
# ---------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"  # still used by call_groq_web_search only
MAX_HISTORY_MESSAGES = 12

if not GROQ_API_KEY or GROQ_API_KEY == "your-groq-key-here":
    print(
        "Missing GROQ_API_KEY.\n"
        "Get a free key at https://console.groq.com/keys, then open the .env file\n"
        "in this folder and paste it in like this:\n"
        "  GROQ_API_KEY=your-actual-key-here\n"
        "Save the file and run this again.\n",
        file=sys.stderr,
    )
    sys.exit(1)

# Gemini is optional at startup on purpose: the app can still run Groq-only
# (exactly like web_chatbot.py) if no Gemini key is set yet -- it just
# won't have a fallback to catch a Groq rate-limit until one is added.
_GEMINI_AVAILABLE = bool(GOOGLE_API_KEY and GOOGLE_API_KEY != "your-gemini-key-here")
if not _GEMINI_AVAILABLE:
    print(
        "NOTE: No GOOGLE_API_KEY found in .env -- running Groq-only, no fallback "
        "provider. Add GOOGLE_API_KEY=<your key> (free at "
        "https://aistudio.google.com/apikey) to enable automatic Gemini fallback "
        "when Groq's rate limit is hit.",
        file=sys.stderr,
    )

# ---------------------------------------------------------
# Token usage log -- same format/behavior as web_chatbot.py, with one
# addition: which provider actually answered (groq or gemini), so you can
# see in this one file exactly when/how often the fallback kicked in.
# ---------------------------------------------------------
TOKEN_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "token_usage.txt")
_token_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _log_token_usage(usage: dict, model: str) -> None:
    if not usage:
        return
    prompt = usage.get("prompt_tokens", 0)
    completion = usage.get("completion_tokens", 0)
    total = usage.get("total_tokens", prompt + completion)
    _token_totals["prompt_tokens"] += prompt
    _token_totals["completion_tokens"] += completion
    _token_totals["total_tokens"] += total
    line = (
        f"{datetime.datetime.now().isoformat(timespec='seconds')} "
        f"model={model} | this call: prompt={prompt} completion={completion} total={total} "
        f"| session running total: prompt={_token_totals['prompt_tokens']} "
        f"completion={_token_totals['completion_tokens']} total={_token_totals['total_tokens']}\n"
    )
    try:
        with open(TOKEN_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass


# ---------------------------------------------------------
# Persistent chat history -- identical to web_chatbot.py.
# ---------------------------------------------------------
HISTORY_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_history.jsonl")


def _log_history_turn(question: str, answer, chart, session_id: str) -> None:
    try:
        entry = {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "session_id": session_id,
            "question": question,
            "answer": answer or "",
            "has_chart": bool(chart),
            "chart_title": (chart or {}).get("title", "") if chart else "",
        }
        with open(HISTORY_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _read_history(session_id: str, limit: int = 200) -> list:
    """Only returns turns that belong to the requesting browser's session_id
    -- this is the piece that keeps two different tabs/visitors from seeing
    each other's chat history. Older log lines written before this change
    (or by web_chatbot.py, which has no session concept) have no
    session_id field and are intentionally excluded here, rather than
    shown to everyone, since there's no way to know who they belonged to."""
    if not os.path.exists(HISTORY_LOG_PATH):
        return []
    entries = []
    try:
        with open(HISTORY_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("session_id") == session_id:
                    entries.append(entry)
    except OSError:
        return []
    entries.reverse()
    return entries[:limit]


# Computed once when the server starts, so the model knows "today" without
# guessing -- refreshes automatically each time you restart the app.
TODAY_STR = datetime.date.today().strftime("%A, %B %d, %Y")

# ---------------------------------------------------------
# System prompt -- byte-for-byte the same rules as web_chatbot.py. Nothing
# about how the model should behave changes just because the underlying
# provider might switch from Groq to Gemini mid-conversation.
# ---------------------------------------------------------
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        f"Today's date is {TODAY_STR}. Use this as the current date whenever a "
        "question depends on it (e.g. 'today', 'this week', or a tool that "
        "defaults to today), instead of guessing or assuming an earlier date. "
        "You are a helpful assistant with access to tools for weather (current, "
        "forecast, history), currency exchange rates, jokes, random facts, NASA's "
        "picture of the day, Wikipedia summaries, live web search, and GitHub user "
        "info. Use a tool whenever the user's question matches what it does. If "
        "the user asks for multiple different things in one message, call the "
        "tools one at a time. If no tool fits (e.g. air quality/AQI — not "
        "available), say so plainly rather than guessing or calling something "
        "irrelevant."
        "\n\n"
        "Picking the RIGHT tool matters more than just picking A tool: "
        "(1) Weather — always call get_coordinates on a named place FIRST (never "
        "guess lat/long from memory), then pick get_weather (right now), "
        "get_weather_forecast (upcoming days), or get_weather_history (past "
        "dates) based on the time frame actually asked about. "
        "(2) Exchange rates — get_exchange_rate is TODAY's single rate only; "
        "any date range, trend, or 'last N days' needs get_exchange_rate_history "
        "instead. For multiple currencies at once, call the right tool ONCE PER "
        "currency and combine the results yourself — never refuse a "
        "multi-currency question just because one call can't cover all of them. "
        "Currency codes must be exact 3-letter ISO codes; if the user names an "
        "ambiguous currency (Dinar, Peso, Rand, plain Dollar — shared by several "
        "countries with different codes), don't guess or refuse — ask a short "
        "clarifying question naming the 2-3 likely candidates. "
        "(3) get_wikipedia_summary vs search_the_web — Wikipedia for stable, "
        "historical, or reference topics (faster, directly sourced); "
        "search_the_web only when the question depends on what's true right now "
        "or recently (current events, recent news, anything that could have "
        "changed after your training data). Never use search_the_web for "
        "something Wikipedia already covers well. "
        "(4) get_random_fact and get_random_joke take NO topic or query input — "
        "the underlying API always returns one completely random result, with "
        "no way to filter by subject, place, or person. If the user asks for a "
        "fact/joke ABOUT something specific (e.g. 'a fact about Andhra', 'a "
        "joke for a programmer'), do NOT call get_random_fact/get_random_joke — "
        "that would silently ignore their actual request. Instead, either "
        "answer with a real, relevant fact yourself (using get_wikipedia_summary "
        "or search_the_web if it needs sourcing) or say plainly that the fact/"
        "joke tool is fully random and can't be targeted, and offer a random "
        "one instead. Only call get_random_fact/get_random_joke when the "
        "request is genuinely open-ended ('tell me a fact', 'tell me a joke')."
        "\n\n"
        "CRITICAL: never narrate your own tool-calling process to the user — no "
        "'I would need to call X', no mentioning tool/function names, call "
        "counts, or your internal plan. Just call whichever tool(s) answer the "
        "question and reply with the real answer. If a question is missing "
        "information a tool needs (e.g. no place named), ask a short clarifying "
        "question instead of guessing. If a tool's result is an error, explain "
        "in one plain sentence what went wrong (never repeat raw error text) "
        "and suggest a next step if there's an obvious one."
        "\n\n"
        "Data honesty, this is important: never state a specific fact, number, "
        "or result, and never put a number into a chart, unless it's real — "
        "either it came back from a tool call in this conversation, or the "
        "user directly typed/pasted it themselves. Both count as real data you "
        "can use; anything else, say you don't have it rather than inventing a "
        "plausible-sounding answer or chart. If the real data you have covers "
        "less than the question assumes (e.g. asked about a full year but you "
        "only have part of it), say so plainly rather than silently answering "
        "as if it were complete."
        "\n\n"
        "Reply in plain text only — no markdown (no **bold**, no bullet/numbered "
        "lists, no headers), since the chat window displays plain text and "
        "markdown symbols would show up as literal asterisks/hashes. Keep "
        "answers brief; only go longer when the data genuinely needs it. Short "
        "follow-up messages ('what about tomorrow?', 'and in Fahrenheit?') refer "
        "back to the most recent relevant topic in this conversation, not a "
        "brand new unrelated question."
        "\n\n"
        "Formatting real data: for a list of comparable items with the same "
        "fields (multiple currencies, a multi-day forecast, a short comparison), "
        "use a simple markdown table with | columns and a |---|---| separator "
        "row, e.g.:\n"
        "| Day | High | Low |\n| --- | --- | --- |\n| Mon | 25C | 14C |\n"
        "For a chart, graph, histogram, or pie chart of real numeric data, call "
        "the show_chart tool — never write a ```chart fenced block or any other "
        "text/code representation of a chart yourself. More generally: never "
        "respond to a computation, counting, categorizing, filtering, or "
        "charting request by writing Python/code for the user to run, and never "
        "ask them to do the math — do the actual work yourself (count it, "
        "bucket it, sum it) using whatever real data is available, and give the "
        "final answer directly, calling show_chart if a chart was asked for."
        "\n\n"
        "After a substantive answer, you may end with ONE short, natural "
        "follow-up suggestion about a related thing the user might want next "
        "(e.g. 'Want me to also check tomorrow's forecast?'). Keep it to a "
        "single plain sentence, only offer something genuinely answerable by "
        "an available tool, and skip it for small talk or answers that already "
        "fully close out the question."
        "\n\n"
        "CRITICAL: the follow-up suggestion is ALWAYS an addition to the real "
        "answer, never a replacement for it. If a tool returned an actual "
        "result (a joke, a fact, a number, a summary, etc.), you must state "
        "that real result in full first. Never reply with only a question "
        "like 'Want to hear another one?' or 'Want me to look that up?' when "
        "you already have the real content sitting in the tool result and "
        "haven't said it yet — that leaves the user with no answer at all."
    ),
}

# ---------------------------------------------------------
# show_chart / search_the_web -- identical schemas to web_chatbot.py.
# ---------------------------------------------------------
SHOW_CHART_TOOL = {
    "type": "function",
    "function": {
        "name": "show_chart",
        "description": (
            "Display a bar or pie chart of real numeric data you already have "
            "in this conversation — either from a tool result, or from numbers "
            "the user directly typed or pasted into the chat themselves. Call "
            "this tool instead of writing a fenced code block, Python, or any "
            "other code to show a chart. Never invent numbers — only chart "
            "real values you actually have. If asked to count, bucket, or "
            "categorize raw data before charting it, do that work yourself "
            "and pass the final counts as the items here."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short chart title."},
                "type": {
                    "type": "string",
                    "enum": ["bar", "pie"],
                    "description": (
                        "'bar' for comparisons, trends, or values over time. "
                        "'pie' for a breakdown of parts making up a whole "
                        "(proportions, percentages, 'how is X split by "
                        "category'). Only use 'pie' when the user specifically "
                        "wants a pie chart or a proportion/percentage "
                        "breakdown — for anything else, use 'bar'."
                    ),
                },
                "unit": {
                    "type": "string",
                    "description": "Optional unit label for the values, e.g. 'INR' or 'days'. Omit if not applicable.",
                },
                "items": {
                    "type": "array",
                    "description": "The data points to chart, in the order they should appear.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string", "description": "Category or x-axis label."},
                            "value": {"type": "number", "description": "The numeric value for this item."},
                        },
                        "required": ["label", "value"],
                    },
                    "minItems": 1,
                },
            },
            "required": ["title", "type", "items"],
        },
    },
}

SEARCH_WEB_TOOL = {
    "type": "function",
    "function": {
        "name": "search_the_web",
        "description": (
            "Search the live web for current, real-time information — recent "
            "news, events, prices, releases, or anything that could have "
            "changed after your training data, or that you're not confident "
            "is still accurate. Do NOT use this for stable, historical, or "
            "reference topics (biographies, established facts, general "
            "knowledge, how something works) — use get_wikipedia_summary for "
            "those instead, since it's faster and more directly sourced. Use "
            "search_the_web specifically when the question depends on what is "
            "true right now or recently, not on unchanging background "
            "information."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query — phrase it like a real web search, not a restatement of the whole user question.",
                },
            },
            "required": ["query"],
        },
    },
}

EXPORT_PDF_TOOL = {
    "type": "function",
    "function": {
        "name": "export_pdf",
        "description": (
            "Export a piece of text (a summary, report, or any content already "
            "discussed in this conversation) as a real downloadable PDF file. "
            "Use this ONLY when the user explicitly asks to save, export, or "
            "download something as a PDF -- not for every answer. Pass the "
            "exact text to put in the PDF body, plus a short title (used as "
            "the PDF's heading and filename)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short title for the PDF -- used as its heading and filename.",
                },
                "content": {
                    "type": "string",
                    "description": "The full text content to place in the PDF body.",
                },
            },
            "required": ["title", "content"],
        },
    },
}

# Generated PDFs are saved directly into the Flask app's own static folder,
# which is already served at the site root with no extra route needed (see
# Flask("...", static_folder="static", static_url_path="") below) -- so a
# file saved here is automatically downloadable the moment it's written, the
# same way capimg/*.png already are.
GENERATED_PDF_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "static", "generated_pdfs"
)
os.makedirs(GENERATED_PDF_DIR, exist_ok=True)


def _generate_pdf(title: str, content: str, host_url: str) -> str:
    """Builds a real PDF file from plain text and returns its full,
    absolute, downloadable URL. host_url is passed in explicitly (rather
    than read from Flask's `request` object here) because this function
    runs on the background asyncio thread via process_message/call_llm's
    tool-dispatch loop -- NOT on the Flask request-handling thread. Flask's
    `request` is thread-local and tied to the thread that received the HTTP
    request; touching it from the background thread would raise "Working
    outside of request context". Capturing request.host_url once, on the
    Flask thread, in the /chat route (where it's valid) and threading it
    through as a plain argument sidesteps that entirely."""
    safe_title = re.sub(r"[^A-Za-z0-9_-]+", "_", (title or "document").strip())[:50] or "document"
    filename = f"{safe_title}_{uuid.uuid4().hex[:8]}.pdf"
    filepath = os.path.join(GENERATED_PDF_DIR, filename)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, title or "Document")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 12)
    pdf.multi_cell(0, 8, content or "")
    pdf.output(filepath)

    return host_url.rstrip("/") + "/generated_pdfs/" + filename


def call_groq_web_search(query: str) -> str:
    """Unchanged from web_chatbot.py: one isolated, stateless call to Groq's
    own groq/compound-mini (built-in Tavily-powered web search). This is
    NOT part of the LangChain primary/fallback chain -- Gemini has no
    equivalent built-in live-search model, so if Groq itself is completely
    unreachable, this specific tool will surface that as a plain error
    message (same as before), rather than silently falling back to a
    different search mechanism that doesn't exist yet."""
    try:
        with httpx.Client(timeout=30.0) as client:
            res = client.post(
                GROQ_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                },
                json={"model": "groq/compound-mini", "messages": [{"role": "user", "content": query}]},
            )
    except httpx.TimeoutException as err:
        raise RuntimeError(f"Web search timed out: {err}") from err
    except httpx.ConnectError as err:
        raise RuntimeError(f"Couldn't connect to Groq for web search: {err}") from err
    except httpx.HTTPError as err:
        raise RuntimeError(f"Web search request failed: {err}") from err
    except Exception as err:  # noqa: BLE001
        raise RuntimeError(f"Web search failed unexpectedly: {err}") from err

    if res.status_code >= 400:
        raise RuntimeError(f"Web search API error {res.status_code}: {res.text[:300]}")
    try:
        data = res.json()
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as err:
        raise RuntimeError(f"Web search returned an unexpected response shape: {err}") from err
    if not content:
        raise RuntimeError("Web search returned an empty answer.")
    return content


# ---------------------------------------------------------
# LangChain message <-> plain-dict conversion.
#
# The rest of this file (trim_history, process_message, the Flask routes)
# all work with the SAME plain-dict message shape web_chatbot.py already
# used ({"role": ..., "content": ..., ["tool_calls"|"tool_call_id"]: ...}).
# That shape is kept as the single source of truth for `messages` on
# purpose -- trim_history's turn-boundary logic is already proven correct
# against exactly this shape, so converting to/from LangChain's message
# objects ONLY at the LLM call boundary (right here) means none of that
# proven logic has to be rewritten or re-verified.
# ---------------------------------------------------------


def _dicts_to_lc_messages(msgs: list) -> list:
    lc_msgs = []
    for m in msgs:
        role = m.get("role")
        if role == "system":
            lc_msgs.append(SystemMessage(content=m.get("content") or ""))
        elif role == "user":
            lc_msgs.append(HumanMessage(content=m.get("content") or ""))
        elif role == "assistant":
            tool_calls = []
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(
                    {"name": fn.get("name", ""), "args": args, "id": tc.get("id", ""), "type": "tool_call"}
                )
            lc_msgs.append(AIMessage(content=m.get("content") or "", tool_calls=tool_calls))
        elif role == "tool":
            lc_msgs.append(ToolMessage(content=m.get("content") or "", tool_call_id=m.get("tool_call_id", "")))
    return lc_msgs


def _extract_text_content(content) -> str:
    """Groq's responses always come back as a plain string, but Gemini
    (via langchain_google_genai) can return `content` as a list of
    structured blocks instead -- e.g. [{"type": "text", "text": "...",
    ...}] -- especially from the newer Gemini 3.x models. The rest of this
    app (and the frontend) expects `content` to always be a plain string,
    exactly like web_chatbot.py's Groq-only version. Passing a raw
    list/dict through unchanged crashed the chat page outright (React
    error #31: "Objects are not valid as a React child") the first time a
    Gemini fallback response used this structured shape instead of a bare
    string. This normalizes any shape back down to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text") or "")
        return "".join(parts)
    if isinstance(content, dict):
        return content.get("text") or ""
    return str(content)


def _ai_message_to_message_dict(ai_msg: AIMessage) -> dict:
    text = _extract_text_content(ai_msg.content)
    message = {"role": "assistant", "content": text if text else None}
    if getattr(ai_msg, "tool_calls", None):
        message["tool_calls"] = [
            {
                "id": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": tc.get("name", ""),
                    "arguments": json.dumps(tc.get("args") or {}),
                },
            }
            for tc in ai_msg.tool_calls
        ]
    return message


def _usage_from_ai_message(ai_msg: AIMessage) -> dict:
    """LangChain standardizes token counts across providers as
    `usage_metadata` on the returned AIMessage (input_tokens/output_tokens/
    total_tokens), when the underlying provider reports them -- true for
    both Groq and Gemini here. Falls back to zeros rather than raising if a
    provider ever omits it, since logging must never be able to break the
    actual chat (same principle as web_chatbot.py's _log_token_usage)."""
    usage = getattr(ai_msg, "usage_metadata", None) or {}
    return {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


# ---------------------------------------------------------
# Globals populated once the MCP connection + LLMs are ready.
# ---------------------------------------------------------
mcp_session = None
groq_tools = []  # OpenAI-format tool schemas -- same shape/role as in web_chatbot.py
_groq_llm = None
_gemini_llm = None

# Tool-less fallback model -- built immediately at import time (it only
# needs GROQ_API_KEY/GROQ_MODEL, both available before MCP ever connects),
# unlike _groq_llm above which needs the live MCP session's tool schemas
# before it can be bound and is therefore None until _finish_setup() runs.
# Without this, any chat message that arrives while MCP is still connecting
# (or stuck retrying, e.g. the other Render service being asleep/down)
# hits `_groq_llm.invoke(...)` on a None object and surfaces a raw
# "'NoneType' object has no attribute 'invoke'" error to the user instead
# of an actual reply. Using this fallback means the bot can still hold a
# plain conversation (no tool calls -- it wasn't given any schemas) while
# the real, tool-capable model finishes connecting in the background.
_groq_llm_notools = (
    ChatGroq(groq_api_key=GROQ_API_KEY, model_name=GROQ_MODEL, temperature=0.3, max_tokens=1600)
    if GROQ_API_KEY
    else None
)


def _extract_balanced_json(text: str):
    """Ported verbatim from web_chatbot.py. Scan forward from the first '{'
    in `text` and return exactly the first balanced {...} object, ignoring
    whatever comes after it (a stray '>', a '</function>' tag, truncation,
    etc). String literals are tracked so a brace inside a quoted value never
    miscounts the depth. Returns None if no '{' is found or it's never
    closed."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _parse_failed_generation(err_json: dict):
    """Ported verbatim from web_chatbot.py's parse_failed_generation.
    Recovers a malformed '<function=NAME{...}>' call from Groq's
    failed_generation text -- this is what Groq sends back inside a
    tool_use_failed 400 error when its own constrained decoding produces
    broken function-call syntax instead of valid JSON."""
    text = err_json.get("error", {}).get("failed_generation", "")
    match = re.search(r"<function=(\w+)", text)
    if not match:
        return None
    name = match.group(1)
    args_str = _extract_balanced_json(text[match.end():])
    if not args_str:
        return None
    try:
        args = json.loads(args_str)
    except json.JSONDecodeError:
        return None
    return name, args


def _extract_failed_generation_text(err_json: dict):
    """Ported verbatim from web_chatbot.py's extract_failed_generation_text.
    Sometimes 'failed_generation' isn't a malformed function call at all --
    it's the model's genuine final answer (plain text, maybe a table or a
    2+2-style direct answer), and Groq's tool-forcing just didn't like that
    the model chose to answer directly instead of calling a function. If the
    text doesn't look like an attempted function call, treat it as the real
    answer instead of throwing it away."""
    text = err_json.get("error", {}).get("failed_generation", "")
    if not text or "<function=" in text:
        return None
    return text.strip()


def _strip_null_tool_calls(data: dict) -> dict:
    """Ported verbatim from web_chatbot.py. Groq's response format can
    include an explicit "tool_calls": null on an ordinary answer that didn't
    call any tool -- fine as a response shape, but if sent back to Groq
    later as conversation history, Groq's request-side validation rejects
    the explicit null ('tool_calls: Value is not nullable'), breaking every
    later question in the session. Strip the key when it's null."""
    try:
        msg = data["choices"][0]["message"]
        if msg.get("tool_calls") is None and "tool_calls" in msg:
            del msg["tool_calls"]
    except (KeyError, IndexError, TypeError):
        pass
    return data


def _call_groq_raw_recovery(msgs: list, tools: list, retries: int = 2) -> Optional[dict]:
    """Raw-HTTP recovery path for Groq's tool_use_failed 400 error, used
    ONLY when LangChain's ChatGroq raises groq.BadRequestError with
    'tool_use_failed' in it. This is the exact, already-proven fix from
    web_chatbot.py's call_groq (see task history: "Structural fix:
    tool_choice=none retry + strip null tool_calls everywhere") -- it was
    never carried over when this LangChain file was first built, which is
    why this exact error was still happening here even though web_chatbot.py
    had already fixed it. Ported here rather than re-derived, so it's the
    same tested behavior: retry the identical request a couple times first
    (transient), then retry with tool_choice="none" (removes the function-
    call constraint so the model can finish its answer without being cut off
    mid-generation), then try to recover a malformed function call from
    failed_generation, then try to recover a genuine plain-text answer from
    failed_generation. Returns an OpenAI-format {"choices": [...]} dict on
    success (already ready to use, no AIMessage conversion needed), or None
    if truly unrecoverable -- in which case call_llm falls back to Gemini,
    same as it already does for a rate limit."""
    with httpx.Client(timeout=60.0) as client:
        for attempt in range(retries + 1):
            res = client.post(
                GROQ_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": msgs,
                    "tools": tools,
                    "tool_choice": "auto",
                    "temperature": 0.3,
                    "max_tokens": 1600,
                },
            )
            if res.status_code == 400 and "tool_use_failed" in res.text:
                if attempt < retries:
                    continue
                try:
                    retry_res = client.post(
                        GROQ_URL,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {GROQ_API_KEY}",
                        },
                        json={
                            "model": GROQ_MODEL,
                            "messages": msgs,
                            "tools": tools,
                            "tool_choice": "none",
                            "temperature": 0.3,
                            "max_tokens": 1600,
                        },
                    )
                    if retry_res.status_code < 400:
                        retry_data = retry_res.json()
                        _log_token_usage(retry_data.get("usage", {}), f"{GROQ_MODEL} (tool_choice=none recovery)")
                        return _strip_null_tool_calls(retry_data)
                except Exception:  # noqa: BLE001
                    pass  # fall through to the text-scraping fallback below

                try:
                    err_json = res.json()
                except Exception:  # noqa: BLE001
                    err_json = {}
                fallback = _parse_failed_generation(err_json)
                if fallback:
                    name, args = fallback
                    return {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": f"call_{uuid.uuid4().hex[:8]}",
                                            "type": "function",
                                            "function": {
                                                "name": name,
                                                "arguments": json.dumps(args),
                                            },
                                        }
                                    ],
                                }
                            }
                        ]
                    }
                recovered_text = _extract_failed_generation_text(err_json)
                if recovered_text:
                    return {"choices": [{"message": {"role": "assistant", "content": recovered_text}}]}
                return None  # genuinely unrecoverable -- caller falls back to Gemini
            if res.status_code >= 400:
                return None
            data = res.json()
            _log_token_usage(data.get("usage", {}), f"{GROQ_MODEL} (raw)")
            return _strip_null_tool_calls(data)
    return None


def call_llm(msgs: list) -> dict:
    """Replaces web_chatbot.py's call_groq(msgs, tools). Groq is tried
    FIRST, exactly like before -- same model, same tools, same messages.
    The ONLY new behavior: if Groq raises groq.RateLimitError (the real
    exception the Groq SDK raises on a 429, after LangChain's own internal
    retries are exhausted), this catches specifically that error and
    retries the exact same request against Gemini instead of surfacing the
    rate-limit error to the user. Any OTHER exception (a genuine bug, a bad
    request, an auth error) is NOT caught here and propagates up exactly
    like web_chatbot.py's call_groq did for its own non-429 errors --
    deliberately not papering over real problems by silently switching
    providers on every possible failure, only on the specific one this was
    built for.

    Returns the same {"choices": [{"message": {...}}]} shape call_groq
    returned, so process_message's tool-calling loop below is COMPLETELY
    unchanged from web_chatbot.py."""
    lc_msgs = _dicts_to_lc_messages(msgs)

    if _groq_llm is None:
        # MCP hasn't connected yet (still retrying in the background, e.g.
        # a hosted MCP server waking up or down) -- answer in plain
        # conversation with the tool-less fallback instead of throwing a
        # raw AttributeError. Tool-dependent questions won't be fulfilled
        # until the real connection finishes, but the chat stays usable.
        if _groq_llm_notools is None:
            raise RuntimeError(
                "The assistant isn't ready yet (no GROQ_API_KEY configured). "
                "Check the server's .env and restart."
            )
        ai_msg = _groq_llm_notools.invoke(lc_msgs)
        _log_token_usage(_usage_from_ai_message(ai_msg), f"{GROQ_MODEL} (no-tools fallback, MCP not ready)")
        return {"choices": [{"message": _ai_message_to_message_dict(ai_msg)}]}

    provider_label = GROQ_MODEL
    try:
        ai_msg = _groq_llm.invoke(lc_msgs)
    except groq.RateLimitError as err:
        if not _GEMINI_AVAILABLE:
            # No fallback configured -- surface the same clear message
            # web_chatbot.py already gave, rather than a raw stack trace.
            raise RuntimeError(
                "Groq's free-tier rate limit was hit, and no GOOGLE_API_KEY is "
                "set for the Gemini fallback yet. Add one to .env to enable "
                "automatic fallback, or wait a bit and try again."
            ) from err
        logger.info("PROVIDER FALLBACK: Groq rate-limited (%s) -> retrying with Gemini (%s)", err, GEMINI_MODEL)
        provider_label = f"{GEMINI_MODEL} (fallback from groq)"
        ai_msg = _gemini_llm.invoke(lc_msgs)
        _log_token_usage(_usage_from_ai_message(ai_msg), provider_label)
        return {"choices": [{"message": _ai_message_to_message_dict(ai_msg)}]}
    except groq.BadRequestError as err:
        # This is the "tool_use_failed" 400 error -- Groq's constrained
        # decoding for tool-calling occasionally produces malformed function-
        # call syntax, or rejects a perfectly good plain-text answer just
        # because the model chose not to call a function. web_chatbot.py
        # already has a proven fix for exactly this (retry, then retry with
        # tool_choice="none", then recover the real answer from
        # failed_generation) -- it just never got carried into this file
        # until now. Any OTHER 400 (a real bad request, not this specific
        # quirk) is NOT swallowed here and still propagates as a real error.
        if "tool_use_failed" not in str(err):
            raise
        logger.info("GROQ tool_use_failed (%s) -> attempting raw recovery (retry / tool_choice=none / failed_generation)", err)
        recovered = _call_groq_raw_recovery(msgs, groq_tools)
        if recovered is not None:
            return recovered
        # Raw recovery couldn't produce a usable answer either -- fall back
        # to Gemini rather than surfacing the raw Groq error to the user,
        # same principle as the rate-limit fallback above.
        if not _GEMINI_AVAILABLE:
            raise RuntimeError(
                "Groq had trouble formatting a tool call for this question, and "
                "no GOOGLE_API_KEY is set for a Gemini fallback yet. Try "
                "rephrasing the question, or add a Gemini key to .env to enable "
                "automatic fallback."
            ) from err
        logger.info("PROVIDER FALLBACK: Groq tool_use_failed (unrecoverable) -> retrying with Gemini (%s)", GEMINI_MODEL)
        provider_label = f"{GEMINI_MODEL} (fallback from groq tool_use_failed)"
        ai_msg = _gemini_llm.invoke(lc_msgs)
        _log_token_usage(_usage_from_ai_message(ai_msg), provider_label)
        return {"choices": [{"message": _ai_message_to_message_dict(ai_msg)}]}

    _log_token_usage(_usage_from_ai_message(ai_msg), provider_label)
    return {"choices": [{"message": _ai_message_to_message_dict(ai_msg)}]}


NOTE_PREFIX = "[Earlier context] "


def trim_history(msgs: list) -> list:
    """Unchanged from web_chatbot.py -- see that file for the full
    reasoning comment. Still operates on whole turns of plain dicts, still
    completely independent of which provider (Groq or Gemini) produced any
    given assistant message."""
    system_msg = msgs[0]
    rest = msgs[1:]

    existing_note = None
    if rest and rest[0].get("role") == "system" and rest[0].get("content", "").startswith(NOTE_PREFIX):
        existing_note = rest[0]
        rest = rest[1:]

    turns = []
    for m in rest:
        if m.get("role") == "user" or not turns:
            turns.append([m])
        else:
            turns[-1].append(m)

    kept_turns = []
    kept_count = 0
    for turn in reversed(turns):
        if kept_turns and kept_count + len(turn) > MAX_HISTORY_MESSAGES:
            break
        kept_turns.insert(0, turn)
        kept_count += len(turn)

    dropped_turns = turns[: len(turns) - len(kept_turns)]
    kept = [m for turn in kept_turns for m in turn]
    dropped_topics = [
        m["content"][:80]
        for turn in dropped_turns
        for m in turn
        if m.get("role") == "user" and m.get("content")
    ]

    if not dropped_topics:
        return [system_msg] + ([existing_note] if existing_note else []) + kept

    prior = existing_note["content"][len(NOTE_PREFIX):] if existing_note else "Earlier in this conversation, topics included:"
    combined = (prior + " " + "; ".join(dropped_topics))[-500:]
    note = {"role": "system", "content": NOTE_PREFIX + combined}
    return [system_msg, note] + kept


# ---------------------------------------------------------
# Per-session conversation store.
#
# web_chatbot.py (and this file, before this change) used ONE global
# `messages` list shared by every visitor -- fine for a solo demo, but it
# means two different browser tabs/people would see and add to the exact
# same conversation. This replaces that single list with a dict keyed by
# a per-browser session_id (see /chat and /history below for how that ID
# is created/read via a cookie), so each visitor gets their own isolated
# conversation in memory. `threading.Lock` guards the dict itself (not
# each session's list) since Flask's dev server currently handles one
# request at a time anyway -- this is just cheap insurance if that ever
# changes.
# ---------------------------------------------------------
_sessions: dict = {}
_sessions_lock = threading.Lock()


def _get_session_messages(session_id: str) -> list:
    with _sessions_lock:
        if session_id not in _sessions:
            _sessions[session_id] = [SYSTEM_PROMPT]
        return _sessions[session_id]


def _set_session_messages(session_id: str, msgs: list) -> None:
    with _sessions_lock:
        _sessions[session_id] = msgs


async def process_message(user_input: str, session_id: str, host_url: str):
    """Same logic as before, except it now reads/writes THIS session's own
    message list (via _get_session_messages/_set_session_messages) instead
    of one shared global list. Every branch (show_chart, search_the_web,
    real MCP tools via mcp_session.call_tool) is otherwise identical.

    host_url is only used by the new export_pdf branch below, to build an
    absolute download link -- see _generate_pdf's docstring for why it has
    to be captured on the Flask thread and passed in as a plain argument
    rather than read from `flask.request` here."""
    messages = _get_session_messages(session_id)

    activated = []
    chart_payload = None
    image_payload = None
    pdf_payload = None
    # Tracks (tool_name, error_message) for any tool call that actually
    # failed this turn. Models sometimes gloss over a tool error with a
    # vague, non-committal reply instead of stating what went wrong (e.g.
    # "here are some videos... want a specific topic?" after a real
    # failure) -- same "don't trust the model to faithfully relay the real
    # result" principle already applied to images/PDFs, extended to errors
    # so the user always sees the ground truth instead of a hallucinated
    # non-answer.
    tool_errors = []
    # Tracks (tool_name, raw_result_text, urls_found) for successful tool
    # calls whose real result contains links (search_youtube, search_arxiv,
    # etc.) -- see the "else" branch below for why this exists alongside
    # tool_errors.
    tool_link_results = []
    messages.append({"role": "user", "content": user_input})

    data = call_llm(messages)
    choice = data["choices"][0]

    while choice["message"].get("tool_calls"):
        messages.append(choice["message"])

        for tool_call in choice["message"]["tool_calls"]:
            fn = tool_call["function"]
            args = json.loads(fn.get("arguments") or "{}")
            activated.append({"name": fn["name"], "args": args})

            if fn["name"] == "show_chart":
                chart_payload = args
                result_text = "Chart displayed to the user."
            elif fn["name"] == "export_pdf":
                try:
                    pdf_url = _generate_pdf(
                        args.get("title", "Document"), args.get("content", ""), host_url
                    )
                    pdf_payload = {"url": pdf_url, "title": args.get("title", "Document")}
                    result_text = f"PDF generated: {pdf_url}"
                except Exception as err:  # noqa: BLE001
                    result_text = f"Error generating PDF: {err}"
                    tool_errors.append((fn["name"], str(err)))
                    logger.info(
                        "TOOL ERROR (exception) name=%s args=%s -> %s: %s",
                        fn["name"], {"title": args.get("title")}, type(err).__name__, err,
                    )
            elif fn["name"] == "search_the_web":
                try:
                    result_text = call_groq_web_search(args.get("query", ""))
                except Exception as err:  # noqa: BLE001
                    result_text = f"Error calling tool: {err}"
                    tool_errors.append((fn["name"], str(err)))
                    logger.info(
                        "TOOL ERROR (exception) name=%s args=%s -> %s: %s",
                        fn["name"], args, type(err).__name__, err,
                    )
            else:
                try:
                    result = await mcp_session.call_tool(fn["name"], args)
                    result_text = "\n".join(
                        getattr(c, "text", str(c)) for c in result.content
                    )
                    if getattr(result, "isError", False):
                        tool_errors.append((fn["name"], result_text))
                        logger.info(
                            "TOOL ERROR (isError) name=%s args=%s -> %s",
                            fn["name"], args, result_text,
                        )
                    elif fn["name"] in ("generate_image", "create_diagram"):
                        # Don't rely on the model faithfully copying the raw
                        # URL into its visible reply text -- it sometimes
                        # paraphrases or reformats it instead (exactly what
                        # was happening: a plain link showing up instead of
                        # an actual displayed image). Same fix pattern as
                        # show_chart above: capture the real value straight
                        # from the tool's own result and send it to the
                        # frontend as its own dedicated field, so display
                        # never depends on the LLM echoing text correctly.
                        url_match = re.search(
                            r"https?://(?:image\.pollinations\.ai|kroki\.io)/\S+",
                            result_text,
                        )
                        if url_match:
                            image_payload = {
                                "url": url_match.group(0),
                                "kind": "image" if fn["name"] == "generate_image" else "diagram",
                            }
                    else:
                        # Same reliability principle again, generalized: tools
                        # like search_youtube/search_arxiv can succeed (real
                        # results, no error at all) and the model can STILL
                        # give a vague non-answer that drops the actual
                        # results instead of listing them (e.g. "here are
                        # some videos... want a specific topic?" with no
                        # links at all). That's not a tool error -- isError
                        # is False -- so tool_errors won't catch it. Track
                        # any URL-bearing successful result here so it can be
                        # force-included below if the model's reply doesn't
                        # actually contain those URLs.
                        result_urls = re.findall(r"https?://\S+", result_text)
                        if result_urls:
                            tool_link_results.append((fn["name"], result_text, result_urls))
                except Exception as err:  # noqa: BLE001
                    result_text = f"Error calling tool: {err}"
                    tool_errors.append((fn["name"], str(err)))
                    logger.info(
                        "TOOL ERROR (exception) name=%s args=%s -> %s: %s",
                        fn["name"], args, type(err).__name__, err,
                    )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result_text,
                }
            )

        data = call_llm(messages)
        choice = data["choices"][0]

    messages.append(choice["message"])
    messages = trim_history(messages)
    _set_session_messages(session_id, messages)

    final_content = choice["message"].get("content") or ""
    if tool_errors:
        # If a real tool failed this turn, make sure the actual reason shows
        # up somewhere in the reply -- don't let a vague, upbeat-sounding
        # non-answer ("here are some videos... want a specific topic?")
        # silently stand in for a genuine failure the user should know
        # about. Only append if the model's own reply doesn't already
        # mention the failed tool by name, so a model that DID explain the
        # error correctly doesn't get a redundant note tacked on.
        missing = [
            (name, msg) for name, msg in tool_errors
            if name.replace("_", " ") not in final_content.lower() and name not in final_content
        ]
        if missing:
            note_lines = [f"({name} failed: {msg})" for name, msg in missing]
            final_content = (final_content + "\n\n" + "\n".join(note_lines)).strip()

    # Same idea, for successful-but-dropped results: if a tool genuinely
    # returned real links and NONE of them made it into the model's reply,
    # the model glossed over real data -- append the actual raw result so
    # the user sees real results instead of a vague non-answer.
    for name, text, urls in tool_link_results:
        if not any(u.rstrip(").,!?") in final_content for u in urls):
            final_content = (final_content + "\n\n" + text).strip()

    return final_content, activated, chart_payload, image_payload, pdf_payload


def run_coro(coro):
    future = asyncio.run_coroutine_threadsafe(coro, bg_loop)
    return future.result()


_ready = threading.Event()
bg_loop = None


async def _server_task():
    """Same MCP connection lifecycle as web_chatbot.py. The only addition:
    once the tool schemas are built (identical code to web_chatbot.py --
    dynamically pulled from the live MCP session, never hand-typed), the
    two LangChain LLM objects are constructed ONCE and have those exact
    schemas bound to them via bind_tools(). LangChain's bind_tools accepts
    the same OpenAI-format {"type": "function", "function": {...}} dicts
    web_chatbot.py already builds, so the tool definitions themselves are
    reused verbatim, not redefined in a new format."""

    async def _finish_setup(session: ClientSession, connection_desc: str):
        """Shared by both connection modes below -- everything from here on
        (tool schema building, binding the two LLMs, the ready signal) is
        identical no matter HOW the session was established, since MCP
        itself works the same way regardless of transport.

        IMPORTANT: this is a NESTED function, so it needs its OWN global
        declaration for every name it assigns to (mcp_session, _groq_llm,
        _gemini_llm) -- _server_task's own `global` statement above only
        applies within _server_task's own body, not automatically to
        functions defined inside it. Without this line, those three names
        would silently become new LOCAL variables scoped only to this
        nested function, and every one of their assignments below would be
        thrown away the moment this function returns -- leaving the real
        module-level mcp_session/_groq_llm/_gemini_llm stuck at None
        forever, breaking every single chat request. groq_tools doesn't
        need this because it's only ever mutated with .extend()/.append()
        here, never reassigned -- mutating a list doesn't require a global
        declaration, only rebinding the name to a new object does."""
        global mcp_session, _groq_llm, _gemini_llm
        await session.initialize()

        tools_result = await session.list_tools()
        groq_tools.extend(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema,
                },
            }
            for t in tools_result.tools
        )
        groq_tools.append(SHOW_CHART_TOOL)
        groq_tools.append(SEARCH_WEB_TOOL)
        groq_tools.append(EXPORT_PDF_TOOL)
        mcp_session = session

        _groq_llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL,
            temperature=0.3,
            max_tokens=1600,
        ).bind_tools(groq_tools)

        if _GEMINI_AVAILABLE:
            _gemini_llm = ChatGoogleGenerativeAI(
                google_api_key=GOOGLE_API_KEY,
                model=GEMINI_MODEL,
                temperature=0.3,
                max_output_tokens=1600,
            ).bind_tools(groq_tools)

        print(
            f"Connected to MCP server ({connection_desc}). Loaded {len(groq_tools)} tools "
            f"({len(groq_tools) - 3} from server.py + show_chart + search_the_web + export_pdf). "
            f"Primary provider: Groq ({GROQ_MODEL}). Fallback provider: "
            f"{'Gemini (' + GEMINI_MODEL + ')' if _GEMINI_AVAILABLE else 'NONE -- add GOOGLE_API_KEY to enable'}."
        )

        _ready.set()

        stop_event = asyncio.Event()
        await stop_event.wait()

    async def _connect_once():
        if MCP_SERVER_URL:
            # Standalone-hosted mode: MCP_SERVER_URL points at a server.py
            # running elsewhere with MCP_TRANSPORT=streamable-http (its own
            # separate deployment, e.g. https://your-mcp.onrender.com/mcp).
            # Nothing about the tools or their behavior differs -- only how
            # the connection itself is established. If the hosted server has
            # auth enabled, send the same bearer token back on every request
            # -- if MCP_AUTH_TOKEN isn't set locally, no header is sent at
            # all, which is correct/expected when the hosted server also has
            # no auth set.
            http_headers = {"Authorization": f"Bearer {MCP_AUTH_TOKEN}"} if MCP_AUTH_TOKEN else None
            async with streamablehttp_client(MCP_SERVER_URL, headers=http_headers) as (read, write, _get_session_id):
                async with ClientSession(read, write) as session:
                    await _finish_setup(session, f"HTTP: {MCP_SERVER_URL}")
        else:
            # Default, unchanged local mode: start server.py as our own
            # private subprocess over stdio, exactly as this file has always
            # done.
            server_params = StdioServerParameters(command=sys.executable, args=["server.py"])
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await _finish_setup(session, "stdio subprocess")

    # Auto-reconnect loop -- a hosted MCP server (e.g. a free-tier Render
    # service that spins down after inactivity, or any connection that
    # simply resets after being idle) can drop or expire this session.
    # Before this loop, that would leave `mcp_session` pointed at a dead
    # object forever -- every tool call would fail with a "session
    # terminated"-style error until someone manually restarted the whole
    # process. Now, any failure that escapes `_connect_once()` (a broken
    # pipe, a closed HTTP session, anything) is caught here, logged, and
    # retried after a short pause. `_finish_setup` reassigns the
    # module-level `mcp_session` to the fresh session on every successful
    # reconnect, so tool calls just start working again on their own --
    # no manual restart needed, and only the one in-flight request at the
    # moment of the drop sees an error.
    global mcp_session
    while True:
        try:
            await _connect_once()
        except Exception as err:  # noqa: BLE001
            mcp_session = None
            _ready.clear()
            print(f"MCP connection lost/failed ({err!r}) -- reconnecting in 5s...", flush=True)
            await asyncio.sleep(5)


def start_background_loop():
    global bg_loop
    bg_loop = asyncio.new_event_loop()

    def _run_loop():
        asyncio.set_event_loop(bg_loop)
        bg_loop.run_forever()

    threading.Thread(target=_run_loop, daemon=True).start()
    asyncio.run_coroutine_threadsafe(_server_task(), bg_loop)

    # 30s wasn't enough margin for a hosted MCP server on a free tier that's
    # gone to sleep after being idle -- Render's own cold-start delay alone
    # can run 30-50s, before the connection handshake even starts. 90s gives
    # real headroom for that.
    #
    # IMPORTANT: this used to raise (and crash the whole process) if the MCP
    # server didn't answer within 90s. That's exactly backwards for a hosted
    # deployment -- a slow-to-wake or momentarily-down MCP server would take
    # the ENTIRE chatbot down with it, in a crash-restart-crash loop, and the
    # hosting platform would never even see an open port to mark the service
    # "Live". The connection loop in _server_task() already retries forever
    # in the background and reassigns mcp_session once it succeeds -- so
    # instead of crashing here, just log a warning and let the Flask app
    # start regardless. General questions and anything not needing a tool
    # still work immediately; tool calls simply degrade gracefully (via the
    # existing try/except around every tool call) until the MCP connection
    # comes up on its own, with zero manual restarts needed either way.
    if not _ready.wait(timeout=90):
        print(
            "WARNING: MCP server hasn't connected yet after 90s -- starting "
            "the web app anyway. Tool calls will fail until it connects, but "
            "it keeps retrying in the background and will pick up on its "
            "own once the MCP server responds. If MCP_SERVER_URL points at "
            "a free-tier host, it may just be waking up from sleep.",
            flush=True,
        )


# ---------------------------------------------------------
# Flask app -- routes identical to web_chatbot.py. HTML_PAGE and
# WIDGET_HTML (the /classic and /widget legacy templates, unused by the
# real "/" demo page) are imported from web_chatbot.py rather than
# duplicated here, wrapped in a try/except so those two optional routes
# simply won't be registered if that import ever fails for any reason --
# the main app (the one actually used for the demo) never depends on it.
# ---------------------------------------------------------
app = Flask(__name__, static_folder="static", static_url_path="")

try:
    from web_chatbot import HTML_PAGE, WIDGET_HTML  # noqa: E402
    _LEGACY_TEMPLATES_AVAILABLE = True
except Exception:  # noqa: BLE001
    _LEGACY_TEMPLATES_AVAILABLE = False


SESSION_COOKIE_NAME = "jamaru_session_id"


def _get_or_create_session_id():
    """Reads the browser's session_id cookie if it already has one, else
    generates a fresh one. Returns (session_id, is_new) so callers know
    whether they need to set the cookie on the response."""
    sid = request.cookies.get(SESSION_COOKIE_NAME)
    if sid:
        return sid, False
    return uuid.uuid4().hex, True


def _attach_session_cookie(resp, session_id: str):
    # 30-day cookie, HttpOnly (JS on the page never needs to read it -- it's
    # only ever sent back to this same server automatically by the browser).
    resp.set_cookie(
        SESSION_COOKIE_NAME,
        session_id,
        max_age=60 * 60 * 24 * 30,
        httponly=True,
        samesite="Lax",
    )
    return resp


@app.route("/")
def index():
    # Assign the session cookie here too (not just in /chat), so it's set
    # the moment the page loads, before the first chat message is even sent.
    session_id, is_new = _get_or_create_session_id()
    resp = make_response(app.send_static_file("jamaru.dc.html"))
    if is_new:
        _attach_session_cookie(resp, session_id)
    return resp


if _LEGACY_TEMPLATES_AVAILABLE:
    @app.route("/classic")
    def classic():
        return render_template_string(HTML_PAGE)

    @app.route("/widget")
    def widget():
        return render_template_string(WIDGET_HTML)


@app.route("/chat", methods=["POST"])
def chat():
    session_id, is_new = _get_or_create_session_id()
    # request.host_url is only valid here, on the Flask request thread --
    # captured once and passed down as a plain argument since process_message
    # (and export_pdf inside it) actually runs on the separate background
    # asyncio thread, where flask.request would raise an error. See
    # _generate_pdf's docstring for the full explanation.
    host_url = request.host_url
    user_input = (request.json or {}).get("message", "").strip()
    if not user_input:
        resp = jsonify({"reply": "", "tools": [], "chart": None, "image": None, "pdf": None})
        return _attach_session_cookie(resp, session_id) if is_new else resp
    try:
        reply, activated, chart, image, pdf = run_coro(
            process_message(user_input, session_id, host_url)
        )
    except Exception as err:  # noqa: BLE001
        logger.info("Q: %s | TOOLS: error before completion | ERROR: %s", user_input, err)
        resp = jsonify({"reply": f"Error: {err}", "tools": [], "chart": None, "image": None, "pdf": None})
        return _attach_session_cookie(resp, session_id) if is_new else resp
    # Same reliability principle as generate_image/create_diagram: don't
    # trust the model to faithfully paste the real PDF link into its reply
    # text. If it didn't, append it here so the download link always shows
    # up -- reusing the frontend's existing link-detection (linkify) means
    # no new frontend code is needed for this at all. This only touches what
    # gets shown to the browser, not what's stored in message history for
    # future LLM context.
    if pdf and pdf.get("url") and (not reply or pdf["url"] not in reply):
        reply = (reply or "").rstrip() + f"\n\nDownload your PDF: {pdf['url']}"
    tool_names = ", ".join(t.get("name", "?") for t in activated) or "none"
    logger.info("Q: %s | TOOLS: %s | A: %s | session=%s", user_input, tool_names, (reply or "")[:300], session_id)
    _log_history_turn(user_input, reply, chart, session_id)
    resp = jsonify({"reply": reply, "tools": activated, "chart": chart, "image": image, "pdf": pdf})
    return _attach_session_cookie(resp, session_id) if is_new else resp


@app.route("/history", methods=["GET"])
def history():
    session_id, is_new = _get_or_create_session_id()
    resp = jsonify({"turns": _read_history(session_id)})
    return _attach_session_cookie(resp, session_id) if is_new else resp


if __name__ == "__main__":
    start_background_loop()
    # host="127.0.0.1" (Flask's default) only accepts connections FROM this
    # same machine -- fine for local testing, but on any hosting platform
    # (Render, Railway, Fly.io, etc.) the platform's own router connects to
    # your app from outside the container, which gets silently refused by
    # 127.0.0.1. "0.0.0.0" means "accept connections on any network
    # interface", which is what every hosting platform requires.
    #
    # The port is also no longer hardcoded to 5000: hosting platforms assign
    # a port at deploy time and tell your app which one via the PORT
    # environment variable -- if the app ignores that and always binds to
    # 5000, the platform's router can't reach it and the deployment fails
    # with something like "no open port detected". Falling back to 5000 when
    # PORT isn't set keeps local `python web_chatbot_langchain.py` working
    # exactly as before.
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
