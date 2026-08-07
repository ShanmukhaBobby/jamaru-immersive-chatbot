#!/usr/bin/env python3
"""
web_chatbot_immersive.py
---------------------------------------------------------
Another SEPARATE chatbot variant, same safety pattern as
web_chatbot_stream.py before it: nothing about web_chatbot_langchain.py
(the stable one) or web_chatbot_stream.py (the previous experimental one)
is touched or overwritten. This file is a copy of web_chatbot_stream.py's
backend logic, only pointed at the new "Jamaru immersive" frontend
(static/jamaru_immersive_preview.html) instead of static/jamaru_stream.html,
running on its OWN port with its OWN history log file, so all three
chatbots can run side by side with zero interference:
  - web_chatbot_langchain.py  -> the original, stable, demo-proven one
  - web_chatbot_stream.py     -> the streaming "Steps taken" variant
  - web_chatbot_immersive.py  -> THIS one: the dark, motherboard-reveal,
                                  liquid-logo design, now wired to the
                                  real backend instead of the frontend's
                                  earlier simulated/fake demo timers.

Run:
    python web_chatbot_immersive.py
Then open http://127.0.0.1:5070 (or whatever IMMERSIVE_PORT is set to).

Uses the exact same .env keys already set up for the other two chatbots
(GROQ_API_KEY, GEMINI_API_KEY/GOOGLE_API_KEY, MCP_SERVER_URL,
MCP_AUTH_TOKEN) -- no new keys needed.
---------------------------------------------------------
"""

import datetime
import json
import os
import re
import threading
import uuid

from flask import Flask, Response, jsonify, make_response, request

# Reuse, don't reinvent -- same proven functions web_chatbot_stream.py
# already reuses from the stable chatbot (call_llm's full Groq retry /
# Gemini fallback logic, trim_history, SYSTEM_PROMPT, the MCP connection
# setup). Importing this module does NOT start its own Flask server (its
# app.run() is guarded by `if __name__ == "__main__":`), so this is safe.
import web_chatbot_langchain as base

# ---------------------------------------------------------
# Per-tool human-readable "why" descriptions, shown live while loading --
# identical to web_chatbot_stream.py's version.
# ---------------------------------------------------------
# Short, human-readable label for each tool -- used in both the "matched
# tool" and "activating" reasoning lines below, so the two stay consistent
# without repeating a full description every time.
_TOOL_LABEL = {
    "get_coordinates": "location lookup",
    "get_weather": "live weather data",
    "get_weather_forecast": "forecast data",
    "get_weather_history": "historical weather",
    "get_exchange_rate": "currency exchange",
    "get_exchange_rate_history": "currency history",
    "get_random_joke": "joke generator",
    "get_random_fact": "fact generator",
    "get_nasa_apod": "NASA daily photo",
    "get_wikipedia_summary": "Wikipedia lookup",
    "get_github_user": "GitHub lookup",
    "list_capabilities": "capability list",
    "generate_image": "image generation",
    "create_diagram": "diagram generation",
    "search_youtube": "YouTube search",
    "search_arxiv": "Arxiv search",
    "show_chart": "chart builder",
    "export_pdf": "PDF export",
    "search_the_web": "web search",
}


def _tool_request_line(name: str, args: dict) -> str:
    """One short line (aim for ~4-7 words) naming the SPECIFIC thing the
    user's question was asking for, using the actual arguments the model
    already extracted -- e.g. the place name, the currency codes, the
    search query. This is what shows up FIRST in the Reasoning steps, so
    it should read like "here's what I picked out of your question", not
    a generic "using a tool" placeholder."""
    if name == "get_coordinates":
        return f"Need location of {args.get('place', 'that place')}"
    if name == "get_weather":
        return "Need current weather right now"
    if name == "get_weather_forecast":
        return f"Need {args.get('days', 'a few')}-day forecast"
    if name == "get_weather_history":
        return "Need past weather records"
    if name == "get_exchange_rate":
        return f"Need {args.get('from_currency', '?')}→{args.get('to_currency', '?')} rate"
    if name == "get_exchange_rate_history":
        return "Need exchange rate history"
    if name == "get_random_joke":
        return "Asked for a joke"
    if name == "get_random_fact":
        return "Asked for a random fact"
    if name == "get_nasa_apod":
        return "Asked for NASA's daily photo"
    if name == "get_wikipedia_summary":
        return f"Need summary on {args.get('topic', 'that topic')}"
    if name == "get_github_user":
        return f"Need GitHub profile {args.get('username', '')}"
    if name == "list_capabilities":
        return "Asked what tools exist"
    if name == "generate_image":
        return f"Need image: {args.get('prompt', args.get('description', 'requested scene'))}"[:40]
    if name == "create_diagram":
        return "Need a diagram made"
    if name == "search_youtube":
        return f"Need YouTube results: {args.get('query', '')}"[:40]
    if name == "search_arxiv":
        return f"Need papers on {args.get('query', '')}"[:40]
    if name == "show_chart":
        return "Asked to visualize this data"
    if name == "export_pdf":
        return "Asked to export as PDF"
    if name == "search_the_web":
        return f"Need current info: {args.get('query', '')}"[:40]
    return f"Need data from {name}"


def _reasoning_lines(name: str, args: dict) -> list:
    """Three short lines shown in the Reasoning dropdown for one tool call:
    (1) what was detected in the question, (2) which tool matched it, (3)
    that it's now being activated. Deliberately short (a handful of words
    each) and specific to THIS question -- not a rundown of every tool the
    bot has available."""
    label = _TOOL_LABEL.get(name, name.replace("_", " "))
    return [
        _tool_request_line(name, args),
        f"Matched tool: {label}",
        f"Activating {label} now",
    ]


# ---------------------------------------------------------
# Which board "chip" lights up for a given real tool name -- purely
# cosmetic, read by the frontend to drive the trace-to-chip activation on
# the motherboard visual. Kept server-side so the mapping lives in one
# place and both this file and the frontend agree on tool -> chip.
# ---------------------------------------------------------
_TOOL_CHIP_MAP = {
    # RAM -- location/weather (fast, frequently re-read data)
    "get_coordinates": "ram1",
    "get_weather": "ram1",
    "get_weather_forecast": "ram1",
    "get_weather_history": "ram1",
    # RAM2 -- currency
    "get_exchange_rate": "ram2",
    "get_exchange_rate_history": "ram2",
    # ROM -- reference/research lookups
    "get_wikipedia_summary": "rom",
    "search_arxiv": "rom",
    "list_capabilities": "rom",
    # NET -- anything that reaches out across the actual network
    "search_the_web": "net",
    "search_youtube": "net",
    "get_github_user": "net",
    # GPU -- anything that renders/generates visual output
    "generate_image": "gpu",
    "create_diagram": "gpu",
    "show_chart": "gpu",
    # SSD -- persisted output
    "export_pdf": "ssd",
    # NPU -- the odds-and-ends "smart extras" tools
    "get_nasa_apod": "npu",
    "get_random_fact": "npu",
    "get_random_joke": "npu",
}


def _chip_for_tool(name: str) -> str:
    return _TOOL_CHIP_MAP.get(name, "rom")


# ---------------------------------------------------------
# Independent, in-memory per-session chat history -- fully separate from
# both other chatbots' in-memory dicts.
# ---------------------------------------------------------
_sessions: dict[str, list] = {}
_sessions_lock = threading.Lock()


def _get_session_messages(session_id: str) -> list:
    with _sessions_lock:
        if session_id not in _sessions:
            _sessions[session_id] = [base.SYSTEM_PROMPT]
        return _sessions[session_id]


def _set_session_messages(session_id: str, msgs: list) -> None:
    with _sessions_lock:
        _sessions[session_id] = msgs


# ---------------------------------------------------------
# Persistent chat history -- its OWN file, separate from both
# chat_history.jsonl (the main chatbot) and stream_chat_history.jsonl (the
# streaming variant), so this design's conversations never mix with
# either of theirs.
# ---------------------------------------------------------
HISTORY_LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "immersive_chat_history.jsonl"
)


def _log_history_turn(question: str, answer: str, session_id: str, chart=None, image=None, pdf=None, steps=None) -> None:
    try:
        entry = {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "session_id": session_id,
            "question": question,
            "answer": answer or "",
            "chart": chart,
            "image": image,
            "pdf": pdf,
            # Previously the Reasoning trace only existed in the live SSE
            # stream, never saved -- so reloading the page or reopening an
            # older conversation from the sidebar showed the final answer
            # with the "Reasoning" dropdown simply gone, even though it was
            # there moments earlier. Persisting the same step lines the
            # frontend already displayed live means history reload can
            # rebuild an identical dropdown instead of losing it.
            "steps": steps or [],
        }
        with open(HISTORY_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _read_history(session_id: str, limit: int = 200) -> list:
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
    return entries[-limit:]


def _list_conversations(owner_session_id: str, limit: int = 100) -> list:
    if not os.path.exists(HISTORY_LOG_PATH):
        return []
    convos = {}
    order = []
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
                sid = entry.get("session_id")
                if not sid or sid != owner_session_id:
                    continue
                if sid not in convos:
                    convos[sid] = {
                        "session_id": sid,
                        "preview": (entry.get("question") or "")[:60],
                        "last_ts": entry.get("ts", ""),
                        "count": 0,
                    }
                    order.append(sid)
                convos[sid]["last_ts"] = entry.get("ts", convos[sid]["last_ts"])
                convos[sid]["count"] += 1
    except OSError:
        return []
    result = [convos[sid] for sid in order]
    result.sort(key=lambda c: c["last_ts"], reverse=True)
    return result[:limit]


def _sse_line(event_type: str, payload) -> str:
    return json.dumps({"type": event_type, "data": payload}) + "\n"


def _stream_reply(user_input: str, session_id: str, host_url: str):
    """Same shape as web_chatbot_stream.py's _stream_reply -- status/tool/
    tool_done/final events -- with one addition: each "tool" event now
    also carries a "chip" field so the frontend's board visual can light
    up the real sector actually in use, instead of guessing from the
    user's wording."""
    # Mirrors exactly what the frontend's own `steps` array accumulates from
    # this same SSE stream (see streamReply() in the static HTML) -- kept
    # here too so the finished turn can be persisted and an identical
    # Reasoning dropdown can be rebuilt later on history reload, instead of
    # the trace only ever existing transiently in the live page.
    steps = ["Connecting to your tools...", "Reading your question"]
    yield _sse_line("status", "Reading your question")

    messages = _get_session_messages(session_id)
    messages.append({"role": "user", "content": user_input})

    data = base.call_llm(messages)
    choice = data["choices"][0]

    chart_payload = None
    image_payload = None
    pdf_payload = None
    tool_errors = []
    tool_link_results = []

    # Previously the Reasoning dropdown only ever showed steps when a tool
    # was actually used -- a generic question answered straight from the
    # model's own knowledge produced no reasoning trace at all, which made
    # it look like the model skipped thinking about it rather than having
    # deliberately decided no tool was needed. Make that decision visible
    # too, symmetric with the tool-use case, but only when no tool is ever
    # called this turn (checked once here, before the loop, not on every
    # follow-up round after tools already ran).
    if not choice["message"].get("tool_calls"):
        yield _sse_line("status", "No tool needed -- answering from general knowledge")
        steps.append("No tool needed -- answering from general knowledge")

    while choice["message"].get("tool_calls"):
        messages.append(choice["message"])

        for tool_call in choice["message"]["tool_calls"]:
            fn = tool_call["function"]
            args = json.loads(fn.get("arguments") or "{}")
            # Three short, question-specific lines instead of one generic
            # "using X tool" line -- what was detected, which tool matched
            # it, and that it's now activating. See _reasoning_lines above.
            request_line, matched_line, activate_line = _reasoning_lines(fn["name"], args)
            yield _sse_line("status", request_line)
            steps.append(request_line)
            yield _sse_line("status", matched_line)
            steps.append(matched_line)
            yield _sse_line("tool", {
                "name": fn["name"],
                "reason": activate_line,
                "chip": _chip_for_tool(fn["name"]),
            })
            steps.append(activate_line)

            errors_before = len(tool_errors)
            if fn["name"] == "show_chart":
                chart_payload = args
                result_text = "Chart displayed to the user."
            elif fn["name"] == "export_pdf":
                try:
                    pdf_url = base._generate_pdf(
                        args.get("title", "Document"), args.get("content", ""), host_url
                    )
                    pdf_payload = {"url": pdf_url, "title": args.get("title", "Document")}
                    result_text = f"PDF generated: {pdf_url}"
                except Exception as err:  # noqa: BLE001
                    result_text = f"Error generating PDF: {err}"
                    tool_errors.append((fn["name"], str(err)))
            elif fn["name"] == "search_the_web":
                try:
                    result_text = base.call_groq_web_search(args.get("query", ""))
                except Exception as err:  # noqa: BLE001
                    result_text = f"Error calling tool: {err}"
                    tool_errors.append((fn["name"], str(err)))
            else:
                try:
                    result = base.run_coro(base.mcp_session.call_tool(fn["name"], args))
                    result_text = "\n".join(
                        getattr(c, "text", str(c)) for c in result.content
                    )
                    if getattr(result, "isError", False):
                        tool_errors.append((fn["name"], result_text))
                    elif fn["name"] in ("generate_image", "create_diagram"):
                        url_match = re.search(
                            r"https?://(?:image\.pollinations\.ai|kroki\.io)/\S+",
                            result_text,
                        )
                        if url_match:
                            image_payload = {
                                "url": url_match.group(0),
                                "kind": "image" if fn["name"] == "generate_image" else "diagram",
                            }
                    elif fn["name"] == "get_nasa_apod":
                        # NASA's APOD "Media URL" is sometimes a real image
                        # and sometimes a YouTube video link (roughly one day
                        # in several is a video feature instead of a photo).
                        # Only show it as an inline image when it actually IS
                        # one -- a YouTube link stays as a normal clickable
                        # link below, since it can't be embedded as an <img>.
                        media_match = re.search(r"Media URL:\s*(\S+)", result_text)
                        if media_match:
                            media_url = media_match.group(1)
                            is_video_link = "youtube.com" in media_url or "youtu.be" in media_url
                            if not is_video_link and re.search(r"\.(jpe?g|png|gif|webp)(\?\S*)?$", media_url, re.IGNORECASE):
                                image_payload = {"url": media_url, "kind": "image"}
                            else:
                                tool_link_results.append((fn["name"], result_text, [media_url]))
                    else:
                        result_urls = re.findall(r"https?://\S+", result_text)
                        if result_urls:
                            tool_link_results.append((fn["name"], result_text, result_urls))
                except Exception as err:  # noqa: BLE001
                    result_text = f"Error calling tool: {err}"
                    tool_errors.append((fn["name"], str(err)))

            # Only THIS call's error (if any) -- tool_errors is shared across
            # every tool call in the whole reply, so comparing lengths
            # before/after isolates whether this specific call is the one
            # that failed, rather than reporting a stale error from an
            # earlier, unrelated tool call.
            this_call_error = tool_errors[-1][1] if len(tool_errors) > errors_before else None
            yield _sse_line("tool_done", {"name": fn["name"], "error": this_call_error})
            if this_call_error:
                steps.append(f"⚠ {fn['name']} failed: {this_call_error}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result_text,
                }
            )

        yield _sse_line("status", "Preparing final answer")
        steps.append("Preparing final answer")
        data = base.call_llm(messages)
        choice = data["choices"][0]

    messages.append(choice["message"])
    messages = base.trim_history(messages)
    _set_session_messages(session_id, messages)

    final_content = choice["message"].get("content") or ""
    # Pull off the hidden [[APPROACH: ...]] method tag (see SYSTEM_PROMPT) --
    # present only when the model actually solved something via a specific
    # method/formula, whether or not a tool was involved. Surfaced as one
    # more Reasoning step, right before the answer, without disturbing any
    # of the existing tool-detection steps above.
    final_content, approach_note = base.extract_approach_tag(final_content)
    if approach_note:
        yield _sse_line("status", f"Approach: {approach_note}")
        steps.append(f"Approach: {approach_note}")

    # Only bolt a tool-failure note onto the VISIBLE reply when the model
    # genuinely couldn't produce a real answer despite the failure(s) --
    # if it fell back successfully (e.g. Wikipedia failed but a web search
    # still answered the question), that failure is expected/handled and
    # belongs in the "Reasoning" steps trace only (see the tool_done
    # event's "error" field above), not bolted onto the chat reply as
    # noise the user didn't ask to see.
    if tool_errors and len(final_content.strip()) < 20:
        missing = [
            (name, msg) for name, msg in tool_errors
            if name.replace("_", " ") not in final_content.lower() and name not in final_content
        ]
        if missing:
            note_lines = [f"({name} failed: {msg})" for name, msg in missing]
            final_content = (final_content + "\n\n" + "\n".join(note_lines)).strip()
    for name, text, urls in tool_link_results:
        if not any(u.rstrip(").,!?") in final_content for u in urls):
            final_content = (final_content + "\n\n" + text).strip()

    _log_history_turn(
        user_input, final_content, session_id,
        chart=chart_payload, image=image_payload, pdf=pdf_payload,
        # Only worth persisting/showing the dropdown if there's more than
        # the two baseline lines every turn always has -- matches the
        # frontend's own `steps.length > 1` check for whether to render a
        # Reasoning toggle at all.
        steps=steps if len(steps) > 2 else [],
    )

    yield _sse_line("final", {
        "content": final_content,
        "chart": chart_payload,
        "image": image_payload,
        "pdf": pdf_payload,
    })


# ---------------------------------------------------------
# Flask app -- its own independent process/port, separate from BOTH other
# chatbots' apps entirely.
# ---------------------------------------------------------
app = Flask(__name__, static_folder="static", static_url_path="")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


SESSION_COOKIE_NAME = "jamaru_immersive_session"


def _get_or_create_session_id():
    sid = request.cookies.get(SESSION_COOKIE_NAME)
    is_new = False
    if not sid:
        sid = uuid.uuid4().hex
        is_new = True
    return sid, is_new


def _attach_session_cookie(resp, session_id: str):
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
    session_id, is_new = _get_or_create_session_id()
    with open(os.path.join(STATIC_DIR, "jamaru_immersive_preview.html"), "r", encoding="utf-8") as f:
        resp = make_response(f.read())
    return _attach_session_cookie(resp, session_id) if is_new else resp


@app.route("/history", methods=["GET"])
def history():
    session_id, is_new = _get_or_create_session_id()
    resp = jsonify({"turns": _read_history(session_id), "session_id": session_id})
    return _attach_session_cookie(resp, session_id) if is_new else resp


@app.route("/new-chat", methods=["POST"])
def new_chat():
    new_session_id = uuid.uuid4().hex
    resp = jsonify({"ok": True, "session_id": new_session_id})
    return _attach_session_cookie(resp, new_session_id)


@app.route("/conversations", methods=["GET"])
def conversations():
    session_id, is_new = _get_or_create_session_id()
    resp = jsonify({"conversations": _list_conversations(session_id)})
    return _attach_session_cookie(resp, session_id) if is_new else resp


@app.route("/delete-conversation", methods=["POST"])
def delete_conversation():
    target_id = (request.json or {}).get("session_id", "").strip()
    if not target_id:
        return jsonify({"ok": False, "error": "session_id required"}), 400
    try:
        if os.path.exists(HISTORY_LOG_PATH):
            with open(HISTORY_LOG_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
            kept = []
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError:
                    kept.append(line)
                    continue
                if entry.get("session_id") != target_id:
                    kept.append(line)
            with open(HISTORY_LOG_PATH, "w", encoding="utf-8") as f:
                f.writelines(kept)
    except OSError as err:
        return jsonify({"ok": False, "error": str(err)}), 500
    with _sessions_lock:
        _sessions.pop(target_id, None)
    return jsonify({"ok": True})


@app.route("/about", methods=["GET"])
def about():
    return jsonify({
        "primary_model": f"Groq — {base.GROQ_MODEL}",
        "fallback_model": (
            f"Gemini — {base.GEMINI_MODEL} (used automatically if Groq rate-limits or errors)"
            if base._GEMINI_AVAILABLE else "Not configured"
        ),
        "framework": "LangChain (model orchestration + tool binding)",
        "protocol": "MCP (Model Context Protocol) over Streamable HTTP",
        "tool_count": len(base.groq_tools) if base.groq_tools else 0,
        "mcp_server": base.MCP_SERVER_URL or "local subprocess (server.py)",
    })


@app.route("/open-conversation", methods=["POST"])
def open_conversation():
    target_id = (request.json or {}).get("session_id", "").strip()
    if not target_id:
        return jsonify({"ok": False, "error": "session_id required"}), 400

    rebuilt = [base.SYSTEM_PROMPT]
    for turn in _read_history(target_id):
        rebuilt.append({"role": "user", "content": turn.get("question", "")})
        if turn.get("answer"):
            rebuilt.append({"role": "assistant", "content": turn.get("answer", "")})
    _set_session_messages(target_id, rebuilt)

    resp = jsonify({"ok": True})
    return _attach_session_cookie(resp, target_id)


@app.route("/chat", methods=["POST"])
def chat():
    session_id, is_new = _get_or_create_session_id()
    host_url = request.host_url
    user_input = (request.json or {}).get("message", "").strip()

    if not user_input:
        def _empty():
            yield _sse_line("final", {"content": "", "chart": None, "image": None, "pdf": None})
        resp = Response(_empty(), mimetype="application/x-ndjson")
        if is_new:
            resp = _attach_session_cookie(resp, session_id)
        return resp

    def _generate():
        try:
            yield from _stream_reply(user_input, session_id, host_url)
        except GeneratorExit:
            # The client disconnected/aborted mid-stream (the pause/stop
            # button) -- nothing to clean up, just let the generator end.
            raise
        except Exception as err:  # noqa: BLE001
            # Never surface raw Python exception text to the user -- log the
            # real error server-side (visible in Render's logs) for
            # debugging, and show a plain, non-technical message in the
            # chat itself. This is a catch-all for anything NOT already
            # handled inside _stream_reply's own per-tool try/excepts (e.g.
            # a genuinely unexpected bug), so no matter what breaks, the
            # user only ever sees a clean sentence, never a stack trace or
            # "'NoneType' object has no attribute ..."-style internals.
            print(f"Unhandled error in /chat stream: {err!r}", flush=True)
            yield _sse_line("final", {
                "content": (
                    "Something went wrong on my end while working on that -- "
                    "please try asking again in a moment."
                ),
                "chart": None, "image": None, "pdf": None,
            })

    resp = Response(_generate(), mimetype="application/x-ndjson")
    resp.headers["X-Accel-Buffering"] = "no"
    resp.headers["Cache-Control"] = "no-cache"
    if is_new:
        resp = _attach_session_cookie(resp, session_id)
    return resp


if __name__ == "__main__":
    base.start_background_loop()
    # Render (and most hosting platforms) assign the port at deploy time
    # via the standard PORT env var and expect the app to bind to it --
    # ignoring that and always using IMMERSIVE_PORT's default would make
    # the deploy fail with "no open port detected" since the platform's
    # router can't reach the port the app actually opened. Checking PORT
    # first means it works correctly once hosted, while IMMERSIVE_PORT
    # still lets you pick a custom port for local runs, and 5070 remains
    # the fallback for local runs when neither is set.
    port = int(os.environ.get("PORT") or os.environ.get("IMMERSIVE_PORT") or 5070)
    print(f"web_chatbot_immersive running at http://0.0.0.0:{port}", flush=True)
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)
