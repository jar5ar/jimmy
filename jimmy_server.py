"""
JIMMY — private single-user AI backend
Run: pip install -r requirements.txt
Set environment variables before starting:
  JIMMY_ACCESS_TOKEN=<long-random-secret>
  MODEL_BASE_URL=https://your-openai-compatible-provider/v1
  MODEL_API_KEY=<provider-key>
  MODEL_NAME=<model-name>
  ALLOWED_ORIGIN=https://your-jimmy-site.example
  BRAVE_SEARCH_API_KEY=<optional, for full web search; without it search uses Wikipedia>
  JIMMY_PERSONALITY=<optional, overrides the built-in personality>
Then: uvicorn jimmy_server:app --host 0.0.0.0 --port 7860
This backend does not include a model. It connects to an OpenAI-compatible endpoint.

Systems (all driven by the model through tools — no buttons):
- Personality: one system prompt, the same every chat (JIMMY_PERSONALITY).
- Memory: long-term memories table + save_memory / recall_memory tools,
  recalled into context by keyword match.
- Research: search_web tool. Brave Search API when BRAVE_SEARCH_API_KEY is
  set (free tier), else Wikipedia (keyless, facts only).
- Agent brain: tool loop. The model answers with {"tool": ..., "args": {...}},
  the server runs it and returns the result, up to 4 rounds.
- Tasks: tasks table + add_task / list_tasks / complete_task tools.
- Security: Bearer auth on every endpoint. Consequential actions (spending,
  publishing, sending, deleting) only happen through request_approval, and
  only after the user approves in chat ("approve") or via /api/approvals.
  Jimmy has no spend/publish/send integration yet, so an approved action of
  that kind is recorded as approved and reported honestly as not sent.
"""
import html
import json
import os
import re
import secrets
import sqlite3
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

DB_PATH = os.getenv("JIMMY_DB_PATH", "jimmy_memory.sqlite3")
ACCESS_TOKEN = os.getenv("JIMMY_ACCESS_TOKEN", "")
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "").rstrip("/")
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
# Strip a trailing slash: "https://site.example/" never matches the
# browser's Origin header "https://site.example", and CORS would fail
# silently with no useful error in the app.
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "").rstrip("/")
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY", "")

JIMMY_PERSONALITY = os.getenv("JIMMY_PERSONALITY", (
    "You are Jimmy, the user's personal AI assistant. Be direct and "
    "plain-spoken. Short answers. Do the work instead of talking about it. "
    "Remember what the user tells you to remember. When you do not know "
    "something, search the web instead of guessing — never invent facts, "
    "scores, dates, or prices. You have tools: save and recall memories, "
    "search the web, manage tasks. Use them when they help; do not narrate "
    "the tools, just use them and give the answer. For anything "
    "consequential — spending money, publishing, sending messages, deleting "
    "data — call request_approval and wait for the user to approve. Never "
    "claim you did something unless a tool confirmed it. Never reveal "
    "secrets, tokens, or API keys."
))

if not ACCESS_TOKEN or len(ACCESS_TOKEN) < 32:
    raise RuntimeError("Set JIMMY_ACCESS_TOKEN to a random secret of at least 32 characters.")
if not MODEL_BASE_URL or not MODEL_NAME:
    # App can start for health/config checks, but chat will report missing model config.
    print("WARNING: Configure MODEL_BASE_URL and MODEL_NAME before using chat.")

app = FastAPI(title="Jimmy Personal AI", version="0.2.0")
origins = [ALLOWED_ORIGIN] if ALLOWED_ORIGIN else []
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

MAX_TOOL_ROUNDS = 4


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    conversation_id: Optional[str] = None


class TaskCreate(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    due: Optional[str] = None


def db():
    # timeout=30: a slow disk or a second request writing at the same
    # moment used to fail with "database is locked" after 5s.
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db() as con:
        # WAL mode: reads don't block on a write, and a crash is less
        # likely to leave the database locked.
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("""CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY, owner TEXT NOT NULL, created_at TEXT NOT NULL
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL,
            role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id)
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'fact', text TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open', due TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS approvals (
            id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL,
            action TEXT NOT NULL, details TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL, decided_at TEXT
        )""")


init_db()


def require_auth(authorization: Optional[str]):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing bearer token")
    supplied = authorization[7:].strip()
    if not secrets.compare_digest(supplied, ACCESS_TOKEN):
        raise HTTPException(401, "Invalid access token")


def get_or_create_conversation(conversation_id: Optional[str]) -> str:
    with db() as con:
        if conversation_id:
            row = con.execute(
                "SELECT id FROM conversations WHERE id=? AND owner='owner'",
                (conversation_id,)
            ).fetchone()
            if row:
                return row["id"]
        cid = str(uuid.uuid4())
        con.execute(
            "INSERT INTO conversations(id,owner,created_at) VALUES(?,?,?)",
            (cid, "owner", datetime.now(timezone.utc).isoformat())
        )
        return cid


def save_message(cid: str, role: str, content: str):
    with db() as con:
        con.execute(
            "INSERT INTO messages(conversation_id,role,content,created_at) VALUES(?,?,?,?)",
            (cid, role, content, datetime.now(timezone.utc).isoformat())
        )


def history(cid: str):
    with db() as con:
        rows = con.execute(
            "SELECT role,content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT 24",
            (cid,)
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


# ---- memories ----

def save_memory(cid: str, kind: str, text: str) -> int:
    with db() as con:
        cur = con.execute(
            "INSERT INTO memories(conversation_id,kind,text,created_at) VALUES(?,?,?,?)",
            (cid, kind or "fact", text, datetime.now(timezone.utc).isoformat())
        )
        return cur.lastrowid


def recall_memories(query: str, limit: int = 5):
    words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2}
    with db() as con:
        rows = con.execute(
            "SELECT id,kind,text FROM memories ORDER BY id DESC LIMIT 100"
        ).fetchall()
    scored = []
    for r in rows:
        text_words = set(re.findall(r"[a-z0-9]+", r["text"].lower()))
        score = len(words & text_words)
        if score:
            scored.append((score, r))
    scored.sort(key=lambda s: (-s[0], -s[1]["id"]))
    return [{"id": r["id"], "kind": r["kind"], "text": r["text"]}
            for _, r in scored[:limit]]


def delete_memory(memory_id: int) -> bool:
    with db() as con:
        cur = con.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        return cur.rowcount > 0


# ---- tasks ----

def add_task(title: str, due: Optional[str]) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with db() as con:
        cur = con.execute(
            "INSERT INTO tasks(title,status,due,created_at,updated_at) VALUES(?,?,?,?,?)",
            (title, "open", due or "", now, now)
        )
        return cur.lastrowid


def list_tasks(status: Optional[str] = None):
    with db() as con:
        if status:
            rows = con.execute(
                "SELECT id,title,status,due FROM tasks WHERE status=? ORDER BY id",
                (status,)
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT id,title,status,due FROM tasks ORDER BY id"
            ).fetchall()
    return [{"id": r["id"], "title": r["title"], "status": r["status"],
             "due": r["due"]} for r in rows]


def open_tasks(limit: int = 5):
    return list_tasks("open")[:limit]


def complete_task(task_id: int) -> bool:
    with db() as con:
        cur = con.execute(
            "UPDATE tasks SET status='done', updated_at=? WHERE id=? AND status='open'",
            (datetime.now(timezone.utc).isoformat(), task_id)
        )
        return cur.rowcount > 0


def delete_task(task_id: int) -> bool:
    with db() as con:
        cur = con.execute("DELETE FROM tasks WHERE id=?", (task_id,))
        return cur.rowcount > 0


# ---- approvals ----

def create_approval(cid: str, action: str, details: dict) -> int:
    with db() as con:
        cur = con.execute(
            "INSERT INTO approvals(conversation_id,action,details,created_at) VALUES(?,?,?,?)",
            (cid, action, json.dumps(details),
             datetime.now(timezone.utc).isoformat())
        )
        return cur.lastrowid


def pending_approval(cid: str):
    with db() as con:
        row = con.execute(
            "SELECT id,action,details FROM approvals "
            "WHERE conversation_id=? AND status='pending' ORDER BY id DESC LIMIT 1",
            (cid,)
        ).fetchone()
    if not row:
        return None
    return {"id": row["id"], "action": row["action"],
            "details": json.loads(row["details"] or "{}")}


def decide_approval(approval_id: int, status: str) -> bool:
    with db() as con:
        cur = con.execute(
            "UPDATE approvals SET status=?, decided_at=? WHERE id=? AND status='pending'",
            (status, datetime.now(timezone.utc).isoformat(), approval_id)
        )
        return cur.rowcount > 0


def execute_approval(approval: dict) -> str:
    """Run an approved action. Only local actions really execute; anything
    needing an outside connection is recorded as approved and reported
    honestly as not sent, because no such integration exists yet."""
    action = approval["action"]
    details = approval["details"]
    if action == "delete_memory":
        ok = delete_memory(int(details.get("memory_id", -1)))
        return f"Deleted memory #{details.get('memory_id')}." if ok else "That memory was already gone."
    if action == "delete_task":
        ok = delete_task(int(details.get("task_id", -1)))
        return f"Deleted task #{details.get('task_id')}." if ok else "That task was already gone."
    if action == "clear_history":
        cid = details.get("conversation_id", "")
        with db() as con:
            cur = con.execute(
                "DELETE FROM messages WHERE conversation_id=?", (cid,))
            return f"Cleared {cur.rowcount} messages from this conversation."
    return (f"Approved and recorded. Jimmy has no connection to "
            f"{action} yet, so nothing was sent, published, or spent.")


# ---- web search (stdlib only; Brave key or keyless Wikipedia) ----

def _strip_html(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()


def _web_search(query: str, max_results: int = 5) -> str:
    if BRAVE_SEARCH_API_KEY:
        try:
            url = ("https://api.search.brave.com/res/v1/web/search?q="
                   + urllib.parse.quote(query) + f"&count={max_results}")
            req = urllib.request.Request(
                url, headers={"X-Subscription-Token": BRAVE_SEARCH_API_KEY})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
            lines = []
            for i, r in enumerate(data.get("results", [])[:max_results], 1):
                lines.append(f"{i}. {r.get('title', '')} — "
                             f"{_strip_html(r.get('description', ''))} "
                             f"(source: {r.get('url', '')})")
            return "\n".join(lines)
        except Exception:
            pass
    try:
        url = ("https://en.wikipedia.org/w/api.php?action=query&list=search"
               f"&srsearch={urllib.parse.quote(query)}&srlimit={max_results}&format=json")
        req = urllib.request.Request(url, headers={"User-Agent": "Jimmy/0.2"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        lines = []
        for i, r in enumerate(
                data.get("query", {}).get("search", [])[:max_results], 1):
            title = r.get("title", "")
            lines.append(f"{i}. {title} — {_strip_html(r.get('snippet', ''))} "
                         f"(source: https://en.wikipedia.org/wiki/"
                         f"{urllib.parse.quote(title.replace(' ', '_'))})")
        return "\n".join(lines)
    except Exception:
        return ""


# ---- tool-call parsing ----

def _parse_tool_call(text: str):
    """Return (tool_name, args) for the first JSON object with a 'tool' key,
    or None when the reply is plain text."""
    if not isinstance(text, str):
        return None
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[m.start():])
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and isinstance(obj.get("tool"), str):
            args = obj.get("args")
            return obj["tool"], args if isinstance(args, dict) else {}
    return None


def _clean_final_answer(text: str) -> str:
    """Strip a leading tool-call JSON object (and any junk before real text)
    so the user never sees raw tool JSON."""
    if not isinstance(text, str):
        return ""
    out = text.strip()
    decoder = json.JSONDecoder()
    m = re.search(r"\{", out)
    if m:
        try:
            obj, end = decoder.raw_decode(out[m.start():])
            if isinstance(obj, dict) and "tool" in obj:
                out = out[m.start() + end:].strip()
        except (json.JSONDecodeError, ValueError):
            pass
    cut = re.search(r'\{\s*"tool"\s*:', out)
    if cut:
        out = out[:cut.start()].strip()
    return out.strip(" ,\"'").strip()


# ---- agent brain: system prompt, tool runner, model calls ----

TOOLS_TEXT = """You can use these tools by replying with exactly one JSON object and nothing else:
{"tool": "save_memory", "args": {"text": "...", "kind": "fact"}}
{"tool": "recall_memory", "args": {"query": "..."}}
{"tool": "search_web", "args": {"query": "..."}}
{"tool": "add_task", "args": {"title": "...", "due": "..."}}
{"tool": "list_tasks", "args": {}}
{"tool": "complete_task", "args": {"task_id": 1}}
{"tool": "request_approval", "args": {"action": "delete_memory", "details": {"memory_id": 1}}}
Rules: call one tool per reply. After a tool result comes back, answer the
user in plain text. Never invent a tool result. For spending, publishing,
sending, or deleting anything, call request_approval first — never do those
directly."""


def build_system_prompt(cid: str, user_message: str) -> str:
    parts = [JIMMY_PERSONALITY]
    parts.append(f"Today is {datetime.now(timezone.utc).date().isoformat()} (UTC).")
    parts.append(TOOLS_TEXT)
    mems = recall_memories(user_message, limit=3)
    if mems:
        lines = "\n".join(f"- [{m['kind']}] {m['text']}" for m in mems)
        parts.append("Things you remember (use them when relevant):\n" + lines)
    tasks = open_tasks(5)
    if tasks:
        lines = "\n".join(
            f"- #{t['id']} {t['title']}"
            + (f" (due {t['due']})" if t["due"] else "")
            for t in tasks)
        parts.append("Open tasks (mention them when the user asks what is unfinished):\n" + lines)
    return "\n\n".join(parts)


def run_tool(name: str, args: dict, cid: str) -> str:
    if name == "save_memory":
        text = (args.get("text") or "").strip()
        if not text:
            return "save_memory needs a text to remember."
        mid = save_memory(cid, (args.get("kind") or "fact").strip() or "fact", text)
        return f"Saved memory #{mid}."
    if name == "recall_memory":
        query = (args.get("query") or "").strip()
        if not query:
            return "recall_memory needs a query."
        mems = recall_memories(query)
        if not mems:
            return "No matching memories."
        return "\n".join(f"{m['id']}. [{m['kind']}] {m['text']}" for m in mems)
    if name == "search_web":
        query = (args.get("query") or "").strip()
        if not query:
            return "search_web needs a query."
        res = _web_search(query)
        return res or "No results found."
    if name == "add_task":
        title = (args.get("title") or "").strip()
        if not title:
            return "add_task needs a title."
        tid = add_task(title, (args.get("due") or "").strip() or None)
        due = f" (due {args.get('due')})" if args.get("due") else ""
        return f"Task #{tid} added: {title}{due}."
    if name == "list_tasks":
        status = (args.get("status") or "").strip() or None
        tasks = list_tasks(status)
        if not tasks:
            return "No open tasks." if status == "open" else "No tasks yet."
        return "\n".join(
            f"#{t['id']} [{t['status']}] {t['title']}"
            + (f" (due {t['due']})" if t["due"] else "")
            for t in tasks)
    if name == "complete_task":
        try:
            tid = int(args.get("task_id"))
        except (TypeError, ValueError):
            return "complete_task needs a numeric task_id."
        if complete_task(tid):
            return f"Task #{tid} done."
        return f"No open task #{tid}."
    if name == "request_approval":
        action = (args.get("action") or "").strip()
        if not action:
            return "request_approval needs an action."
        raw_details = args.get("details")
        if isinstance(raw_details, dict):
            details = raw_details
        else:
            details = {"description": str(raw_details or "").strip()}
        aid = create_approval(cid, action, details)
        return (f"Approval #{aid} requested for {action}. Tell the user exactly "
                f"what you want to do and that they need to reply 'approve' or 'deny'.")
    return f"Unknown tool: {name}."


async def _call_model(messages: list) -> str:
    headers = {"Content-Type": "application/json"}
    if MODEL_API_KEY:
        headers["Authorization"] = f"Bearer {MODEL_API_KEY}"
    try:
        async with httpx.AsyncClient(timeout=90) as client:
            result = await client.post(
                f"{MODEL_BASE_URL}/chat/completions",
                headers=headers,
                json={"model": MODEL_NAME, "messages": messages, "temperature": 0.3}
            )
        result.raise_for_status()
        payload = result.json()
        reply = payload["choices"][0]["message"]["content"]
        # Some providers return content: null (tool calls, empty reply).
        # The old code turned that into the literal string "None" and
        # saved it as Jimmy's answer. Treat it as a bad response instead.
        if not isinstance(reply, str) or not reply.strip():
            raise HTTPException(502, "Could not get a valid response from the model provider.")
        return reply
    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        raise HTTPException(502, f"Model provider returned HTTP {exc.response.status_code}.")
    except (httpx.HTTPError, KeyError, IndexError, ValueError):
        raise HTTPException(502, "Could not get a valid response from the model provider.")


APPROVE_WORDS = {"approve", "approved", "yes", "ok", "okay", "go", "do it", "go ahead"}
DENY_WORDS = {"deny", "denied", "no", "nope", "cancel", "don't", "dont", "stop"}


@app.get("/")
def root():
    return {"status": "online", "app": "Jimmy", "model_configured": bool(MODEL_BASE_URL and MODEL_NAME)}


@app.get("/health")
def health():
    return {"status": "online", "app": "Jimmy", "model_configured": bool(MODEL_BASE_URL and MODEL_NAME)}


@app.post("/api/chat")
async def chat(body: ChatRequest, authorization: Optional[str] = Header(default=None)):
    require_auth(authorization)
    if not MODEL_BASE_URL or not MODEL_NAME:
        raise HTTPException(503, "Jimmy's model endpoint is not configured yet.")
    cid = get_or_create_conversation(body.conversation_id)

    # In-chat approval: a one-word approve/deny answers a pending request.
    pending = pending_approval(cid)
    word = body.message.strip().lower()
    if pending and word in APPROVE_WORDS:
        decide_approval(pending["id"], "approved")
        result = execute_approval(pending)
        reply = f"Approved. {result}"
        save_message(cid, "user", body.message)
        save_message(cid, "assistant", reply)
        return {"conversation_id": cid, "reply": reply}
    if pending and word in DENY_WORDS:
        decide_approval(pending["id"], "denied")
        reply = "Cancelled — nothing was done."
        save_message(cid, "user", body.message)
        save_message(cid, "assistant", reply)
        return {"conversation_id": cid, "reply": reply}

    save_message(cid, "user", body.message)
    messages = [
        {"role": "system", "content": build_system_prompt(cid, body.message)}
    ] + history(cid)

    # Tool loop: the model calls tools, the server runs them, results go back.
    final = ""
    last_tool_result = ""
    for _ in range(MAX_TOOL_ROUNDS):
        reply = await _call_model(messages)
        tool = _parse_tool_call(reply)
        if tool is None:
            final = _clean_final_answer(reply)
            break
        name, args = tool
        last_tool_result = run_tool(name, args, cid)
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": (
            f"Tool {name} returned:\n{last_tool_result}\n"
            "Continue: answer the user in plain text now, or call another "
            "tool if you still need one.")})
    else:
        final = _clean_final_answer(last_tool_result) or (
            "I ran the tools but didn't get a clean answer back — "
            "try asking again.")
    if not final:
        final = "I didn't get a usable answer from the model — try asking again."
    save_message(cid, "assistant", final)
    return {"conversation_id": cid, "reply": final}


@app.get("/api/history/{conversation_id}")
def get_history(conversation_id: str, authorization: Optional[str] = Header(default=None)):
    require_auth(authorization)
    with db() as con:
        exists = con.execute(
            "SELECT id FROM conversations WHERE id=? AND owner='owner'", (conversation_id,)
        ).fetchone()
    if not exists:
        raise HTTPException(404, "Conversation not found")
    return {"conversation_id": conversation_id, "messages": history(conversation_id)}


@app.get("/api/tasks")
def get_tasks(authorization: Optional[str] = Header(default=None)):
    require_auth(authorization)
    return {"tasks": list_tasks()}


@app.post("/api/tasks")
def create_task(body: TaskCreate, authorization: Optional[str] = Header(default=None)):
    require_auth(authorization)
    tid = add_task(body.title, body.due)
    return {"id": tid, "title": body.title, "status": "open", "due": body.due or ""}


@app.post("/api/tasks/{task_id}/complete")
def complete_task_endpoint(task_id: int, authorization: Optional[str] = Header(default=None)):
    require_auth(authorization)
    if not complete_task(task_id):
        raise HTTPException(404, "No open task with that id")
    return {"id": task_id, "status": "done"}


@app.get("/api/memories")
def get_memories(authorization: Optional[str] = Header(default=None)):
    require_auth(authorization)
    with db() as con:
        rows = con.execute(
            "SELECT id,kind,text,created_at FROM memories ORDER BY id DESC LIMIT 100"
        ).fetchall()
    return {"memories": [
        {"id": r["id"], "kind": r["kind"], "text": r["text"],
         "created_at": r["created_at"]} for r in rows]}


@app.delete("/api/memories/{memory_id}")
def forget_memory(memory_id: int, authorization: Optional[str] = Header(default=None)):
    require_auth(authorization)
    if not delete_memory(memory_id):
        raise HTTPException(404, "No memory with that id")
    return {"id": memory_id, "deleted": True}


@app.get("/api/approvals")
def get_approvals(authorization: Optional[str] = Header(default=None)):
    require_auth(authorization)
    with db() as con:
        rows = con.execute(
            "SELECT id,conversation_id,action,details,status,created_at "
            "FROM approvals WHERE status='pending' ORDER BY id"
        ).fetchall()
    return {"approvals": [
        {"id": r["id"], "conversation_id": r["conversation_id"],
         "action": r["action"], "details": json.loads(r["details"] or "{}"),
         "status": r["status"], "created_at": r["created_at"]} for r in rows]}


@app.post("/api/approvals/{approval_id}/approve")
def approve_endpoint(approval_id: int, authorization: Optional[str] = Header(default=None)):
    require_auth(authorization)
    with db() as con:
        row = con.execute(
            "SELECT id,conversation_id,action,details FROM approvals "
            "WHERE id=? AND status='pending'", (approval_id,)
        ).fetchone()
    if not row:
        raise HTTPException(404, "No pending approval with that id")
    approval = {"id": row["id"], "action": row["action"],
                "details": json.loads(row["details"] or "{}")}
    decide_approval(approval_id, "approved")
    return {"id": approval_id, "status": "approved",
            "result": execute_approval(approval)}


@app.post("/api/approvals/{approval_id}/deny")
def deny_endpoint(approval_id: int, authorization: Optional[str] = Header(default=None)):
    require_auth(authorization)
    if not decide_approval(approval_id, "denied"):
        raise HTTPException(404, "No pending approval with that id")
    return {"id": approval_id, "status": "denied"}
