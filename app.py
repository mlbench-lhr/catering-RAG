import os
import json
from datetime import datetime
from bson import ObjectId
import dotenv
import pymongo
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import gradio as gr

dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "catering")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.1")

client_openai = OpenAI(api_key=OPENAI_API_KEY)

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def json_safe(o):
    if isinstance(o, dict):
        return {k: json_safe(v) for k, v in o.items()}
    if isinstance(o, list):
        return [json_safe(i) for i in o]
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, ObjectId):
        return str(o)
    return o

# ------------------------------------------------------------
# MONGO + EMBEDDINGS
# ------------------------------------------------------------
mongo = pymongo.MongoClient(MONGO_URI)
db = mongo[MONGO_DB]

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def load_documents():
    collections = [
        "menuitems", "deals", "cateringrequests", "orders",
        "notifications", "servicegroups", "users"
    ]
    docs = []
    for coll in collections:
        if coll not in db.list_collection_names():
            continue
        for d in db[coll].find({}):
            c = json_safe(d)
            t = " ".join(str(v) for v in c.values())
            docs.append({"collection": coll, "id": c["_id"], "text": t, "raw": c})
    return docs

DOCUMENTS = load_documents()
corpus_embeddings = model.encode([d["text"] for d in DOCUMENTS], normalize_embeddings=True)

# ------------------------------------------------------------
# LANGUAGE DETECTION
# ------------------------------------------------------------
def detect_lang(text):
    try:
        out = client_openai.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "Return only 'en' or 'de'."},
                {"role": "user", "content": text}
            ],
            max_completion_tokens=2,
            temperature=0
        ).choices[0].message.content.strip().lower()
        return out if out in ["en", "de"] else "en"
    except:
        return "en"

# ------------------------------------------------------------
# QUOTE CALCULATOR
# ------------------------------------------------------------
def compute_quote(doc):
    guests = None
    price_pp = None

    for k in ["guestCount", "guests", "totalGuests"]:
        if k in doc and isinstance(doc[k], (int, float)):
            guests = doc[k]

    for k in ["pricePerPerson", "budgetPerPerson", "pp"]:
        if k in doc and isinstance(doc[k], (int, float)):
            price_pp = doc[k]

    if price_pp is None and "price" in doc:
        try: price_pp = float(str(doc["price"]).replace(",", "").replace("$", ""))
        except: pass

    if guests is None and price_pp is None:
        return None

    subtotal = (guests or 1)*(price_pp or 0)
    service_fee = subtotal*0.10
    tax = subtotal*0.05
    final = subtotal + service_fee + tax

    return {
        "guests": guests,
        "price_per_person": price_pp,
        "subtotal": round(subtotal,2),
        "service_fee": round(service_fee,2),
        "tax": round(tax,2),
        "total": round(final,2)
    }

# ------------------------------------------------------------
# RETRIEVAL
# ------------------------------------------------------------
def retrieve(query, top_k):
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    sims = np.dot(corpus_embeddings, q_emb)
    idx = np.argsort(sims)[::-1][:top_k]
    return [{
        "collection": DOCUMENTS[i]["collection"],
        "id": DOCUMENTS[i]["id"],
        "score": float(sims[i]),
        "doc": DOCUMENTS[i]["raw"],
    } for i in idx]

# ------------------------------------------------------------
# ANSWER GENERATION (MEMORY + TYPING SIMULATION)
# ------------------------------------------------------------
def generate_answer(query, lang, retrieved, history):
    # RAG + Quotes
    context = "NO_DB_CONTEXT"
    qc = ""
    if retrieved:
        ctx_rows = []
        for r in retrieved:
            ctx_rows.append(f"[{r['collection']}] {json.dumps(json_safe(r['doc']), ensure_ascii=False)[:1500]}")
            q = compute_quote(r["doc"])
            if q: qc += f"\n\nQUOTE_ESTIMATE: {json.dumps(q)}"
        context = "\n\n".join(ctx_rows) + qc

    # Build message history
    messages = [
        {
            "role": "system",
            "content": f"You are a multilingual catering assistant. Always respond in {lang}. You have full conversation memory."
        }
    ]

    # Previous conversation
    for h_user, h_bot in history:
        messages.append({"role": "user", "content": h_user})
        messages.append({"role": "assistant", "content": h_bot})

    # New message
    messages.append({
        "role": "user",
        "content": f"User query:\n{query}\n\nContext:\n{context}\n\nAnswer now in {lang}."
    })

    # STREAMING / TYPING SIMULATION
    stream = client_openai.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        stream=True,
        temperature=0.2,
        max_completion_tokens=700,
    )

    full_answer = ""
    for chunk in stream:
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta.content
            if delta:
                full_answer += delta
                yield full_answer  # send partial answer to UI "typing"
    yield full_answer


# ------------------------------------------------------------
# CHAT LOOP
# ------------------------------------------------------------
def chat_fn(query, history, top_k):
    lang = detect_lang(query)
    retrieved = retrieve(query, top_k)

    # Stream partial results
    answer_stream = generate_answer(query, lang, retrieved, history)

    final = ""
    for partial in answer_stream:
        final = partial
        yield partial, history  # update typing output

    # Save final step to memory
    history.append((query, final))
    yield final, history


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
with gr.Blocks() as app:
    gr.Markdown("# Catering RAG Agent — GPT-5.1 • Quotes • Memory • Typing…")

    history_state = gr.State([])
    topk = gr.Slider(1, 12, value=5, label="Top-K Retrieval")

    chat = gr.Chatbot(label="Chat with Catering Agent", height=450)

    inp = gr.Textbox(label="Ask something…", placeholder="Type your message", lines=2)
    send = gr.Button("Send")
    clear = gr.Button("Reset Conversation")

    def wrapper(message, history, topk):
        chat_history = [(u, b) for u, b in history]
        stream = chat_fn(message, chat_history, topk)
        for partial, hstate in stream:
            chat_pairs = [(u, b) for u, b in hstate]
            chat_pairs.append((message, partial))
            yield chat_pairs, hstate

    send.click(
        wrapper,
        inputs=[inp, history_state, topk],
        outputs=[chat, history_state],
        concurrency_limit=1
    )

    clear.click(lambda: ([], []), None, [chat, history_state])

app.launch(server_name="0.0.0.0", server_port=7860)
