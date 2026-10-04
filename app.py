"""Alfred WhatsApp pilot: Twilio webhook + GLM-5.3-Flash on Nebius Token Factory.

Flow: consent -> 5-question onboarding -> free-text dinner requests -> 3 options
(cook / order / pick up) -> reply 1/2/3 for details, "other" for new options.
Safety: allergies and diet are hard filters in code; the model only parses intent.
"""
import json
import math
import os
import re
import time

import psycopg
from fastapi import FastAPI, HTTPException, Request, Response
from openai import OpenAI
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from twilio.request_validator import RequestValidator
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

# ---------- config ----------
DATABASE_URL = os.environ["DATABASE_URL"]
PUBLIC_URL = os.getenv("PUBLIC_URL", "").rstrip("/")          # e.g. https://alfred.up.railway.app
VALIDATE_TWILIO = os.getenv("VALIDATE_TWILIO", "true").lower() == "true"
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "")  # e.g. whatsapp:+14155238886
TEAM_ALERT_NUMBER = os.getenv("TEAM_ALERT_NUMBER", "")        # e.g. whatsapp:+316...
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", "")
NEBIUS_BASE_URL = os.getenv("NEBIUS_BASE_URL", "")            # copy from Token Factory console
NEBIUS_MODEL = os.getenv("NEBIUS_MODEL", "")                  # exact GLM-5.3-Flash model id
PRIVACY_URL = os.getenv("PRIVACY_URL", "[privacy link]")
SEED_PATH = os.getenv("SEED_PATH", os.path.join(os.path.dirname(__file__), "seed.json"))

llm = OpenAI(api_key=NEBIUS_API_KEY, base_url=NEBIUS_BASE_URL) if NEBIUS_API_KEY and NEBIUS_BASE_URL else None
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else None

with open(SEED_PATH, encoding="utf-8") as f:
    SEED = json.load(f)
ITEMS = {i["id"]: i for i in SEED["items"]}

ALLERGEN_WORDS = {
    "nuts": ["nut", "nuts", "peanut", "noten", "pinda", "pindas"],
    "dairy": ["dairy", "milk", "lactose", "melk", "zuivel", "lactose"],
    "gluten": ["gluten", "wheat", "tarwe", "coeliac", "celiac"],
    "shellfish": ["shellfish", "shrimp", "prawn", "schaaldier", "schaaldieren", "garnalen"],
    "egg": ["egg", "eggs", "ei", "eieren"],
    "fish": ["fish", "vis"],
    "soy": ["soy", "soya", "soja"],
    "sesame": ["sesame", "sesam"],
}
DIETS = ["vegan", "vegetarian", "halal"]
LANE_LABEL = {"cook": "Cook", "order": "Order", "pickup": "Pick up"}

app = FastAPI()


# ---------- database ----------
def db():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row, autocommit=True)


@app.on_event("startup")
def init_db():
    with db() as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS wa_users (
            phone TEXT PRIMARY KEY,
            state TEXT NOT NULL DEFAULT 'new',
            consent_at TIMESTAMPTZ,
            adults INT, kids INT,
            allergies TEXT[] NOT NULL DEFAULT '{}',
            diet TEXT NOT NULL DEFAULT 'none',
            budget_eur NUMERIC,
            postcode TEXT,
            last_options JSONB,
            last_intent JSONB,
            excluded JSONB NOT NULL DEFAULT '[]',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS wa_events (
            id BIGSERIAL PRIMARY KEY,
            phone TEXT NOT NULL,
            kind TEXT NOT NULL,
            body TEXT,
            meta JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """)


def get_user(phone):
    with db() as c:
        u = c.execute("SELECT * FROM wa_users WHERE phone=%s", (phone,)).fetchone()
        if not u:
            u = c.execute("INSERT INTO wa_users(phone) VALUES (%s) RETURNING *", (phone,)).fetchone()
        return u


def update_user(phone, **fields):
    if not fields:
        return
    cols = ", ".join(f"{k}=%s" for k in fields)
    vals = [Jsonb(v) if isinstance(v, (dict, list)) and k not in ("allergies",) else v for k, v in fields.items()]
    with db() as c:
        c.execute(f"UPDATE wa_users SET {cols}, updated_at=NOW() WHERE phone=%s", (*vals, phone))


def log(phone, kind, body=None, meta=None):
    with db() as c:
        c.execute("INSERT INTO wa_events(phone, kind, body, meta) VALUES (%s,%s,%s,%s)",
                  (phone, kind, body, Jsonb(meta or {})))


def delete_user(phone):
    with db() as c:
        c.execute("DELETE FROM wa_events WHERE phone=%s", (phone,))
        c.execute("DELETE FROM wa_users WHERE phone=%s", (phone,))


# ---------- helpers ----------
def alert_team(text):
    if twilio_client and TEAM_ALERT_NUMBER and TWILIO_WHATSAPP_FROM:
        try:
            twilio_client.messages.create(from_=TWILIO_WHATSAPP_FROM, to=TEAM_ALERT_NUMBER, body=text[:1500])
        except Exception as e:  # never break the user flow
            print("alert failed:", e)


def parse_allergies(text):
    t = text.lower()
    if re.search(r"\b(none|no|geen|nee|nothing)\b", t) and not any(w in t for ws in ALLERGEN_WORDS.values() for w in ws if len(w) > 2):
        return []
    found = [a for a, words in ALLERGEN_WORDS.items() if any(re.search(rf"\b{re.escape(w)}\b", t) for w in words)]
    return found if found else None  # None = could not understand


def parse_household(text):
    t = text.lower()
    a = re.search(r"(\d+)\s*(adult|volwassen|grown)", t)
    k = re.search(r"(\d+)\s*(kid|child|children|kind|kinderen)", t)
    if a or k:
        return int(a.group(1)) if a else 1, int(k.group(1)) if k else 0
    n = re.search(r"\d+", t)
    return (int(n.group()), 0) if n else None


def regex_intent(text):
    t = text.lower()
    eur = re.search(r"(?:€|eur|euro)\s*(\d+(?:[.,]\d+)?)|(\d+(?:[.,]\d+)?)\s*(?:€|eur|euro)", t)
    mins = re.search(r"(\d+)\s*(?:min|minutes|minuten)", t)
    ppl = re.search(r"(?:for|voor)\s*(\d+)", t)
    lane = "cook" if re.search(r"\b(cook|koken)\b", t) else "order" if re.search(r"\b(order|takeaway|bezorg|deliver)", t) else "pickup" if re.search(r"\b(pick ?up|kit|ophalen)\b", t) else None
    return {
        "people": int(ppl.group(1)) if ppl else None,
        "max_total_eur": float((eur.group(1) or eur.group(2)).replace(",", ".")) if eur else None,
        "max_minutes": int(mins.group(1)) if mins else None,
        "cuisines": [],
        "craving": None,
        "lane": lane,
    }


INTENT_PROMPT = """You turn a family's dinner request into JSON. Reply with ONLY a JSON object, no prose:
{"people": int|null, "max_total_eur": number|null, "max_minutes": int|null,
 "cuisines": [lowercase strings from: italian, japanese, korean, thai, indian, middle_eastern, mediterranean, mexican, dutch, chinese, american],
 "craving": short lowercase dish or ingredient keyword or null,
 "lane": "cook"|"order"|"pickup"|null}
Use null when the user did not say. Never invent numbers. The request may be in Dutch or English."""


def llm_intent(phone, text):
    """Returns (intent, meta). Falls back to regex when the model is unavailable or invalid."""
    fallback = regex_intent(text)
    if not llm or not NEBIUS_MODEL:
        return fallback, {"source": "regex", "reason": "no_llm"}
    t0 = time.time()
    try:
        r = llm.chat.completions.create(
            model=NEBIUS_MODEL, temperature=0, max_tokens=200,
            messages=[{"role": "system", "content": INTENT_PROMPT}, {"role": "user", "content": text[:500]}],
        )
        latency = round((time.time() - t0) * 1000)
        raw = r.choices[0].message.content or ""
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group()) if m else None
        valid = isinstance(data, dict)
        meta = {"source": "llm", "latency_ms": latency, "json_valid": valid,
                "prompt_tokens": getattr(r.usage, "prompt_tokens", None),
                "completion_tokens": getattr(r.usage, "completion_tokens", None)}
        if not valid:
            return fallback, meta
        # merge: regex wins for explicit numbers the user typed; model fills the rest
        intent = {**{k: data.get(k) for k in fallback}, **{k: v for k, v in fallback.items() if v not in (None, [])}}
        intent["cuisines"] = [c for c in (intent.get("cuisines") or []) if isinstance(c, str)][:3]
        return intent, meta
    except Exception as e:
        return fallback, {"source": "regex", "reason": f"llm_error: {type(e).__name__}", "latency_ms": round((time.time() - t0) * 1000)}


# ---------- recommendation ----------
def allowed(item, u):
    allergies = set(u["allergies"] or [])
    if allergies:
        if not item.get("allergens_known", False):
            return False  # unknown allergens + allergic household = drop, never guess
        if allergies & set(item.get("allergens", [])):
            return False
    if u["diet"] in DIETS and u["diet"] not in item.get("diet", []):
        return False
    areas = item.get("areas")
    if areas and u["postcode"] and not any(u["postcode"].startswith(a) for a in areas):
        return False
    return True


def total_cost(item, people):
    serves = item.get("serves", 1) or 1
    if item["lane"] == "order":
        return round(item["price_per_person_eur"] * people, 2)
    if item["lane"] == "pickup":
        return round(math.ceil(people / serves) * item["price_eur"], 2)
    return round(item["missing_cost_eur"] * people / serves, 2)  # cook: cost of missing items


def score(item, intent, people, budget):
    s, why = 50, []
    cuisines = intent.get("cuisines") or []
    if cuisines and item.get("cuisine") in cuisines:
        s += 30; why.append(f"{item['cuisine'].replace('_', ' ')} as asked")
    craving = (intent.get("craving") or "").lower()
    if craving and craving in (item["name"] + " " + " ".join(item.get("tags", []))).lower():
        s += 30; why.append(f"matches '{craving}'")
    cost = total_cost(item, people)
    if budget:
        if cost <= budget:
            s += 20
        else:
            s -= 40
    if intent.get("max_minutes") and item.get("minutes"):
        s += 10 if item["minutes"] <= intent["max_minutes"] else -30
    if item.get("offer"):
        s += 10; why.append(item["offer"])
    if intent.get("lane") == item["lane"]:
        s += 15
    return s, cost, why


def recommend(u, intent):
    people = intent.get("people") or ((u["adults"] or 1) + (u["kids"] or 0))
    budget = intent.get("max_total_eur") or (float(u["budget_eur"]) if u["budget_eur"] else None)
    excluded = set(u["excluded"] or [])
    picks = {}
    for lane in ("cook", "order", "pickup"):
        cands = []
        for it in SEED["items"]:
            if it["lane"] != lane or it["id"] in excluded or not allowed(it, u):
                continue
            s, cost, why = score(it, intent, people, budget)
            if budget and cost > budget * 1.25:
                continue
            cands.append((s, cost, why, it))
        if cands:
            cands.sort(key=lambda x: (-x[0], x[1]))
            picks[lane] = cands[0]
    return picks, people, budget


def format_options(u, picks, people, budget):
    count = {1: "One option", 2: "Two options", 3: "Three options"}[len(picks)]
    lines = [f"{count} for {people}" + (f", around €{budget:.0f}" if budget else "") + ":"]
    options = {}
    n = 0
    for lane in ("cook", "order", "pickup"):
        if lane not in picks:
            continue
        n += 1
        s, cost, why, it = picks[lane]
        options[str(n)] = it["id"]
        extra = f" · {it['minutes']} min" if it.get("minutes") else ""
        where = f" · {it['store']}" if it.get("store") else ""
        cost_txt = f"€{cost:.2f} extra" if lane == "cook" else f"€{cost:.2f}"
        reason = f"\n   _{'; '.join(why[:2])}_" if why else ""
        lines.append(f"*{n} {LANE_LABEL[lane]}:* {it['name']}{where} · {cost_txt}{extra}{reason}")
    keys = list(options)
    pick_txt = keys[0] if len(keys) == 1 else ", ".join(keys[:-1]) + " or " + keys[-1]
    lines.append(f"\nReply {pick_txt} for details. Reply *other* for new options.")
    return "\n".join(lines), options


def detail(item, people):
    if item["lane"] == "cook":
        ing = "\n".join(f"• {x}" for x in item.get("missing_items", []))
        steps = "\n".join(f"{i+1}. {s}" for i, s in enumerate(item.get("steps", [])))
        return (f"*{item['name']}* ({item.get('minutes','?')} min, serves {item.get('serves','?')})\n\n"
                f"*To buy* ({item.get('store','nearby store')}, ≈€{total_cost(item, people):.2f}):\n{ing}\n\n*Steps:*\n{steps}"
                + (f"\n\nRecipe: {item['url']}" if item.get("url") else ""))
    if item["lane"] == "order":
        return (f"*{item['name']}* — ≈€{total_cost(item, people):.2f} for {people}"
                + (f", {item['minutes']} min delivery" if item.get("minutes") else "")
                + f"\nOrder here: {item['url']}\n\nI never order for you, so you stay in control. Enjoy!")
    return (f"*{item['name']}* at {item.get('store','the store')} — ≈€{total_cost(item, people):.2f}"
            + (f"\n{item['offer']}" if item.get("offer") else "")
            + (f"\nDetails: {item['url']}" if item.get("url") else ""))


def plan_week(u):
    people = (u["adults"] or 1) + (u["kids"] or 0)
    budget = float(u["budget_eur"]) if u["budget_eur"] else None
    cands = []
    for it in SEED["items"]:
        if not allowed(it, u):
            continue
        s, cost, why = score(it, {}, people, budget)
        if budget and cost > budget * 1.25:
            continue
        cands.append((s, cost, it))
    cands.sort(key=lambda x: (-x[0], x[1]))
    week, seen_cuisines = [], set()
    for s, cost, it in cands:  # prefer variety of cuisines
        if it.get("cuisine") in seen_cuisines and len(cands) > 7:
            continue
        week.append((cost, it)); seen_cuisines.add(it.get("cuisine"))
        if len(week) == 5:
            break
    if not week:
        return None
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    lines = ["*Your week:*"] + [f"{days[i]}: {LANE_LABEL[it['lane']]} — {it['name']} (≈€{c:.2f})" for i, (c, it) in enumerate(week)]
    groceries = sorted({x for _, it in week if it["lane"] == "cook" for x in it.get("missing_items", [])})
    if groceries:
        lines += ["", "*Shopping list:*"] + [f"• {g}" for g in groceries]
    lines.append(f"\nTotal ≈€{sum(c for c, _ in week):.2f}")
    return "\n".join(lines)


# ---------- conversation ----------
WELCOME = ("Good evening, I'm Alfred. I decide dinner for your family: cook, order or pick up, "
           "within your budget.\n\nTo do that I store your household size, allergies, diet, budget and postcode. "
           f"Privacy: {PRIVACY_URL}\nReply *YES* to agree. You can reply *DELETE* any time to erase everything.")
HELP = ("Ask me things like:\n• dinner for 4 under €20 in 30 min\n• something spicy, cook\n• plan my week\n"
        "Other commands: *profile*, *reset*, *DELETE*")


def handle(phone, body):
    text = body.strip()
    low = text.lower()
    if low in ("delete", "delete my data", "verwijder"):
        delete_user(phone)
        return "Done. Everything I stored about you is deleted. Send any message to start again."

    u = get_user(phone)
    state = u["state"]

    if low == "reset" and state != "new":
        update_user(phone, state="household", excluded=[])
        return "Let's redo your profile. How many people eat dinner at home? e.g. *2 adults 2 kids*"

    if state == "new":
        update_user(phone, state="consent")
        return WELCOME
    if state == "consent":
        if low in ("yes", "ja", "y", "ok", "akkoord"):
            update_user(phone, state="household", consent_at="now")
            return "Thank you. How many people eat dinner at home? e.g. *2 adults 2 kids*"
        return "Reply *YES* to agree so I can help, or *DELETE* to stop."
    if state == "household":
        hh = parse_household(text)
        if not hh:
            return "Sorry, I didn't catch that. e.g. *2 adults 2 kids* or just *4*"
        update_user(phone, adults=hh[0], kids=hh[1], state="allergies")
        return ("Any allergies at home? Reply *none*, or list them: nuts, dairy, gluten, shellfish, egg, fish, soy, sesame. "
                "I treat these as strict rules.")
    if state == "allergies":
        al = parse_allergies(text)
        if al is None:
            return "I couldn't match that. Reply *none* or e.g. *nuts, dairy*"
        update_user(phone, allergies=al, state="diet")
        return "Any diet? Reply *none*, *vegetarian*, *vegan* or *halal*."
    if state == "diet":
        d = next((x for x in DIETS if x in low or (x == "vegetarian" and "vega" in low and "vegan" not in low)), "none")
        update_user(phone, diet=d, state="budget")
        return "Typical dinner budget for the whole family, in euros? e.g. *20*"
    if state == "budget":
        n = re.search(r"\d+(?:[.,]\d+)?", text)
        if not n:
            return "Just a number please, e.g. *20*"
        update_user(phone, budget_eur=float(n.group().replace(",", ".")), state="postcode")
        return "Last one: your postcode (the 4 digits are enough), so I only suggest places nearby."
    if state == "postcode":
        pc = re.search(r"\d{4}", text)
        if not pc:
            return "Please send the 4 digits of your postcode, e.g. *2132*"
        update_user(phone, postcode=pc.group(), state="ready")
        return "All set. " + HELP

    # ---- ready ----
    if low in ("help", "?", "hi", "hello", "hoi", "hallo"):
        return HELP
    if low == "profile":
        return (f"Household: {u['adults']} adults, {u['kids']} kids\nAllergies: {', '.join(u['allergies']) or 'none'}\n"
                f"Diet: {u['diet']}\nBudget: €{u['budget_eur']}\nPostcode: {u['postcode']}\nReply *reset* to change.")
    if low in ("1", "2", "3"):
        opts = u["last_options"] or {}
        item = ITEMS.get(opts.get(low, ""))
        if not item:
            if opts:
                return f"Please pick one of the numbers shown ({', '.join(sorted(opts))}), or reply *other*."
            return "Ask me for dinner first, e.g. *dinner for 4 under €20*."
        people = (u["last_intent"] or {}).get("people") or ((u["adults"] or 1) + (u["kids"] or 0))
        log(phone, "accepted", low, {"item_id": item["id"], "lane": item["lane"]})
        return detail(item, people)
    if "plan" in low and "week" in low:
        out = plan_week(u)
        log(phone, "plan_week", None, {"ok": bool(out)})
        if not out:
            alert_team(f"[Alfred] No week plan possible for {phone}. Please reply manually.")
            return "I couldn't build a full week within your rules. A teammate will message you shortly."
        return out

    is_other = low in ("other", "👎", "not tonight", "anders", "nee")
    if is_other:
        if not u["last_intent"]:
            return "Ask me for dinner first, e.g. *dinner for 4 under €20*."
        excluded = list(set((u["excluded"] or []) + list((u["last_options"] or {}).values())))
        update_user(phone, excluded=excluded)
        u = {**u, "excluded": excluded}
        intent, meta = u["last_intent"], {"source": "reuse"}
        log(phone, "rejected", text, {"options": u["last_options"]})
    else:
        intent, meta = llm_intent(phone, text)
        update_user(phone, excluded=[])
        u = {**u, "excluded": []}

    picks, people, budget = recommend(u, intent)
    log(phone, "request", text, {**meta, "intent": intent, "n_picks": len(picks)})
    if not picks:
        alert_team(f"[Alfred] No options for {phone}: '{text}'. Intent {json.dumps(intent)}. Please reply manually.")
        return "Nothing nearby fits all your rules right now. A teammate will send you options shortly."
    msg, options = format_options(u, picks, people, budget)
    update_user(phone, last_options=options, last_intent={**intent, "people": people})
    return msg


@app.post("/whatsapp")
async def whatsapp(request: Request):
    form = dict(await request.form())
    if VALIDATE_TWILIO:
        url = (PUBLIC_URL + request.url.path) if PUBLIC_URL else str(request.url)
        sig = request.headers.get("X-Twilio-Signature", "")
        if not RequestValidator(TWILIO_AUTH_TOKEN).validate(url, form, sig):
            raise HTTPException(status_code=403, detail="invalid signature")
    phone, body = form.get("From", ""), form.get("Body", "")
    try:
        reply = handle(phone, body)
    except Exception as e:
        print("handler error:", repr(e))
        alert_team(f"[Alfred] Error for {phone}: {type(e).__name__}. Message: '{body}'")
        reply = "Sorry, something went wrong on my side. A teammate will help you shortly."
    if phone and not body.strip().lower().startswith("delete"):
        try:
            log(phone, "reply", reply[:1500])
        except Exception:
            pass
    resp = MessagingResponse()
    resp.message(reply[:1550])
    return Response(content=str(resp), media_type="application/xml")


@app.get("/health")
def health():
    return {"ok": True, "items": len(SEED["items"]), "llm": bool(llm and NEBIUS_MODEL)}


@app.get("/metrics")
def metrics(key: str = ""):
    """Pilot numbers for the pitch. Protect with METRICS_KEY."""
    if key != os.getenv("METRICS_KEY", "change-me"):
        raise HTTPException(status_code=403)
    with db() as c:
        q = lambda sql: c.execute(sql).fetchone()
        return {
            "users_onboarded": q("SELECT COUNT(*) n FROM wa_users WHERE state='ready'")["n"],
            "requests": q("SELECT COUNT(*) n FROM wa_events WHERE kind='request'")["n"],
            "accepted": q("SELECT COUNT(*) n FROM wa_events WHERE kind='accepted'")["n"],
            "rejected": q("SELECT COUNT(*) n FROM wa_events WHERE kind='rejected'")["n"],
            "llm_calls": q("SELECT COUNT(*) n FROM wa_events WHERE kind='request' AND meta->>'source'='llm'")["n"],
            "llm_json_valid_rate": q("SELECT AVG(CASE WHEN (meta->>'json_valid')::bool THEN 1 ELSE 0 END) r FROM wa_events WHERE kind='request' AND meta->>'source'='llm'")["r"],
            "llm_median_latency_ms": q("SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (meta->>'latency_ms')::int) m FROM wa_events WHERE kind='request' AND meta ? 'latency_ms'")["m"],
            "avg_prompt_tokens": q("SELECT AVG((meta->>'prompt_tokens')::int) a FROM wa_events WHERE meta ? 'prompt_tokens'")["a"],
            "avg_completion_tokens": q("SELECT AVG((meta->>'completion_tokens')::int) a FROM wa_events WHERE meta ? 'completion_tokens'")["a"],
        }
