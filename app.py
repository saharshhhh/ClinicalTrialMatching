from flask import Flask, render_template, request, jsonify
import csv
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = "supersecretkey"

# ── In-memory stores ────────────────────────────────────────────────────────
CONSENTS = []
PATIENTS = []


# ── Load trials from CSV ────────────────────────────────────────────────────
def load_trials():
    trials = []
    csv_path = os.path.join(os.path.dirname(__file__), 'trails.csv')
    if not os.path.exists(csv_path):
        print(f"WARNING: trails.csv not found at {csv_path}")
        return []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            trials.append({
                "id":           i + 1,
                "title":        row.get("Brief Title") or "No Title Available",
                "condition":    row.get("Conditions") or "",
                "description":  row.get("Intervention Description") or "",
                "eligibility":  row.get("Standard Age") or "",
                "phase":        row.get("Phases") or "",
                "status":       row.get("Overall Status") or "",
                "location":     row.get("Organization Full Name") or "",
                "duration":     row.get("Start Date") or "N/A",
                "compensation": "Contact sponsor",
            })
    return trials

ALL_TRIALS = load_trials()
# Only count recruiting trials as "active" for the dashboard stat
TRIALS = ALL_TRIALS  # show all in lists
ACTIVE_TRIALS = [t for t in ALL_TRIALS if t["status"] in ("RECRUITING", "NOT_YET_RECRUITING")]
print(f"[STARTUP] Loaded {len(ALL_TRIALS)} total trials, {len(ACTIVE_TRIALS)} active")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("landing.html")


@app.route("/patient", methods=["GET", "POST"])
def patient():
    if request.method == "POST":
        name      = request.form.get("name", "").strip()
        email     = request.form.get("email", "").strip()
        condition = request.form.get("condition", "").strip()

        if not any(p["email"] == email for p in PATIENTS):
            PATIENTS.append({"name": name, "email": email, "condition": condition})

        user_condition = condition.lower()
        filtered = [t for t in TRIALS if user_condition in t.get("condition", "").lower()]
        trials_to_show = filtered[:10] if filtered else TRIALS[:10]

        return render_template("patient.html",
            name=name, email=email, condition=condition, trials=trials_to_show)

    return render_template("patient.html", name="", email="", condition="", trials=None)


@app.route("/consent", methods=["POST"])
def consent():
    data      = request.get_json()
    trial_id  = data.get("trial_id")
    decision  = data.get("decision")
    name      = data.get("name",      "Anonymous")
    email     = data.get("email",     "unknown@email.com")
    condition = data.get("condition", "")

    trial = next((t for t in TRIALS if t["id"] == trial_id), None)
    trial_title = trial["title"] if trial else "Unknown Trial"

    # Remove duplicate
    global CONSENTS
    CONSENTS = [c for c in CONSENTS
                if not (c["patient_email"] == email and c["trial_id"] == trial_id)]

    # ✅ Keys match EXACTLY what doctor.html uses in the template
    CONSENTS.append({
        "patient_name":  name,
        "patient_email": email,
        "condition":     condition,
        "trial_id":      trial_id,
        "trial_title":   trial_title,
        "decision":      decision,
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M"),
        "enrolled":      False,
    })

    print(f"[CONSENT] {name} ({email}) → {decision} → {trial_title}")
    print(f"[CONSENTS] Total={len(CONSENTS)}, Accepted={len([c for c in CONSENTS if c['decision']=='accepted'])}")

    return jsonify({"status": "success", "decision": decision})


@app.route("/enroll", methods=["POST"])
def enroll():
    data = request.get_json()
    key  = data.get("key")  # format: "email_trialid"

    print(f"[ENROLL] Received key: {key}")
    available_keys = [str(c["patient_email"]) + "_" + str(c["trial_id"]) for c in CONSENTS]
    print(f"[ENROLL] Available keys: {available_keys}")

    for c in CONSENTS:
        # Compare as strings to avoid int/str mismatch
        consent_key = f"{c['patient_email']}_{c['trial_id']}"
        if consent_key == str(key):
            c["enrolled"] = True
            print(f"[ENROLL] Success: {c['patient_name']} enrolled in {c['trial_title']}")
            return jsonify({"status": "enrolled", "name": c["patient_name"], "trial": c["trial_title"]})

    print(f"[ENROLL] No match found for key: {key}")
    return jsonify({"status": "error", "message": f"Record not found for key: {key}"}), 404


@app.route("/doctor")
def doctor():
    accepted       = [c for c in CONSENTS if c["decision"] == "accepted"]
    enrolled_count = len([c for c in CONSENTS if c.get("enrolled")])
    active_trials  = len(set(c["trial_id"] for c in accepted)) if accepted else 0

    print(f"[DOCTOR PAGE] Showing {len(accepted)} accepted consents")

    return render_template("doctor.html",
        trials         = TRIALS,
        consents       = accepted,
        enrolled_count = enrolled_count,
        active_trials  = len(ACTIVE_TRIALS),   # ← only recruiting trials
        patient_count  = len(PATIENTS),
    )


@app.route("/debug")
def debug():
    accepted = [c for c in CONSENTS if c["decision"] == "accepted"]
    return jsonify({
        "total_consents":    len(CONSENTS),
        "accepted_consents": len(accepted),
        "total_patients":    len(PATIENTS),
        "consents":          CONSENTS,
    })


@app.route("/chat", methods=["POST"])
def chat():
    import json as json_lib
    import urllib.request
    import urllib.error
    import ssl

    data     = request.get_json()
    messages = data.get("messages", [])
    role     = data.get("role", "patient")

    if not messages:
        messages = [{"role": "user", "content": "Hello"}]

    trial_context = "\n".join([
        f"- {t['title']} | Condition: {t['condition']} | Phase: {t['phase']} | Location: {t['location']}"
        for t in TRIALS[:5]
    ])

    if role == "patient":
        preamble = (
            "You are a compassionate clinical trial assistant helping patients understand clinical trials. "
            "Explain everything in simple friendly language. Help patients understand what participation involves, "
            "what consent means, risks and benefits. Be warm, empathetic and never pushy. "
            "Always recommend consulting their doctor for medical decisions. "
            "Give clear, helpful answers to any question.\n\n"
            "Available trials on this platform:\n" + trial_context
        )
    else:
        preamble = (
            "You are an expert clinical research assistant for doctors and scientists. "
            "Help with trial protocols, patient eligibility, enrollment strategies, "
            "adverse event reporting, and research data interpretation. "
            "Be precise, evidence-based and professional.\n\n"
            "Available trials on this platform:\n" + trial_context
        )

    # Build chat history for Cohere format
    chat_history = []
    for m in messages[:-1]:
        chat_history.append({
            "role": "USER" if m["role"] == "user" else "CHATBOT",
            "message": m["content"]
        })

    # Last message is the current user question
    last_message = messages[-1]["content"] if messages else "Hello"

    payload = json_lib.dumps({
        "model": "command-r7b-12-2024",
        "message": last_message,
        "chat_history": chat_history,
        "preamble": preamble,
        "max_tokens": 300,
        "temperature": 0.7,
    }).encode("utf-8")

    api_key = "AZZS1J9Iq8zUPoUrCZTWBuoutwV92Ix4YP4n6IF6"

    req = urllib.request.Request(
        "https://api.cohere.com/v1/chat",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "X-Client-Name": "TrialBridge",
        },
        method="POST",
    )

    try:
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
            result = json_lib.loads(resp.read().decode("utf-8"))
            reply = result["text"]
            return jsonify({"reply": reply})
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        print(f"[CHAT ERROR] HTTP {e.code}: {body}")
        return jsonify({"reply": f"AI error {e.code}: {body}"})
    except Exception as e:
        print(f"[CHAT ERROR] {type(e).__name__}: {e}")
        return jsonify({"reply": f"AI unavailable: {e}"})


@app.route("/test-groq")
def test_groq():
    """Visit /test-groq to check if Groq API is reachable."""
    import urllib.request, json as j, ssl
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return jsonify({"status": "error", "message": "GROQ_API_KEY is empty in app.py"})
    payload = j.dumps({
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 10,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}", "User-Agent": "TrialBridge/1.0"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, context=ssl.create_default_context(), timeout=15) as resp:
            result = j.loads(resp.read().decode("utf-8"))
            return jsonify({"status": "success", "reply": result["choices"][0]["message"]["content"]})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})



if __name__ == "__main__":
    # ⚠️ use_reloader=False is CRITICAL
    # With reloader=True, Flask starts TWO processes — the second one has
    # its own empty CONSENTS list, so data written to one process is invisible
    # to the other. This is why consents appear to disappear!
    app.run(debug=True, use_reloader=False)
