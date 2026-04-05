from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv  
import csv
import os
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables from the .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")

# ── Persistent storage ──────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

PATIENTS_FILE = os.path.join(DATA_DIR, 'patients.json')
DOCTORS_FILE = os.path.join(DATA_DIR, 'doctors.json')
CONSENTS_FILE = os.path.join(DATA_DIR, 'consents.json')

def load_json(filepath, default_value):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return default_value
    return default_value

def save_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

CONSENTS = load_json(CONSENTS_FILE, [])
PATIENTS = load_json(PATIENTS_FILE, [])
DOCTORS = load_json(DOCTORS_FILE, [])

# ── Load trials from CSV ────────────────────────────────────────────────────
def load_trials():
    trials = []
    csv_path = os.path.join(os.path.dirname(__file__), 'trials.csv')
    if not os.path.exists(csv_path):
        print(f"WARNING: trials.csv not found at {csv_path}")
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
TRIALS = ALL_TRIALS  
ACTIVE_TRIALS = [t for t in ALL_TRIALS if t["status"] in ("RECRUITING", "NOT_YET_RECRUITING")]
print(f"[STARTUP] Loaded {len(ALL_TRIALS)} total trials, {len(ACTIVE_TRIALS)} active")

# ── NLP Startup Logic ───────────────────────────────────────────────────────
print("[STARTUP] Loading SentenceTransformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2') 

def embed_text(text):
    return model.encode([str(text)], show_progress_bar=False)[0]

MATCHING_SAMPLE_SIZE = 50000 
matching_trials = ALL_TRIALS[:MATCHING_SAMPLE_SIZE]

print("[STARTUP] Loading pre-computed trial embeddings...")
embeddings_path = os.path.join(DATA_DIR, 'trial_embeddings.npy')
try:
    # Load the vectors directly from disk in milliseconds
    trial_vectors = np.load(embeddings_path)
    print(f"[STARTUP] Successfully loaded {len(trial_vectors)} trial embeddings!")
except FileNotFoundError:
    print(f"\n[ERROR] Could not find {embeddings_path}!")
    print("Please run `python precompute.py` first to generate the embeddings.\n")
    # Fallback empty array so the app doesn't crash entirely, but search won't work
    trial_vectors = np.empty((0, 384))

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("landing.html")

@app.route("/patient_login", methods=["GET", "POST"])
def patient_login():
    if session.get("role") == "patient":
        return redirect(url_for("patient"))
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        patient = next((p for p in PATIENTS if p["email"] == email), None)
        if patient and check_password_hash(patient.get("password", ""), password):
            session["role"] = "patient"
            session["email"] = email
            session["name"] = patient["name"]
            session["condition"] = patient.get("condition", "")
            return redirect(url_for("patient"))
        else:
            flash("Invalid credentials. Please try again.")
            return redirect(url_for("patient_login"))
    return render_template("patient_login.html")

@app.route("/patient_signup", methods=["GET", "POST"])
def patient_signup():
    if session.get("role") == "patient":
        return redirect(url_for("patient"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        condition = request.form.get("condition", "").strip()
        password = request.form.get("password", "")
        
        if any(p["email"] == email for p in PATIENTS):
            flash("Email already exists. Please log in.")
            return redirect(url_for("patient_login"))
            
        PATIENTS.append({
            "name": name, 
            "email": email, 
            "condition": condition,
            "password": generate_password_hash(password)
        })
        save_json(PATIENTS_FILE, PATIENTS)
        session["role"] = "patient"
        session["email"] = email
        session["name"] = name
        session["condition"] = condition
        return redirect(url_for("patient"))
    return render_template("patient_signup.html")

@app.route("/doctor_login", methods=["GET", "POST"])
def doctor_login():
    if session.get("role") == "doctor":
        return redirect(url_for("doctor"))
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        doctor = next((d for d in DOCTORS if d["email"] == email), None)
        if doctor and check_password_hash(doctor.get("password", ""), password):
            session["role"] = "doctor"
            session["email"] = email
            session["name"] = doctor["name"]
            return redirect(url_for("doctor"))
        else:
            flash("Invalid credentials. Please try again.")
            return redirect(url_for("doctor_login"))
    return render_template("doctor_login.html")

@app.route("/doctor_signup", methods=["GET", "POST"])
def doctor_signup():
    if session.get("role") == "doctor":
        return redirect(url_for("doctor"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        organization = request.form.get("organization", "").strip()
        password = request.form.get("password", "")
        
        if any(d["email"] == email for d in DOCTORS):
            flash("Email already exists. Please log in.")
            return redirect(url_for("doctor_login"))
            
        DOCTORS.append({
            "name": name, 
            "email": email, 
            "organization": organization,
            "password": generate_password_hash(password)
        })
        save_json(DOCTORS_FILE, DOCTORS)
        session["role"] = "doctor"
        session["email"] = email
        session["name"] = name
        return redirect(url_for("doctor"))
    return render_template("doctor_signup.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


@app.route("/patient", methods=["GET", "POST"])
def patient():
    if session.get("role") != "patient":
        return redirect(url_for("patient_login"))
        
    if request.method == "POST":
        session["condition"] = request.form.get("condition", "").strip()
        session["age"] = request.form.get("age", "").strip()
        session["gender"] = request.form.get("gender", "").strip()

    name = session.get("name", "")
    email = session.get("email", "")
    condition = session.get("condition", "")
    age = session.get("age", "")
    gender = session.get("gender", "")

    user_condition = condition.lower()
    
    # Matching algorithm
    filtered = []
    if user_condition or age or gender:
        query_text = " ".join(filter(None, [user_condition, gender]))
        
        age_group = ""
        if age and age.isdigit():
            age_int = int(age)
            if age_int < 18:
                age_group = "CHILD"
            elif age_int >= 65:
                age_group = "OLDER_ADULT"
            else:
                age_group = "ADULT"

        if query_text and len(trial_vectors) > 0:
            # Semantic search using SentenceTransformer cosine similarity
            query_vec = embed_text(query_text)
            sims = cosine_similarity([query_vec], trial_vectors).flatten()
            
            # Get top matches where similarity is > 0, ordered by best match
            top_indices = np.argsort(sims)[::-1]
            
            for idx in top_indices:
                trial = matching_trials[idx]
                if age_group and trial.get("eligibility") and age_group not in trial.get("eligibility", ""):
                    continue
                if sims[idx] > 0.01:
                    filtered.append(trial)
                if len(filtered) >= 15:
                    break
    
    trials_to_show = filtered if filtered else TRIALS[:15]

    return render_template("patient.html",
        name=name, email=email, condition=condition, age=age, gender=gender, trials=trials_to_show)


@app.route("/my-status")
def my_status():
    if session.get("role") != "patient":
        return redirect(url_for("patient_login"))
    
    email = session.get("email")
    name = session.get("name")
    enrollments = [c for c in CONSENTS if c["patient_email"] == email]
    
    return render_template("patient_dashboard.html", name=name, email=email, enrollments=enrollments)


@app.route("/consent", methods=["POST"])
def consent():
    if session.get("role") != "patient":
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    data      = request.get_json()
    trial_id  = data.get("trial_id")
    decision  = data.get("decision")
    name      = data.get("name",      "Anonymous")
    email     = data.get("email",     "unknown@email.com")
    condition = data.get("condition", "")
    age       = data.get("age", "")
    gender    = data.get("gender", "")

    trial = next((t for t in TRIALS if t["id"] == trial_id), None)
    trial_title = trial["title"] if trial else "Unknown Trial"

    # Remove duplicate
    global CONSENTS
    CONSENTS = [c for c in CONSENTS
                if not (c["patient_email"] == email and c["trial_id"] == trial_id)]

    CONSENTS.append({
        "patient_name":  name,
        "patient_email": email,
        "condition":     condition,
        "patient_age":   age,
        "patient_gender":gender,
        "trial_id":      trial_id,
        "trial_title":   trial_title,
        "decision":      decision,
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M"),
        "enrolled":      False,
    })
    save_json(CONSENTS_FILE, CONSENTS)

    print(f"[CONSENT] {name} ({email}) → {decision} → {trial_title}")
    print(f"[CONSENTS] Total={len(CONSENTS)}, Accepted={len([c for c in CONSENTS if c['decision']=='accepted'])}")

    return jsonify({"status": "success", "decision": decision})


@app.route("/enroll", methods=["POST"])
def enroll():
    if session.get("role") != "doctor":
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    data = request.get_json()
    key  = data.get("key") 

    print(f"[ENROLL] Received key: {key}")
    available_keys = [str(c["patient_email"]) + "_" + str(c["trial_id"]) for c in CONSENTS]
    
    for c in CONSENTS:
        consent_key = f"{c['patient_email']}_{c['trial_id']}"
        if consent_key == str(key):
            c["enrolled"] = True
            save_json(CONSENTS_FILE, CONSENTS)
            print(f"[ENROLL] Success: {c['patient_name']} enrolled in {c['trial_title']}")
            return jsonify({"status": "enrolled", "name": c["patient_name"], "trial": c["trial_title"]})

    return jsonify({"status": "error", "message": f"Record not found for key: {key}"}), 404


@app.route("/doctor", methods=["GET", "POST"])
def doctor():
    if session.get("role") != "doctor":
        return redirect(url_for("doctor_login"))

    accepted       = [c for c in CONSENTS if c["decision"] == "accepted"]
    enrolled_count = len([c for c in CONSENTS if c.get("enrolled")])
    
    search_query = ""
    trials_to_show = TRIALS[:50]
    
    if request.method == "POST":
        search_query = request.form.get("search_query", "").strip()
        if search_query and len(trial_vectors) > 0:
            query_vec = embed_text(search_query)
            sims = cosine_similarity([query_vec], trial_vectors).flatten()
            top_indices = np.argsort(sims)[::-1]
            
            filtered = []
            for idx in top_indices:
                if sims[idx] > 0.01:
                    filtered.append(matching_trials[idx])
                if len(filtered) >= 50:
                    break
            if filtered:
                trials_to_show = filtered

    print(f"[DOCTOR PAGE] Showing {len(accepted)} accepted consents. Search query: '{search_query}'")

    return render_template("doctor.html",
        trials         = trials_to_show,
        consents       = accepted,
        enrolled_count = enrolled_count,
        active_trials  = len(ACTIVE_TRIALS),
        patient_count  = len(PATIENTS),
        search_query   = search_query,
        all_consents   = CONSENTS,
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
    import markdown
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
            "adverse event reporting, and research data interpretation."
            "Be precise, evidence-based and professional.\n\n"
            "Available trials on this platform:\n" + trial_context
        )

    chat_history = []
    for m in messages[:-1]:
        chat_history.append({
            "role": "USER" if m["role"] == "user" else "CHATBOT",
            "message": m["content"]
        })

    last_message = messages[-1]["content"] if messages else "Hello"

    payload = json_lib.dumps({
        "model": "command-r7b-12-2024",
        "message": last_message,
        "chat_history": chat_history,
        "preamble": preamble,
        "temperature": 0.7,
    }).encode("utf-8")

    COHERE_API_KEY = os.getenv("COHERE_API_KEY")

    if not COHERE_API_KEY:
        print("[CHAT ERROR] COHERE_API_KEY not found in environment variables.")
        return jsonify({"reply": "AI service is currently unavailable (API key missing)."}), 500

    req = urllib.request.Request(
        "https://api.cohere.com/v1/chat",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {COHERE_API_KEY}",
            "X-Client-Name": "TrialBridge",
        },
        method="POST",
    )

    try:
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
            result = json_lib.loads(resp.read().decode("utf-8"))
            reply = result["text"]
            html_reply = markdown.markdown(reply)
            return jsonify({"reply": html_reply})
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        print(f"[CHAT ERROR] HTTP {e.code}: {body}")
        return jsonify({"reply": f"AI error {e.code}: {body}"})
    except Exception as e:
        print(f"[CHAT ERROR] {type(e).__name__}: {e}")
        return jsonify({"reply": f"AI unavailable: {e}"})

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)