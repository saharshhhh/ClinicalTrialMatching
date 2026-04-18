from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv  
import csv
import os
import json
import urllib.request
import urllib.error
import ssl
import markdown
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables from the .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")

def send_actual_email(to_email, subject, body):
    """Sends an actual email using SMTP settings from environment variables."""
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = os.getenv("SMTP_PORT", "587")
    smtp_user = os.getenv("SMTP_USERNAME")
    smtp_pass = os.getenv("SMTP_PASSWORD")

    if not all([smtp_server, smtp_user, smtp_pass]):
        print("[EMAIL] Skipping real email sending: SMTP credentials missing in environment.")
        return False

    try:
        msg = MIMEMultipart()
        msg['From'] = smtp_user
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(smtp_server, int(smtp_port))
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)
        server.quit()
        print(f"[EMAIL] Successfully sent email to {to_email}")
        return True
    except Exception as e:
        print(f"[EMAIL ERROR] Failed to send email to {to_email}: {e}")
        return False

# ── Persistent storage ──────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

DB_FILE = os.path.join(DATA_DIR, 'database.db')

def get_db_connection():
    import sqlite3
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    import sqlite3
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    email TEXT UNIQUE,
                    condition TEXT,
                    organization TEXT,
                    password TEXT
                 )''')
    c.execute('''CREATE TABLE IF NOT EXISTS doctors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    email TEXT UNIQUE,
                    organization TEXT,
                    password TEXT
                 )''')
    c.execute('''CREATE TABLE IF NOT EXISTS consents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_name TEXT,
                    patient_email TEXT,
                    condition TEXT,
                    patient_age TEXT,
                    patient_gender TEXT,
                    trial_id TEXT,
                    trial_title TEXT,
                    decision TEXT,
                    timestamp TEXT,
                    enrolled BOOLEAN
                 )''')

    # Check if organization column exists in patients table
    c.execute("PRAGMA table_info(patients)")
    columns = [column[1] for column in c.fetchall()]
    if 'organization' not in columns:
        c.execute('ALTER TABLE patients ADD COLUMN organization TEXT')

    conn.commit()
    
    # Migration logic
    c.execute('SELECT COUNT(*) FROM patients')
    if c.fetchone()[0] == 0:
        import json
        for table, file_name in [('patients', 'patients.json'), ('doctors', 'doctors.json'), ('consents', 'consents.json')]:
            file_path = os.path.join(DATA_DIR, file_name)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    try:
                        data = json.load(f)
                        for item in data:
                            if table == 'patients':
                                c.execute('''INSERT OR IGNORE INTO patients (name, email, condition, organization, password)
                                             VALUES (?, ?, ?, ?, ?)''',
                                          (item.get('name'), item.get('email'), item.get('condition'), item.get('organization'), item.get('password')))
                            elif table == 'doctors':
                                c.execute('''INSERT OR IGNORE INTO doctors (name, email, organization, password) 
                                             VALUES (?, ?, ?, ?)''', 
                                          (item.get('name'), item.get('email'), item.get('organization'), item.get('password')))
                            elif table == 'consents':
                                c.execute('''INSERT INTO consents (patient_name, patient_email, condition, patient_age, patient_gender, trial_id, trial_title, decision, timestamp, enrolled)
                                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                                          (item.get('patient_name'), item.get('patient_email'), item.get('condition'), item.get('patient_age'), item.get('patient_gender'), str(item.get('trial_id')), item.get('trial_title'), item.get('decision'), item.get('timestamp'), item.get('enrolled')))
                    except Exception as e:
                        print("Migration error:", e)
        conn.commit()
    conn.close()

init_db()

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
        conn = get_db_connection()
        patient = conn.execute("SELECT * FROM patients WHERE email = ?", (email,)).fetchone()
        conn.close()
        if patient and check_password_hash(patient["password"], password):
            session["role"] = "patient"
            session["email"] = email
            session["name"] = patient["name"]
            session["condition"] = patient["condition"] or ""
            session["organization"] = patient["organization"] or ""
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
        organization = request.form.get("organization", "").strip()
        password = request.form.get("password", "")
        
        conn = get_db_connection()
        existing = conn.execute("SELECT * FROM patients WHERE email = ?", (email,)).fetchone()
        if existing:
            conn.close()
            flash("Email already exists. Please log in.")
            return redirect(url_for("patient_login"))
            
        conn.execute("INSERT INTO patients (name, email, condition, organization, password) VALUES (?, ?, ?, ?, ?)",
                     (name, email, condition, organization, generate_password_hash(password)))
        conn.commit()
        conn.close()
        session["role"] = "patient"
        session["email"] = email
        session["name"] = name
        session["condition"] = condition
        session["organization"] = organization
        return redirect(url_for("patient"))
    return render_template("patient_signup.html")

@app.route("/doctor_login", methods=["GET", "POST"])
def doctor_login():
    if session.get("role") == "doctor":
        return redirect(url_for("doctor"))
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        conn = get_db_connection()
        doctor = conn.execute("SELECT * FROM doctors WHERE email = ?", (email,)).fetchone()
        conn.close()
        if doctor and check_password_hash(doctor["password"], password):
            session["role"] = "doctor"
            session["email"] = email
            session["name"] = doctor["name"]
            session["organization"] = doctor["organization"]
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
        
        conn = get_db_connection()
        existing = conn.execute("SELECT * FROM doctors WHERE email = ?", (email,)).fetchone()
        if existing:
            conn.close()
            flash("Email already exists. Please log in.")
            return redirect(url_for("doctor_login"))
            
        conn.execute("INSERT INTO doctors (name, email, organization, password) VALUES (?, ?, ?, ?)",
                     (name, email, organization, generate_password_hash(password)))
        conn.commit()
        conn.close()
        session["role"] = "doctor"
        session["email"] = email
        session["name"] = name
        session["organization"] = organization
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
    
    conn = get_db_connection()
    enrollments = [dict(row) for row in conn.execute("SELECT * FROM consents WHERE patient_email = ?", (email,)).fetchall()]
    conn.close()
    
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

    conn = get_db_connection()
    conn.execute("DELETE FROM consents WHERE patient_email = ? AND trial_id = ?", (email, str(trial_id)))
    
    conn.execute("""
        INSERT INTO consents (patient_name, patient_email, condition, patient_age, patient_gender, trial_id, trial_title, decision, timestamp, enrolled)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (name, email, condition, age, gender, str(trial_id), trial_title, decision, datetime.now().strftime("%Y-%m-%d %H:%M"), False))
    conn.commit()

    c_total = conn.execute("SELECT COUNT(*) FROM consents").fetchone()[0]
    c_accepted = conn.execute("SELECT COUNT(*) FROM consents WHERE decision = 'accepted'").fetchone()[0]
    conn.close()

    print(f"[CONSENT] {name} ({email}) → {decision} → {trial_title}")
    print(f"[CONSENTS] Total={c_total}, Accepted={c_accepted}")

    return jsonify({"status": "success", "decision": decision})


@app.route("/enroll", methods=["POST"])
def enroll():
    if session.get("role") != "doctor":
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    data = request.get_json()
    key  = data.get("key") 

    print(f"[ENROLL] Received key: {key}")
    
    conn = get_db_connection()
    try:
        if "_" in str(key):
            patient_email, trial_id = str(key).rsplit("_", 1)
            c = conn.execute("SELECT * FROM consents WHERE patient_email = ? AND trial_id = ?", (patient_email, trial_id)).fetchone()
            if c:
                conn.execute("UPDATE consents SET enrolled = 1 WHERE id = ?", (c['id'],))
                conn.commit()
                conn.close()
                print(f"[ENROLL] Success: {c['patient_name']} enrolled in {c['trial_title']}")
                return jsonify({"status": "enrolled", "name": c["patient_name"], "trial": c["trial_title"]})
    except Exception as e:
        print("Enrollment error", e)
        
    conn.close()

    return jsonify({"status": "error", "message": f"Record not found for key: {key}"}), 404


@app.route("/doctor", methods=["GET", "POST"])
def doctor():
    if session.get("role") != "doctor":
        return redirect(url_for("doctor_login"))

    organization = session.get("organization")
    conn = get_db_connection()
    all_consents = [dict(row) for row in conn.execute("SELECT * FROM consents").fetchall()]

    # Fetch patients in the same organization
    org_patients = []
    if organization:
        org_patients = [dict(row) for row in conn.execute("SELECT * FROM patients WHERE organization = ?", (organization,)).fetchall()]

    patient_count = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    conn.close()
    
    accepted       = [c for c in all_consents if c["decision"] == "accepted"]
    enrolled_count = len([c for c in all_consents if c.get("enrolled")])
    
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
        patient_count  = patient_count,
        search_query   = search_query,
        all_consents   = all_consents,
        org_patients   = org_patients,
        organization   = organization
    )


@app.route("/request_consent", methods=["POST"])
def request_consent():
    if session.get("role") != "doctor":
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    data = request.get_json()
    patient_email = data.get("patient_email")
    trial_id = int(data.get("trial_id"))
    doctor_name = session.get("name")
    doctor_email = session.get("email")
    doctor_org = session.get("organization")

    # Security check: verify patient belongs to doctor's organization
    conn = get_db_connection()
    patient = conn.execute("SELECT * FROM patients WHERE email = ? AND organization = ?", (patient_email, doctor_org)).fetchone()
    conn.close()

    if not patient:
        return jsonify({"status": "error", "message": "Patient not found in your organization"}), 403

    trial = next((t for t in TRIALS if t["id"] == trial_id), None)
    if not trial:
        return jsonify({"status": "error", "message": "Trial not found"}), 404

    # Summarize trial description using Cohere
    summary = "No summary available."
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")

    if COHERE_API_KEY:
        prompt = f"Please provide a concise, patient-friendly summary of the following clinical trial description in 2-3 sentences:\n\n{trial['description']}"
        payload = json.dumps({
            "model": "command-r7b-12-2024",
            "message": prompt,
            "temperature": 0.3,
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.cohere.com/v1/chat",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {COHERE_API_KEY}",
            },
            method="POST",
        )

        try:
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                summary = result["text"]
        except Exception as e:
            print(f"[SUMMARIZATION ERROR] {e}")
            summary = trial["description"][:200] + "..."

    # Simulate sending email
    email_content = f"""
    To: {patient_email}
    From: {doctor_email} ({doctor_name})
    Subject: Consent Request for Clinical Trial: {trial['title']}

    Dear Patient,

    Dr. {doctor_name} has requested your consent to participate in the following clinical trial:
    Title: {trial['title']}

    Trial Summary:
    {summary}

    Please log in to the TrialBridge portal to review the full details and provide your decision.

    Best regards,
    TrialBridge Team
    """

    # Log the "email"
    with open("server.log", "a") as f:
        f.write(f"\n--- EMAIL REQUEST AT {datetime.now()} ---\n")
        f.write(email_content)
        f.write("\n-----------------------------------\n")

    # Send actual email
    subject = f"Consent Request for Clinical Trial: {trial['title']}"
    sent = send_actual_email(patient_email, subject, email_content)

    print(f"[REQUEST CONSENT] Email workflow completed for {patient_email} regarding trial {trial_id}. Actual sent: {sent}")

    return jsonify({"status": "success", "summary": summary, "email_sent": sent})


@app.route("/debug")
def debug():
    conn = get_db_connection()
    all_consents = [dict(row) for row in conn.execute("SELECT * FROM consents").fetchall()]
    accepted = [c for c in all_consents if c["decision"] == "accepted"]
    patient_count = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    conn.close()
    
    return jsonify({
        "total_consents":    len(all_consents),
        "accepted_consents": len(accepted),
        "total_patients":    patient_count,
        "consents":          all_consents,
    })


@app.route("/chat", methods=["POST"])
def chat():
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

    payload = json.dumps({
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
            result = json.loads(resp.read().decode("utf-8"))
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