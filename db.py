import json
import os
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

DATA_FILE = os.path.join(os.path.dirname(__file__), 'data.json')

class Database:
    def __init__(self):
        self.patients = []
        self.doctors = []
        self.consents = []
        self.load()

    def load(self):
        if os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.patients = data.get("patients", [])
                    self.doctors = data.get("doctors", [])
                    self.consents = data.get("consents", [])
            except Exception as e:
                print(f"Error loading data: {e}")

    def save(self):
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                "patients": self.patients,
                "doctors": self.doctors,
                "consents": self.consents
            }, f, indent=4)

    def get_patient(self, email):
        for p in self.patients:
            if p["email"].lower() == email.lower():
                return p
        return None

    def add_patient(self, name, email, condition, password):
        if self.get_patient(email):
            return False
        self.patients.append({
            "name": name,
            "email": email.lower(),
            "condition": condition,
            "password": generate_password_hash(password)
        })
        self.save()
        return True

    def get_doctor(self, email):
        for d in self.doctors:
            if d["email"].lower() == email.lower():
                return d
        return None

    def add_doctor(self, name, email, organization, password):
        if self.get_doctor(email):
            return False
        self.doctors.append({
            "name": name,
            "email": email.lower(),
            "organization": organization,
            "password": generate_password_hash(password)
        })
        self.save()
        return True
        
    def add_consent(self, name, email, condition, age, gender, trial_id, trial_title, decision):
        self.consents = [c for c in self.consents
                    if not (c["patient_email"].lower() == email.lower() and str(c["trial_id"]) == str(trial_id))]
                    
        self.consents.append({
            "patient_name": name,
            "patient_email": email.lower(),
            "condition": condition,
            "patient_age": age,
            "patient_gender": gender,
            "trial_id": trial_id,
            "trial_title": trial_title,
            "decision": decision,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "enrolled": False,
        })
        self.save()
        
    def enroll_consent(self, key):
        for c in self.consents:
            consent_key = f"{c['patient_email'].lower()}_{c['trial_id']}"
            if consent_key == str(key).lower():
                c["enrolled"] = True
                self.save()
                return c
        return None

db = Database()
