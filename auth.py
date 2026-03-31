from flask import Blueprint, render_template, request, session, redirect, url_for, flash
from werkzeug.security import check_password_hash
from db import db

auth_bp = Blueprint('auth', __name__)

@auth_bp.route("/patient_login", methods=["GET", "POST"])
def patient_login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        patient = db.get_patient(email)
        if patient and check_password_hash(patient.get("password", ""), password):
            session["role"] = "patient"
            session["email"] = patient["email"]
            session["name"] = patient["name"]
            session["condition"] = patient.get("condition", "")
            return redirect(url_for("patient"))
        else:
            return "Invalid credentials. Please try again.", 401
    return render_template("patient_login.html")

@auth_bp.route("/patient_signup", methods=["GET", "POST"])
def patient_signup():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        condition = request.form.get("condition", "").strip()
        password = request.form.get("password", "")
        
        if not db.add_patient(name, email, condition, password):
            return "Email already exists. Please log in.", 400
            
        session["role"] = "patient"
        session["email"] = email.lower()
        session["name"] = name
        session["condition"] = condition
        return redirect(url_for("patient"))
    return render_template("patient_signup.html")

@auth_bp.route("/doctor_login", methods=["GET", "POST"])
def doctor_login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        doctor = db.get_doctor(email)
        if doctor and check_password_hash(doctor.get("password", ""), password):
            session["role"] = "doctor"
            session["email"] = doctor["email"]
            session["name"] = doctor["name"]
            return redirect(url_for("doctor"))
        else:
            return "Invalid credentials. Please try again.", 401
    return render_template("doctor_login.html")

@auth_bp.route("/doctor_signup", methods=["GET", "POST"])
def doctor_signup():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        organization = request.form.get("organization", "").strip()
        password = request.form.get("password", "")
        
        if not db.add_doctor(name, email, organization, password):
            return "Email already exists. Please log in.", 400
            
        session["role"] = "doctor"
        session["email"] = email.lower()
        session["name"] = name
        return redirect(url_for("doctor"))
    return render_template("doctor_signup.html")

@auth_bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))
