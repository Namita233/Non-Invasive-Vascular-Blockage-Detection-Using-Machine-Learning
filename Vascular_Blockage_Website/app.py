from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from database import init_db, add_user, get_user
init_db()
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from fpdf import FPDF
import datetime

app = Flask(__name__)


from flask import session
app.secret_key = "supersecretkey"  # required for session storage

# ------------------------------
# PATHS
# ------------------------------
MODEL_PATH = "saved_model/blockage_cnn.h5"
UPLOAD_FOLDER = "static/uploads"
REPORT_FOLDER = "reports"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# ------------------------------
# LOAD MODEL
# ------------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# ------------------------------
# ROUTES
# ------------------------------
@app.route("/")
def index():
    return render_template("index.html")



@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        # Check if username already exists
        existing_user = get_user(username)
        if existing_user:
            # If username exists, show error on register page
            error = "Username already exists. Please choose another."
            return render_template("register.html", error=error)

        # If username is new, add to DB
        add_user(username, password)
        return redirect(url_for("login"))  # after successful registration

    return render_template("register.html")



@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        user = get_user(username)
        if user:
            stored_username, stored_password = user
            if password == stored_password:
                session['username'] = stored_username
                return redirect(url_for("welcome_page"))
            else:
                error = "Incorrect password! Try again."
        else:
            error = "Username does not exist! Please register first."

    return render_template("login.html", error=error)



@app.route("/welcome", methods=["GET", "POST"])
def welcome_page():
    if request.method == "POST":
        patient_name = request.form.get("patient_name")
        option = request.form.get("option")

        if option == "individual":
            # store name in session
            session['patient_name'] = patient_name
            return redirect(url_for("patient_info"))
        elif option == "medical":
            # store doctor name in session
            session['doctor_name'] = patient_name
            return redirect(url_for("doctor_info"))
        else:
            return "Invalid option selected!"

    return render_template("welcome.html")

@app.route("/doctor_info", methods=["GET", "POST"])
def doctor_info():
    if request.method == "POST":
        session['doctor_name'] = request.form.get("doctor_name")      # ADD THIS LINE
        session['doctor_specialty'] = request.form.get("specialty")
        session['doctor_license'] = request.form.get("license")
        return redirect(url_for("patient_info"))
    return render_template("doctor_info.html")
# ------------------------------
# Patient Info Page for Individual
# ------------------------------
from flask import session  # make sure this import is at the top of your file

app.secret_key = "supersecretkey"  # required for session storage

@app.route("/patient_info", methods=["GET", "POST"])
def patient_info():
    if request.method == "POST":
        # Store patient info in session
        session['patient_name'] = request.form.get("patient_name")
        session['age'] = request.form.get("age")
        session['gender'] = request.form.get("gender")
        session['bp'] = request.form.get("bp")
        session['sugar'] = request.form.get("sugar")
        session['symptoms'] = request.form.getlist("symptoms")

        # ---- CLEAR doctor info IF this is individual patient ----
        # If doctor info is NOT provided in this session, remove old doctor info
        if 'doctor_name' not in session:
            session.pop('doctor_name', None)
            session.pop('doctor_specialty', None)
            session.pop('doctor_license', None)

        return redirect(url_for("upload_mri"))

    return render_template("patient_info.html")

# ------------------------------
# MRI Upload Page
# ------------------------------
@app.route("/upload_mri", methods=["GET", "POST"])
def upload_mri():
    if request.method == "POST":
        # Save uploaded MRI file
        file = request.files["image"]
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Store filename in session
        session['mri_file'] = filename

        # Redirect to predict route
        return redirect(url_for("predict"))

    # GET request: pass patient info from session to template
    patient_name = session.get("patient_name", "")
    age = session.get("age", "")
    gender = session.get("gender", "")
    bp = session.get("bp", "")
    sugar = session.get("sugar", "")
    symptoms = session.get("symptoms", [])

    return render_template("upload_mri.html",
                           patient_name=patient_name,
                           age=age,
                           gender=gender,
                           bp=bp,
                           sugar=sugar,
                           symptoms=symptoms)

# ------------------------------
# Upload page
# ------------------------------
@app.route("/upload", methods=["GET", "POST"])
def upload_page():
    if request.method == "POST":
        user_type = request.form.get("user_type")

        # --------- Individual flow ----------
        if user_type == "individual":
            # Get data from form
            patient_name = request.form.get("patient_name")
            age = request.form.get("age")
            gender = request.form.get("gender")
            bp = request.form.get("bp")
            sugar = request.form.get("sugar")
            symptoms = request.form.get("symptoms")

            # Render MRI upload page
            return render_template("upload.html",
                                   step="mri_upload",
                                   user_type="individual",
                                   patient_name=patient_name,
                                   age=age,
                                   gender=gender,
                                   bp=bp,
                                   sugar=sugar,
                                   symptoms=symptoms.split(","))

        # --------- Doctor flow ----------
        elif user_type == "doctor":
            # Step 1: Doctor info submitted, now render patient info form
            if "doctor_name" in request.form and "hospital" in request.form and "license" in request.form and "patient_name" not in request.form:
                doctor_name = request.form.get("doctor_name")
                hospital = request.form.get("hospital")
                license = request.form.get("license")
                return render_template("upload.html",
                                       step="patient_info",
                                       doctor_name=doctor_name,
                                       hospital=hospital,
                                       license=license)
            # Step 2: Patient info submitted, now render MRI upload page
            elif "patient_name" in request.form:
                doctor_name = request.form.get("doctor_name")
                hospital = request.form.get("hospital")
                license = request.form.get("license")
                patient_name = request.form.get("patient_name")
                age = request.form.get("age")
                gender = request.form.get("gender")
                bp = request.form.get("bp")
                sugar = request.form.get("sugar")
                symptoms = request.form.get("symptoms").split(",")
                return render_template("upload.html",
                                       step="mri_upload",
                                       user_type="doctor",
                                       doctor_name=doctor_name,
                                       hospital=hospital,
                                       license=license,
                                       patient_name=patient_name,
                                       age=age,
                                       gender=gender,
                                       bp=bp,
                                       sugar=sugar,
                                       symptoms=symptoms)

    # Default GET request -> decide step based on query params or initial selection
    user_type = request.args.get("user_type")
    if user_type == "doctor":
        return render_template("upload.html", step="doctor_info")
    else:
        return render_template("upload.html", step="individual_info")
    


# ------------------------------
# IMAGE PREDICTION FUNCTION
# ------------------------------
def predict_blockage(image_path):
    img = Image.open(image_path).convert("RGB").resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return model.predict(img)[0][0]


from fpdf import FPDF
from flask import session
import datetime

# ------------------------------------------------------
#  PDF GENERATION FUNCTION
# ------------------------------------------------------


def generate_pdf(patient_name, age, gender, bp, sugar, symptoms, diagnosis,
                 doctor_name=None, doctor_specialty=None, doctor_license=None):
    from fpdf import FPDF
    import os
    import datetime

    
    # print("Diagnosis going to PDF:", diagnosis)
    REPORT_FOLDER = "reports"
    os.makedirs(REPORT_FOLDER, exist_ok=True)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    y = 10  # initial y position

    # -------------------
    # Border
    # -------------------
    pdf.set_draw_color(180, 180, 180)
    pdf.set_line_width(0.8)
    pdf.rect(8, 8, 194, 280)

    # -------------------
    # Date
    # -------------------
    pdf.set_font("Arial", "", 12)
    pdf.set_xy(140, y)
    pdf.cell(0, 10, f"Date: {datetime.date.today()}", ln=False, align="R")
    y += 15


    # -------------------
    # Title
    # -------------------
    pdf.set_font("Arial", "B", 18)
    pdf.set_xy(8, y)
    pdf.cell(0, 10, "Medical Blockage Report", ln=True, align="C")
    y += 12

    # -------------------
    # Doctor Info
    # -------------------
    
    if doctor_name and doctor_name.strip() != "":
        pdf.set_font("Arial", "B", 14)
        pdf.set_xy(10, y)
        pdf.cell(0, 8, "Doctor Information:", ln=True)
        y += 8

        pdf.set_font("Arial", "", 12)
        pdf.set_xy(10, y)
        pdf.cell(0, 6, f"Name: {doctor_name}", ln=True)
        y += 6
        if doctor_specialty:
            pdf.set_xy(10, y)
            pdf.cell(0, 6, f"Specialty: {doctor_specialty}", ln=True)
            y += 6
        if doctor_license:
            pdf.set_xy(10, y)
            pdf.cell(0, 6, f"License: {doctor_license}", ln=True)
            y += 8

        pdf.set_line_width(0.5)
        pdf.line(10, y, 198, y)
        y += 5

    # -------------------
    # Patient Info
    # -------------------
    pdf.set_font("Arial", "B", 14)
    pdf.set_xy(10, y)
    pdf.cell(0, 8, "Patient Information:", ln=True)
    y += 8

    pdf.set_font("Arial", "", 12)
    pdf.set_xy(10, y)
    pdf.cell(0, 6, f"Name: {patient_name}", ln=True)
    y += 6
    pdf.set_xy(10, y)
    pdf.cell(0, 6, f"Age: {age}", ln=True)
    y += 6
    pdf.set_xy(10, y)
    pdf.cell(0, 6, f"Gender: {gender}", ln=True)
    y += 6
    pdf.set_xy(10, y)
    pdf.cell(0, 6, f"Blood Pressure: {bp}", ln=True)
    y += 6
    pdf.set_xy(10, y)
    pdf.cell(0, 6, f"Sugar Level: {sugar}", ln=True)
    y += 8

    pdf.line(10, y, 198, y)
    y += 5

    # -------------------
    # Health Status
    # -------------------
    pdf.set_font("Arial", "B", 14)
    pdf.set_xy(10, y)
    pdf.cell(0, 8, "Health Status Summary:", ln=True)
    y += 8

    pdf.set_font("Arial", "", 12)

    # BP Status
    bp_status = "Normal"
    try:
        bp_value = int(bp)
        if bp_value < 90:
            bp_status = "Low"
        elif bp_value > 120:
            bp_status = "High"
    except:
        bp_status = "Unknown"

    pdf.set_xy(10, y)
    pdf.cell(0, 6, f"Blood Pressure Status: {bp_status}", ln=True)
    y += 6

    # Sugar Status
    sugar_status = "Normal"
    try:
        sugar_value = int(sugar)
        if sugar_value < 90:
            sugar_status = "Low"
        elif sugar_value > 140:
            sugar_status = "High"
    except:
        sugar_status = "Unknown"

    pdf.set_xy(10, y)
    pdf.cell(0, 6, f"Sugar Level Status: {sugar_status}", ln=True)
    y += 8

    pdf.line(10, y, 198, y)
    y += 5

    # -------------------
    # Symptoms
    # -------------------
    pdf.set_font("Arial", "B", 14)
    pdf.set_xy(10, y)
    pdf.cell(0, 8, "Symptoms:", ln=True)
    y += 8

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 6, ", ".join(symptoms))
    y = pdf.get_y() + 2

    pdf.line(10, y, 198, y)
    y += 5

    # -------------------
    # MRI Results (FIXED)
    # -------------------
    pdf.set_font("Arial", "B", 14)
    pdf.set_xy(10, y)
    pdf.cell(0, 8, "MRI Results:", ln=True)
    y += 8

    pdf.set_font("Arial", "B", 12)

    
    # --------- FIXED PART ---------
    # print("diagnosis in condition:"+diagnosis.lower().strip())
    diagnosis_clean = diagnosis.split("\n")[0].lower().strip().replace(".", "")
    # print("diagnosisclean"+diagnosis_clean)
    if diagnosis_clean == "blockage detected":
        pdf.multi_cell(0, 6, "Blockage Detected: Immediate medical attention suggested.")
        y = pdf.get_y()
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 6,
            "The MRI scan shows clear signs of blockage. Please consult a specialist immediately for further evaluation and required treatment."
        )
    else:
        pdf.multi_cell(0, 6, "No Blockage Detected.")
        y = pdf.get_y()
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 6,
            "The MRI shows no signs of blockage. Continue maintaining healthy lifestyle habits and do regular checkups for safety."
        )
    # --------- END FIX ---------

    pdf.ln(20)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "This is a computer-generated report and should be verified by a doctor.", ln=True, align="C")

    # Save PDF
    filename = f"report_{patient_name.replace(' ', '_')}.pdf"
    file_path = os.path.join(REPORT_FOLDER, filename)
    pdf.output(file_path)

    return filename
# ------------------------------------------------------
#  
# ----------------------------------------------------------------------------------------

   


# ------------------------------
# PREDICTION + REPORT ROUTE
# ------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    # -------------------------
    # GET PATIENT INFO FROM SESSION
    # -------------------------
    patient_name = session.get("patient_name")
    age = session.get("age")
    gender = session.get("gender")
    bp = session.get("bp")
    sugar = session.get("sugar")
    symptoms = session.get("symptoms", [])

    # -------------------------
    # GET MRI FILE
    # -------------------------
    file = request.files["image"]
    if not file:
        return "No file uploaded!"

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # -------------------------
    # PREDICT
    # -------------------------
    pred_value = predict_blockage(filepath)

    pred_value = predict_blockage(filepath)
# Convert model output to a clear string
    diagnosis = "Blockage Detected. \nImmediate medical attention suggested." if pred_value > 0.5 else "No Blockage.\nEverything looks normal"


    # Fetch doctor info from session if it exists
    doctor_name = session.get('doctor_name', None)
    doctor_specialty = session.get('doctor_specialty', None)
    doctor_license = session.get('doctor_license', None)


    # -------------------------
    # GENERATE REPORT (WITH DOCTOR INFO)
    # -------------------------
    pdf_name = generate_pdf(
        patient_name, age, gender, bp, sugar, symptoms, diagnosis,
        doctor_name=session.get("doctor_name"),
        doctor_specialty=session.get("doctor_specialty"),
        doctor_license=session.get("doctor_license")
    )


    # Clear doctor info from session so it doesn't affect individual reports
    session.pop('doctor_name', None)
    session.pop('doctor_specialty', None)
    session.pop('doctor_license',None)

    # -------------------------
    # SHOW DASHBOARD
    # -------------------------
    return redirect(url_for(
    'summary',
    patient_name=patient_name,
    age=age,
    gender=gender,
    bp=bp,
    sugar=sugar,
    symptoms=",".join(symptoms),
    prediction=diagnosis,
    pdf_name=pdf_name,
    doctor_note="AI-generated medical summary based on MRI scan."
))



@app.route('/summary')
def summary():
    return render_template(
        "summary.html",
        patient_name=request.args.get("patient_name"),
        age=request.args.get("age"),
        gender=request.args.get("gender"),
        bp=request.args.get("bp"),
        sugar=request.args.get("sugar"),
        symptoms=request.args.get("symptoms"),
        prediction=request.args.get("prediction"),
        doctor_note=request.args.get("doctor_note"),
        pdf_name=request.args.get("pdf_name")
    )


@app.route("/reports/<filename>")
def send_report(filename):
    return send_from_directory(REPORT_FOLDER, filename, as_attachment=True)


# ------------------------------
# RUN SERVER
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)