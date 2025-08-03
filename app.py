from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import os
import joblib

app = Flask(__name__)
app.secret_key = 'your_secret_key'

MODEL_FILE = "model/model.pkl"
USER_FILE = "users.csv"
DATA_FILE = "data.csv"

# Home route
@app.route('/')
def home():
    return redirect(url_for('login'))

# Register new user
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        password = request.form['password']

        users = pd.read_csv(USER_FILE) if os.path.exists(USER_FILE) else pd.DataFrame(columns=["name", "password"])

        if name in users['name'].values:
            return render_template("login.html", error="User already exists")

        new_user = pd.DataFrame([{"name": name, "password": password}])
        users = pd.concat([users, new_user], ignore_index=True)
        users.to_csv(USER_FILE, index=False)

        return render_template("login.html", success="Registration successful. Please log in.")

    return render_template("login.html")

# Login existing user
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        name = request.form['name']
        password = request.form['password']

        if not os.path.exists(USER_FILE):
            return render_template("login.html", error="No users found.")

        users = pd.read_csv(USER_FILE)
        user = users[(users['name'] == name) & (users['password'] == password)]

        if not user.empty:
            session['user'] = name
            return redirect(url_for('dashboard'))
        else:
            return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")

# Dashboard to predict
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))

    prediction = None
    if request.method == 'POST':
        age = request.form['age']
        symptoms = request.form['symptoms']

        # Load model and vectorizer
        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load("model/vectorizer.pkl")

        features = vectorizer.transform([symptoms])
        prediction = model.predict(features)[0]

        # Save to CSV
        data = pd.DataFrame([{
            "Name": session['user'],
            "Age": age,
            "Symptoms": symptoms,
            "Predicted Disease": prediction
        }])

        if os.path.exists(DATA_FILE):
            existing = pd.read_csv(DATA_FILE)
            data = pd.concat([existing, data], ignore_index=True)

        data.to_csv(DATA_FILE, index=False)

    return render_template("dashboard.html", name=session['user'], prediction=prediction)

# View personal records
@app.route('/records')
def my_records():
    if 'user' not in session:
        return redirect(url_for('login'))

    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        if "Name" in df.columns:
            user_df = df[df['Name'] == session['user']]
            records = user_df.to_dict(orient='records')
        else:
            records = []
    else:
        records = []

    return render_template("records.html", records=records)

# Alias for url_for('my_records')
app.add_url_rule('/records', 'my_records', my_records)

# Logout
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)