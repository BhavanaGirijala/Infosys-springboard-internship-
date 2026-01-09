import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

# ---------------- DATABASE ----------------
conn = sqlite3.connect("student_app.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    username TEXT,
    study REAL,
    work REAL,
    play REAL,
    sleep REAL,
    marks REAL
)
""")
conn.commit()

# ---------------- AUTH FUNCTIONS ----------------
def signup_user(u, p):
    try:
        c.execute("INSERT INTO users VALUES (?,?)", (u, p))
        conn.commit()
        return True
    except:
        return False

def login_user(u, p):
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (u, p))
    return c.fetchone()

def reset_password(u, p):
    c.execute("UPDATE users SET password=? WHERE username=?", (p, u))
    conn.commit()

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# ---------------- LOGIN PAGE ----------------
if not st.session_state.logged_in:
    st.title("🔐 Student Performance System")

    tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Forgot Password"])

    with tab1:
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if login_user(u, p):
                st.session_state.logged_in = True
                st.session_state.username = u
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        u = st.text_input("New Username", key="signup_user")
        p = st.text_input("New Password", type="password", key="signup_pass")
        if st.button("Create Account"):
            if signup_user(u, p):
                st.success("Account created. Please login.")
            else:
                st.error("Username already exists")

    with tab3:
        u = st.text_input("Username", key="forgot_user")
        p = st.text_input("New Password", type="password", key="forgot_pass")
        if st.button("Reset Password"):
            reset_password(u, p)
            st.success("Password updated")

    st.stop()

# ---------------- DASHBOARD ----------------
st.title("🎓 Student Performance Prediction Dashboard")

# ---------- LOAD DATA ----------
df = pd.read_csv("old_dataset.csv")

features = ["StudyHours", "WorkHours", "PlayHours", "SleepHours"]
X = df[features]
y = df["Marks"]

# ---------------- AI MODEL ----------------
ai_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
ai_model.fit(X, y)

# Accuracy
y_pred = ai_model.predict(X)
accuracy = r2_score(y, y_pred) * 100

# ---------------- CLUSTERING ----------------
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)

# ---------------- RECOMMENDATION ----------------
def recommend(marks):
    if marks >= 85:
        return "🌟 Excellent performance. Maintain consistency."
    elif marks >= 60:
        return "👍 Good performance. Improve study hours."
    else:
        return "⚠️ Needs improvement. Focus more on studies."

# ---------------- MENU ----------------
menu = st.sidebar.radio("Menu", ["Single Prediction", "Bulk Prediction", "Analytics"])

# ---------------- SINGLE PREDICTION ----------------
if menu == "Single Prediction":
    st.subheader("🔹 AI-Based Single Prediction")

    study = st.number_input("Study Hours", 0, 10, 4)
    work = st.number_input("Work Hours", 0, 10, 1)
    play = st.number_input("Play Hours", 0, 10, 2)
    sleep = st.number_input("Sleep Hours", 0, 10, 7)

    if st.button("Predict"):
        input_data = np.array([[study, work, play, sleep]])
        marks = ai_model.predict(input_data)[0]

        c.execute(
            "INSERT INTO predictions VALUES (?,?,?,?,?,?)",
            (st.session_state.username, study, work, play, sleep, marks)
        )
        conn.commit()

        st.success(f"🎯 Predicted Marks: {marks:.2f}")
        st.info(f"📌 Recommendation: {recommend(marks)}")
        st.metric("🤖 AI Model Accuracy", f"{accuracy:.2f}%")

# ---------------- BULK PREDICTION ----------------
elif menu == "Bulk Prediction":
    st.subheader("📂 AI-Based Bulk Prediction")

    file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    if file:
        bulk_df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)

        bulk_df["Predicted Marks"] = ai_model.predict(bulk_df[features])
        bulk_df["Recommendation"] = bulk_df["Predicted Marks"].apply(recommend)

        st.dataframe(bulk_df)

        st.metric("🤖 AI Model Accuracy", f"{accuracy:.2f}%")

        st.download_button(
            "⬇️ Download Predictions",
            bulk_df.to_csv(index=False),
            "bulk_predictions.csv"
        )

# ---------------- ANALYTICS ----------------
else:
    st.subheader("📊 Analytics & Visualization")

    fig1 = px.scatter(
        df,
        x="StudyHours",
        y="Marks",
        color=df["Cluster"].astype(str),
        title="Study Hours vs Marks (Clusters)"
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(
        df,
        x="Marks",
        nbins=10,
        title="Marks Distribution"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.metric("🤖 AI Model Accuracy", f"{accuracy:.2f}%")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("✅ **Milestone 4 | AI-Based Student Performance System | Streamlit**")