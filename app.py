import streamlit as st
import pandas as pd
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance & Attention Prediction",
    layout="wide"
)

# ---------------- DATABASE ----------------
conn = sqlite3.connect("student_app.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
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
    c.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (u, p)
    )
    return c.fetchone()

def reset_password(u, p):
    c.execute(
        "UPDATE users SET password=? WHERE username=?",
        (p, u)
    )
    conn.commit()

# ---------------- MARKS CALCULATION ----------------
def calculate_marks(study, work, play, sleep, attention):
    base = (study * 10) + (sleep * 5) - (play * 2) - (work * 1)

    if attention == "High":
        base += 15
    elif attention == "Medium":
        base += 8
    else:
        base += 2

    return min(max(base, 0), 100)

# ---------------- RECOMMENDATION ----------------
def get_recommendation(attention, marks):
    if attention == "High" and marks >= 80:
        return "🌟 Excellent performance. Maintain your routine."
    elif attention == "Medium":
        return "👍 Good performance. Improve focus and consistency."
    else:
        return "⚠️ Needs improvement. Increase study time and sleep quality."

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- LOGIN PAGE ----------------
if not st.session_state.logged_in:
    st.title("🔐 Student Performance System")

    tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Forgot Password"])

    with tab1:
        u = st.text_input("Username", key="login_u")
        p = st.text_input("Password", type="password", key="login_p")
        if st.button("Login"):
            if login_user(u, p):
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        u = st.text_input("New Username", key="signup_u")
        p = st.text_input("New Password", type="password", key="signup_p")
        if st.button("Create Account"):
            if signup_user(u, p):
                st.success("Account created successfully")
            else:
                st.error("Username already exists")

    with tab3:
        u = st.text_input("Username", key="forgot_u")
        p = st.text_input("New Password", type="password", key="forgot_p")
        if st.button("Reset Password"):
            reset_password(u, p)
            st.success("Password updated")

    st.stop()

# ---------------- DATASET (NO NAMES) ----------------
df = pd.DataFrame({
    "StudyHours": [4,6,3,2,5,1,7,4,2,4,6,5,3,7,1,4,6,2],
    "WorkHours":  [0,1,0,2,0,3,1,1,3,2,1,2,3,1,4,2,1,3],
    "PlayHours":  [2,1,3,4,2,5,0,2,3,2,1,2,3,1,4,2,1,3],
    "SleepHours": [7,6,8,5,7,4,6,7,8,7,6,7,8,6,9,7,6,8],
    "Attention_Level": [
        "Medium","High","Medium","Low","High","Low","High","Medium",
        "Low","Medium","High","High","Medium","High","Low","Medium","High","Low"
    ]
})

# ---------------- MODEL PREP (BACKGROUND) ----------------
le = LabelEncoder()
df["Attention_Label"] = le.fit_transform(df["Attention_Level"])

X = df[["StudyHours", "WorkHours", "PlayHours", "SleepHours"]]
y = df["Attention_Label"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

accuracy = model.score(X, y) * 100

# ---------------- SIDEBAR ----------------
menu = st.sidebar.radio(
    "Menu",
    [
        "Project Overview",
        "Single Student Prediction",
        "Bulk Prediction",
        "Future Student Prediction",
        "Analytics"
    ]
)

# ---------------- PROJECT OVERVIEW ----------------
if menu == "Project Overview":
    st.title("📘 Project Overview")
    st.write("""
    This AI-based system predicts:
    - **Student Attention Level**
    - **Student Marks (derived logically)**

    It supports:
    - Single student prediction
    - Bulk prediction
    - Future student prediction

    The system provides **personalized recommendations**.
    """)
    st.metric("Model Accuracy", f"{accuracy:.2f}%")

# ---------------- SINGLE STUDENT ----------------
elif menu == "Single Student Prediction":
    st.subheader("🎯 Single Student Prediction")

    study = st.number_input("Study Hours", 0, 10, 4)
    work = st.number_input("Work Hours", 0, 10, 1)
    play = st.number_input("Play Hours", 0, 10, 2)
    sleep = st.number_input("Sleep Hours", 0, 12, 7)

    if st.button("Predict"):
        att_label = model.predict([[study, work, play, sleep]])[0]
        attention = le.inverse_transform([att_label])[0]
        marks = calculate_marks(study, work, play, sleep, attention)

        st.success(f"Attention Level: {attention}")
        st.success(f"Predicted Marks: {marks:.2f}")
        st.info(get_recommendation(attention, marks))

# ---------------- BULK PREDICTION ----------------
elif menu == "Bulk Prediction":
    st.subheader("📂 Bulk Prediction")

    file = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])
    if file:
        bulk_df = (
            pd.read_excel(file)
            if file.name.endswith("xlsx")
            else pd.read_csv(file)
        )

        preds = model.predict(
            bulk_df[["StudyHours","WorkHours","PlayHours","SleepHours"]]
        )
        bulk_df["Attention_Level"] = le.inverse_transform(preds)

        bulk_df["Predicted_Marks"] = bulk_df.apply(
            lambda r: calculate_marks(
                r["StudyHours"],
                r["WorkHours"],
                r["PlayHours"],
                r["SleepHours"],
                r["Attention_Level"]
            ),
            axis=1
        )

        bulk_df["Recommendation"] = bulk_df.apply(
            lambda r: get_recommendation(
                r["Attention_Level"],
                r["Predicted_Marks"]
            ),
            axis=1
        )

        st.dataframe(bulk_df)

        st.download_button(
            "⬇️ Download Results",
            bulk_df.to_csv(index=False),
            "bulk_predictions.csv"
        )

# ---------------- FUTURE STUDENT ----------------
elif menu == "Future Student Prediction":
    st.subheader("🔮 Future Student Prediction")

    student_name = st.text_input("Student Name")

    study = st.number_input("Expected Study Hours", 0.0, 10.0)
    work = st.number_input("Expected Work Hours", 0.0, 10.0)
    play = st.number_input("Expected Play Hours", 0.0, 10.0)
    sleep = st.number_input("Expected Sleep Hours", 0.0, 12.0)

    if st.button("Predict Future Performance"):
        att_label = model.predict([[study, work, play, sleep]])[0]
        attention = le.inverse_transform([att_label])[0]
        marks = calculate_marks(study, work, play, sleep, attention)

        st.success(f"Student: {student_name}")
        st.success(f"Predicted Attention Level: {attention}")
        st.success(f"Expected Marks: {marks:.2f}")
        st.info(get_recommendation(attention, marks))

# ---------------- ANALYTICS ----------------
elif menu == "Analytics":
    st.subheader("📊 Analytics")

    fig1 = px.histogram(
        df,
        x="Attention_Level",
        title="Attention Level Distribution"
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(
        df,
        x="StudyHours",
        y="SleepHours",
        color="Attention_Level",
        title="Study Hours vs Sleep Hours"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "✅ **Milestone 4 | Attention + Marks + Bulk + Future Prediction (Internship Ready)**"
)