import sqlite3
import hashlib

def get_connection():
    return sqlite3.connect("student.db", check_same_thread=False)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_tables():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        study REAL,
        work REAL,
        play REAL,
        sleep REAL,
        predicted_marks REAL,
        cluster INTEGER
    )
    """)

    conn.commit()
    conn.close()