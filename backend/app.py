
import os
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd

app = Flask(__name__)
CORS(app) # Enable CORS for the frontend to communicate with the backend

# Define the path to the database and exports directory
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BACKEND_DIR, 'emotion_tracker.db')
EXPORTS_DIR = os.path.join(BACKEND_DIR, '../exports')
os.makedirs(EXPORTS_DIR, exist_ok=True)

def init_db():
    """Initializes the SQLite database with users and emotions tables."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Drop existing tables to ensure the new schema is applied
    c.execute("DROP TABLE IF EXISTS users")
    c.execute("DROP TABLE IF EXISTS emotions")
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS emotions (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            username TEXT NOT NULL,
            emotion TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()

# Initialize the database on startup
init_db()

@app.route('/signup', methods=['POST'])
def signup():
    """Handles user registration. Hashes the password and stores the new user in the database."""
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'message': 'Username and password are required.'}), 400

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        # Hash the password for secure storage
        password_hash = generate_password_hash(password)
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
        conn.commit()
        return jsonify({'message': 'User registered successfully!'}), 201
    except sqlite3.IntegrityError:
        return jsonify({'message': 'Username already exists.'}), 409
    finally:
        conn.close()

@app.route('/login', methods=['POST'])
def login():
    """Handles user authentication. Verifies the password against the stored hash."""
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'message': 'Username and password are required.'}), 400

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()

    if user and check_password_hash(user[1], password):
        return jsonify({'message': 'Login successful!', 'user_id': user[0]}), 200
    else:
        return jsonify({'message': 'Invalid username or password.'}), 401

@app.route('/emotion', methods=['POST'])
def save_emotion():
    """Saves a new emotion entry with user ID, username and timestamp."""
    data = request.json
    user_id = data.get('user_id')
    username = data.get('username')
    emotion = data.get('emotion')
    timestamp = data.get('timestamp')

    if not user_id or not username or not emotion or not timestamp:
        return jsonify({'message': 'Missing data.'}), 400

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO emotions (user_id, username, emotion, timestamp) VALUES (?, ?, ?, ?)", (user_id, username, emotion, timestamp))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Emotion saved successfully!'}), 201

@app.route('/history/<int:user_id>', methods=['GET'])
def get_history(user_id):
    """Fetches the entire emotion history for a given user ID."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT username, emotion, timestamp FROM emotions WHERE user_id = ? ORDER BY timestamp DESC", (user_id,))
    history = c.fetchall()
    conn.close()

    history_list = [{'username': row[0], 'emotion': row[1], 'timestamp': row[2]} for row in history]
    return jsonify(history_list), 200

@app.route('/history/<int:user_id>/download', methods=['GET'])
def download_history(user_id):
    """Generates a CSV file of the user's emotion history and allows download."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT username, emotion, timestamp FROM emotions WHERE user_id = {user_id}", conn)
    conn.close()

    if df.empty:
        return jsonify({'message': 'No data to download.'}), 404

    # Generate a unique filename and save to the exports folder
    filename = f"emotion_history_{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    filepath = os.path.join(EXPORTS_DIR, filename)
    df.to_csv(filepath, index=False)
    
    # In a real-world scenario, you would send this file back.
    # For this simplified example, we'll just indicate it was saved.
    return jsonify({'message': 'CSV file generated successfully!', 'filepath': filepath}), 200

if __name__ == '__main__':
    app.run(port=5000, debug=True)
