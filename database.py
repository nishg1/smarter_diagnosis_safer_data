import sqlite3
import bcrypt
from datetime import datetime
import json

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  profile_data TEXT)''')
    
    # Create documents table
    c.execute('''CREATE TABLE IF NOT EXISTS documents
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  filename TEXT NOT NULL,
                  file_type TEXT NOT NULL,
                  upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  file_path TEXT NOT NULL,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Create actions table
    c.execute('''CREATE TABLE IF NOT EXISTS actions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  action_type TEXT NOT NULL,
                  action_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  details TEXT,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    conn.commit()
    conn.close()

def create_user(username, password, email, profile_data=None):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Hash the password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    try:
        c.execute('''INSERT INTO users (username, password, email, profile_data)
                     VALUES (?, ?, ?, ?)''',
                  (username, hashed_password, email, profile_data))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    c.execute('SELECT id, password FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    conn.close()
    
    if result:
        user_id, hashed_password = result
        if bcrypt.checkpw(password.encode('utf-8'), hashed_password):
            return user_id
    return None

def get_user_profile(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    c.execute('SELECT id, username, email, created_at, profile_data FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    conn.close()
    
    if result:
        return {
            'id': result[0],
            'username': result[1],
            'email': result[2],
            'created_at': result[3],
            'profile_data': result[4]
        }
    return None

def update_profile(username, profile_data):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    c.execute('UPDATE users SET profile_data = ? WHERE username = ?', (profile_data, username))
    conn.commit()
    conn.close()

def log_action(user_id, action_type, details=None):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    details_json = json.dumps(details) if details else None
    c.execute('''INSERT INTO actions (user_id, action_type, details)
                 VALUES (?, ?, ?)''', (user_id, action_type, details_json))
    conn.commit()
    conn.close()

def get_user_actions(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    c.execute('''SELECT action_type, action_date, details 
                 FROM actions 
                 WHERE user_id = ? 
                 ORDER BY action_date DESC''', (user_id,))
    actions = c.fetchall()
    conn.close()
    
    return [{
        'type': action[0],
        'date': action[1],
        'details': json.loads(action[2]) if action[2] else None
    } for action in actions]

def save_document(user_id, filename, file_type, file_path):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    c.execute('''INSERT INTO documents (user_id, filename, file_type, file_path)
                 VALUES (?, ?, ?, ?)''', (user_id, filename, file_type, file_path))
    conn.commit()
    conn.close()

def get_user_documents(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    c.execute('''SELECT filename, file_type, upload_date, file_path 
                 FROM documents 
                 WHERE user_id = ? 
                 ORDER BY upload_date DESC''', (user_id,))
    documents = c.fetchall()
    conn.close()
    
    return [{
        'filename': doc[0],
        'type': doc[1],
        'upload_date': doc[2],
        'path': doc[3]
    } for doc in documents]

# Initialize the database when the module is imported
init_db() 