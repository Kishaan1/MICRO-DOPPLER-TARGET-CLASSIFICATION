import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "radar_logs.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS classification_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            target_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            doppler_shift_hz REAL,
            mod_bandwidth_hz REAL,
            mod_rate_hz REAL,
            harmonic_ratio REAL,
            timestamp TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def log_classification(filename, target_class, confidence, doppler_shift_hz, mod_bandwidth_hz, mod_rate_hz, harmonic_ratio):
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO classification_logs 
        (filename, target_class, confidence, doppler_shift_hz, mod_bandwidth_hz, mod_rate_hz, harmonic_ratio, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (filename, target_class, confidence, doppler_shift_hz, mod_bandwidth_hz, mod_rate_hz, harmonic_ratio, timestamp_str))
    conn.commit()
    log_id = cursor.lastrowid
    conn.close()
    return log_id

def get_history(limit=50):
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, filename, target_class, confidence, doppler_shift_hz, mod_bandwidth_hz, mod_rate_hz, harmonic_ratio, timestamp
        FROM classification_logs
        ORDER BY id DESC
        LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_stats():
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM classification_logs')
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM classification_logs WHERE target_class = 'Drone'")
    drones = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM classification_logs WHERE target_class = 'Bird'")
    birds = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM classification_logs WHERE target_class = 'Noise'")
    noise = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(confidence) FROM classification_logs")
    avg_conf_row = cursor.fetchone()[0]
    avg_confidence = round(avg_conf_row, 2) if avg_conf_row else 0.0

    conn.close()
    return {
        "total_scans": total,
        "drone_threats": drones,
        "bird_targets": birds,
        "noise_clutter": noise,
        "avg_confidence": avg_confidence
    }

def clear_history():
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM classification_logs')
    conn.commit()
    conn.close()
    return True
