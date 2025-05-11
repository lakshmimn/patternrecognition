import sqlite3
from datetime import datetime
import json
import os

class MetricsTracker:
    def __init__(self):
        self.db_path = 'metrics.db'
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create metrics table
        c.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                event_type TEXT,
                details TEXT
            )
        ''')
        
        # Create daily_stats table
        c.execute('''
            CREATE TABLE IF NOT EXISTS daily_stats (
                date DATE PRIMARY KEY,
                page_views INTEGER DEFAULT 0,
                analyses_performed INTEGER DEFAULT 0,
                files_uploaded INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_event(self, event_type, details=None):
        """Log an event with timestamp"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Log the event
        c.execute('''
            INSERT INTO metrics (timestamp, event_type, details)
            VALUES (?, ?, ?)
        ''', (datetime.now(), event_type, json.dumps(details) if details else None))
        
        # Update daily stats
        today = datetime.now().date()
        if event_type == 'page_view':
            c.execute('''
                INSERT INTO daily_stats (date, page_views)
                VALUES (?, 1)
                ON CONFLICT(date) DO UPDATE SET page_views = page_views + 1
            ''', (today,))
        elif event_type == 'analysis':
            c.execute('''
                INSERT INTO daily_stats (date, analyses_performed)
                VALUES (?, 1)
                ON CONFLICT(date) DO UPDATE SET analyses_performed = analyses_performed + 1
            ''', (today,))
        elif event_type == 'file_upload':
            c.execute('''
                INSERT INTO daily_stats (date, files_uploaded)
                VALUES (?, 1)
                ON CONFLICT(date) DO UPDATE SET files_uploaded = files_uploaded + 1
            ''', (today,))
        
        conn.commit()
        conn.close()
    
    def get_daily_stats(self, days=7):
        """Get statistics for the last n days"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            SELECT date, page_views, analyses_performed, files_uploaded
            FROM daily_stats
            ORDER BY date DESC
            LIMIT ?
        ''', (days,))
        
        stats = c.fetchall()
        conn.close()
        
        return [{
            'date': row[0],
            'page_views': row[1],
            'analyses_performed': row[2],
            'files_uploaded': row[3]
        } for row in stats]
    
    def get_total_stats(self):
        """Get total statistics"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            SELECT 
                SUM(page_views) as total_views,
                SUM(analyses_performed) as total_analyses,
                SUM(files_uploaded) as total_uploads
            FROM daily_stats
        ''')
        
        stats = c.fetchone()
        conn.close()
        
        return {
            'total_views': stats[0] or 0,
            'total_analyses': stats[1] or 0,
            'total_uploads': stats[2] or 0
        } 