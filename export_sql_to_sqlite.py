import pyodbc
import sqlite3
import json
import struct
import sys
import os

# Configuration
SQL_SERVER = '(local)'  # Try (local) or localhost
DATABASE = 'EduBusDB'   # Guessing the name, will try to find it if fails
SQLITE_DB = 'students.db'

def get_sql_connection():
    drivers = [driver for driver in pyodbc.drivers() if 'SQL Server' in driver]
    if not drivers:
        print("❌ No ODBC drivers found!")
        sys.exit(1)
    
    driver = drivers[0]
    print(f"Using ODBC Driver: {driver}")
    
    try:
        # Try connecting to master to find database if needed
        conn_str = f'DRIVER={{{driver}}};SERVER={SQL_SERVER};DATABASE={DATABASE};Trusted_Connection=yes;'
        return pyodbc.connect(conn_str)
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("Trying localhost...")
        try:
            conn_str = f'DRIVER={{{driver}}};SERVER=localhost;DATABASE={DATABASE};Trusted_Connection=yes;'
            return pyodbc.connect(conn_str)
        except Exception as e2:
            print(f"❌ Connection failed: {e2}")
            sys.exit(1)

def main():
    print(f"Exporting from SQL Server ({DATABASE}) to SQLite ({SQLITE_DB})...")
    
    # 1. Connect to SQL Server
    cnxn = get_sql_connection()
    cursor = cnxn.cursor()
    
    # 2. Query Data
    # Note: Using FirstName + LastName for FullName
    query = """
    SELECT s.Id, s.FirstName, s.LastName, f.EmbeddingJson
    FROM Students s
    JOIN FaceEmbeddings f ON s.Id = f.StudentId
    WHERE f.ModelVersion = 'MobileFaceNet_V1'
    """
    
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
    except Exception as e:
        print(f"❌ Query failed: {e}")
        # Try to list tables to debug
        print("Listing tables...")
        cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES")
        for row in cursor.fetchall():
            print(f" - {row[0]}")
        sys.exit(1)
        
    print(f"✓ Found {len(rows)} students with embeddings.")
    
    # 3. Create SQLite DB
    if os.path.exists(SQLITE_DB):
        os.remove(SQLITE_DB)
        
    sqlite_conn = sqlite3.connect(SQLITE_DB)
    sqlite_cursor = sqlite_conn.cursor()
    
    sqlite_cursor.execute("""
        CREATE TABLE students (
            id TEXT PRIMARY KEY,
            name TEXT,
            embedding BLOB
        )
    """)
    
    count = 0
    for row in rows:
        student_id = str(row.Id)
        first_name = row.FirstName
        last_name = row.LastName
        full_name = f"{first_name} {last_name}".strip()
        embedding_json = row.EmbeddingJson
        
        try:
            embedding_list = json.loads(embedding_json)
            # Convert to float32 bytes
            embedding_bytes = struct.pack(f'{len(embedding_list)}f', *embedding_list)
            
            sqlite_cursor.execute(
                "INSERT INTO students (id, name, embedding) VALUES (?, ?, ?)",
                (student_id, full_name, embedding_bytes)
            )
            count += 1
        except Exception as e:
            print(f"⚠️ Error processing student {full_name}: {e}")
            
    sqlite_conn.commit()
    sqlite_conn.close()
    cnxn.close()
    
    print(f"\n✅ Successfully exported {count} students to {SQLITE_DB}")
    print(f"Now run: scp {SQLITE_DB} edubus@192.168.1.202:~/jetson-face-recognition/data/")

if __name__ == "__main__":
    main()
