import os
import sqlite3
import face_recognition
import pickle

def setup_db():
    conn = sqlite3.connect('face_system.db')
    c = conn.cursor()
    
    # Updated Table: Added 'age' and 'gender'
    c.execute('''CREATE TABLE IF NOT EXISTS people (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    age TEXT,
                    gender TEXT,
                    encoding BLOB
                )''')

    c.execute('''CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    image_path TEXT,
                    FOREIGN KEY(person_id) REFERENCES people(id)
                )''')
    
    conn.commit()
    return conn

def enroll_people(gallery_path='gallery_db'):
    # Delete old DB to ensure clean slate (Optional, but recommended if schema changes)
    if os.path.exists('face_system.db'):
        print("Note: Appending to existing database.")
        
    conn = setup_db()
    c = conn.cursor()
    
    for person_name in os.listdir(gallery_path):
        person_folder = os.path.join(gallery_path, person_name)
        
        if not os.path.isdir(person_folder):
            continue

        # Check if person already exists to avoid re-entering info
        c.execute("SELECT name FROM people WHERE name=?", (person_name,))
        if c.fetchone():
            print(f"Skipping {person_name} (Already in DB).")
            continue

        print(f"\n--- Found New Person: {person_name} ---")
        images_in_folder = os.listdir(person_folder)
        
        if not images_in_folder:
            continue
            
        # 1. Ask for Details
        age = input(f"Enter Age for {person_name}: ")
        gender = input(f"Enter Gender for {person_name}: ")

        # 2. Process First Image for Encoding
        ref_image_path = os.path.join(person_folder, images_in_folder[0])
        image = face_recognition.load_image_file(ref_image_path)
        
        try:
            encoding = face_recognition.face_encodings(image)[0]
            encoding_blob = pickle.dumps(encoding)
            
            # Insert with new fields
            c.execute("INSERT INTO people (name, age, gender, encoding) VALUES (?, ?, ?, ?)", 
                      (person_name, age, gender, encoding_blob))
            person_id = c.lastrowid
            
            # 3. Save all images
            for img_file in images_in_folder:
                full_path = os.path.join(person_folder, img_file)
                c.execute("INSERT INTO images (person_id, image_path) VALUES (?, ?)", (person_id, full_path))
                
            print(f" -> Saved {person_name} ({age}, {gender}) with {len(images_in_folder)} images.")
            
        except IndexError:
            print(f" -> No face found in {ref_image_path}. Skipping.")

    conn.commit()
    conn.close()
    print("\nDatabase setup complete.")

if __name__ == "__main__":
    enroll_people()