import face_recognition
import cv2
import sqlite3
import pickle
import numpy as np
import os
import time

# --- CONFIGURATION ---
STRICT_TOLERANCE = 0.40
GALLERY_ROOT = "gallery_db"
FRAME_SKIP = 3
BOX_PADDING = 30  # Increase to make box bigger

def get_db_connection():
    return sqlite3.connect('face_system.db')

def get_known_faces():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id, name, age, gender, encoding FROM people")
    rows = c.fetchall()
    conn.close()

    known_faces = []
    for r in rows:
        known_faces.append({
            "id": r[0], "name": r[1], "age": r[2], "gender": r[3],
            "encoding": pickle.loads(r[4])
        })
    return known_faces

def get_gallery_collage(person_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT image_path FROM images WHERE person_id=?", (person_id,))
    paths = [row[0] for row in c.fetchall()]
    conn.close()

    images = []
    for p in paths:
        if os.path.exists(p):
            img = cv2.imread(p)
            if img is not None:
                img = cv2.resize(img, (150, 150))
                images.append(img)
    if not images: return None
    return np.hstack(images)

def add_new_person(frame, face_encoding):
    cv2.destroyAllWindows()
    try:
        name = input("\n-> Enter Name: ").strip()
        if not name: return False
        age = input("-> Enter Age: ").strip()
        gender = input("-> Enter Gender: ").strip()
        
        person_dir = os.path.join(GALLERY_ROOT, name)
        if not os.path.exists(person_dir): os.makedirs(person_dir)
        
        full_path = os.path.join(person_dir, f"{name}_{int(time.time())}.jpg")
        cv2.imwrite(full_path, frame)
        
        conn = get_db_connection()
        c = conn.cursor()
        encoding_blob = pickle.dumps(face_encoding)
        c.execute("INSERT INTO people (name, age, gender, encoding) VALUES (?, ?, ?, ?)", 
                  (name, age, gender, encoding_blob))
        person_id = c.lastrowid
        c.execute("INSERT INTO images (person_id, image_path) VALUES (?, ?)", (person_id, full_path))
        conn.commit()
        conn.close()
        print("-> Saved!")
        return True
    except:
        return False

def main():
    if not os.path.exists(GALLERY_ROOT): os.makedirs(GALLERY_ROOT)
    
    known_people = get_known_faces()
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    face_locations = []
    face_encodings = []
    face_names = []
    current_face_data = [] 
    
    last_seen_person_id = None
    cached_collage = None

    print("System Started. Press 'a' to add face, 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret: break

        # --- FIX: FLIP THE FRAME HORIZONTALLY (Mirror View) ---
        frame = cv2.flip(frame, 1)
        # ------------------------------------------------------

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        frame_count += 1
        
        if frame_count % FRAME_SKIP == 0:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            current_face_data = []
            known_encodings = [p["encoding"] for p in known_people]
            found_known_person_this_frame = False

            for face_encoding, face_location in zip(face_encodings, face_locations):
                name = "Unknown"
                info = "Unknown (Press 'a')"
                color = (0, 0, 255)
                collage_to_show = None

                if known_encodings:
                    distances = face_recognition.face_distance(known_encodings, face_encoding)
                    best_match_index = np.argmin(distances)
                    if distances[best_match_index] < STRICT_TOLERANCE:
                        p = known_people[best_match_index]
                        name = p["name"]
                        info = f"{name} | {p['age']} | {p['gender']}"
                        color = (0, 255, 0)
                        
                        if p["id"] != last_seen_person_id:
                            cached_collage = get_gallery_collage(p["id"])
                            last_seen_person_id = p["id"]
                        collage_to_show = cached_collage
                        found_known_person_this_frame = True

                face_names.append((name, info, color, collage_to_show))
                current_face_data.append((face_encoding, face_location))
            
            if not found_known_person_this_frame:
                last_seen_person_id = None
                cached_collage = None
                try: cv2.destroyWindow("Database Matches")
                except: pass

        for (top, right, bottom, left), (name, info, color, collage) in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Apply Padding
            top = max(0, top - BOX_PADDING)
            left = max(0, left - BOX_PADDING)
            bottom = min(frame.shape[0], bottom + BOX_PADDING)
            right = min(frame.shape[1], right + BOX_PADDING)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, info, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            if collage is not None:
                cv2.imshow("Database Matches", collage)

        cv2.imshow('Live Scanner', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('a'):
            if len(current_face_data) == 1:
                if add_new_person(frame, current_face_data[0][0]):
                    known_people = get_known_faces()
                    last_seen_person_id = None
            elif len(current_face_data) == 0:
                print("No face found.")

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()