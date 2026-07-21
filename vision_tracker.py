import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import socket
import json

# --- UDP SOCKET SETUP ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def get_vector(p1, p2):
    return np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])

def calculate_angle(a, b, c):
    ba = get_vector(b, a)
    bc = get_vector(b, c)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def calculate_spread(base1, tip1, base2, tip2):
    v1 = get_vector(base1, tip1)
    v2 = get_vector(base2, tip2)
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def normalize(angle, min_angle, max_angle):
    return float(np.clip((angle - min_angle) / (max_angle - min_angle), 0.0, 1.0))

url = "http://192.168.101.30:8080/shot.jpg"
print(f"Transmitting 16-DoF data over UDP ({UDP_IP}:{UDP_PORT})... Press 'q' to quit.")

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while True:
        try:
            img_resp = urllib.request.urlopen(url, timeout=1.0)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            image = cv2.imdecode(imgnp, -1)
            
            if image is None: 
                continue

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    lm = hand_landmarks.landmark

                    # 1. BASE FLEXION
                    t_base = 1.0 - normalize(calculate_angle(lm[0], lm[1], lm[2]), 120.0, 170.0)
                    i_base = 1.0 - normalize(calculate_angle(lm[0], lm[5], lm[6]), 90.0, 170.0)
                    m_base = 1.0 - normalize(calculate_angle(lm[0], lm[9], lm[10]), 90.0, 170.0)
                    r_base = 1.0 - normalize(calculate_angle(lm[0], lm[13], lm[14]), 90.0, 170.0)
                    p_base = 1.0 - normalize(calculate_angle(lm[0], lm[17], lm[18]), 90.0, 170.0)

                    # 2. TIP CURLING
                    t_curl = 1.0 - normalize(calculate_angle(lm[2], lm[3], lm[4]), 120.0, 170.0)
                    i_curl = 1.0 - normalize(calculate_angle(lm[5], lm[6], lm[8]), 70.0, 170.0)
                    m_curl = 1.0 - normalize(calculate_angle(lm[9], lm[10], lm[12]), 70.0, 170.0)
                    r_curl = 1.0 - normalize(calculate_angle(lm[13], lm[14], lm[16]), 70.0, 170.0)
                    p_curl = 1.0 - normalize(calculate_angle(lm[17], lm[18], lm[20]), 70.0, 170.0)

                    # 3. SPREAD
                    t_spread = normalize(calculate_spread(lm[1], lm[2], lm[9], lm[10]), 20.0, 60.0)
                    i_spread = normalize(calculate_spread(lm[5], lm[6], lm[9], lm[10]), 5.0, 25.0)
                    r_spread = normalize(calculate_spread(lm[13], lm[14], lm[9], lm[10]), 5.0, 25.0)
                    p_spread = normalize(calculate_spread(lm[17], lm[18], lm[9], lm[10]), 5.0, 35.0)

                    # 4. WRIST ORIENTATION
                    wrist_to_index = get_vector(lm[0], lm[5])
                    wrist_to_pinky = get_vector(lm[0], lm[17])
                    palm_normal = np.cross(wrist_to_index, wrist_to_pinky)
                    palm_normal = palm_normal / np.linalg.norm(palm_normal)
                    
                    wrist_pitch = normalize(palm_normal[1], -0.5, 0.5)
                    wrist_roll = normalize(palm_normal[0], -0.5, 0.5)

                    state = {
                        "Thumb":  [round(t_base, 2), round(t_curl, 2), round(t_spread, 2)],
                        "Index":  [round(i_base, 2), round(i_curl, 2), round(i_spread, 2)],
                        "Middle": [round(m_base, 2), round(m_curl, 2)],
                        "Ring":   [round(r_base, 2), round(r_curl, 2), round(r_spread, 2)],
                        "Pinky":  [round(p_base, 2), round(p_curl, 2), round(p_spread, 2)],
                        "Wrist":  [round(wrist_pitch, 2), round(wrist_roll, 2)]
                    }
                    
                    payload = json.dumps(state).encode('utf-8')
                    sock.sendto(payload, (UDP_IP, UDP_PORT))

            cv2.imshow('MediaPipe 16-DoF Kinematics', image)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
                
        except Exception as e:
            continue

cv2.destroyAllWindows()
sock.close()