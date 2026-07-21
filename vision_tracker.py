import cv2
import mediapipe as mp
import numpy as np
import socket
import json
import urllib.request

# --- SETUP UDP SOCKET & STREAM URL ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

url = "http://192.168.101.30:8080/shot.jpg"
use_ip_cam = True

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

if not use_ip_cam:
    cap = cv2.VideoCapture(0)

def get_angle(v1, v2):
    v1_u = v1 / (np.linalg.norm(v1) + 1e-6)
    v2_u = v2 / (np.linalg.norm(v2) + 1e-6)
    dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return float(np.arccos(dot_product))

print(f"Transmitting True 19-DoF Biomechanical Angles over UDP ({UDP_IP}:{UDP_PORT})...")

while True:
    if use_ip_cam:
        try:
            img_resp = urllib.request.urlopen(url, timeout=2.0)
            img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            frame = cv2.imdecode(img_np, -1)
            if frame is None: continue
        except Exception:
            continue
    else:
        success, frame = cap.read()
        if not success: continue
        frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    hand_state = {}

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm = np.array([[pt.x, pt.y, pt.z] for pt in hand_landmarks.landmark])
            
            # --- COMPUTE 19 SERVO ANGLES FROM 21 LANDMARKS ---
            
            # Define directional vectors for the base of each finger
            v_idx = lm[6] - lm[5]
            v_mid = lm[10] - lm[9]
            v_rng = lm[14] - lm[13]
            v_pnk = lm[18] - lm[17]

            # Thumb (FIXED: Sequential bone mapping!)
            t_abd = get_angle(lm[2] - lm[1], lm[5] - lm[0]) 
            t_f1 = get_angle(lm[2] - lm[1], lm[3] - lm[2]) # Corrected to use lm[2]
            t_f2 = get_angle(lm[3] - lm[2], lm[4] - lm[3])
            
            # Index
            i_abd = get_angle(v_idx, v_mid)
            i_f1 = get_angle(lm[5] - lm[0], lm[6] - lm[5])
            i_f2 = get_angle(lm[6] - lm[5], lm[7] - lm[6])
            i_f3 = get_angle(lm[7] - lm[6], lm[8] - lm[7])
            
            # Middle
            m_abd = get_angle(v_mid, v_rng)
            m_f1 = get_angle(lm[9] - lm[0], lm[10] - lm[9])
            m_f2 = get_angle(lm[10] - lm[9], lm[11] - lm[10])
            m_f3 = get_angle(lm[11] - lm[10], lm[12] - lm[11])
            
            # Ring
            r_abd = get_angle(v_rng, v_mid)
            r_f1 = get_angle(lm[13] - lm[0], lm[14] - lm[13])
            r_f2 = get_angle(lm[14] - lm[13], lm[15] - lm[14])
            r_f3 = get_angle(lm[15] - lm[14], lm[16] - lm[15])
            
            # Pinky
            p_abd = get_angle(v_pnk, v_rng)
            p_f1 = get_angle(lm[17] - lm[0], lm[18] - lm[17])
            p_f2 = get_angle(lm[18] - lm[17], lm[19] - lm[18])
            p_f3 = get_angle(lm[19] - lm[18], lm[20] - lm[19])
            hand_state = {
                "thumb": [t_abd, t_f1, t_f2],
                "index": [i_abd, i_f1, i_f2, i_f3],
                "middle": [m_abd, m_f1, m_f2, m_f3],
                "ring": [r_abd, r_f1, r_f2, r_f3],
                "pinky": [p_abd, p_f1, p_f2, p_f3]
            }

    try:
        sock.sendto(json.dumps(hand_state).encode('utf-8'), (UDP_IP, UDP_PORT))
    except Exception:
        pass

    cv2.imshow("Full 21-Joint Raw Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if not use_ip_cam and 'cap' in locals(): cap.release()
cv2.destroyAllWindows()