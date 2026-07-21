from controller import Robot, Node
import socket
import json

# --- 1. SETUP UDP RECEIVER SOCKET ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

# --- 2. INITIALIZE WEBOTS ROBOT & MOTORS ---
robot = Robot()
timestep = int(robot.getBasicTimeStep())

print("--- INITIALIZING HEY5 HAND MOTORS ---")
motor_names = [
    "hand_right_thumb_abd_joint", "hand_right_thumb_flex_1_joint", "hand_right_thumb_flex_2_joint",
    "hand_right_index_flex_1_joint", "hand_right_index_flex_2_joint",
    "hand_right_middle_flex_1_joint", "hand_right_middle_flex_2_joint",
    "hand_right_ring_flex_1_joint", "hand_right_ring_flex_2_joint",
    "hand_right_little_flex_1_joint", "hand_right_little_flex_2_joint"
]

motors = {}
for name in motor_names:
    motor = robot.getDevice(name)
    if motor is not None:
        motors[name] = motor
        min_l = motor.getMinPosition()
        max_l = motor.getMaxPosition()
        
        if min_l != max_l:
            motor.setPosition(min_l + 0.05)
        else:
            motor.setPosition(0.0)
            
        motor.setVelocity(2.0)
        print(f"Successfully linked: {name}")
    else:
        print(f"Warning: Could not find motor -> {name}")

print("Initialization complete. Listening for UDP packets...")

# --- 3. MAIN SIMULATION LOOP ---
while robot.step(timestep) != -1:
    try:
        data, addr = sock.recvfrom(2048)
        hand_state = json.loads(data.decode('utf-8'))
        
        # Fixed function definition accepting 'scale' and expanding the input window
        def apply_mapping(motor_name, val, invert=False, scale=1.0):
            if motor_name in motors:
                motor = motors[motor_name]
                
                if invert:
                    val = 1.0 - val
                
                # Expand sensitivity window so normal hand movements span full range
                val = (val - 0.2) / (0.8 - 0.2)
                val = max(0.0, min(val, 1.0))
                
                min_l = motor.getMinPosition()
                max_l = motor.getMaxPosition()
                
                if min_l != max_l:
                    target = min_l + (val * (max_l - min_l) * scale)
                    target = max(min_l, min(target, max_l))
                else:
                    target = val * scale
                    
                motor.setPosition(target)

        # --- THUMB ---
        if "Thumb" in hand_state:
            t = hand_state["Thumb"]
            apply_mapping("hand_right_thumb_abd_joint", t[2], invert=False, scale=1.0)
            apply_mapping("hand_right_thumb_flex_1_joint", t[0], invert=False, scale=1.0)
            apply_mapping("hand_right_thumb_flex_2_joint", t[1], invert=False, scale=1.0)

        # --- INDEX ---
        if "Index" in hand_state:
            i = hand_state["Index"]
            apply_mapping("hand_right_index_flex_1_joint", i[0], invert=False, scale=1.0)
            apply_mapping("hand_right_index_flex_2_joint", i[1], invert=False, scale=1.0)

        # --- MIDDLE ---
        if "Middle" in hand_state:
            m = hand_state["Middle"]
            apply_mapping("hand_right_middle_flex_1_joint", m[0], invert=False, scale=1.0)
            apply_mapping("hand_right_middle_flex_2_joint", m[1], invert=False, scale=1.0)

        # --- RING ---
        if "Ring" in hand_state:
            r = hand_state["Ring"]
            apply_mapping("hand_right_ring_flex_1_joint", r[0], invert=False, scale=1.0)
            apply_mapping("hand_right_ring_flex_2_joint", r[1], invert=False, scale=1.0)

        # --- PINKY ---
        if "Pinky" in hand_state:
            p = hand_state["Pinky"]
            apply_mapping("hand_right_little_flex_1_joint", p[0], invert=False, scale=1.0)
            apply_mapping("hand_right_little_flex_2_joint", p[1], invert=False, scale=1.0)

    except BlockingIOError:
        pass
    except Exception as e:
        if getattr(e, 'winerror', None) != 10035:
            print(f"Receiver error: {e}")