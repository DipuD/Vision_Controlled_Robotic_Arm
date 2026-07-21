from controller import Robot
import socket
import json

# --- SETUP UDP RECEIVER ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

robot = Robot()
timestep = int(robot.getBasicTimeStep())

print("--- INITIALIZING PRECISION 19-DOF CONTROLLER ---")

motor_map = {
    "thumb_abd": "hand_right_thumb_abd_joint",
    "thumb_f1": "hand_right_thumb_flex_1_joint",
    "thumb_f2": "hand_right_thumb_flex_2_joint",
    
    "index_abd": "hand_right_index_abd_joint",
    "index_f1": "hand_right_index_flex_1_joint",
    "index_f2": "hand_right_index_flex_2_joint",
    "index_f3": "hand_right_index_flex_3_joint",
    
    "middle_abd": "hand_right_middle_abd_joint",
    "middle_f1": "hand_right_middle_flex_1_joint",
    "middle_f2": "hand_right_middle_flex_2_joint",
    "middle_f3": "hand_right_middle_flex_3_joint",
    
    "ring_abd": "hand_right_ring_abd_joint",
    "ring_f1": "hand_right_ring_flex_1_joint",
    "ring_f2": "hand_right_ring_flex_2_joint",
    "ring_f3": "hand_right_ring_flex_3_joint",
    
    "pinky_abd": "hand_right_little_abd_joint",
    "pinky_f1": "hand_right_little_flex_1_joint",
    "pinky_f2": "hand_right_little_flex_2_joint",
    "pinky_f3": "hand_right_little_flex_3_joint"
}

motors = {}
for key, name in motor_map.items():
    dev = robot.getDevice(name)
    if dev is not None:
        motors[key] = dev
        dev.setVelocity(2.0)  
        print(f"Linked: {name}")

print("Precision controller active. Listening for vision data...")

while robot.step(timestep) != -1:
    try:
        data, addr = sock.recvfrom(4096)
        state = json.loads(data.decode('utf-8'))

        def apply_mapped_joint(key, raw_rad, human_min, human_max, robot_min, robot_max, invert=False):
            if key in motors:
                motor = motors[key]
                
                # 1. Clamp human input so it doesn't exceed our expected bounds
                clamped_rad = max(human_min, min(raw_rad, human_max))
                
                # 2. Calculate the exact percentage of closure (0.0 to 1.0)
                ratio = (clamped_rad - human_min) / (human_max - human_min)
                
                # Apply inversion if requested (crucial for the thumb!)
                if invert:
                    ratio = 1.0 - ratio
                
                # 3. Map that percentage exactly to the ROBOT'S safe physical limits
                target_pos = robot_min + (ratio * (robot_max - robot_min))
                
                # 4. Final safety clamp against actual Webots motor limits
                target_pos = max(motor.getMinPosition(), min(target_pos, motor.getMaxPosition()))
                
                motor.setPosition(target_pos)

        # --- THUMB ---
        if "thumb" in state:
            apply_mapped_joint("thumb_abd", state["thumb"][0], human_min=0.10, human_max=0.55, robot_min=0.0, robot_max=1.20, invert=False)
            apply_mapped_joint("thumb_f1", state["thumb"][1], human_min=0.05, human_max=0.65, robot_min=0.0, robot_max=1.35)
            apply_mapped_joint("thumb_f2", state["thumb"][2], human_min=0.05, human_max=0.65, robot_min=0.0, robot_max=1.20)

        # --- INDEX ---
        if "index" in state:
            apply_mapped_joint("index_abd", state["index"][0], human_min=0.05, human_max=0.25, robot_min=0.0, robot_max=0.15)
            # Pushed to 2.05 for a fully compressed, flush fist
            apply_mapped_joint("index_f1", state["index"][1], human_min=0.10, human_max=1.45, robot_min=0.0, robot_max=2.05)
            apply_mapped_joint("index_f2", state["index"][2], human_min=0.10, human_max=1.45, robot_min=0.0, robot_max=1.80)
            apply_mapped_joint("index_f3", state["index"][3], human_min=0.10, human_max=1.45, robot_min=0.0, robot_max=1.10)

        # --- MIDDLE ---
        if "middle" in state:
            apply_mapped_joint("middle_abd", state["middle"][0], human_min=0.05, human_max=0.25, robot_min=0.0, robot_max=0.10)
            apply_mapped_joint("middle_f1", state["middle"][1], human_min=0.10, human_max=1.45, robot_min=0.0, robot_max=2.05)
            apply_mapped_joint("middle_f2", state["middle"][2], human_min=0.10, human_max=1.45, robot_min=0.0, robot_max=1.80)
            apply_mapped_joint("middle_f3", state["middle"][3], human_min=0.10, human_max=1.45, robot_min=0.0, robot_max=1.10)

        # --- RING ---
        if "ring" in state:
            apply_mapped_joint("ring_abd", state["ring"][0], human_min=0.05, human_max=0.25, robot_min=0.0, robot_max=0.15)
            apply_mapped_joint("ring_f1", state["ring"][1], human_min=0.10, human_max=1.45, robot_min=0.0, robot_max=2.05)
            apply_mapped_joint("ring_f2", state["ring"][2], human_min=0.10, human_max=1.45, robot_min=0.0, robot_max=1.80)
            apply_mapped_joint("ring_f3", state["ring"][3], human_min=0.10, human_max=1.45, robot_min=0.0, robot_max=1.10)

        # --- PINKY ---
        if "pinky" in state:
            apply_mapped_joint("pinky_abd", state["pinky"][0], human_min=0.05, human_max=0.25, robot_min=0.0, robot_max=0.20)
            apply_mapped_joint("pinky_f1", state["pinky"][1], human_min=0.10, human_max=1.45, robot_min=0.0, robot_max=2.05)
            apply_mapped_joint("pinky_f2", state["pinky"][2], human_min=0.10, human_max=1.45, robot_min=0.0, robot_max=1.80)
            apply_mapped_joint("pinky_f3", state["pinky"][3], human_min=0.10, human_max=1.45, robot_min=0.0, robot_max=1.10)

    except BlockingIOError:
        pass
    except Exception as e:
        if getattr(e, 'winerror', None) != 10035:
            print(f"Error: {e}")