from controller import Robot
import cv2
import struct
import zlib
import json
import numpy as np
import base64
import heapq
import socket
import time

# --- UDP Configuration ---
simulink_ip = "127.0.0.1"
simulink_port = 25001
local_port = 25002  # Webots listening port
sock = None
is_socket_connected = False
is_path_sent = False
is_pathx_sent = False
is_pathy_sent = False
is_pathHead_sent = False

# --- Webots Setup ---
robot = Robot()
receiver = robot.getDevice("mooseReceiver")
receiver.enable(int(robot.getBasicTimeStep()))

# --- Global variables ---
vis_binary = None
qr_positions = None
supervisor_data = None
path_generated = False
path_to_follow = []

import numpy as np
import struct

def send_array_to_matlab(array, socket, marker):
    """
    Send array to MATLAB/Simulink via UDP socket
    
    Args:
        array: Input array (any iterable)
        socket: UDP socket object
        marker: Data marker (integer 0-255)
        simulink_ip: Target IP address
        simulink_port: Target port
    """
    
    # Convert to numpy array for consistency
    array = np.array(array, dtype=np.float64)
    
    # Create message
    marker_byte = marker.encode('utf-8') # Convert integer to single byte
    array_length = len(array)
    data_type = b'd'  # double (8 bytes)
    
    # Pack header: marker (1B) + length (4B) + data_type (1B)
    header = marker_byte + struct.pack('<I', array_length) + data_type
    
    # Pack data
    data = struct.pack('<%dd' % array_length, *array)
    
    # Combine and send
    message = header + data
    socket.sendto(message, (simulink_ip, simulink_port))
    
    print(f"📤 Sent array: {array_length} elements, {len(message)} bytes, marker: {marker}")


# --- A* Pathfinding ---
def astar(start, goal, grid):
    neighbors = [
        (0,1),(1,0),(0,-1),(-1,0),
        (1,1),(1,-1),(-1,1),(-1,-1)
    ]
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start, [start]))
    visited = set()

    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            return path

        for dx, dy in neighbors:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]:
                if grid[ny, nx] == 1 and (nx, ny) not in visited:
                    cost = g + (1.4142 if dx != 0 and dy != 0 else 1)
                    heapq.heappush(open_set, (
                        cost + heuristic((nx, ny), goal),
                        cost,
                        (nx, ny),
                        path + [(nx, ny)]
                    ))
    return []

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def compute_astar_path(map_img, qr_positions, supervisor_data, vehicle_radius=50):
    if map_img is None or map_img.size == 0:
        return []
    if not qr_positions or supervisor_data is None:
        return []

    map_h, map_w = map_img.shape

    # Convert world coordinates to grid
    start_x = round(map_w/2 + supervisor_data['x'] / 20.0 * map_w)
    start_y = round(map_h/2 - supervisor_data['y'] / 20.0 * map_h)
    start = (start_x, start_y)

    goal_label, goal_x, goal_y = qr_positions[0]
    goal_x_grid = round(map_w/2 + goal_x / 20.0 * map_w)
    goal_y_grid = round(map_h/2 - goal_y / 20.0 * map_h)
    goal = (goal_x_grid, goal_y_grid)

    # Inflate walls for vehicle size
    walls = (map_img > 0).astype(np.uint8)
    inflated_walls = cv2.dilate(
        walls,
        np.ones((vehicle_radius*2+1, vehicle_radius*2+1), np.uint8)
    )
    grid_for_astar = (inflated_walls == 0).astype(np.uint8)

    path = astar(start, goal, grid_for_astar)
    return path

def compute_path_with_heading(path):
    path_with_heading = []
    for i in range(len(path)):
        x, y = path[i]
        if i < len(path) - 1:
            nx, ny = path[i+1]
            dx = nx - x
            dy = ny - y
            heading = np.arctan2(dy, dx)
        else:
            heading = 0
        path_with_heading.append({'x': x, 'y': y, 'heading': heading})
    return path_with_heading

# --- Main loop ---
while robot.step(int(robot.getBasicTimeStep())) != -1:
    # --- Receive data from Webots ---
    if receiver.getQueueLength() > 0:
        try:
            data = receiver.getBytes()
            if len(data) < 4:
                receiver.nextPacket()
                continue

            json_len = struct.unpack('I', data[:4])[0]
            json_payload = data[4:4+json_len]
            payload = json.loads(json_payload.decode('utf-8'))

            if payload["id"] == "supervisor":
                supervisor_data = payload["position"]
                supervisor_data['heading'] = payload["heading"]

            elif payload["id"] == "camera":
                width = payload["width"]
                height = payload["height"]

                compressed_map_b64 = payload["cam_data"]
                compressed_map = base64.b64decode(compressed_map_b64)
                decompressed_map = zlib.decompress(compressed_map)
                qr_positions = payload["qr_data"]

                raw = np.frombuffer(decompressed_map, dtype=np.uint8)
                if raw.size == width * height:
                    vis_binary = raw.reshape((height, width))
                else:
                    vis_binary = None

        except Exception as e:
            print("❌ Error decoding packet:", e)

        receiver.nextPacket()

    # --- Compute path once ---
    if not path_generated:
        raw_path = compute_astar_path(vis_binary, qr_positions, supervisor_data, vehicle_radius=50)
        if raw_path:
            path_to_follow = compute_path_with_heading(raw_path)
            path_generated = True

    # --- Visualize path ---
    if vis_binary is not None and path_to_follow:
        vis = vis_binary.copy()
        for point in path_to_follow:
            x, y, heading = int(point['x']), int(point['y']), point['heading']
            cv2.circle(vis, (x, y), 1, 128, -1)
            arrow_length = 20
            end_x = int(x + arrow_length * np.cos(heading))
            end_y = int(y + arrow_length * np.sin(heading))
            cv2.arrowedLine(vis, (x, y), (end_x, end_y), 255, 1, tipLength=0.3)
        cv2.imshow("Binary Occupancy Map with Path & Heading", vis)
        cv2.waitKey(1)

    # --- Wait for Simulink to send "READY" ---
        # --- Non-blocking wait for Simulink to send "READY" ---
    if not is_socket_connected and path_generated:
        if sock is None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("", local_port))
            sock.settimeout(0.01)  # very short timeout
            print("⏳ Waiting for Simulink to send 'READY'...")

        try:
            data, addr = sock.recvfrom(1024)
            msg = data.decode("utf-8").strip().upper()
            if msg == "READY":
                print(f"✅ Simulink ready at {addr}")
                is_socket_connected = True
                simulink_ip = addr[0]
        except socket.timeout:
            pass  # no data this step, continue Webots simulation

    # --- Send path once ---
    if is_socket_connected and not is_path_sent:
        path_x = [p['x'] for p in path_to_follow]
        path_y = [p['y'] for p in path_to_follow]
        path_heading = [p['heading'] for p in path_to_follow]
        
        # Send X and wait for acknowledgment
        if not is_pathx_sent:
            send_array_to_matlab(path_x, sock, 'X')
            try:
                sock.settimeout(2.0)  # 2 second timeout
                data, addr = sock.recvfrom(1024)
                msg = data.decode("utf-8").strip().upper()
                if msg == "XOK":
                    is_pathx_sent = True
                    print("✅ path_X sent success")
                    time.sleep(0.1)  # Small delay before sending Y
                else:
                    print(f"Unexpected reply: {msg}")
            except socket.timeout:
                print("Timeout waiting for XOK - will retry")
            except Exception as e:
                print(f"Error receiving XOK: {e}")
            finally:
                sock.settimeout(0.1)  # Reset to original timeout
        
        # Send Y only after X is acknowledged
        if is_pathx_sent and not is_pathy_sent:
            send_array_to_matlab(path_y, sock, 'Y')
            try:
                sock.settimeout(2.0)  # 2 second timeout
                data, addr = sock.recvfrom(1024)
                msg = data.decode("utf-8").strip().upper()
                if msg == "YOK":
                    is_pathy_sent = True
                    print("✅ path_Y sent success")
                else:
                    print(f"Unexpected reply: {msg}")
            except socket.timeout:
                print("Timeout waiting for YOK - will retry")
            except Exception as e:
                print(f"Error receiving YOK: {e}")
            finally:
                sock.settimeout(0.1)  # Reset to original timeout
                
        if is_pathx_sent and is_pathy_sent and not is_path_sent:
            send_array_to_matlab(path_heading, sock, 'H')
            try:
                sock.settimeout(2.0)  # 2 second timeout
                data, addr = sock.recvfrom(1024)
                msg = data.decode("utf-8").strip().upper()
                if msg == "HOK":
                    is_path_sent = True
                    print("✅ path_Heading sent success")
                else:
                    print(f"Unexpected reply: {msg}")
            except socket.timeout:
                print("Timeout waiting for HOK - will retry")
            except Exception as e:
                print(f"Error receiving HOK: {e}")
            finally:
                sock.settimeout(0.1)  # Reset to original timeout
                
                
    