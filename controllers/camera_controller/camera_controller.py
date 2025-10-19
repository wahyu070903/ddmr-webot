from controller import Robot
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import struct
import json
import zlib
import base64

robot = Robot()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("TopCamera")
camera.enable(timestep)

emitter = robot.getDevice("cameraEmitter")

# map size in meters
MAP_SIZE = 20.0  

while robot.step(timestep) != -1:
    image = camera.getImage()
    width = camera.getWidth()
    height = camera.getHeight()
    img = np.frombuffer(image, np.uint8).reshape((height, width, 4))
    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # === STEP 1: Detect and mask QR codes (pyzbar + OpenCV) ===
    decoded_objects = decode(frame)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    qr_positions = []  # store (label, world_x, world_y)

    for obj in decoded_objects:
        (x, y, w, h) = obj.rect
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
        
        # Fill mask to remove QR later
        # Expand the mask to also remove the white QR border (quiet zone)
        pad = int(0.3 * max(w, h))  # expand mask by 30% of QR size
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(width, x + w + pad)
        y2 = min(height, y + h + pad)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # Compute QR center
        cx = x + w // 2
        cy = y + h // 2

        # Convert to world coordinates (centered map)
        scale_x = MAP_SIZE / width
        scale_y = MAP_SIZE / height
        wx = (cx - width / 2) * scale_x
        wy = (height / 2 - cy) * scale_y

        # Decode QR content (as text)
        label = obj.data.decode('utf-8', errors='ignore')
        qr_positions.append((label, wx, wy))

        # Draw center and label
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
        cv2.putText(frame, f"{label} ({wx:.2f},{wy:.2f})", (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Remove QR from the frame (for wall detection)
    img_no_qr = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))

    # === STEP 2: Wall detection using color filtering ===
    hsv = cv2.cvtColor(img_no_qr, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 12, 255])
    mask_wall = cv2.inRange(hsv, lower_white, upper_white)

    # --- Morphological cleaning to reduce QR edges and noise ---
    mask_wall = cv2.morphologyEx(mask_wall, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask_wall = cv2.morphologyEx(mask_wall, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # --- Connected-component filtering to remove small blobs (QR borders etc.) ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_wall)
    min_area = 800  # adjust depending on map resolution
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            mask_wall[labels == i] = 0

    contours, _ = cv2.findContours(mask_wall, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = frame.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

    # === STEP 3: Create Binary Map (for path planning) ===
    binary_map = np.zeros_like(mask_wall, dtype=np.uint8)
    binary_map[mask_wall > 0] = 1  # wall = 1

    # === Visualization ===
    vis_binary = (binary_map * 255).astype(np.uint8)
    # cv2.imshow("Binary Occupancy Map", vis_binary)
    cv2.imshow("Webots Wall Detection + QR", output)
    
    # == Send data ===
    compressed_map = zlib.compress(vis_binary.tobytes())
    compressed_map_b64 = base64.b64encode(compressed_map).decode('utf-8') 

    
    payload = {
        "id": "camera",
        "width": width,
        "height": height,
        "cam_data": compressed_map_b64,
        "qr_data": qr_positions
    }
    
    json_data = json.dumps(payload).encode('utf-8')
    packet = struct.pack('I', len(json_data)) + json_data
    emitter.send(packet)
    
    # === Log QR Data ===
    if qr_positions:
        # print("QR Code world positions:")
        for label, x, y in qr_positions:
            # print(f"  {label}: x={x:.2f}, y={y:.2f}")
            pass
    else:
        # print("No QR detected")
        pass
        
    # print(f"Sent map ({len(compressed_map)} bytes) + QR data ({len(qr_json)} bytes)")
    
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cv2.destroyAllWindows()
