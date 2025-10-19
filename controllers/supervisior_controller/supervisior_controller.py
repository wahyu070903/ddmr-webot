from controller import Supervisor
import math
import json
import struct

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
emitter = supervisor.getDevice("supervisiorEmitter")

robot_node = supervisor.getFromDef("moose")

if robot_node is None:
    print("❌ Robot node not found! Make sure DEF name is 'moose'")
    exit()

while supervisor.step(timestep) != -1:
    position = robot_node.getField("translation").getSFVec3f()
    rotation = robot_node.getField("rotation").getSFRotation()
    heading_deg = math.degrees(rotation[3] * rotation[2])
    payload = {
        "id": "supervisor",
        "position": {"x": position[0], "y": position[1], "z": position[2]},
        "heading": heading_deg
    }

    json_data = json.dumps(payload).encode('utf-8')
    packet = struct.pack('I', len(json_data)) + json_data
    emitter.send(packet)

    print(f"📡 Sent position x={position[0]:.2f}, y={position[1]:.2f}, heading={heading_deg:.2f}°")
