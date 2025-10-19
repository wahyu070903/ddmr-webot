from controller import Robot

# Create the Robot instance
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Get and enable the GPS
gps = robot.getDevice("gps")
gps.enable(timestep)

# Main control loop
while robot.step(timestep) != -1:
    position = gps.getValues()
    print(f"x={position[0]:.2f}, y={position[1]:.2f}, z={position[2]:.2f}")
