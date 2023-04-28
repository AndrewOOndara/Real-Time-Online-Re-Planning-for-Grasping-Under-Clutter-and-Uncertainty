import pybullet as p
import pybullet_data as pd
import numpy as np
import time


# open the GUI
p.connect(p.GUI)

p.setAdditionalSearchPath(pd.getDataPath())

p.setRealTimeSimulation(0)

target_pos = [6, 1, 0]
# load files and place them at the offsets
start_position = [0, 0, 1]
goal_state = np.array(target_pos)

# calculate the angle between the robot and the target
angle = np.arctan2(goal_state[1] - start_position[1], goal_state[0] - start_position[0])

# load files and place them at the offsets
# set the orientation of the robot
ori = p.getQuaternionFromEuler([0, 0, angle])
turtle = p.loadURDF("urdf/most_simple_turtle.urdf",[0,0,1], ori)
plane = p.loadURDF("plane100.urdf")
target = p.loadURDF("urdf/target.urdf", target_pos)
obstacle_positions = [
    [3, 0, 1],
    [3, -1, 1],
    [4, 0.5, 1],
    [5, -0.5, 1],
    [6, 1.5, 1],
    [6, -1.5, 1],
    [4.5, 1, 1],
    [4.5, -1, 1],
    [5.5, 0, 1],
    [2, 1, 1],
    [2, -1, 1],
    [4, -1.5, 1],
    [5, 1.5, 1],
    [1, 0.5, 1],
    [1, -0.5, 1],
    [1.5, 0, 1]
]

obstacles = []
for position in obstacle_positions:
    obstacle = p.loadURDF("urdf/box.urdf", position)
    obstacles.append(obstacle)




# disable real time simulation
p.setRealTimeSimulation(0)

# define gravity
p.setGravity(0,0,-10)

# Helper function to get the robot's state
def get_robot_state(robot):
    pos, ori = p.getBasePositionAndOrientation(robot)
    euler = p.getEulerFromQuaternion(ori)
    vel, ang_vel = p.getBaseVelocity(robot)
    return np.array([pos[0], pos[1], euler[2], vel[0], vel[1], ang_vel[2]])

# Helper function to get the robot's state
def get_target_state(target):
    pos, ori = p.getBasePositionAndOrientation(target)
    euler = p.getEulerFromQuaternion(ori)
    vel, ang_vel = p.getBaseVelocity(target)
    return np.array([pos[0], pos[1], euler[2], vel[0], vel[1], ang_vel[2]])

# Helper function to set the robot's state
def set_robot_state(robot, state):
    pos = [state[0], state[1], 0.1]
    ori = p.getQuaternionFromEuler([0, 0, state[2]])
    p.resetBasePositionAndOrientation(robot, pos, ori)
    p.resetBaseVelocity(robot, [state[3], state[4], 0], [0, 0, state[5]])

def reset_simulation():
    p.setTimeStep(1/35)
    p.resetBasePositionAndOrientation(turtle, [0, 0, 1], ori)
    print(ori)
    p.resetBasePositionAndOrientation(target, target_pos, [0, 0, 0, 1])
    for i, obstacle_pos in enumerate(obstacle_positions):
        p.resetBasePositionAndOrientation(obstacles[i], obstacle_pos, [0, 0, 0, 1])

def reset_whole_simulation():
    p.resetSimulation()
    p.setGravity(0,0,-10)
    # load files and place them at the offsets
    start_position = [0, 0, 1]
    goal_state = np.array(target_pos)

    # calculate the angle between the robot and the target
    angle = np.arctan2(goal_state[1] - start_position[1], goal_state[0] - start_position[0])

    # load files and place them at the offsets
    # set the orientation of the robot
    ori = p.getQuaternionFromEuler([0, 0, angle])
    turtle = p.loadURDF("urdf/most_simple_turtle.urdf",[0,0,1], ori)
    plane = p.loadURDF("plane100.urdf")
    target = p.loadURDF("urdf/target.urdf", target_pos)
    obstacle_positions = [        [3, 0, 1],
        [3, -1, 1],
        [4, 0.5, 1],
        [5, -0.5, 1],
        [6, 1.5, 1],
        [6, -1.5, 1],
        [4.5, 1, 1],
        [4.5, -1, 1],
        [5.5, 0, 1],
        [2, 1, 1],
        [2, -1, 1],
        [4, -1.5, 1],
        [5, 1.5, 1],
        [1, 0.5, 1],
        [1, -0.5, 1],
        [1.5, 0, 1]
    ]

    obstacles = []
    for position in obstacle_positions:
        obstacle = p.loadURDF("urdf/box.urdf", position)
        obstacles.append(obstacle)


def psto(robot, start_state, initial_control_sequence, num_iters, path_std):
    
    # Initialize the path and control input arrays
    control_input = np.zeros((num_iters, initial_control_sequence.shape[1]))
    
    
    for i in range(num_iters):
        # Add Gaussian noise to the initial control sequence
        noise = np.random.normal(loc=0, scale=path_std, size=initial_control_sequence.shape[1])
        noisy_control_input = initial_control_sequence[i] + noise
        
        control_input[i] = noisy_control_input
    
    # Return the generated path and the control input
    return control_input




# generate random actions until the turtle reaches the target
# initialize robot and goal state
robot_state = get_robot_state(turtle)

# set up PBSTO parameters
delta_t = 0.5
num_samples = 50
num_iters = 15
path_std = 0.05
speed = 10

p.setTimeStep(1/35)

# initialize robot and goal state
start_state = get_robot_state(turtle)
max = 0
time_step = 1/35
success_threshold = 0.60

successful_control_input = {}

state_id = p.saveState()
initial_test = 0
while np.linalg.norm(get_robot_state(turtle)[:2] - get_target_state(target)[:2]) >= success_threshold:
    # Reset the simulation
    reset_simulation()
    print("reset")
    

    # Generate initial control sequence straight toward the obstacle
    initial_control_sequence = np.zeros((num_iters, 2))
    initial_control_sequence[:, 0] = 1.0
    initial_control_sequence[:, 1] = 0.0
    
    # Generate trajectory using psto
    if(initial_test == 0):
        control_input = initial_control_sequence
        print(control_input)
    else:
         control_input = psto(turtle, start_state, initial_control_sequence, num_iters, path_std)
         print(control_input)

    initial_test += 1

    if np.linalg.norm(robot_state[:2] - get_target_state(target)[:2]) >= success_threshold:
        # Follow the generated trajectory
        for i in range(num_iters):
            forward = control_input[i, 0]
        
            turn = control_input[i, 1]
            
            # Calculate the duration of the control input
            control_duration = delta_t

            # Calculate the number of simulation steps required to execute the control input
            num_sim_steps = int(np.ceil(control_duration / (time_step)))
            print(num_sim_steps)


            # Call stepSimulation for the calculated number of steps
            for _ in range(num_sim_steps):
                p.stepSimulation()

            p.setJointMotorControl2(turtle, 0, p.VELOCITY_CONTROL, targetVelocity=(forward-turn)*speed, force=1000)
            p.setJointMotorControl2(turtle, 1, p.VELOCITY_CONTROL, targetVelocity=(forward+turn)*speed, force=1000)
            
            
            # Get the robot's new state after applying the control input
            robot_state = get_robot_state(turtle)
            target_state = get_target_state(target)
            
            # Check if the robot has reached the goal
            if np.linalg.norm(robot_state[:2] - target_state[:2]) < success_threshold:
                successful_control_input = control_input
                break

        # If the robot has reached the target, stop the simulation
        if np.linalg.norm(robot_state[:2] - target_state[:2]) < success_threshold:
            break

# stop the simulation once the turtle reaches the target
time.sleep(5)
reset_simulation()

print("success")
print(successful_control_input)

p.setRealTimeSimulation(0)
p.setTimeStep(time_step)

p.setPhysicsEngineParameter(fixedTimeStep=1/35, numSubSteps=1, enableFileCaching=0, deterministicOverlappingPairs=1)

print(successful_control_input)
while True:
    reset_whole_simulation()
    # Follow the generated trajectory
    for i in range(num_iters):
        forward = successful_control_input[i, 0]
        

        turn = successful_control_input[i, 1]
        
        
        # Calculate the duration of the control input
        control_duration = delta_t

        # Calculate the number of simulation steps required to execute the control input
        num_sim_steps = int(np.ceil(control_duration / (time_step)))
        print(num_sim_steps)

        # Call stepSimulation for the calculated number of steps
        for _ in range(num_sim_steps):
            p.setJointMotorControl2(turtle, 0, p.VELOCITY_CONTROL, targetVelocity=(forward-turn)*speed, force=1000)
            p.setJointMotorControl2(turtle, 1, p.VELOCITY_CONTROL, targetVelocity=(forward+turn)*speed, force=1000)
            p.stepSimulation()
        time.sleep(time_step)
p.disconnect()
