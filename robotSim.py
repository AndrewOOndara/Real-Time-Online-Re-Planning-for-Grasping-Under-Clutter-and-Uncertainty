import pybullet as p
import pybullet_data as pd
import numpy as np


# open the GUI
p.connect(p.GUI)

p.setAdditionalSearchPath(pd.getDataPath())

p.setRealTimeSimulation(0)

# load files and place them at the offsets
start_position = [0, 0, 1]
goal_state = np.array([3, 0, 0])

# calculate the angle between the robot and the target
angle = np.arctan2(goal_state[1] - start_position[1], goal_state[0] - start_position[0])

# load files and place them at the offsets
# set the orientation of the robot
ori = p.getQuaternionFromEuler([0, 0, angle])
turtle = p.loadURDF("urdf/most_simple_turtle.urdf",[0,0,1], ori)
plane = p.loadURDF("plane100.urdf")
target = p.loadURDF("urdf/target.urdf", [3,0,1])
obstacle = p.loadURDF("urdf/box.urdf", [2,0,1])



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

# Helper function to reset the simulation
def reset_simulation():
    p.resetBasePositionAndOrientation(turtle, [0,0,1], [0,0,0,1])
    p.resetBasePositionAndOrientation(target, [3,0,1], [0,0,0,1])
    p.resetBasePositionAndOrientation(obstacle, [2,0,1], [0,0,0,1])

def add_gaussian_noise(path, std):
    return path + np.random.normal(loc=0, scale=std, size=path.shape)

def psto(robot, start_state, initial_control_sequence, num_iters, path_std):
    
    # Initialize the path and control input arrays
    control_input = np.zeros((num_iters, initial_control_sequence.shape[1]))
    
    
    for i in range(num_iters):
        # Add Gaussian noise to the initial control sequence
        noise = np.random.normal(loc=0, scale=path_std, size=initial_control_sequence.shape[1])
        if i == 0 or i == 1:
            noise[1] = 0.0
        noisy_control_input = initial_control_sequence[i] + noise
        
        control_input[i] = noisy_control_input
    
    # Return the generated path and the control input
    return control_input




# generate random actions until the turtle reaches the target
# initialize robot and goal state
robot_state = get_robot_state(turtle)

# set up PBSTO parameters
delta_t = 0.01
num_samples = 100
num_iters = 50
num_paths = 10
path_std = 0.1
speed = 1

p.setTimeStep(1/35)

# initialize robot and goal state
start_state = get_robot_state(turtle)
max = 0

initial_test = 0
while np.linalg.norm(robot_state[:2] - get_target_state(target)[:2]) >= 0.6:
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
    else:
         control_input = psto(turtle, start_state, initial_control_sequence, num_iters, path_std)

    initial_test += 1

    if np.linalg.norm(robot_state[:2] - get_target_state(target)[:2]) >= 0.65:
        # Follow the generated trajectory
        for i in range(num_iters):
            forward = control_input[i, 0]
        
            turn = control_input[i, 1]
            
            # Calculate the duration of the control input
            control_duration = delta_t * num_samples

            # Calculate the number of simulation steps required to execute the control input
            num_sim_steps = int(np.ceil(control_duration / p.getPhysicsEngineParameters()['fixedTimeStep']))


            # Call stepSimulation for the calculated number of steps
            for _ in range(num_sim_steps):
                p.stepSimulation()

            p.setJointMotorControl2(turtle, 0, p.VELOCITY_CONTROL, targetVelocity=(forward-turn)*speed, force=1000)
            p.setJointMotorControl2(turtle, 1, p.VELOCITY_CONTROL, targetVelocity=(forward+turn)*speed, force=1000)
            
            
            # Get the robot's new state after applying the control input
            robot_state = get_robot_state(turtle)
            target_state = get_target_state(target)
            
            # Check if the robot has reached the goal
            if np.linalg.norm(robot_state[:2] - target_state[:2]) < 0.65:
                break

        # If the robot has reached the target, stop the simulation
        if np.linalg.norm(robot_state[:2] - target_state[:2]) < 0.65:
            break

# stop the simulation once the turtle reaches the target
p.disconnect()