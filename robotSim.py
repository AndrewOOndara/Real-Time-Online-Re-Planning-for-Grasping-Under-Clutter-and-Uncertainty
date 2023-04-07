import pybullet as p
import pybullet_data as pd
import numpy as np


# open the GUI
p.connect(p.GUI)

p.setAdditionalSearchPath(pd.getDataPath())

# load files and place them at the offsets
turtle = p.loadURDF("urdf/most_simple_turtle.urdf",[0,0,1])
plane = p.loadURDF("plane100.urdf")
target = p.loadURDF("urdf/target.urdf", [3,0,1])
obstacle = p.loadURDF("urdf/target.urdf", [2,0,1])


# disable real time simulation
p.setRealTimeSimulation(0)

# define gravity
p.setGravity(0,0,-10)

foward = 0
turn=0
speed=10

goal_state = np.array([3, 0, 0])

# Helper function to get the robot's state
def get_robot_state(robot):
    pos, ori = p.getBasePositionAndOrientation(robot)
    euler = p.getEulerFromQuaternion(ori)
    vel, ang_vel = p.getBaseVelocity(robot)
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
    p.resetBasePositionAndOrientation(target, [3,0,1], [0,0,0,1])


def psto(robot, start_state, goal_state, delta_t, num_samples, num_iters):
    # Initialize the robot's state
    robot_state = start_state
    
    # Loop over the specified number of iterations
    for i in range(num_iters):
        # Generate next state samples using dynamics
        control_input_samples = np.random.uniform(low=-1.0, high=1.0, size=(num_samples, 2))
        next_state_samples = np.zeros((num_samples, 6))
        for j in range(num_samples):
            next_state_samples[j] = robot_state + delta_t*np.hstack((np.cos(robot_state[2])*control_input_samples[j, 0], np.sin(robot_state[2])*control_input_samples[j, 0], 0, control_input_samples[j, 1], 0, 0))
        
        # Compute costs for each next state sample
        costs = np.linalg.norm(next_state_samples[:, :2] - goal_state[:2], axis=1)
        
        # Get the best next state and control input
        best_idx = np.argmin(costs)
        best_next_state = next_state_samples[best_idx]
        best_control_input = control_input_samples[best_idx]
        
        # Update the robot's state
        robot_state = best_next_state
        
        # Check if the goal has been reached
        if np.linalg.norm(robot_state[:2] - goal_state[:2]) < 0.1:
            break
    
    # Return the best control input
    return best_control_input




# generate random actions until the turtle reaches the target
# initialize robot and goal state
robot_state = get_robot_state(turtle)

# set up PBSTO parameters
delta_t = 0.01
num_samples = 100
num_iters = 100
num_paths = 10
path_std = 0.1

p.setTimeStep(0.01)

# initialize robot and goal state
start_state = get_robot_state(turtle)
goal_state = np.array([2, 0, 0])

while True:
    # Reset the simulation
    reset_simulation()
    
    # Generate initial path straight toward the obstacle
    path = np.zeros((num_iters, 2))
    path[:, 0] = 1.0
    path[:, 1] = 0.0
    
    # Follow the initial path
    for i in range(num_iters):
        forward = path[i, 0]
        turn = path[i, 1]
        p.setJointMotorControl2(turtle, 0, p.VELOCITY_CONTROL, targetVelocity=(forward-turn)*speed, force=1000)
        p.setJointMotorControl2(turtle, 1, p.VELOCITY_CONTROL, targetVelocity=(forward+turn)*speed, force=1000)
        p.stepSimulation()
        
        # Get the robot's new state after applying the control input
        robot_state = get_robot_state(turtle)
        
        # Check if the robot has reached the goal
        if np.linalg.norm(robot_state[:2] - goal_state[:2]) < 0.1:
            break
        
    # If the robot hasn't reached the target, try again with a perturbed path
    if np.linalg.norm(robot_state[:2] - goal_state[:2]) >= 0.1:
        for i in range(num_paths):
            # Generate perturbed path
            perturbed_path = path + np.random.normal(scale=path_std, size=(num_iters, 2))
            
            # Generate trajectory using PBSTO
            control_input = psto(turtle, start_state, goal_state, delta_t, num_samples, num_iters)
            
            # Follow the generated trajectory
            for j in range(num_iters):
                forward = control_input[j % 2]
                turn = 0
                p.setJointMotorControl2(turtle, 0, p.VELOCITY_CONTROL, targetVelocity=(forward-turn)*speed, force=1000)
                p.setJointMotorControl2(turtle, 1, p.VELOCITY_CONTROL, targetVelocity=(forward+turn)*speed, force=1000)
                p.stepSimulation()
                
                # Get the robot's new state after applying the control input
                robot_state = get_robot_state(turtle)
                
                # Check if the robot has reached the goal
                if np.linalg.norm(robot_state[:2] - goal_state[:2]) < 0.1:
                    break
            
            # If the robot has reached the target, stop trying new paths
            if np.linalg.norm(robot_state[:2] - goal_state[:2]) < 0.1:
                break
        
    # If the robot has reached the target, stop the simulation
    if np.linalg.norm(robot_state[:2] - goal_state[:2]) < 0.1:
        break

# stop the simulation once the turtle reaches the target
p.disconnect()
