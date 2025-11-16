import os
import numpy as np
import mengine as m
import control as control


# ---------------------------------------
# Defining environment
# ---------------------------------------

env = m.Env(time_step=0.005, seed=300)
ground = m.Ground()
m.visualize_coordinate_frame()

table = m.URDF(
    filename=os.path.join(m.directory, 'table', 'table.urdf'),
    static=True,
    position=[0, 0, 0],
    orientation=[0, 0, 0, 1],
)
wall = m.Shape(
    m.Box(half_extents=[0.2, 0.2, 0.01]),
    static=False,
    position=[-0.2, 0, 0.745],
    orientation=[0, 0, 0, 1],
    rgba=[0, 1, 1, 0.75],
)
pen = m.Shape(
    m.Cylinder(radius=0.01, length=0.1), # m.Box(half_extents=[0.02, 0.1, 0.02]),
    static=False,
    mass=0.1,
    position=[0.0, -0.3, 1.5],
    orientation=m.get_quaternion(euler=[0, np.pi/2, 0]),
    rgba=[0, 0, 1, 0.75],
)
pen.set_whole_body_frictions(
    lateral_friction=1.0,
    spinning_friction=1.0,
    rolling_friction=1.0,
)
# let pen drop on the table
m.step_simulation(steps=100, realtime=True)

# create robot
robot = m.Robot.Jaco(position=[0.5, 0, 0.76])
robot.motor_forces = 100
robot.set_whole_body_frictions(
    lateral_friction=1.0, 
    spinning_friction=1.0, 
    rolling_friction=1.0)

# ---------------------------------------
# Controller definition
# ---------------------------------------

controller = control.Controller(
    robot=robot,
    motor_gains=0.05
)


# Home pose: above the table, looking forward
home_pos = [-0.2, 0.2, 1.2]
default_euler = np.array([np.pi/2, 0.0, 0.0])   
home_orient_quat = m.get_quaternion(default_euler)

home_joints = robot.ik(
    robot.end_effector,
    target_pos=home_pos,
    target_orient=home_orient_quat,
)
robot.control(home_joints, set_instantly=True)



# Open gripper
robot.set_gripper_position([0.7, 0.7, 0.7], set_instantly=True)

# grasp pose
pen_pos, pen_orient = pen.get_base_pos_orient()
print(pen_orient)
pen_euler = m.get_euler(pen_orient)
print(pen_euler)
pen_yaw = pen_euler[-1]
gripper_euler = default_euler + [np.pi/2, 0, 0]
gripper_orient_quat = m.get_quaternion(gripper_euler)

# move to pre-grasp pose
pre_grasp_offset = np.array([0.0, 0.0, 0.15])
pre_grasp_pos = pen_pos + pre_grasp_offset
controller.moveto(pre_grasp_pos, gripper_orient_quat)

# m.step_simulation(steps=10000, realtime=True)


# move to grasp pose
grasp_offset = np.array([0.0, 0.0, 0.03])
grasp_pos = pen_pos + grasp_offset
controller.moveto(grasp_pos, gripper_orient_quat)

# close gripper (pen pick up)
m.step_simulation(steps=100, realtime=True)

robot.set_gripper_position([1.35, 1.35, 1.35], force=5000)
m.step_simulation(steps=100, realtime=True)

# lift pen
lift_offset = np.array([0.0, 0.0, 0.25])
lift_pos = pen_pos + lift_offset
controller.moveto(lift_pos, gripper_orient_quat)

print('Done picking up the pen.')

# run sim
m.step_simulation(steps=10000, realtime=True)
