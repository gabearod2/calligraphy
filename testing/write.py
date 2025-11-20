import os
import time
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
    static=True, position=[0, 0, 0], orientation=[0, 0, 0, 1],
)
writing_pad = m.Shape(
    m.Box(half_extents=[0.2, 0.2, 0.01]),static=False, 
    position=[-0.2, 0, 0.745],orientation=[0, 0, 0, 1], rgba=[0, 1, 1, 0.75],
)
pen = m.Shape(
    m.Box(half_extents=[0.015, 0.1, 0.015]), # m.Cylinder(radius=0.015, length=0.1), 
    static=False, mass=1.0, position=[0.0, -0.3, 1.5],
    orientation=m.get_quaternion(euler=[0, np.pi/2, np.pi/2]),
    rgba=[0, 0, 1, 0.75],
)
pen.set_whole_body_frictions(
    lateral_friction=2000, 
    spinning_friction=2000, 
    rolling_friction=0.5
)
# let pen drop on the table
m.step_simulation(steps=100, realtime=True)
pen_tip_offset = np.array([0.0, -0.05, 0.0])  # depends on orientation!


# create robot
robot = m.Robot.Panda(position=[0.5, 0, 0.76])
robot.motor_forces = 100

# ---------------------------------------
# intialize controller definition
# ---------------------------------------

controller = control.Controller(
    robot=robot,
    motor_gains=0.05,
    dt=0.02,
    horizon=10,
    w_e=1.0,
    w_d=0.25,
    w_a=0.1,
    w_q=0.5
)

# ---------------------------------------
# Pick up writing utensil
# ---------------------------------------

# set to home robot position instantly
home_pos = [-0.2, 0.2, 1.2]
default_euler = np.array([np.pi/2, 0.0, 0.0])   
home_orient_quat = m.get_quaternion(default_euler)

home_joints = robot.ik(
    robot.end_effector,
    target_pos=home_pos,
    target_orient=home_orient_quat,
)
robot.control(home_joints, set_instantly=True)
robot.set_gripper_position([1]*2, set_instantly=True)

# defining grasp pose
pen_pos, pen_orient = pen.get_base_pos_orient()
pen_euler = m.get_euler(pen_orient)
pen_yaw = pen_euler[-1]
gripper_euler = default_euler + [np.pi/2, 0, 0]
gripper_orient_quat = m.get_quaternion(gripper_euler)

# move to pre-grasp pose
pre_grasp_offset = np.array([0.0, 0.0, 0.15])
pre_grasp_pos = pen_pos + pre_grasp_offset
controller.moveto(pre_grasp_pos, gripper_orient_quat)

# move to grasp pose
grasp_offset = np.array([0.0, 0.0, -0.005])
grasp_pos = pen_pos + grasp_offset
controller.moveto(grasp_pos, gripper_orient_quat)

# close gripper
m.step_simulation(steps=100, realtime=True)
robot.set_gripper_position([0]*2, force=10000)
m.step_simulation(steps=100, realtime=True)

# move to post-grasp pose
lift_offset = np.array([0.0, 0.0, 0.25])
lift_pos = pen_pos + lift_offset
controller.moveto(lift_pos, gripper_orient_quat)

# rotate to writing orientation
gripper_euler = default_euler + [np.pi/2, np.pi/2, 0]
gripper_orient_quat = m.get_quaternion(gripper_euler)
controller.moveto(lift_pos, gripper_orient_quat)

# run sim
m.step_simulation(steps=10000, realtime=True)

# ---------------------------------------
# Generate handwriting trajectory
# ---------------------------------------
'''
For now, just a circle on the writing_pad surface.
'''

# Writing surface height
pad_z = 0.745 + 0.01  # top of pad

# draw circle of radius 5 cm
radius = 0.05
thetas = np.linspace(0, 2*np.pi, 200)

traj_world = []
for th in thetas:
    x = -0.2              # constant X over pad
    y = 0.0 + radius*np.cos(th)
    z = pad_z + 0.002     # barely touching

    traj_world.append([x, y, z])

traj_world = np.array(traj_world)
traj_ee = traj_world - pen_tip_offset

# ---------------------------------------
# Follow the generated trajectory
# ---------------------------------------
'''
Need to account for FK from the ee to the tip of the pen.
Should be a -90 degree rotation about the y axis
'''
H = controller.N

for t in range(len(traj_ee)):
    seg = traj_ee[t:t+H]
    if len(seg) < H:
        seg = np.vstack([seg, np.tile(seg[-1], (H-len(seg), 1))])

    controller.mpc_step(seg)
    m.step_simulation(steps=1, realtime=False)
    time.sleep(controller.dt)
