import os
import time
import numpy as np
import mengine as m
import control as c


# ---------------------------------------
# Defining environment
# ---------------------------------------

sim_dt = 0.005
env = m.Env(time_step=sim_dt, seed=300)
ground = m.Ground()
m.visualize_coordinate_frame()

# world objects 
table = m.URDF(
    filename=os.path.join(m.directory, 'table', 'table.urdf'),
    static=True, position=[0, 0, 0], orientation=[0, 0, 0, 1],
)
writing_pad = m.Shape(
    m.Box(half_extents=[0.3, 0.4, 0.01]), static=True, 
    position=[-0.1, 0, 0.745], orientation=[0, 0, 0, 1], rgba=[0, 0, 0, 0.75],
)
pen = m.Shape(
    m.Box(half_extents=[0.02, 0.05, 0.01]), # m.Cylinder(radius=0.015, length=0.1), 
    static=False, mass=1.0, position=[-0.2, -0.3, 1.5],
    orientation=m.get_quaternion(euler=[0, np.pi/2, 0]),
    rgba=[1, 1, 1, 1],
)

# setting friction
writing_pad.set_whole_body_frictions(
    lateral_friction=0, 
    spinning_friction=0, 
    rolling_friction=0
)
pen.set_whole_body_frictions(
    lateral_friction=2000, 
    spinning_friction=2000, 
    rolling_friction=2000
)
# let pen drop on the table
m.step_simulation(steps=100, realtime=True)

# create robot
robot = m.Robot.Panda(position=[0.5, 0, 0.76])
robot.motor_forces = 100

# ---------------------------------------
# intialize controller definition
# ---------------------------------------

controller = c.Controller(
    robot=robot,
    pen=pen,
    writing_pad=writing_pad, 
    motor_gains=0.05,
    dt=0.02,
    horizon=10,
    position_weight=1.0,
    velocity_weight=0.1,
    acceleration_weight=0.1,
    reference_weight=0.5
)

# ---------------------------------------
# Pick up writing utensil
# ---------------------------------------

print("Picking up the pen.")
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
gripper_euler = default_euler + [np.pi/2, 0, -np.pi/2]
gripper_orient_quat = m.get_quaternion(gripper_euler)

# move to pre-grasp pose
pre_grasp_offset = np.array([0.0, 0.0, 0.15])
pre_grasp_pos = pen_pos + pre_grasp_offset
controller.ik_move_to(pre_grasp_pos, gripper_orient_quat)

# move to grasp pose
grasp_offset = np.array([0.0, 0.015, -0.0065])
grasp_pos = pen_pos + grasp_offset
controller.ik_move_to(grasp_pos, gripper_orient_quat)

# close gripper
m.step_simulation(steps=100, realtime=True)
robot.set_gripper_position([0]*2, force=10000)
m.step_simulation(steps=100, realtime=True)

# move to post-grasp pose
lift_offset = np.array([0.0, 0.0, 0.35])
lift_pos = pen_pos + lift_offset
controller.ik_move_to(lift_pos, gripper_orient_quat)

# rotate to writing orientation
gripper_euler = default_euler + [np.pi/2, np.pi/2, 0]
gripper_orient_quat = m.get_quaternion(gripper_euler)
controller.ik_move_to(lift_pos, gripper_orient_quat)

# ---------------------------------------
# Generate handwriting and thickness trajectory
# ---------------------------------------

print("Generating the writing trajectory.")
writing_trajectory = []
thickness_trajectory = []
for i in range(50):
    x = -0.3 
    y = -0.2 + i * 0.01
    z = 0.765
    thickness = 0.005

    writing_trajectory.append([x, y, z])
    thickness_trajectory.append([thickness])
    m.Shape(
        m.Sphere(radius=thickness),
        static=True,
        collision=False,
        position=[x, y, z],
        rgba=[1, 0, 0, 0.5]
    )
writing_trajectory = np.array(writing_trajectory)
thickness_trajectory = np.array(thickness_trajectory)
first_point = writing_trajectory[0] + np.array([0.0, 0.0, 0.09])

# ---------------------------------------
# Follow the generated trajectory
# ---------------------------------------

print("IK to above the first writing waypoint. ")
controller.ik_move_to(first_point, gripper_orient_quat)
m.step_simulation(steps=100, realtime=True)
print("Now, move down until first contact. ")
controller.move_to_first_contact(first_point, gripper_orient_quat)
print("Made contact. ")

print("Starting MPC.")
H = controller.n_p  
N = len(writing_trajectory)
for k in range(N):
    # pull segments
    writing_seg = writing_trajectory[k : k + H].copy()
    thickness_seg = thickness_trajectory[k : k + H].copy()

    # pad if needed
    if len(thickness_seg) < H:
        pad_count = H - len(thickness_seg)
        thickness_seg = np.vstack([thickness_seg, np.tile(thickness_seg[-1], (pad_count, 1))])
        writing_seg = np.vstack([writing_seg, np.tile(writing_seg[-1], (pad_count, 1))])

    # Run MPC for this window
    controller.mpc_step(writing_seg, thickness_seg)
    m.step_simulation(steps=round(controller.dt / sim_dt), realtime=True)

