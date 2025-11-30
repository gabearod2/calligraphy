import os
import time
import numpy as np
import mengine as m
import control.control as c
import scipy.spatial.transform as sst
from trajectory_generation.handwriting_to_points import handwriting_to_points


# ---------------------------------------
# Control type and logging for evaluation
# ---------------------------------------

mode = "mpc_f" # "ik" "mpc" "mpc_f"
pos_des_hist = []
pos_meas_hist = []
thick_des_hist = []
thick_meas_hist = []

# ---------------------------------------
# Defining environment
# ---------------------------------------

sim_dt = 0.005
env = m.Env(time_step=sim_dt, seed=300)
ground = m.Ground()
m.visualize_coordinate_frame()
env.set_gui_camera(
    pitch=-30,
    distance=0.6,
    yaw=-60
)

# world objects 
table = m.URDF(
    filename=os.path.join(m.directory, 'table', 'table.urdf'),
    static=True, position=[0, 0, 0], orientation=[0, 0, 0, 1],
)
writing_pad = m.Shape(
    m.Box(half_extents=[0.3, 0.4, 0.01]), static=True, 
    position=[-0.1, 0, 0.745], orientation=[0, 0, 0, 1], rgba=[0, 0, 0, 1],
)
pen = m.URDF(
    filename=os.path.join(m.directory, 'pen', 'pen.urdf'),
    static=False, position=[-0.25, -0.3, 1.5],
    orientation=m.get_quaternion(euler=[0, np.pi/2, np.pi/2]),
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
    orientation_weight=1.0,
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
gripper_euler = default_euler + [np.pi/2, -np.pi/4, -np.pi/2]
gripper_orient_quat = m.get_quaternion(gripper_euler)

# move to pre-grasp pose
pre_grasp_offset = np.array([0.0, 0.0, 0.15])
pre_grasp_pos = pen_pos + pre_grasp_offset
controller.ik_move_to(pre_grasp_pos, gripper_orient_quat)

# move to grasp pose
grasp_offset = np.array([0.0, 0.0, -0.0055])
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
gripper_euler = default_euler + [np.pi/2, np.pi/4, 0]
gripper_orient_quat = m.get_quaternion(gripper_euler)
controller.ik_move_to(lift_pos, gripper_orient_quat)

# ---------------------------------------
# Generate handwriting and thickness trajectory
# ---------------------------------------

print("Generating the writing trajectory.")
thickness_scale = 5000
writing_pad_z = 0.755
x_displacement = -0.1


# # Testing staight-line trajectory
# for i in range(50):
#     x = -0.3 
#     y = -0.2 + i * 0.01
#     z = 0.755
#     thickness = 0.005
#     writing_trajectory.append([x, y, z])
#     thickness_trajectory.append([thickness])
#     m.Shape(
#         m.Sphere(radius=thickness),
#         static=True,
#         collision=False,
#         position=[x, y, z],z
#         rgba=[1, 0, 0, 0.5]
#     )

# Generating handwriting points
Xs, Ys, Ts, num_letters = handwriting_to_points(
    image_path="trajectory_generation/handwriting/G.jpg",
    plot=False,
)

# Find global means
all_xs = np.concatenate([np.array(xs) for xs in Xs])
all_ys = np.concatenate([np.array(ys) for ys in Ys])
global_x_mean = np.nanmean(all_xs)
global_y_mean = np.nanmean(all_ys)

for xs, ys, ts in zip(Xs, Ys, Ts):
    writing_trajectory = []
    thickness_trajectory = []

    # rotating handwriting points
    xs = np.array(xs)
    ys = np.array(ys)
    ts = np.array(ts)/thickness_scale
    valid_xs = xs[~np.isnan(xs)]
    valid_ys = ys[~np.isnan(ys)]
    xs_centered = valid_xs - global_x_mean
    ys_centered = valid_ys - global_y_mean
    rot = sst.Rotation.from_euler("zyx", [-np.pi/2, 0, np.pi])
    R = rot.as_matrix()
    X = np.array([xs_centered, ys_centered, np.zeros_like(xs_centered)])

    # plotting in simulator and adding to writing_trajectory list
    for i in range(len(valid_xs)):
        xi = R @ X[:, i].T
        x = xi[0] + x_displacement
        y = xi[1] 
        t = ts[i]
        z = writing_pad_z

        writing_trajectory.append([x, y, z])
        thickness_trajectory.append([t])
        if i % 20 == 0:
            m.Shape(
                m.Sphere(radius=t),
                static=True,
                collision=False,
                position=[x, y, z],
                rgba=[1, 1, 1, 0.25]
            )

    print("Finished plotting the desired contour.")
    writing_trajectory = np.array(writing_trajectory)
    thickness_trajectory = np.array(thickness_trajectory)
    first_point = writing_trajectory[0] + np.array([0.0, 0.0, 0.06]) 

    # ---------------------------------------
    # Follow the generated trajectory
    # ---------------------------------------

    print("IK to above the first writing waypoint. ")
    controller.ik_move_to(first_point, gripper_orient_quat)
    m.step_simulation(steps=100, realtime=True)
    print("Now, move down until first contact. ")
    controller.move_to_first_contact(first_point, gripper_orient_quat)
    print("Made contact. ")


    if mode=="mpc_f":
        force = True

        print("Starting MPC with Force Control.")
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

            # run mpc for this window
            controller.mpc_step(writing_seg, thickness_seg, force)
            m.step_simulation(steps=round(controller.dt / sim_dt), realtime=True)

            # logging
            pos_des = writing_seg[0]          
            thick_des = thickness_seg[0, 0] 
            pos_meas = controller.get_pen_tip_world()
            thick_meas = controller.get_thickness_meas()
            pos_des_hist.append(pos_des)
            pos_meas_hist.append(pos_meas)
            thick_des_hist.append(thick_des)
            thick_meas_hist.append(thick_meas)
    
    elif mode=="mpc":
        force = False

        print("Starting MPC without Force Control.")
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

            # run mpc for this window
            controller.mpc_step(writing_seg, thickness_seg, force)
            m.step_simulation(steps=round(controller.dt / sim_dt), realtime=True)

            # logging
            pos_des = writing_seg[0]          
            thick_des = thickness_seg[0, 0] 
            pos_meas = controller.get_pen_tip_world()
            thick_meas = controller.get_thickness_meas()
            pos_des_hist.append(pos_des)
            pos_meas_hist.append(pos_meas)
            thick_des_hist.append(thick_des)
            thick_meas_hist.append(thick_meas)

    elif mode=="ik":
        force = False

        print("Starting Inverse Kinematic Controller.")
        N = len(writing_trajectory)
        for k in range(N):
            writing_point = writing_trajectory[k]

            # run ik controller
            controller.ik_move_to_write(writing_point, gripper_orient_quat)

            # logging
            pos_des = writing_point
            pos_meas = controller.get_pen_tip_world()
            thick_des = thickness_trajectory[k, 0]
            thick_meas = controller.get_thickness_meas()
            pos_des_hist.append(pos_des)
            pos_meas_hist.append(pos_meas)
            thick_des_hist.append(thick_des)
            thick_meas_hist.append(thick_meas)

controller.ik_move_to(lift_pos, gripper_orient_quat)
m.step_simulation(steps=10000, realtime=True)

# error calculation
pos_des_hist = np.array(pos_des_hist)      
pos_meas_hist = np.array(pos_meas_hist)    
thick_des_hist = np.array(thick_des_hist)  
thick_meas_hist = np.array(thick_meas_hist)

# position error
e_pos = pos_meas_hist[:, :2] - pos_des_hist[:, :2] 
e_pos_norm = np.linalg.norm(e_pos, axis=1)

rmse_pos = np.sqrt(np.mean(e_pos_norm**2))
max_pos_err = np.max(e_pos_norm)

# thickness error
e_thick = thick_meas_hist - thick_des_hist 
rmse_thick = np.sqrt(np.mean(e_thick**2))
max_thick_err = np.max(np.abs(e_thick))

print(f"Method: {mode}")
print(f"Position RMSE (xy): {rmse_pos:.4f} m")
print(f"Position Max Error (xy): {max_pos_err:.4f} m")
print(f"Thickness RMSE: {rmse_thick:.6f} m")
print(f"Thickness Max Error: {max_thick_err:.6f} m")

# save results
np.savez(f"results_{mode}.npz",
         pos_des=pos_des_hist,
         pos_meas=pos_meas_hist,
         thick_des=thick_des_hist,
         thick_meas=thick_meas_hist,
         rmse_pos=rmse_pos,
         rmse_thick=rmse_thick)



