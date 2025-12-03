import os
import numpy as np
import mengine as m
import control.control as c
import scipy.spatial.transform as sst
from trajectory_generation.handwriting_to_points import handwriting_to_points



handwriting_dir = "trajectory_generation/handwriting"
thickness_scale = 5000
writing_pad_z = 0.755
x_displacement = -0.1
sim_dt = 0.005


def build_environment():
    """Construct a fresh simulation environment, robot, pen, table, pad, and controller."""
    env = m.Env(time_step=sim_dt, seed=300)
    ground = m.Ground()
    m.visualize_coordinate_frame()
    env.set_gui_camera(pitch=-30, distance=0.6, yaw=-60)

    # table + writing pad
    table = m.URDF(
        filename=os.path.join(m.directory, 'table', 'table.urdf'),
        static=True, position=[0, 0, 0], orientation=[0, 0, 0, 1],
    )

    writing_pad = m.Shape(
        m.Box(half_extents=[0.3, 0.4, 0.01]), static=True,
        position=[-0.1, 0, 0.745], orientation=[0, 0, 0, 1],
        rgba=[0, 0, 0, 1],
    )

    # pen
    pen = m.URDF(
        filename=os.path.join(m.directory, 'pen', 'pen.urdf'),
        static=False, position=[-0.25, -0.3, 1.5],
        orientation=m.get_quaternion(euler=[0, np.pi/2, np.pi/2]),
    )

    # frictions
    writing_pad.set_whole_body_frictions(0, 0, 0)
    pen.set_whole_body_frictions(2000, 2000, 2000)

    m.step_simulation(steps=100, realtime=True)

    # Panda robot
    robot = m.Robot.Panda(position=[0.5, 0, 0.76])
    robot.motor_forces = 100

    # Controller
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
        reference_weight=0.5,
    )

    return env, robot, pen, writing_pad, controller

def pen_not_in_contact(robot, pen):
    cps = robot.get_contact_points(bodyB=pen)
    return (cps is None) or (len(cps) == 0)

def pickup_pen(robot, pen, controller):
    """Move robot into position, grasp pen, lift, and rotate into writing orientation."""

    print("Picking up the pen.")

    home_pos = [-0.2, 0.2, 1.2]
    default_euler = np.array([np.pi/2, 0.0, 0.0])
    home_orient_quat = m.get_quaternion(default_euler)

    # Move instantly to home
    home_joints = robot.ik(robot.end_effector, home_pos, home_orient_quat)
    robot.control(home_joints, set_instantly=True)
    robot.set_gripper_position([1]*2, set_instantly=True)

    # Pre-grasp → grasp pose
    pen_pos, pen_orient = pen.get_base_pos_orient()
    gripper_euler = default_euler + [np.pi/2, -np.pi/4, -np.pi/2]
    gripper_orient_quat = m.get_quaternion(gripper_euler)

    pre_grasp = pen_pos + np.array([0, 0, 0.15])
    controller.ik_move_to(pre_grasp, gripper_orient_quat)

    grasp_pos = pen_pos + np.array([0, 0, -0.0055])
    controller.ik_move_to(grasp_pos, gripper_orient_quat)
    m.step_simulation(steps=100, realtime=True)

    robot.set_gripper_position([0]*2, force=10000)
    m.step_simulation(steps=100, realtime=True)

    # Lift pen
    lift_pos = pen_pos + np.array([0, 0, 0.35])
    controller.ik_move_to(lift_pos, gripper_orient_quat)

    # rotate into writing orientation
    gripper_euler = default_euler + [np.pi/2, np.pi/4, 0]
    writing_orient = m.get_quaternion(gripper_euler)
    controller.ik_move_to(lift_pos, writing_orient)

    return writing_orient, lift_pos

image_files = sorted(
    f for f in os.listdir(handwriting_dir)
    if f.lower().endswith((".jpg", ".png"))
)
indexed_files = list(enumerate(image_files, start=1))

print("\n============== HANDWRITING IMAGES FOUND ==============")
for idx, f in indexed_files:
    print(f"  img{idx:03d} → {f}")
print("=======================================================\n")

for file_idx, fname in indexed_files:

    print(f"\n=========== RESETTING SIM FOR IMAGE #{file_idx:03d} — {fname} ===========")

    # fresh environment every word
    try:
        env.disconnect()
    except:
        pass
    env, robot, pen, writing_pad, controller = build_environment()

    # track indices for this word
    pos_des_hist = []
    pos_meas_hist = []
    thick_des_hist = []
    thick_meas_hist = []
    start_idx = len(pos_des_hist)
    lost_pen = False

    # pick up pen
    writing_orient, lift_pos = pickup_pen(robot, pen, controller)

    # load handwriting
    image_path = os.path.join(handwriting_dir, fname)
    Xs, Ys, Ts, num_letters = handwriting_to_points(image_path, plot=False)

    # center handwriting globally
    all_xs = np.concatenate([np.array(xs) for xs in Xs])
    all_ys = np.concatenate([np.array(ys) for ys in Ys])
    global_x_mean = np.nanmean(all_xs)
    global_y_mean = np.nanmean(all_ys)

    # -----------------------------------------
    # Loop through contours within this word
    # -----------------------------------------
    for contour_idx, (xs, ys, ts) in enumerate(zip(Xs, Ys, Ts), start=1):

        print(f"\n--- Contour {contour_idx}/{num_letters} in img{file_idx:03d} ---")

        xs = np.array(xs)
        ys = np.array(ys)
        ts = np.array(ts) / thickness_scale

        valid = ~np.isnan(xs)
        xs = xs[valid] - global_x_mean
        ys = ys[valid] - global_y_mean
        ts = ts[valid]

        # rotate
        R = sst.Rotation.from_euler("zyx", [-np.pi/2, 0, np.pi]).as_matrix()
        X = np.vstack([xs, ys, np.zeros_like(xs)])

        writing_trajectory = []
        thickness_trajectory = []

        for i in range(len(xs)):
            xi = R @ X[:, i]
            writing_trajectory.append([
                xi[0] + x_displacement,
                xi[1],
                writing_pad_z
            ])
            thickness_trajectory.append([ts[i]])

        writing_trajectory = np.array(writing_trajectory)
        thickness_trajectory = np.array(thickness_trajectory)

        # move above start point
        first_point = writing_trajectory[0] + np.array([0,0,0.06])
        controller.ik_move_to(first_point, writing_orient)
        m.step_simulation(steps=100, realtime=True)

        controller.move_to_first_contact(first_point, writing_orient)


        # follow trajectory
        H = controller.n_p
        N = len(writing_trajectory)

        for k in range(N):

            # safety: pen lost
            if pen_not_in_contact(robot, pen):
                print("\n!!! PEN LOST — CANCELING CURRENT WORD !!!\n")
                lost_pen = True
                break

            w_seg = writing_trajectory[k:k+H].copy()
            t_seg = thickness_trajectory[k:k+H].copy()

            # pad to horizon
            if len(w_seg) < H:
                pad = H - len(w_seg)
                w_seg = np.vstack([w_seg, np.tile(w_seg[-1], (pad,1))])
                t_seg = np.vstack([t_seg, np.tile(t_seg[-1], (pad,1))])

            controller.mpc_step(w_seg, t_seg, True)
            m.step_simulation(steps=round(controller.dt/sim_dt), realtime=True)

            pos_des_hist.append(w_seg[0])
            pos_meas_hist.append(controller.get_pen_tip_world())
            thick_des_hist.append(t_seg[0,0])
            thick_meas_hist.append(controller.get_thickness_meas())

        # break out of contour loop if pen lost
        if lost_pen:
            break

        print(f"Completed contour {contour_idx}/{num_letters}.")

    print(f"Finished processing image img{file_idx:03d}.\n")
    word_success = not lost_pen

    # record this word’s index range and success
    end_idx = len(pos_des_hist)

    # compute errors for the current word
    pos_des_hist = np.array(pos_des_hist)
    pos_meas_hist = np.array(pos_meas_hist)
    thick_des_hist = np.array(thick_des_hist)
    thick_meas_hist = np.array(thick_meas_hist)

    # compute only on valid (non-NaN) entries
    valid = ~np.isnan(pos_des_hist[:,0])

    e_pos = pos_meas_hist[valid, :2] - pos_des_hist[valid, :2]
    e_norm = np.linalg.norm(e_pos, axis=1)
    rmse_pos = np.sqrt(np.mean(e_norm**2))
    max_pos_err = np.max(e_norm)

    e_th = thick_meas_hist[valid] - thick_des_hist[valid]
    rmse_thick = np.sqrt(np.mean(e_th**2))
    max_thick_err = np.max(np.abs(e_th))

    print("\n================ FINAL RESULTS ================")
    print(f"Word: {fname}")
    print(f"Position RMSE (xy):   {rmse_pos:.4f} m")
    print(f"Max Position Error:   {max_pos_err:.4f} m")
    print(f"Thickness RMSE:       {rmse_thick:.6f}")
    print(f"Max Thickness Error:  {max_thick_err:.6f}")
    print("================================================\n")

    # save
    os.makedirs("results", exist_ok=True)
    word =fname.removesuffix(".jpg")

    np.savez(
        f"results/results_{word}.npz",
        pos_des=pos_des_hist,
        pos_meas=pos_meas_hist,
        thick_des=thick_des_hist,
        thick_meas=thick_meas_hist,
        rmse_pos=rmse_pos,
        rmse_thick=rmse_thick,
        success=word_success,  
        filename=fname,
    )
    print(f"Saved results ==> results/results_{word}_.npz")
