import os
import time
import numpy as np
import mengine as m
import control as control


# ---------------------------------------
# Defining environment
# ---------------------------------------

env = m.Env(time_step=0.02, seed=300)
ground = m.Ground()
m.visualize_coordinate_frame()

table = m.URDF(
    filename=os.path.join(m.directory, 'table', 'table.urdf'),
    static=True,
    position=[0, 0, 0],
    orientation=[0, 0, 0, 1],
)

# create robot
robot = m.Robot.Panda(position=[0.5, 0, 0.76])
robot.motor_forces = 100
robot.set_whole_body_frictions(
    lateral_friction=0.5, 
    spinning_friction=1.0, 
    rolling_friction=1.0
)

# ---------------------------------------
# Controller definition
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
# Setting home pose
# ---------------------------------------

home_pos = [-0.4, 0.0, 1.2]
default_euler = np.array([0.0, -np.pi/2, 0.0])   
home_orient_quat = m.get_quaternion(default_euler)
home_joints = robot.ik(
    robot.end_effector,
    target_pos=home_pos,
    target_orient=home_orient_quat,
)
robot.control(home_joints, set_instantly=True)
m.step_simulation(steps=100, realtime=True)

# ---------------------------------------
# Follow trajectory
# ---------------------------------------
# TODO: add desired thickness of the written segment.
# would require moving the trajectory onto a surface.
# would also require getting the normals on the pen. 

""" STRAIGHT LINE TRAJECTORY """
traj = []
N_traj = 20
for i in range(N_traj):
    traj.append([
        home_pos[0] ,  
        home_pos[1],              
        home_pos[2] - 0.01 * i,              
    ])
    m.Shape(m.Sphere(radius=0.01), static=True, collision=False,
        position=[
            home_pos[0],  
            home_pos[1],              
            home_pos[2] - 0.01 * i,              
        ], rgba=[1, 0, 0, 1]
    )
traj = np.array(traj)

""" CIRCLE TRAJECTORY """
# traj = []
# N_traj = 20
# diameter = 0.01 * (N_traj - 1)
# radius = diameter / 2
# thetas = np.linspace(0, 2*np.pi, N_traj)

# for theta in thetas:
#     point = [
#         home_pos[0],
#         home_pos[1] + radius * np.cos(theta),
#         home_pos[2] + radius * np.sin(theta)
#     ]
#     traj.append(point)

#     m.Shape(
#         m.Sphere(radius=0.01),
#         static=True,
#         collision=False,
#         position=point,
#         rgba=[1, 0, 0, 1]
#     )
# traj = np.array(traj)


# TODO: Find out how to make sure it is time consistent? Like how do I know how far along I am?
# Should I be manually stepping the simulation to match the defined dt of the controller?
total_steps = 500
m.step_simulation(realtime=False)

for t in range(total_steps):
    traj_time = t * controller.dt
    idx = int(traj_time / controller.dt)
    idx = min(idx, N_traj - 1)

    # Local horizon
    H = controller.N
    ref_segment = traj[idx:idx + H]
    if ref_segment.shape[0] < H:
        last = ref_segment[-1]
        pad = np.tile(last, (H - ref_segment.shape[0], 1))
        ref_segment = np.vstack([ref_segment, pad])

    # control using mpc...
    controller.mpc_step(ref_segment)
    time.sleep(controller.dt)
    m.step_simulation(steps=1, realtime=False)

m.step_simulation(steps=10000, realtime=True)
