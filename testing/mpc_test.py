import os
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
wall = m.Shape(
    m.Box(half_extents=[0.2, 0.2, 0.01]),
    static=False,
    position=[-0.2, 0, 0.745],
    orientation=[0, 0, 0, 1],
    rgba=[0, 1, 1, 0.75],
)
pen = m.Shape(
    m.Cylinder(radius=0.015, length=0.1), # m.Box(half_extents=[0.02, 0.1, 0.02]),
    static=False,
    mass=0.1,
    position=[0.0, -0.3, 1.5],
    orientation=m.get_quaternion(euler=[0, np.pi/2, 0]),
    rgba=[0, 0, 1, 0.75],
)
pen.set_whole_body_frictions(
    lateral_friction=0.5,
    spinning_friction=0.5,
    rolling_friction=0.5,
)
# let pen drop on the table
m.step_simulation(steps=100, realtime=True)

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
    motor_gains=0.05
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
# Follow Trajectory
# ---------------------------------------

print(robot.get_motor_joint_states())
print(robot.get_joint_angles())

traj = []
N_traj = 40
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


total_steps = 40
for t in range(total_steps):
    idx = min(t, N_traj - 1)

    # Local horizon
    H = controller.N
    ref_segment = traj[idx:idx + H]
    if ref_segment.shape[0] < H:
        last = ref_segment[-1]
        pad = np.tile(last, (H - ref_segment.shape[0], 1))
        ref_segment = np.vstack([ref_segment, pad])

    # control using mpc...
    controller.mpc_step(ref_segment)
    m.step_simulation(steps=100, realtime=True)

m.step_simulation(steps=10000, realtime=True)
