import os
import numpy as np
import mengine as m


def moveto(robot, pos, orient_quat):
    """Simple IK-based move: go to (pos, orient_quat) and wait until joint error is small."""
    print('[moveto] Using inverse kinematics controller...')
    robot.motor_gains = 0.15
    target_joint_angles = robot.ik(
        robot.end_effector,
        target_pos=pos,
        target_orient=orient_quat,
        use_current_joint_angles=True,
    )
    robot.control(target_joint_angles)
    # Wait until joints reach targets (within a small tolerance)
    while np.linalg.norm(robot.get_joint_angles(robot.controllable_joints) - target_joint_angles) > 0.03:
        m.step_simulation(realtime=True)


# --------------------------------------------------------------------------
# 1) Environment setup
# --------------------------------------------------------------------------
env = m.Env(seed=300)
ground = m.Ground()
m.visualize_coordinate_frame()

# Table
table = m.URDF(
    filename=os.path.join(m.directory, 'table', 'table.urdf'),
    static=True,
    position=[0, 0, 0],
    orientation=[0, 0, 0, 1],
)

# Just a visual wall (not strictly needed for the pen pickup)
wall = m.Shape(
    m.Box(half_extents=[0.2, 0.2, 0.04]),
    static=False,
    position=[-0.2, 0, 0.745],
    orientation=[0, 0, 0, 1],
    rgba=[0, 1, 1, 0.75],
)

# --------------------------------------------------------------------------
# 2) Create the pen and let it drop
# --------------------------------------------------------------------------
pen = m.Shape(
    m.Box(half_extents=[0.02, 0.1, 0.02]),
    static=False,
    mass=1,
    position=[0.0, -0.3, 1.0],
    orientation=m.get_quaternion([0, 0, 0]),  # whatever, just some initial orientation
    rgba=[0, 0, 1, 0.75],
)
pen.set_whole_body_frictions(
    lateral_friction=0.3,
    spinning_friction=0.05,
    rolling_friction=0.05,
)

# Let the pen settle on the table
m.step_simulation(steps=50, realtime=True)

# --------------------------------------------------------------------------
# 3) Create Panda robot and home pose
# --------------------------------------------------------------------------
robot = m.Robot.Panda(position=[0.5, 0, 0.76])
robot.motor_forces = 100

# Home pose: above the table, looking straight down (π about x)
home_pos = [-0.2, 0.2, 1.2]
default_euler = np.array([np.pi/2, 0.0, 0.0])   # roll=π, pitch=0, yaw=0
home_orient_quat = m.get_quaternion(default_euler)

home_joints = robot.ik(
    robot.end_effector,
    target_pos=home_pos,
    target_orient=home_orient_quat,
)
robot.control(home_joints, set_instantly=True)

# Open gripper
robot.set_gripper_position([1.0, 1.0], set_instantly=True)

# --------------------------------------------------------------------------
# 4) Compute grasp orientation from pen yaw
# --------------------------------------------------------------------------
cube_pos, cube_orient_quat = pen.get_base_pos_orient()  # "cube" == pen base
pen_euler = m.get_euler(cube_orient_quat)               # [roll, pitch, yaw] in same units as get_quaternion
pen_yaw = pen_euler[-1]

# Use the default downward orientation but match the pen's yaw
gripper_euler = default_euler + [0, 0, pen_yaw]
gripper_orient_quat = m.get_quaternion(gripper_euler)

# --------------------------------------------------------------------------
# 5) Pick up the pen:
#    - move above the pen
#    - move down to grasp
#    - close gripper
#    - lift pen
# --------------------------------------------------------------------------

# 5a) Move to pre-grasp pose (slightly above the pen)
pre_grasp_offset = np.array([0.0, 0.0, 0.15])
pre_grasp_pos = cube_pos + pre_grasp_offset
moveto(robot, pre_grasp_pos, gripper_orient_quat)

# 5b) Move down to grasp position (just above the table/pen)
grasp_offset = np.array([0.0, 0.0, 0.01])
grasp_pos = cube_pos + grasp_offset
moveto(robot, grasp_pos, gripper_orient_quat)

# 5c) Close the gripper to grab the pen
robot.set_gripper_position([0.0, 0.0], force=5000)
m.step_simulation(steps=100, realtime=True)

# 5d) Lift the pen straight up
lift_offset = np.array([0.0, 0.0, 0.25])
lift_pos = cube_pos + lift_offset
moveto(robot, lift_pos, gripper_orient_quat)

print('Done picking up the pen.')

# --------------------------------------------------------------------------
# 6) Let the simulation run so you can see the result
# --------------------------------------------------------------------------
m.step_simulation(steps=10000, realtime=True)
