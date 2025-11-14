import os
import numpy as np
import mengine as m

class Node:
    def __init__(self, joint_angles, parent=None):
        self.angles = joint_angles
        self.parent = parent

    def retrace(self):
        sequence = []
        node = self
        while node is not None:
            sequence.append(node)
            node = node.parent
        return sequence[::-1] # Reverse the order of sequence


def links_in_collision(body1, link1, body2, link2, max_distance=1e-1):
    _, _, _, _, dists = body1.get_closest_points(body2, distance=max_distance, linkA=link1, linkB=link2)
    return len(dists) != 0


def robot_in_collision(q):
    prev_joint_angles = robot.get_joint_angles(robot.controllable_joints)
    robot.control(q, set_instantly=True)

    # joint limits
    if np.any(q < robot.ik_lower_limits[:7]) or np.any(q > robot.ik_upper_limits[:7]):
        robot.control(prev_joint_angles, set_instantly=True)
        return True

    # robot-obstacle collision
    for obstacle in obstacles:
        if len(robot.get_closest_points(obstacle, distance=0)[-1]) != 0:
            robot.control(prev_joint_angles, set_instantly=True)
            return True
        
    # robot self-collision
    # link_indices = list(range(len(robot.all_joints)))
    # for i in link_indices:
    #     for j in link_indices:
    #         if abs(i - j) <= 1:
    #             continue
    #         if links_in_collision(robot, i, robot, j):
    #             robot.control(prev_joint_angles, set_instantly=True)
    #             return True

    robot.control(prev_joint_angles, set_instantly=True)
    return False


def generate_path(q1, q2, step_size=0.05):
    """Returns a list of robot joint angles from q1 to q2."""
    direction = q2 - q1
    distance = np.linalg.norm(direction)
    num_steps = int(distance / step_size)
    q = q1
    yield q1
    for i in range(num_steps):
        q = (1. / (num_steps - i)) * np.array(q2 - q) + q
        yield q


def extend(tree, target):
    """Takes current tree and extend it towards a new node (`target`).
    """
    closest_node = min(tree, key=lambda n: np.linalg.norm(n.angles - target))
    for q in generate_path(closest_node.angles, target):
        if robot_in_collision(q):
            return closest_node, False
        closest_node = Node(q, parent=closest_node)
        tree.append(closest_node)
    return closest_node, True


def random_sample_config():
    return np.random.uniform(robot.ik_lower_limits[:7], robot.ik_upper_limits[:7])

def distance(q1, q2):
    return np.linalg.norm(q1 - q2)

def nearest(tree, q):
    """
    Return the node in `tree` whose joint-angle vector is closest to q (L2 in joint space).
    """
    q = np.asarray(q, dtype=float)
    return min(tree, key=lambda n: distance(n.angles, q))

def steer(q_from, q_to, eta=0.2):
    """Return configuration moved from q_from toward q_to by at most eta."""
    d = q_to - q_from
    L = np.linalg.norm(d)
    if L < 1e-9:
        return q_from.copy()
    step = d * min(eta / L, 1.0)
    return q_from + step


def is_path_collision_free(q1, q2, step_size=0.05):
    """Check straight-line edge for collisions."""
    for q in generate_path(q1, q2, step_size):
        if robot_in_collision(q):
            return False
    return True


def find_neighbors(tree, q, radius):
    """Return all nodes within radius (joint-space L2)."""
    r2 = radius ** 2
    out = []
    for n in tree:
        if np.sum((n.angles - q) ** 2) <= r2:
            out.append(n)
    return out


def rrt_star(init, goal, max_iterations=2000):
    """Single-tree RRT* in joint space."""
    dof = len(init)
    eta = 0.20          # max step per iteration
    step_size = 0.2    # for collision checking
    gamma = 2.         # radius scaling
    goal_snap = 0.25    # if close enough, connect

    root = Node(np.array(init, dtype=float))
    root.cost = 0.0
    tree = [root]
    q_goal = np.array(goal, dtype=float)
    goal_node = None

    for k in range(max_iterations):
        # Sample
        q_rand = random_sample_config()

        # Nearest
        x_nearest = nearest(tree, q_rand)
        q_new = steer(x_nearest.angles, q_rand, eta)

        # Check edge feasibility
        if not is_path_collision_free(x_nearest.angles, q_new, step_size):
            continue

        # Create new node, temporarily attach to nearest
        x_new = Node(q_new, parent=x_nearest)
        x_new.cost = getattr(x_nearest, "cost", 0.0) + distance(x_nearest.angles, q_new)

        # Find nearby nodes
        n = len(tree) + 1
        radius = min(eta, gamma * (np.log(max(n, 2)) / max(n, 2)) ** (1.0 / dof))
        near_set = find_neighbors(tree, q_new, radius)

        # Choose best parent
        best_parent = x_nearest
        best_cost = x_new.cost
        for x_near in near_set:
            if is_path_collision_free(x_near.angles, q_new, step_size):
                cand_cost = getattr(x_near, "cost", 0.0) + distance(x_near.angles, q_new)
                if cand_cost < best_cost:
                    best_parent = x_near
                    best_cost = cand_cost
        x_new.parent = best_parent
        x_new.cost = best_cost
        tree.append(x_new)

        # Rewire nearby nodes through x_new if cheaper
        for x_near in near_set:
            new_cost = x_new.cost + distance(x_new.angles, x_near.angles)
            if new_cost + 1e-9 < getattr(x_near, "cost", np.inf) and \
               is_path_collision_free(x_new.angles, x_near.angles, step_size):
                x_near.parent = x_new
                x_near.cost = new_cost

        # Try connecting to goal
        if distance(x_new.angles, q_goal) < goal_snap and \
           is_path_collision_free(x_new.angles, q_goal, step_size):
            goal_node = Node(q_goal, parent=x_new)
            goal_node.cost = x_new.cost + distance(x_new.angles, q_goal)
            break

    # Reconstruct path
    if goal_node is None:
        # Pick nearest feasible to goal
        best = min(tree, key=lambda n: distance(n.angles, q_goal))
        goal_node = Node(q_goal, parent=best)
        goal_node.cost = getattr(best, "cost", 0.0) + distance(best.angles, q_goal)

    return [n.angles for n in goal_node.retrace()]



def is_ee_close(robot, joint_angles, pos, orient):
    """Returns True if the end effector is close to the given position and orientation."""
    prev_joint_angles = robot.get_joint_angles(robot.controllable_joints)
    robot.control(joint_angles, set_instantly=True)
    ee_pos, ee_orient = robot.get_link_pos_orient(robot.end_effector)
    robot.control(prev_joint_angles, set_instantly=True)
    return np.linalg.norm(ee_pos - pos) < 0.01 and (
                np.linalg.norm(ee_orient - orient) < 0.01 or np.linalg.norm(ee_orient + orient) < 0.01)


def moveto(robot, pos, orient, avoid_collision=False, max_iter=100, max_path_length=150):
    if not avoid_collision:
        print('Using simple move')
        robot.motor_gains = 0.05
        target_joint_angles = robot.ik(robot.end_effector, target_pos=pos, target_orient=orient,
                                       use_current_joint_angles=True)
        robot.control(target_joint_angles)
        while np.linalg.norm(robot.get_joint_angles(robot.controllable_joints) - target_joint_angles) > 0.03:
            m.step_simulation(realtime=True)
        return

    print('Using RRT connect')
    for i in range(max_iter):
        print('moveto iteration %d' % i)
        target_joint_angles = robot.ik(robot.end_effector, target_pos=pos, target_orient=orient)
        if not is_ee_close(robot, target_joint_angles, pos, m.get_quaternion(orient)):
            print('ik solution too far, try next')
            continue

        current_joint_angles = robot.get_joint_angles(robot.controllable_joints)
        path = rrt_star(current_joint_angles, target_joint_angles)

        if path is not None and len(path) < max_path_length:
            print('found rrt path')
            robot.motor_gains = 0.2
            color = np.random.uniform(0, 1, 3).tolist() + [1]
            for joint_angles in path:
                robot.control(joint_angles)
                # Enable for Debug
                # robot.control(joint_angles, set_instantly=True)
                # ee_pos, ee_orient = robot.get_link_pos_orient(robot.end_effector)
                # m.Shape(m.Sphere(.005), static=True, mass=0, position=ee_pos, orientation=ee_orient,
                #         rgba=color, collision=False)
                step = 0
                while np.linalg.norm(robot.get_joint_angles(robot.controllable_joints) - joint_angles) > 0.02:
                    m.step_simulation(realtime=True)
                    if step % 10 == 0:
                        ee_pos, ee_orient = robot.get_link_pos_orient(robot.end_effector)
                        m.Shape(m.Sphere(.005), static=True, mass=0, position=ee_pos, orientation=ee_orient,
                                rgba=color, collision=False)
                    step += 1
            return
        print('no rrt path found')


# Create environment and ground plane
env = m.Env(seed=300)
ground = m.Ground()
m.visualize_coordinate_frame()

# Create table and wall
table = m.URDF(filename=os.path.join(m.directory, 'table', 'table.urdf'), static=True, position=[0, 0, 0],
               orientation=[0, 0, 0, 1])
wall = m.Shape(m.Box(half_extents=[0.02, 0.2, 0.4]), static=True, position=[-0.2, 0, 0.85],
               orientation=[0, 0, 1, 1],
               rgba=[0, 1, 1, 0.75])
obstacles = [table, wall]

# Create cubes to grasp
cubes = []
for i in range(3):
    # size = 0.025 - i * 0.0025
    size = 0.025
    position = [0.2 - i * 0.1, -0.2, 1]
    yaw = -np.pi / 4 * i
    cubes.append(m.Shape(m.Box(half_extents=[size] * 3), static=False, mass=1, position=position,
                         orientation=m.get_quaternion([0, 0, yaw]), rgba=[0, (i + 1) / 5.0, (i + 1) / 5.0, 0.75]))
    cubes[-1].set_whole_body_frictions(lateral_friction=2000, spinning_friction=2000, rolling_friction=0)

# Let the cube drop onto the table
m.step_simulation(steps=50)

# Create Panda manipulator
robot = m.Robot.Panda(position=[0.5, 0, 0.76])
robot.motor_forces = 100

# Move end effector to a starting position using IK
pos = [-0.2, 0.2, 1.2]
default_euler = np.array([np.pi, 0, 0])
orient = m.get_quaternion(default_euler)
target_joint_angles = robot.ik(robot.end_effector, target_pos=pos, target_orient=orient)
robot.control(target_joint_angles, set_instantly=True)
robot.set_gripper_position([1] * 2, set_instantly=True)  # Open gripper

for i, current_cube in enumerate(range(len(cubes))):
    # MOVETO cube
    cube_pos, cube_orient = cubes[current_cube].get_base_pos_orient()
    gripper_orient = default_euler + [0, 0, m.get_euler(cube_orient)[-1]]
    moveto(robot, cube_pos + [0, 0, 0.2], gripper_orient, avoid_collision=True)
    moveto(robot, cube_pos, gripper_orient)

    # CLOSE
    robot.set_gripper_position([0]*2, force=5000)
    m.step_simulation(steps=100, realtime=True)

    # MOVETO goal
    pos, ori = robot.get_link_pos_orient(robot.end_effector)
    moveto(robot, pos + [0, 0, 0.2], ori)
    moveto(robot, [-0.1, 0.3, 0.8 + i * 0.05], default_euler, avoid_collision=True)

    # OPEN
    robot.set_gripper_position([1]*2)
    m.step_simulation(steps=50, realtime=True)
    pos, ori = robot.get_link_pos_orient(robot.end_effector)
    moveto(robot, pos + [0, 0, 0.1], ori)

print('Done')
m.step_simulation(steps=10000, realtime=True)