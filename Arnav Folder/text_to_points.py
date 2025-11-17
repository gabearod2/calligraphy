import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
import numpy as np
import os, time
import mengine as m

def text_to_points(text, font="DejaVuSans.ttf", size=1.0, samples_per_curve=20):
    tp = TextPath((0, 0), text, size=size, prop={'fname': font})
    vertices = tp.vertices  # (x,y) outline points
    codes = tp.codes        # drawing instructions (move, line, curve)

    points = []
    prev = None
    for v, c in zip(vertices, codes):
        if c == 1:  # MOVETO
            if points and points[-1] != [None, None]:
                points.append([None, None])  # lift pen
            points.append(v.tolist())
        elif c == 2:  # LINETO
            points.append(v.tolist())
        elif c in [3, 4]:  # CURVE3 / CURVE4
            # interpolate curve
            if prev is not None:
                curve = np.linspace(prev, v, samples_per_curve)
                points.extend(curve.tolist())
        prev = v
    
    return np.array(points, dtype=object)  # (x,y) or (None,None) for pen-up

# Example
points = text_to_points("Hi", size=1.5)

points = [[0.1471875, 1.09359375],
 [0.29507812499999997, 1.09359375],
 [0.29507812499999997, 0.645234375],
 [0.8327343749999999, 0.645234375],
 [0.8327343749999999, 1.09359375],
 [0.980625, 1.09359375],
 [0.980625, 0.0],
 [0.8327343749999999, 0.0],
 [0.8327343749999999, 0.52078125],
 [0.29507812499999997, 0.52078125],
 [0.29507812499999997, 0.0],
 [0.1471875, 0.0],
 [0.1471875, 1.09359375],
 [1.2692578125, 0.8203125],
 [1.4040234375, 0.8203125],
 [1.4040234375, 0.0],
 [1.2692578125, 0.0],
 [1.2692578125, 0.8203125],
 [1.2692578125, 1.139765625],
 [1.4040234375, 1.139765625],
 [1.4040234375, 0.96890625],
 [1.2692578125, 0.96890625],
 [1.2692578125, 1.139765625]]

# Plot to visualize
xs, ys = [], []
for p in points:
    if p[0] is None:  # pen up
        xs.append(np.nan)
        ys.append(np.nan)
    else:
        xs.append(p[0])
        ys.append(p[1])
plt.plot(xs, ys, marker='.')
plt.gca().invert_yaxis()
plt.axis('equal')
plt.show()


target_pos = []
for i, point in enumerate(xs):
    target_pos.append([(xs[i]/3)-0.25, ys[i]/3, 0.8])

print(len(target_pos))

# Create environment and ground plane
env = m.Env(time_step=0.1)
ground = m.Ground()

# Create table
table = m.URDF(filename=os.path.join(m.directory, 'table', 'table.urdf'), static=True, position=[0, 0, 0], orientation=[0, 0, 0, 1])

# Create Panda manipulator
robot = m.Robot.Panda(position=[0.5, 0, 0.75])


# Move end effector to a starting position using IK
pos = [0, 0, 0.8]
orient = m.get_quaternion([np.pi, 0, 0])
target_joint_angles = robot.ik(robot.end_effector, target_pos=pos, target_orient=orient)
robot.control(target_joint_angles, set_instantly=True)

for point in target_pos:
    # Move the end effector to a new pose
    for i in range(50):
        target_joint_angles = robot.ik(robot.end_effector, target_pos=point, target_orient=orient, use_current_joint_angles=True)
        robot.control(target_joint_angles, set_instantly=False)
        m.step_simulation(realtime=True)
        m.Shape(m.Sphere(radius=0.02), static=True, collision=False,
                   position=point, rgba=[1, 0, 0, 1])

        # position, orientation = robot.get_link_pos_orient(robot.end_effector)
        # if np.all(position == point):
            