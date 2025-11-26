import numpy as np
import cv2, os
import matplotlib.pyplot as plt
import skimage as ski
import mengine as m

def handwriting_to_points(image_path, lower_bound=150, upper_bound=255, plot=True):

    # Load image
    if image_path == None:
        raise ValueError("Please specify a path")
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Convert to black and white
    _, binary = cv2.threshold(img, lower_bound, upper_bound, cv2.THRESH_BINARY_INV)

    binary = cv2.medianBlur(binary, 3)

    # Skeletonize image (basically remove line thickness and turn it into a single line)
    # Zhang-Suen thinning
    # TODO: Potentially explore Lee method thinning as well? Seems slightly out of the scope of the class but could add
    # some more experimentation potential
    skel = ski.morphology.skeletonize(binary)
    skel_uint8 = (skel.astype(np.uint8))
    contours, _ = cv2.findContours(skel_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("found contours")
    # print(contours)
    # Iterate through all contours and store as points
    points = []
    for c in contours:
        for p in c:
            x, y = p[0]
            points.append((x, y))

    if plot:
        plt.figure(figsize=(10, 10))
        plt.scatter(*zip(*points), s=1)
        plt.gca().invert_yaxis()
        plt.show()

    xs, ys = [], []
    for p in points:
        if p[0] is None:  # pen up
            xs.append(np.nan)
            ys.append(np.nan)
        else:
            # Modulate point values so writing is centered around 0 and append to X and Y value arrays
            xs.append((p[0]/3500))
            ys.append((p[1]/3500)-0.1)
    
    if plot:
        plt.plot(xs, ys, marker='.')
        plt.gca().invert_yaxis()
        plt.axis('equal')
        plt.show()

    print("trajectory generation done.")
    
    return xs, ys


# Plot points
def plot_points(points):
    plt.figure(figsize=(10, 10))
    plt.scatter(*zip(*points), s=1)
    plt.gca().invert_yaxis()
    plt.show()

# Plot modified points
def plot_points_mod(xs, ys):
    plt.plot(xs, ys, marker='.')
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.show()

# Finding the distance between two points in 3D space
def dist(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 +
                     (p2[1] - p1[1])**2 +
                     (p2[2] - p1[2])**2)


if __name__ == "__main__":
    print("Testing Handwriting Tracjectory Generation")

    # Generate points to track
    xs, ys = handwriting_to_points("handwriting/gabriel_print.jpg")
    
    # Plot the points
    # plot_points(points)

    target_pos = []
    for i, point in enumerate(xs):
        target_pos.append([(xs[i]), -(ys[i]), 0.8])

    # Create environment and ground plane
    env = m.Env(time_step=0.1)
    ground = m.Ground()

    # Create table
    table = m.URDF(filename=os.path.join(m.directory, 'table', 'table.urdf'), static=True, position=[0, 0, 0], orientation=[0, 0, 0, 1])

    # Create Panda manipulator
    robot = m.Robot.Panda(position=[0.75, 0, 0.75])

    # Move end effector to a starting position using IK
    pos = [0, 0, 0.8]
    orient = m.get_quaternion([np.pi, 0, 0])
    target_joint_angles = robot.ik(robot.end_effector, target_pos=pos, target_orient=orient)
    robot.control(target_joint_angles, set_instantly=True)

    # Test points
    tol = 0.001

    for index, point in enumerate(target_pos):   
        if index % 10 == 0:

            # IK to solve for joint angles for target position
            target_joint_angles = robot.ik(robot.end_effector, target_pos=point, target_orient=orient, use_current_joint_angles=True)
            
            # Move robot to specified joint angles, set instantly to speed up simulation
            robot.control(target_joint_angles, set_instantly=True)

            # Get actual position of end effector after each time step
            pos, orient = robot.get_link_pos_orient(robot.end_effector)

            # If distance is less than tolerance amount, spawn a sphere at the location the end effector is currently at
            if dist(pos, point) < tol:
                m.Shape(m.Sphere(radius=0.0025), static=True, collision=False, 
                    position=pos, rgba=[0, 1, 0, 1])
                
            m.step_simulation(realtime=True)
                
