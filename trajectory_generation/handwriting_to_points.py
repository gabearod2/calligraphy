import numpy as np
import cv2, os
import matplotlib.pyplot as plt
import skimage as ski
import mengine as m

# This function takes in an image of handwriting, converts it to points on the XY plane,
# and also assigns a thickness value to each XY point. It then plots a thickness scaled
# heatmap as well as a plot with the thickness represented. 

# plot argument is passed as False by default, set to True if you would like plots displayed for debugging or visual feedbacks
def handwriting_to_points(image_path, lower_bound=150, upper_bound=255, plot=False):
    if image_path is None:
        raise ValueError("Please specify a path")
    
    
    # --- Load image --- #
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")

    # --- Convert to black and white --- #
    _, binary = cv2.threshold(img, lower_bound, upper_bound, cv2.THRESH_BINARY_INV)
    binary = cv2.medianBlur(binary, 3)

    # --- Create distance map of distance from edge --- #
    dist_map = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # --- Skeletonize image --- #
    # Removes thickness with Zhang-Suen thinning

    # TODO: Potentially explore Lee method thinning as well? Seems slightly out of the scope of the class but could add some more experimentation potential
    skel = ski.morphology.skeletonize(binary)
    skel_uint8 = (skel.astype(np.uint8))
    contours, _ = cv2.findContours(skel_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(f"Found {len(contours)} contours")

    # --- Sort letters in left to right order --- #
    if len(contours) > 0:
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    # --- Iterate through all contours and store as points, thicknesses --- #
    points = []
    thicknesses = []
    for i, c in enumerate(contours):
        for p in c:
            x, y = p[0]
            points.append((x, y))

            r = dist_map[y,x]
            thicknesses.append(r*2)

        if i < len(contours) - 1:
            points.append((None, None))     # Set values to (None, None) to trigger pen lift at end of letter
            thicknesses.append(0)           # Thickness set to 0 when pen lifted

    # --- Plot thickness results --- #
    if plot:
        plt.figure(figsize=(14, 6))

        # --- Plot A: Thickness Heatmap --- #
        plt.subplot(1, 2, 1)
        plt.title("Skeleton with Thickness Heatmap")
        # c=thicknesses maps color to value
        sc = plt.scatter(*zip(*points), c=thicknesses, cmap='plasma', s=5)
        plt.colorbar(sc, label='Stroke Width (px)')
        plt.gca().invert_yaxis()
        plt.axis('equal')

        # --- Plot B: Stroke Reconstruction (Variable Radius) --- #
        plt.subplot(1, 2, 2)
        plt.title("Reconstructed Stroke (Size scaled by Thickness)")
        
        # Calculate marker size for scatter plot
        # s in scatter is Area (points^2). Area is proportional to Diameter^2.

        scale = 0.05    # Tune this scaling parameter to achieve desired thickness
        sizes = (np.array(thicknesses) ** 2) * scale
        
        plt.scatter(*zip(*points), s=sizes, c='black', alpha=0.6)
        plt.gca().invert_yaxis()
        plt.axis('equal')
        
        plt.tight_layout()
        plt.show()

    # --- Plot time visualization --- #
    if plot:
        # Filter out None values for plotting (Matplotlib can't scatter None)
        valid_points = [p for p in points if p[0] is not None]
        valid_thicknesses = [t for t in thicknesses if t > 0]
        
        # Calculate sizes for the "stroke reconstruction" style plot
        sizes = (np.array(valid_thicknesses) ** 2) * 0.05

        plt.figure(figsize=(14, 6))

        # Plot: Reconstructed Stroke
        plt.title("Sorted Left-to-Right Execution")
        
        # We plot a gradient of colors to prove the order is correct
        # Blue = Start, Red = End
        time_colors = np.arange(len(valid_points))
        
        plt.scatter(*zip(*valid_points), s=sizes, c=time_colors, cmap='copper', alpha=0.6)
        plt.colorbar(label='Execution Order (Blue->Red)')
        plt.gca().invert_yaxis()
        plt.axis('equal')
        
        plt.show()

    xs, ys, ts = [], [], []
    for i, p in enumerate(points):
        if p[0] is None:  # pen up
            xs.append(np.nan)
            ys.append(np.nan)
        else:
            # Modulate point values so writing is centered around 0 and append to X and Y value arrays
            xs.append((p[0]/3500))
            ys.append((p[1]/3500)-0.1)
            ts.append(thicknesses[i])

    print("trajectory generation done.")
    
    return xs, ys, ts

# --- Finding the distance between two points in 3D space --- #
def dist(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 +
                     (p2[1] - p1[1])**2 +
                     (p2[2] - p1[2])**2)


if __name__ == "__main__":
    print("Testing Handwriting Tracjectory Generation")

    # --- Generate points to track --- #
    xs, ys, ts = handwriting_to_points("handwriting/gabriel_print.jpg", plot=True)

    target_pos = []
    for i, point in enumerate(xs):
        target_pos.append([(xs[i]), -(ys[i]), 0.8])

    # --- Environment setup --- #

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

    # Tolerance to determine if sphere is spawned
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
                
