import numpy as np
import mengine as m

class Controller():
    def __init__(
            self,
            robot,
            motor_gains
        ):
        self.robot = robot
        self.robot.motor_gains = motor_gains
        print("Controller initialized...")

    def moveto(self, pos, orient_quat):
        """
        IK-based move: 
            1. go to (pos, orient_quat)
            2. wait until joint error is small
        """
        print('[moveto] Using IK controller...')
        target_joint_angles = self.robot.ik(
            self.robot.end_effector,
            target_pos=pos,
            target_orient=orient_quat,
            use_current_joint_angles=True,
        )
        self.robot.control(target_joint_angles)
        while np.linalg.norm(self.robot.get_joint_angles(self.robot.controllable_joints) - target_joint_angles) > 0.03:
            m.step_simulation(realtime=True)

    def nominal_trajectory():
        return