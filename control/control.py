import os
import numpy as np
import mengine as m


class Controller():
    def __init__(
            self,
            robot,
            pen, 
            writing_pad,
            motor_gains,
            dt=0.02,
            horizon=10,
            position_weight=1.0,
            velocity_weight=0.25,
            acceleration_weight=0.1,
            reference_weight=0.5,
            stiffness=20000.0,
            kp_force=7.5e-5,
            ki_force=1e-5,
            kd_force=5e-7
        ):
        """
        Linear, Kinematic Model Predictive Controller
        """
        # bodies 
        self.robot = robot
        self.pen = pen
        self.writing_pad = writing_pad

        # inverse kinematics controller gains
        self.robot.motor_gains = motor_gains
        self.pen_tip_local = np.array([0.0, -0.05, 0.0]) 

        # time, dof, and constant matrices
        self.n_p = horizon
        self.dt = dt
        self.n_dof = len(self.robot.controllable_joints)
        self.S_v, self.S_a = self._build_S_matrices()

        # force control parameters
        self.stiffness = stiffness
        self.kp_force = kp_force
        self.ki_force = ki_force
        self.kd_force = kd_force
        self.F_prev = 100.0
        self.F_err_sum = 0.0

        # weighting matrices
        self.w_p = position_weight
        self.w_v = velocity_weight
        self.w_a = acceleration_weight
        self.w_q = reference_weight
        self.Q_p = self.w_p * np.eye(3)
        self.Q_v = self.w_v * np.eye(self.n_dof)
        self.Q_a = self.w_a * np.eye(self.n_dof)
        self.Q_q = self.w_q * np.eye(self.n_dof)

    def ik_move_to(self, pos, orient, set_instantly=False):
        """
        IK-based move
        """
        target_joint_angles = self.robot.ik(
            self.robot.end_effector,
            target_pos=pos,
            target_orient=orient,
            use_current_joint_angles=True,
        )
        self.robot.control(target_joint_angles, set_instantly=set_instantly)
        while np.linalg.norm(
            self.robot.get_joint_angles(self.robot.controllable_joints) - target_joint_angles
        ) > 0.03:
            m.step_simulation(realtime=True)

    def move_to_first_contact(self, pos, orient):
        """
        IK-based move till contact
        """
        in_contact = False
        while not in_contact:
            target_joint_angles = self.robot.ik(
                self.robot.end_effector,
                target_pos=pos,
                target_orient=orient,
                use_current_joint_angles=True,
            )
            self.robot.control(target_joint_angles)
            pos = pos + np.array([0.0, 0.0, -0.00001])
            contact_point = self.pen.get_contact_points(bodyB=self.writing_pad, average=True)
            if contact_point is not None:
                in_contact = True
                pos = pos + np.array([0.0, 0.0, 0.001])
                target_joint_angles = self.robot.ik(
                    self.robot.end_effector,
                    target_pos=pos,
                    target_orient=orient,
                    use_current_joint_angles=True,
                )
                self.robot.control(target_joint_angles)
            m.step_simulation(realtime=True)

    def _build_S_matrices(self):
        """
        Build constant matrices for velocity and acceleration linearization
        """
        n_p = self.n_p
        dt = self.dt
        I_n = np.eye(self.n_dof)

        S_v_base = np.zeros((n_p - 1, n_p))
        for k in range(1, n_p):
            S_v_base[k - 1, k] =  1.0 / dt
            S_v_base[k - 1, k - 1] = -1.0 / dt
        S_v = np.kron(S_v_base, I_n) 

        S_a_base = np.zeros((n_p - 2, n_p))
        for k in range(2, n_p):
            S_a_base[k - 2, k]     =  1.0 / (dt ** 2)
            S_a_base[k - 2, k - 1] = -2.0 / (dt ** 2)
            S_a_base[k - 2, k - 2] =  1.0 / (dt ** 2)
        S_a = np.kron(S_a_base, I_n) 
        return S_v, S_a
    
    def get_ee_pose(self):
        ee_position, ee_orientation = self.robot.get_link_pos_orient(
            self.robot.end_effector
        )
        return np.array(ee_position), np.array(ee_orientation)

    def get_ee_jacobian(self, q_hat=None):
        joint_states = self.robot.get_motor_joint_states()
        if q_hat is not None:
            q_gripper = joint_states[1][-2:]
            q = np.concatenate((q_hat, q_gripper))
            q = q.tolist()
        else:
            q = joint_states[1]
        q_dot = [0] * len(q)
        q_ddot = [0] * len(q)

        J_lin, J_ang = m.p.calculateJacobian(
            self.robot.body,
            self.robot.end_effector,
            localPosition=[0] * 3,
            objPositions=q,
            objVelocities=q_dot,
            objAccelerations=q_ddot,
            physicsClientId=self.robot.id
        )
        J_lin = np.array(J_lin)[:, self.robot.controllable_joints]
        J_ang = np.array(J_ang)[:, self.robot.controllable_joints]
        return J_lin, J_ang

    def get_normal_force(self):
        contact_point = self.pen.get_contact_points(bodyB=self.writing_pad, average=True)
        if contact_point is None:
            return None, None
        contact_normal_force, _, _ = self.pen.get_resultant_contact_forces(bodyB=self.writing_pad)
        return contact_point['posA'], contact_normal_force
    
    def apply_pen_tip_displacement(self, ref_positions):
        pen_pos, pen_orient = self.pen.get_base_pos_orient()
        R = np.array(m.p.getMatrixFromQuaternion(pen_orient)).reshape(3, 3)
        pen_tip_pos= pen_pos + R @ self.pen_tip_local
        ee_pos, _ = self.get_ee_pose()
        displacment = ee_pos - pen_tip_pos 
        ref_positions += displacment
        return ref_positions
    
    def apply_force_displacement(self, ref_positions, ref_thickness):
        _, F_vec = self.get_normal_force()
        if F_vec is None:
            F_meas = 0.0
        else:
            F_meas = float(np.linalg.norm(F_vec))
        
        F_des = self.stiffness * ref_thickness[:, 0]
        F_err = F_des - F_meas
        F_dot = (F_meas - self.F_prev)/self.dt  
        dz = - self.kp_force * (F_err) - self.kd_force * (F_dot) - self.ki_force * (self.F_err_sum)

        self.F_prev = F_meas
        self.F_err_sum += F_err

        # shift z reference
        ref_positions = ref_positions.copy()
        ref_positions[:, 2] += dz
        return ref_positions

    def mpc_step(self, ref_positions, ref_thickness):
        """ 
        MPC control step
        """        
        n_p = self.n_p
        n_dof = self.n_dof
        _, orient_curr = self.robot.get_link_pos_orient(self.robot.end_effector)

        # apply displacement from pen tip to end effector
        ref_positions = self.apply_pen_tip_displacement(ref_positions)
        ref_positions = self.apply_force_displacement(ref_positions, ref_thickness)

        for position in ref_positions:
            m.Shape(
                m.Sphere(radius=0.005),
                static=True,
                collision=False,
                position=position,
                rgba=[0, 1, 0, 0.5]
            )

        q_curr = self.robot.get_joint_angles(joints=self.robot.controllable_joints)

        # Build reference for linearization with current pose as starting point. 
        q_hats = []
        J_hats = []
        q_hats.append(q_curr) # q_0 = q
        for k in range(n_p-1):
            q_hat = self.robot.ik(
                self.robot.end_effector,
                target_pos=ref_positions[k+1],
                target_orient=orient_curr,
                use_current_joint_angles=True,
            )
            q_hats.append(q_hat)
        
        for q_hat in q_hats:
            J_hat, _ = self.get_ee_jacobian(q_hat=q_hat)
            J_hats.append(J_hat)

        J_hats = np.array(J_hats)   
        q_hats = np.array(q_hats)  

        # Total Q and p
        Q = np.zeros((n_p * n_dof, n_p * n_dof))
        p = np.zeros((n_p * n_dof,))

        # Get Q and p along horizon
        for k in range(n_p):
            Jk = J_hats[k] # (3, n)
            q_hat_k = q_hats[k] # (n,)

            A_k = Jk.T @ self.Q_p @ Jk # (n, n)
            A_k_full = A_k + self.Q_q 
            p_k = -A_k @ q_hat_k - self.Q_q @ q_hat_k

            row = k * n_dof
            col = k * n_dof
            Q[row:row+n_dof, col:col+n_dof] += A_k_full
            p[row:row+n_dof] += p_k

        # smoothness costs
        Q_v_big = np.kron(np.eye(n_p - 1), self.Q_v)
        Q_a_big = np.kron(np.eye(n_p - 2), self.Q_a)
        Q_smooth = self.S_v.T @ Q_v_big @ self.S_v + self.S_a.T @ Q_a_big @ self.S_a

        # full cost
        Q += Q_smooth

        # regularization
        Q = 0.5*(Q + Q.T)
        eps = 1e-6
        Q = Q + eps*np.eye(Q.shape[0])

        # optimization 
        q_qp = -np.linalg.solve(Q, p)

        # command first step
        q_stack = q_qp
        q_opt = q_stack.reshape(n_p, n_dof)
        q_des = q_opt[0]
        self.robot.control(q_des) # set_instantly=True


if __name__ == "__main__":
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
    for j in robot.controllable_joints:
        robot.enable_force_torque_sensor(joint=j)


    # ---------------------------------------
    # Controller definition
    # ---------------------------------------

    controller = c.Controller(
        robot=robot,
        pen=None,
        writing_pad=None, 
        motor_gains=0.05,
        dt=0.02,
        horizon=10,
        position_weight=1.0,
        velocity_weight=0.1,
        acceleration_weight=0.1,
        reference_weight=0.5
    )

    # ---------------------------------------
    # Setting home pose
    # ---------------------------------------

    home_pos = [-0.4, 0.0, 1.2] 
    home_orient = m.get_quaternion(np.array([0.0, -np.pi/2, 0.0]))
    controller.ik_move_to(pos=home_pos, orient=home_orient, set_instantly=True)
    m.step_simulation(steps=100, realtime=True)

    # ---------------------------------------
    # Follow trajectory
    # ---------------------------------------
    # TODO: add desired thickness of the written segment.
    # would require moving the trajectory onto a surface.
    # would also require getting the normals on the pen. 

    # """ STRAIGHT LINE TRAJECTORY """
    # traj = []
    # N_traj = 20
    # for i in range(N_traj):
    #     traj.append([
    #         home_pos[0] ,  
    #         home_pos[1],              
    #         home_pos[2] - 0.01 * i,              
    #     ])
    #     m.Shape(m.Sphere(radius=0.01), static=True, collision=False,
    #         position=[
    #             home_pos[0],  
    #             home_pos[1],              
    #             home_pos[2] - 0.01 * i,              
    #         ], rgba=[1, 0, 0, 1]
    #     )
    # traj = np.array(traj)

    """ CIRCLE TRAJECTORY """
    traj = []
    N_traj = 100
    diameter = 0.01 * (20 - 1)
    radius = diameter / 2
    thetas = np.linspace(0, 2*np.pi, N_traj)

    for theta in thetas:
        point = [
            home_pos[0],
            home_pos[1] + radius * np.cos(theta) - radius,
            home_pos[2] + radius * np.sin(theta)
        ]
        traj.append(point)

        m.Shape(
            m.Sphere(radius=0.01),
            static=True,
            collision=False,
            position=point,
            rgba=[1, 0, 0, 1]
        )
    traj = np.array(traj)

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
        m.step_simulation(steps=4, realtime=False)

    m.step_simulation(steps=10000, realtime=True)