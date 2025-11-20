import numpy as np
import mengine as m
import pybullet as p
from scipy.optimize import minimize


class Controller():
    def __init__(
            self,
            robot,
            motor_gains,
            dt=0.02,
            horizon=10,
            w_e=1.0,
            w_d=0.25,
            w_a=0.1,
            w_q=0.5
        ):
        """
        Linear, Kinematic Model Predictive Controller
        """
        self.robot = robot
        self.robot.motor_gains = motor_gains
        self.n_dof = len(self.robot.controllable_joints)
        self.N = horizon
        self.dt = dt
        self.w_e = w_e
        self.w_d = w_d
        self.w_a = w_a
        self.w_q = w_q
        self.S_v, self.S_a = self._build_S_matrices()

    def moveto(self, pos, orient_quat):
        """
        IK-based move:
            1. go to (pos, orient_quat)
            2. wait until joint error is small
        """
        target_joint_angles = self.robot.ik(
            self.robot.end_effector,
            target_pos=pos,
            target_orient=orient_quat,
            use_current_joint_angles=True,
        )
        self.robot.control(target_joint_angles)
        while np.linalg.norm(
            self.robot.get_joint_angles(self.robot.controllable_joints) - target_joint_angles
        ) > 0.03:
            m.step_simulation(realtime=True)

    def get_ee_pose(self):
        ee_position, ee_orientation = self.robot.get_link_pos_orient(
            self.robot.end_effector
        )
        return np.array(ee_position), np.array(ee_orientation)

    def _build_S_matrices(self):
        """
        Build constant matrices for velocity and acceleration
        """
        N = self.N
        dt = self.dt
        I_n = np.eye(self.n_dof)

        S_v_base = np.zeros((N - 1, N))
        for k in range(1, N):
            S_v_base[k - 1, k] =  1.0 / dt
            S_v_base[k - 1, k - 1] = -1.0 / dt
        S_v = np.kron(S_v_base, I_n)   # shape: ((N-1)*n, N*n)

        S_a_base = np.zeros((N - 2, N))
        for k in range(2, N):
            S_a_base[k - 2, k]     =  1.0 / (dt ** 2)
            S_a_base[k - 2, k - 1] = -2.0 / (dt ** 2)
            S_a_base[k - 2, k - 2] =  1.0 / (dt ** 2)
        S_a = np.kron(S_a_base, I_n)   # shape: ((N-2)*n, N*n)

        return S_v, S_a

    def _get_jacobian(self, q_hat):
        joint_states = self.robot.get_motor_joint_states()
        if q_hat is not None:
            # append gripper joints to full state
            q_gripper = joint_states[1][-2:]
            q = np.concatenate((q_hat, q_gripper))
            q = q.tolist()
        else:
            q = joint_states[1]
        q_dot = [0] * len(q)
        q_ddot = [0] * len(q)

        J, _ = p.calculateJacobian(
            self.robot.body,
            self.robot.end_effector,
            localPosition=[0] * 3,
            objPositions=q,
            objVelocities=q_dot,
            objAccelerations=q_ddot,
            physicsClientId=self.robot.id
        )
        # Keep only controllable joint columns
        J = np.array(J)[:, self.robot.controllable_joints]
        return J
    
    def apply_virtual_spring(self, ref_positions, stiffness=200.0, desired_force=3.0):
        """ augment z-position with current force feedback
        is this really just stiffness control? """
        ref_new = ref_positions.copy()
        dz = desired_force / stiffness
        ref_new[:, 2] -= dz
        return ref_new
    
    def apply_pen_tip_offset(self, ref_positions, tip_length=0.10):
        """
        Shift reference positions so MPC tracks the pen tip, not the EE frame.
        Pen is aligned -Y after reorientation, so push forward in -Y direction.
        """
        offset = np.array([0.0, -tip_length, 0.0])
        return ref_positions + offset
    
    def mpc_step(self, ref_positions):
        """ Linearized Kinematic MPC step 
        TODO: Add spring-like force model based on the position.
        Where should I add the desired thickness
        """
        ref_positions = self.apply_virtual_spring(ref_positions)
        ref_positions = self.apply_pen_tip_offset(ref_positions)
        
        N = self.N
        n = self.n_dof
        _, orient_curr = self.robot.get_link_pos_orient(self.robot.end_effector)

        # Build reference for linearization
        q_hats = []
        J_hats = []
        for k in range(N):
            q_hat = self.robot.ik(
                self.robot.end_effector,
                target_pos=ref_positions[k],
                target_orient=orient_curr,
                use_current_joint_angles=True,
            )
            q_hats.append(q_hat)
            J_hat = self._get_jacobian(q_hat=q_hat)
            J_hats.append(J_hat)

        J_hats = np.array(J_hats)   
        q_hats = np.array(q_hats)  


        # ---- Q and p initialization ----
        Q_task = np.zeros((N * n, N * n))
        p_task = np.zeros((N * n,))

        Qe = self.w_e * np.eye(3) # position error
        Qd = self.w_d * np.eye(n) # velocity smoothing
        Qa = self.w_a * np.eye(n) # acceleration smoothing
        Qq = self.w_q * np.eye(n) # reference regular ~ orientation

        for k in range(N):
            Jk = J_hats[k]             # (3, n)
            q_hat_k = q_hats[k]        # (n,)

            # original tracking Hessian (rank < 3)
            A_k = Jk.T @ Qe @ Jk        # (n, n)

            # reference trajectory term
            A_k_full = A_k + Qq

            # linear term must also include joint-space part:
            p_k = -A_k @ q_hat_k - self.w_q * q_hat_k

            row = k * n
            col = k * n
            Q_task[row:row+n, col:col+n] += A_k_full
            p_task[row:row+n]            += p_k

        # ---- Smoothness costs ----
        Qd_big = np.kron(np.eye(N - 1), Qd)
        Qa_big = np.kron(np.eye(N - 2), Qa)

        Q_smooth = self.S_v.T @ Qd_big @ self.S_v \
                 + self.S_a.T @ Qa_big @ self.S_a

        # ---- Full cost ----
        Q = Q_task + Q_smooth
        p = p_task

        # ---- Symmetrize / regularize ----
        Q = 0.5*(Q + Q.T)
        eps = 1e-6
        Q_reg = Q + eps*np.eye(Q.shape[0])

        # optimization 
        q_qp = -np.linalg.solve(Q_reg, p)

        # ---- Output first step of MPC ----
        q_stack = q_qp
        q_opt = q_stack.reshape(N, n)
        q_des = q_opt[0]
        self.robot.control(q_des) # , set_instantly=True


