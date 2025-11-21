import numpy as np
import mengine as m
from scipy.optimize import minimize


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
            reference_weight=0.5
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

        # time, dof, and constant matrices
        self.N = horizon
        self.dt = dt
        self.n_dof = len(self.robot.controllable_joints)
        self.S_v, self.S_a = self._build_S_matrices()

        # weighting matrices
        self.w_p = position_weight
        self.w_v = velocity_weight
        self.w_a = acceleration_weight
        self.w_q = reference_weight
        self.Q_p = self.w_p * np.eye(3)
        self.Q_v = self.w_v * np.eye(self.n_dof)
        self.Q_a = self.w_a * np.eye(self.n_dof)
        self.Q_q = self.w_q * np.eye(self.n_dof)

    def ik_move_to(self, pos, orient, set_instantly):
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

    def _get_ee_jacobian(self, q_hat=None):
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
    
    # def apply_virtual_spring(self, ref_positions, stiffness=200.0, desired_force=3.0):
    #     return 
    
    # def apply_pen_tip_offset(self, ref_positions, tip_length=0.10):
    #     return
    
    def vertical_pd_control(self, ref_positions):
        return
    
    def mpc_step(self, ref_positions):
        """ 
        MPC control step
        """        
        N = self.N
        n = self.n_dof
        pos_curr, orient_curr = self.robot.get_link_pos_orient(self.robot.end_effector)

        # get current force

        # apply pen tip offset scenario

        # apply PD control to augment ref_positions to track force


        q_curr = self.robot.get_joint_angles(joints=self.robot.controllable_joints)

        # Build reference for linearization with current pose as starting point. 
        q_hats = []
        J_hats = []
        q_hats.append(q_curr) # q_0 = q
        for k in range(N-1):
            q_hat = self.robot.ik(
                self.robot.end_effector,
                target_pos=ref_positions[k+1],
                target_orient=orient_curr,
                use_current_joint_angles=True,
            )
            q_hats.append(q_hat)
        
        for q_hat in q_hats:
            J_hat, _ = self._get_ee_jacobian(q_hat=q_hat)
            J_hats.append(J_hat)

        J_hats = np.array(J_hats)   
        q_hats = np.array(q_hats)  

        # Total Q and p
        Q = np.zeros((N * n, N * n))
        p = np.zeros((N * n,))

        # Get Q and p along horizon
        for k in range(N):
            Jk = J_hats[k] # (3, n)
            q_hat_k = q_hats[k] # (n,)

            A_k = Jk.T @ self.Q_p @ Jk # (n, n)
            A_k_full = A_k + self.Q_q 
            p_k = -A_k @ q_hat_k - self.Q_q @ q_hat_k

            row = k * n
            col = k * n
            Q[row:row+n, col:col+n] += A_k_full
            p[row:row+n]            += p_k

        # smoothness costs
        Q_v_big = np.kron(np.eye(N - 1), self.Q_v)
        Q_a_big = np.kron(np.eye(N - 2), self.Q_a)
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
        q_opt = q_stack.reshape(N, n)
        q_des = q_opt[0]
        self.robot.control(q_des, set_instantly=True) 


