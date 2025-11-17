import numpy as np
import mengine as m
import pybullet as p
from scipy.linalg import expm

class Controller():
    def __init__(
            self,
            robot,
            motor_gains,
            dt = 0.02,
            horizon=10,
            w_e=1.0,
            w_d=0.1,
            w_a=0.01
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
        self.S_v, self.S_a = self._build_S_matrices()

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
        N = self.N
        dt = self.dt
        I_n = np.eye(self.n_dof)

        S_v_base = np.zeros((N, N))
        for k in range(1, N):
            S_v_base[k, k] = 1.0 / dt
            S_v_base[k, k - 1] = -1.0 / dt
        S_v = np.kron(S_v_base, I_n)

        S_a_base = np.zeros((N, N))
        for k in range(2, N):
            S_a_base[k, k] = 1.0 / (dt ** 2)
            S_a_base[k, k - 1] = -2.0 / (dt ** 2)
            S_a_base[k, k - 2] = 1.0 / (dt ** 2)    
        S_a = np.kron(S_a_base, I_n)
        return S_v, S_a
    
    def _get_position_jacobian(self, q_curr):
        if q_curr is None:
            J = self.robot.get_linear_jacobian(self.robot.end_effector)
        else:
            J, _ = p.calculateJacobian(
                self.robot.body, 
                self.robot.end_effector, 
                localPosition=[0]*3, 
                objPositions=q_curr.tolist(), 
                objVelocities=[0]*len(self.robot.get_motor_joint_states()[1]), 
                objAccelerations=[0]*len(self.robot.get_motor_joint_states()[1]), 
                physicsClientId=self.robot.id
            )
            J = np.array(J)
        
        return J[:, self.robot.controllable_joints]
    
    def mpc_step(self, ref_positions):
        """
        LMPC implementation 
        """

        N = self.N
        n = self.n_dof

        q_curr = self.robot.get_joint_angles(self.robot.controllable_joints)
        _, orient_curr = self.robot.get_link_pos_orient(self.robot.end_effector)

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
            J_hat = self._get_position_jacobian(q_curr=None) # q_curr=np.concatenate((q_hat, np.array([0.0, 0.0])))
            J_hats.append(J_hat)
        J_hats = np.array(J_hats)
        q_hats = np.array(q_hats)

        Q = np.zeros((N*n, N*n))
        p = np.zeros((N*n,))
        Qe = self.w_e * np.eye(3)
        Qd = self.w_d * np.eye(n)
        Qa = self.w_a * np.eye(n)

        # Task cost
        for k in range(N):
            Jk = J_hats[k] # (3, n)
            q_hat_k = q_hats[k] # shape (n,)
            A_k = Jk.T @ Qe @ Jk # (n, n)
            p_k = -A_k @ q_hat_k # (n,)

            row = k * n
            col = k * n
            Q[row:row+n, col:col+n] += A_k
            p[row:row+n] += p_k

        # kron(I_N, Qd) is block-diagonal with Qd repeated N times
        Qd_big = np.kron(np.eye(N), Qd)
        Qa_big = np.kron(np.eye(N), Qa)
        Q += self.S_a.T @ Qa_big @ self.S_a + self.S_v.T @ Qd_big @ self.S_v

        q_stack = -np.linalg.solve(Q, p) # minimizes q^T Q q + 2 p^T q

        q_opt = q_stack.reshape(N, n)
        q_des = q_opt[0] # first step of horizon
        self.robot.control(q_des, set_instantly=True)
