# dynamics.py
import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import solve_continuous_are
import cvxpy as cp


def dynamics(t, o, Ks, Hs, Ps):

    # Results from ARE
    K1, K2, K3, K4 = Ks
    H1, H2, H3, H4 = Hs
    P1, P2, P3, P4 = Ps

    # Number of elements per state vector (e.g., 3 for 3D)
    n = 3

    # --- Unpack the state vector and reshape each segment into a column vector ---
    # Leaders: each becomes (3,1)
    l1 = o[0:3].reshape(n, 1)
    l2 = o[3:6].reshape(n, 1)
    l3 = o[6:9].reshape(n, 1)
    l4 = o[9:12].reshape(n, 1)

    # Followers:
    x1 = o[12:15].reshape(n, 1)
    x2 = o[15:18].reshape(n, 1)
    x3 = o[18:21].reshape(n, 1)
    x4 = o[21:24].reshape(n, 1)

    # Observers:
    zeta1 = o[24:27].reshape(n, 1)
    zeta2 = o[27:30].reshape(n, 1)
    zeta3 = o[30:33].reshape(n, 1)
    zeta4 = o[33:36].reshape(n, 1)

    # Gains (scalars)
    rhohat1 = o[36]
    rhohat2 = o[37]
    rhohat3 = o[38]
    rhohat4 = o[39]

    theta1 = o[40]
    theta2 = o[41]
    theta3 = o[42]
    theta4 = o[43]

    # --- Define matrices (same as MATLAB) ---
    S = np.array([[0, -2, 1],
                  [2, 0, 1],
                  [-1, -1, 0]])

    # System matrices for followers
    A1 = np.array([[-2, 1, 0],
                   [0, -3, 1],
                   [0.5, 0, -1]])
    B1 = np.eye(3)

    A2 = np.array([[-1, 0, 0.5],
                   [0, -2, 1],
                   [0.5, 0, -0.5]])
    B2 = np.array([[0.5, 1, 0],
                   [1, 0.5, 0],
                   [0, 0, 1]])

    A3 = np.array([[-1, 1, 0],
                   [0, -3, 1],
                   [0, 0.5, -1]])
    B3 = np.eye(3)

    A4 = np.array([[-1, 0.5, 0],
                   [0.5, -1.5, 0.5],
                   [-0.5, 0, -2]])
    B4 = np.eye(3)

    # # --- Compute intermediate matrices ---
    # Pi1 = inv(B1) @ (S - A1)
    # Pi2 = inv(B2) @ (S - A2)
    # Pi3 = inv(B3) @ (S - A3)
    # Pi4 = inv(B4) @ (S - A4)
    #
    # P1 = solve_continuous_are(A1, B1, 3 * np.eye(3), np.eye(3))
    # P2 = solve_continuous_are(A2, B2, 3 * np.eye(3), np.eye(3))
    # P3 = solve_continuous_are(A3, B3, 3 * np.eye(3), np.eye(3))
    # P4 = solve_continuous_are(A4, B4, 3 * np.eye(3), np.eye(3))
    #
    # print(f"t = {t}, P1 = {P1}")
    #
    # K1 = -B1.T @ P1
    # K2 = -B2.T @ P2
    # K3 = -B3.T @ P3
    # K4 = -B4.T @ P4
    #
    # H1 = Pi1 - K1
    # H2 = Pi2 - K2
    # H3 = Pi3 - K3
    # H4 = Pi4 - K4

    # --- Attack signals ---
    time_actu = 3
    time_observ = 3
    if t < time_actu:
        gamma_actu1 = np.zeros((n, 1))
        gamma_actu2 = np.zeros((n, 1))
        gamma_actu3 = np.zeros((n, 1))
        gamma_actu4 = np.zeros((n, 1))
    else:
        gamma_actu1 = np.array([5.5 * np.exp(0.11 * t), 1.5 * np.exp(0.23 * t), -6.6 * np.exp(0.08 * t)]).reshape(n, 1)
        gamma_actu2 = np.array([2.3 * np.exp(0.15 * t), -8.7 * np.exp(0.42 * t), 14.5 * np.exp(0.14 * t)]).reshape(n, 1)
        gamma_actu3 = np.array([7.6 * np.exp(0.35 * t), -9.7 * np.exp(0.20 * t), -17.2 * np.exp(0.06 * t)]).reshape(n,
                                                                                                                    1)
        gamma_actu4 = np.array([-2.9 * np.exp(0.16 * t), 5.2 * np.exp(0.15 * t), -7.7 * np.exp(0.10 * t)]).reshape(n, 1)

    if t < time_observ:
        gamma_observ1 = np.zeros((n, 1))
        gamma_observ2 = np.zeros((n, 1))
        gamma_observ3 = np.zeros((n, 1))
        gamma_observ4 = np.zeros((n, 1))
    else:
        gamma_observ1 = np.array([-1.2 * np.exp(0.40 * t), 1.5 * np.exp(0.61 * t), 2.7 * np.exp(0.15 * t)]).reshape(n,
                                                                                                                    1)
        gamma_observ2 = np.array([3.3 * np.exp(0.06 * t), -2.2 * np.exp(0.27 * t), -1.7 * np.exp(0.21 * t)]).reshape(n,
                                                                                                                     1)
        gamma_observ3 = np.array([2.8 * np.exp(0.24 * t), -5.0 * np.exp(0.04 * t), -1.8 * np.exp(0.08 * t)]).reshape(n,
                                                                                                                     1)
        gamma_observ4 = np.array([-5.2 * np.exp(0.04 * t), 2.4 * np.exp(0.13 * t), -2.1 * np.exp(0.18 * t)]).reshape(n,
                                                                                                                     1)

    # --- Compute adaptive terms for control ---
    # Compute differences as column vectors:
    diff1 = (x1 - zeta1).reshape(n, 1)
    diff2 = (x2 - zeta2).reshape(n, 1)
    diff3 = (x3 - zeta3).reshape(n, 1)
    diff4 = (x4 - zeta4).reshape(n, 1)

    term1 = B1.T @ P1 @ diff1  # (3,1)
    term2 = B2.T @ P2 @ diff2
    term3 = B3.T @ P3 @ diff3
    term4 = B4.T @ P4 @ diff4

    alpha1 = 1
    alpha2 = 1
    alpha3 = 1
    alpha4 = 1

    drhohat1 = alpha1*norm(term1)
    drhohat2 = alpha2*norm(term2)
    drhohat3 = alpha3*norm(term3)
    drhohat4 = alpha4*norm(term4)

    gammahat_actu1 = term1 * np.exp(rhohat1) / (norm(term1) + np.exp(-0.01 * t ** 2) + 1e-6)
    gammahat_actu2 = term2 * np.exp(rhohat2) / (norm(term2) + np.exp(-0.01 * t ** 2) + 1e-6)
    gammahat_actu3 = term3 * np.exp(rhohat3) / (norm(term3) + np.exp(-0.01 * t ** 2) + 1e-6)
    gammahat_actu4 = term4 * np.exp(rhohat4) / (norm(term4) + np.exp(-0.01 * t ** 2) + 1e-6)

    # --- Compute reference control inputs for each follower ---
    u_ref1 = K1 @ x1.reshape(n, 1) + H1 @ zeta1 + (-gammahat_actu1) + gamma_actu1
    u_ref2 = K2 @ x2.reshape(n, 1) + H2 @ zeta2 + (-gammahat_actu2) + gamma_actu2
    u_ref3 = K3 @ x3.reshape(n, 1) + H3 @ zeta3 + (-gammahat_actu3) + gamma_actu3
    u_ref4 = K4 @ x4.reshape(n, 1) + H4 @ zeta4 + (-gammahat_actu4) + gamma_actu4
    # Flatten each column before stacking so that u_ref_matrix becomes (3,4)
    u_ref_matrix = np.column_stack((u_ref1.flatten(), u_ref2.flatten(),
                                    u_ref3.flatten(), u_ref4.flatten()))

    # --- Distributed Optimization via cvxpy ---
    delta_ij = 1.8e4
    d_s = 0.25
    x_list = [x1, x2, x3, x4]

    valid_pairs = []
    for i in range(4):
        for j in range(i+1, 4):
            if norm(x_list[i] - x_list[j]) <= d_s:
                valid_pairs.append((i, j))

    N = 4
    num_decision_vars = 3 * N  # 12 variables total
    max_constraints = int(N * (N - 1) / 2)  # up to 6 constraints
    A_ineq = np.zeros((max_constraints, num_decision_vars))
    b_ineq = np.zeros(max_constraints)
    constraint_count = 0
    for (i, j) in valid_pairs:
        A_ineq[constraint_count, (3*i):(3*i+3)] = -2 * (x_list[i] - x_list[j]).T @ B1
        b_ineq[constraint_count] = (+2 * (x_list[i] - x_list[j]).T @ A1 @ x_list[i]
                                    - 2 * (x_list[i] - x_list[j]).T @ B2 @ u_ref_matrix[:, j]
                                    - 2 * (x_list[i] - x_list[j]).T @ A2 @ x_list[j]
                                    - delta_ij * (d_s**2 - norm(x_list[i] - x_list[j])**2))
        constraint_count += 1
    A_ineq = A_ineq[:constraint_count, :]
    b_ineq = b_ineq[:constraint_count]

    u_var = cp.Variable(num_decision_vars)
    H_qp = 2 * np.eye(num_decision_vars)
    f_qp = -2 * u_ref_matrix.flatten(order='F')
    objective = cp.Minimize(0.5 * cp.quad_form(u_var, H_qp) + f_qp.T @ u_var)
    constraints = []
    if constraint_count > 0:
        constraints.append(A_ineq @ u_var <= b_ineq)
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
        if u_var.value is None:
            u_opt = u_ref_matrix.flatten(order='F')
        else:
            u_opt = u_var.value
    except Exception:
        u_opt = u_ref_matrix.flatten(order='F')
    u_opt_matrix = u_opt.reshape((3, N), order='F')

    agents_in_pairs = set(i for pair in valid_pairs for i in pair)
    for i in range(N):
        if i not in agents_in_pairs:
            u_opt_matrix[:, i] = u_ref_matrix[:, i]

    print(f"t = {t}")
    # print(f"t = {t}, gamma_actu1 = {gamma_actu1}")
    # print(f"t = {t}, gammahat_actu1 = {gammahat_actu1}")
    # print(f"t = {t}, u_ref_matrix = {u_ref_matrix}")
    # print(f"t = {t}, u_ref1 = {u_ref1}")

    # # --- Follower Dynamics using the ref control input ---
    # dx1 = A1 @ x1.reshape(n, 1) + B1 @ u_ref_matrix[:, 0].reshape(n, 1)
    # dx2 = A2 @ x2.reshape(n, 1) + B2 @ u_ref_matrix[:, 1].reshape(n, 1)
    # dx3 = A3 @ x3.reshape(n, 1) + B3 @ u_ref_matrix[:, 2].reshape(n, 1)
    # dx4 = A4 @ x4.reshape(n, 1) + B4 @ u_ref_matrix[:, 3].reshape(n, 1)

    # --- Follower Dynamics using the opt control input ---
    dx1 = A1 @ x1.reshape(n, 1) + B1 @ u_opt_matrix[:, 0].reshape(n, 1)
    dx2 = A2 @ x2.reshape(n, 1) + B2 @ u_opt_matrix[:, 1].reshape(n, 1)
    dx3 = A3 @ x3.reshape(n, 1) + B3 @ u_opt_matrix[:, 2].reshape(n, 1)
    dx4 = A4 @ x4.reshape(n, 1) + B4 @ u_opt_matrix[:, 3].reshape(n, 1)

    # print(f"t = {t}, dx1 = {dx1}")

    # --- Leader Dynamics (already computed as column vectors) ---
    dl1 = S @ l1  # (3,1)
    dl2 = S @ l2
    dl3 = S @ l3
    dl4 = S @ l4

    # --- Observer Dynamics ---
    # These differences can remain as 1D arrays if only used for norm calculations.
    xi1 = (zeta2 - zeta1) + (l1 - zeta1) + (l2 - zeta1) + (l3 - zeta1) + (l4 - zeta1)
    xi2 = (l1 - zeta2) + (l2 - zeta2) + (l3 - zeta2) + (l4 - zeta2)
    xi3 = (zeta1 - zeta3) + (l1 - zeta3) + (l2 - zeta3) + (l3 - zeta3) + (l4 - zeta3)
    xi4 = (zeta1 - zeta4) + (l1 - zeta4) + (l2 - zeta4) + (l3 - zeta4) + (l4 - zeta4)

    dtheta1 = 200 * (xi1.T @ xi1).item()
    dtheta2 = 200 * (xi2.T @ xi2).item()
    dtheta3 = 200 * (xi3.T @ xi3).item()
    dtheta4 = 200 * (xi4.T @ xi4).item()

    dzeta1 = S @ zeta1 + np.exp(theta1) * xi1 + gamma_observ1
    dzeta2 = S @ zeta2 + np.exp(theta2) * xi2 + gamma_observ2
    dzeta3 = S @ zeta3 + np.exp(theta3) * xi3 + gamma_observ3
    dzeta4 = S @ zeta4 + np.exp(theta4) * xi4 + gamma_observ4

    # --- Construct the complete derivative vector ---
    # Flatten all column vectors to 1D arrays before concatenation
    do = np.concatenate((
        dl1.flatten(), dl2.flatten(), dl3.flatten(), dl4.flatten(),  # 12 leader elements
        dx1.flatten(), dx2.flatten(), dx3.flatten(), dx4.flatten(),  # 12 follower elements
        dzeta1.flatten(), dzeta2.flatten(), dzeta3.flatten(), dzeta4.flatten(),  # 12 observer elements
        np.array([drhohat1, drhohat2, drhohat3, drhohat4]),  # 4 adaptive coupling gains
        np.array([dtheta1, dtheta2, dtheta3, dtheta4])  # 4 adaptive observer gains
    ))
    return do
