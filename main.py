import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
from dynamics import dynamics  # Ensure your dynamics function is defined in dynamics.py
from numpy.linalg import inv
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def main():
    # -------------------------------
    # 1. Setup and Initial Conditions
    # -------------------------------
    runtime = 10
    Sc = 1
    r = 2.5  # Radius for tetrahedron arrangement

    # Leader coordinates (each leader has 3 elements)
    il11, il12, il13 = r * np.cos(0), r * np.sin(0), 0
    il21, il22, il23 = r * np.cos(2 * np.pi / 3), r * np.sin(2 * np.pi / 3), 0
    il31, il32, il33 = r * np.cos(4 * np.pi / 3), r * np.sin(4 * np.pi / 3), 0
    il41, il42, il43 = 0, 0, r

    # Follower initial coordinates (fixed values from MATLAB)
    if11, if12, if13 = -8.1282, 14.6540, 9.9717
    if21, if22, if23 = -8.8798, 15.4758, 0.7240
    if31, if32, if33 = -7.3144, 12.3904, 2.2326
    if41, if42, if43 = -8.1313, 14.5965, 0.5199

    # Build the 44-element initial state vector:
    # Leaders: 4x3 = 12 elements
    # Followers: 4x3 = 12 elements
    # Observers: 4x3 = 12 elements (initialized to zeros)
    # Adaptive coupling gains: 4 zeros
    # Adaptive observer gains: 4 zeros
    init = np.array([
        # Leaders (indices 0–11)
        il11, il12, il13,
        il21, il22, il23,
        il31, il32, il33,
        il41, il42, il43,
        # Followers (indices 12–23)
        if11, if12, if13,
        if21, if22, if23,
        if31, if32, if33,
        if41, if42, if43,
        # Observers (indices 24–35)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        # Adaptive coupling gains (indices 36–39)
        0, 0, 0, 0,
        # Adaptive observer gains (indices 40–43)
        0, 0, 0, 0
    ]) * Sc

    # ------------------------------
    # ARE results
    #-------------------------------

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

    # --- Compute intermediate matrices ---
    Pi1 = inv(B1) @ (S - A1)
    Pi2 = inv(B2) @ (S - A2)
    Pi3 = inv(B3) @ (S - A3)
    Pi4 = inv(B4) @ (S - A4)

    P1 = solve_continuous_are(A1, B1, 3 * np.eye(3), np.eye(3))
    P2 = solve_continuous_are(A2, B2, 3 * np.eye(3), np.eye(3))
    P3 = solve_continuous_are(A3, B3, 3 * np.eye(3), np.eye(3))
    P4 = solve_continuous_are(A4, B4, 3 * np.eye(3), np.eye(3))

    K1 = -B1.T @ P1
    K2 = -B2.T @ P2
    K3 = -B3.T @ P3
    K4 = -B4.T @ P4

    H1 = Pi1 - K1
    H2 = Pi2 - K2
    H3 = Pi3 - K3
    H4 = Pi4 - K4

    # Prepare the argument containers
    Ks = (K1, K2, K3, K4)
    Hs = (H1, H2, H3, H4)
    Ps = (P1, P2, P3, P4)

    # -------------------------------
    # 2. Solve the ODE
    # -------------------------------
    t_span = (0, runtime)
    t_eval = np.linspace(0, runtime, 100)
    sol = solve_ivp(dynamics, t_span, init, method='LSODA', t_eval=t_eval, rtol=1e-2, atol=1e-4, args=(Ks, Hs, Ps))
    t = sol.t
    o = sol.y.T  # Shape: (number of time steps, 44)

    np.savez('simulation_results.npz', o=o, t=t)

    data = np.load('simulation_results.npz')
    o = data['o']
    t= data['t']

    # Extract state variables using the partition:
    # Leaders:
    l1 = o[:, 0:3]
    l2 = o[:, 3:6]
    l3 = o[:, 6:9]
    l4 = o[:, 9:12]
    # Followers:
    x1 = o[:, 12:15]
    x2 = o[:, 15:18]
    x3 = o[:, 18:21]
    x4 = o[:, 21:24]
    # (Observers and adaptive gains are computed in dynamics but not directly used in these plots.)

    # -------------------------------
    # 3. Plot e₍c₎ Error
    # -------------------------------
    # Define follower network matrices:
    A_mat = np.array([[0, 1, 0, 0],
                      [0, 0, 0, 0],
                      [1, 0, 0, 0],
                      [1, 0, 0, 0]])
    D = np.diag(np.sum(A_mat, axis=1))
    L_mat = D - A_mat

    M = 4  # Number of leaders
    G1 = np.diag([1, 0, 0, 0])
    G2 = np.diag([0, 1, 0, 0])
    G3 = np.diag([0, 0, 1, 0])
    G4 = np.diag([0, 0, 0, 1])
    Phi1 = (1 / M) * L_mat + G1
    Phi2 = (1 / M) * L_mat + G2
    Phi3 = (1 / M) * L_mat + G3
    Phi4 = (1 / M) * L_mat + G4

    # Replicate each leader's state:
    # MATLAB: kron(ones(1, M), l1) produces a (n_time x (3*M)) matrix.
    ones_M = np.ones((1, M))
    x_r_bar1 = np.kron(ones_M, l1)  # shape: (n_time, 3*M)
    x_r_bar2 = np.kron(ones_M, l2)
    x_r_bar3 = np.kron(ones_M, l3)
    x_r_bar4 = np.kron(ones_M, l4)

    # Construct global weighting matrix:
    I3 = np.eye(3)
    Phi_sum = np.kron(Phi1, I3) + np.kron(Phi2, I3) + np.kron(Phi3, I3) + np.kron(Phi4, I3)
    Phi_sum_inv = np.linalg.inv(Phi_sum)

    n_time = len(t)
    col_dim = Phi_sum.shape[1]  # Should be 3*M (i.e., 12)
    Phi_sum_x_r_bar = np.zeros((n_time, col_dim))
    K_Phi1 = np.kron(Phi1, I3).T
    K_Phi2 = np.kron(Phi2, I3).T
    K_Phi3 = np.kron(Phi3, I3).T
    K_Phi4 = np.kron(Phi4, I3).T
    for i in range(n_time):
        Phi_sum_x_r_bar[i, :] = (x_r_bar1[i, :] @ K_Phi1 +
                                 x_r_bar2[i, :] @ K_Phi2 +
                                 x_r_bar3[i, :] @ K_Phi3 +
                                 x_r_bar4[i, :] @ K_Phi4)
    e_c_term1 = np.hstack((x1, x2, x3, x4))  # shape: (n_time, 12)
    e_c = np.zeros_like(e_c_term1)
    for i in range(n_time):
        subtraction_term = Phi_sum_x_r_bar[i, :] @ Phi_sum_inv.T
        e_c[i, :] = e_c_term1[i, :] - subtraction_term

    # Create e_c error plots with inset zoom views.
    num_dimensions = 3
    fig_ec, axs_ec = plt.subplots(num_dimensions, 1, figsize=(10, 8))

    for dim in range(num_dimensions):
        data = e_c[:, dim::num_dimensions]  # Extract data for each coordinate (n_time x 4)

        # Main plot
        axs_ec[dim].plot(t, data, linewidth=3)
        axs_ec[dim].set_ylim([-20, 20])
        axs_ec[dim].set_xlabel('Time (s)', fontsize=14)
        axs_ec[dim].set_ylabel(f'$e_c({dim + 1})$', fontsize=14)
        axs_ec[dim].legend(['Follower 1', 'Follower 2', 'Follower 3', 'Follower 4'], fontsize=12)
        axs_ec[dim].grid(True)

        # Inset axes: specify exact bounding box in normalized parent coordinates
        ax_inset_ec = inset_axes(
            axs_ec[dim],
            width="100%",  # The inset width is 30% of its bounding box
            height="100%",  # The inset height is 30% of its bounding box
            bbox_to_anchor=(0.65, 0.55, 0.3, 0.4),  # (left, bottom, width, height)
            bbox_transform=axs_ec[dim].transAxes,
            borderpad=0
        )

        # Inset plot
        ax_inset_ec.plot(t, data, linewidth=3)
        ax_inset_ec.set_xlim([2.95, 3.2])
        ax_inset_ec.set_ylim([-3, 3])
        ax_inset_ec.set_title("Zoomed-in", fontsize=10)
        ax_inset_ec.grid(True)
    plt.show(block=True)
    # fig_ec.suptitle('Global Containment Error $e_c$', fontsize=16)

    plt.savefig("e_c.png", dpi=300)

    # -------------------------------
    # 4. Plot Pairwise Distances Between Followers
    # -------------------------------
    d12 = np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
    d13 = np.sqrt(np.sum((x1 - x3) ** 2, axis=1))
    d14 = np.sqrt(np.sum((x1 - x4) ** 2, axis=1))
    d23 = np.sqrt(np.sum((x2 - x3) ** 2, axis=1))
    d24 = np.sqrt(np.sum((x2 - x4) ** 2, axis=1))
    d34 = np.sqrt(np.sum((x3 - x4) ** 2, axis=1))

    fig_pd, ax_pd = plt.subplots(figsize=(10, 6))
    ax_pd.plot(t, d12, 'r', linewidth=2.5, label='$d_{12}$')
    ax_pd.plot(t, d13, 'g', linewidth=2.5, label='$d_{13}$')
    ax_pd.plot(t, d14, 'm', linewidth=2.5, label='$d_{14}$')
    ax_pd.plot(t, d23, 'b', linewidth=2.5, label='$d_{23}$')
    ax_pd.plot(t, d24, 'c', linewidth=2.5, label='$d_{24}$')
    ax_pd.plot(t, d34, 'y', linewidth=2.5, label='$d_{34}$')
    ax_pd.axhline(0.2, color='r', linestyle='--', linewidth=2, label='$d_s = 0.2$')
    ax_pd.set_xlabel('Time (s)', fontsize=14)
    ax_pd.set_ylabel('Distance', fontsize=14)
    ax_pd.legend(fontsize=12)
    ax_pd.grid(True)

    # Suppose you want the inset to be at x=0.2, y=0.5, with width=0.4, height=0.3
    # in normalized axes coordinates. You can do:
    ax_inset_pd = inset_axes(
        ax_pd,
        width="100%",  # or a numeric fraction of parent axis
        height="100%",
        bbox_to_anchor=(0.16, 0.7, 0.6, 0.25),  # (x0, y0, width, height) in normalized coords
        bbox_transform=ax_pd.transAxes,
        borderpad=0,  # optional, to reduce or remove padding
    )

    # Then plot as usual
    ax_inset_pd.plot(t, d12, 'r', linewidth=2.5)
    ax_inset_pd.plot(t, d13, 'g', linewidth=2.5)
    ax_inset_pd.plot(t, d14, 'm', linewidth=2.5)
    ax_inset_pd.plot(t, d23, 'b', linewidth=2.5)
    ax_inset_pd.plot(t, d24, 'c', linewidth=2.5)
    ax_inset_pd.plot(t, d34, 'y', linewidth=2.5)
    ax_inset_pd.axhline(0.2, color='r', linestyle='--', linewidth=2)
    ax_inset_pd.set_xlim([0, runtime])
    ax_inset_pd.set_ylim([0, 0.5])
    ax_inset_pd.grid(True)
    plt.show(block=True)

    # plt.savefig("Pairwise_Distances.png", dpi=300)

    # =============================================================================
    # 5. Animation with Clock (Updated to Display the Clock)
    # =============================================================================
    num_timestamps = t.shape[0]
    animation_indices = np.round(np.linspace(0, num_timestamps - 1, 300)).astype(int)
    fig_anim = plt.figure()
    ax_anim = fig_anim.add_subplot(111, projection='3d')
    plt.ion()  # Enable interactive mode for animation

    for idx in animation_indices:
        ax_anim.cla()  # Clear the axis for the new frame

        # Plot leaders (green triangles)
        ax_anim.scatter(l1[idx, 0], l1[idx, 1], l1[idx, 2], marker='^', color='g', s=100)
        ax_anim.scatter(l2[idx, 0], l2[idx, 1], l2[idx, 2], marker='^', color='g', s=100)
        ax_anim.scatter(l3[idx, 0], l3[idx, 1], l3[idx, 2], marker='^', color='g', s=100)
        ax_anim.scatter(l4[idx, 0], l4[idx, 1], l4[idx, 2], marker='^', color='g', s=100)

        # Plot followers (red circles)
        ax_anim.scatter(x1[idx, 0], x1[idx, 1], x1[idx, 2], marker='o', color='r', s=150)
        ax_anim.scatter(x2[idx, 0], x2[idx, 1], x2[idx, 2], marker='o', color='r', s=150)
        ax_anim.scatter(x3[idx, 0], x3[idx, 1], x3[idx, 2], marker='o', color='r', s=150)
        ax_anim.scatter(x4[idx, 0], x4[idx, 1], x4[idx, 2], marker='o', color='r', s=150)

        # Connect leaders to form a tetrahedron
        ax_anim.plot([l1[idx, 0], l2[idx, 0]], [l1[idx, 1], l2[idx, 1]], [l1[idx, 2], l2[idx, 2]], 'k-', linewidth=1.5)
        ax_anim.plot([l1[idx, 0], l3[idx, 0]], [l1[idx, 1], l3[idx, 1]], [l1[idx, 2], l3[idx, 2]], 'k-', linewidth=1.5)
        ax_anim.plot([l1[idx, 0], l4[idx, 0]], [l1[idx, 1], l4[idx, 1]], [l1[idx, 2], l4[idx, 2]], 'k-', linewidth=1.5)
        ax_anim.plot([l2[idx, 0], l3[idx, 0]], [l2[idx, 1], l3[idx, 1]], [l2[idx, 2], l3[idx, 2]], 'k-', linewidth=1.5)
        ax_anim.plot([l2[idx, 0], l4[idx, 0]], [l2[idx, 1], l4[idx, 1]], [l2[idx, 2], l4[idx, 2]], 'k-', linewidth=1.5)
        ax_anim.plot([l3[idx, 0], l4[idx, 0]], [l3[idx, 1], l4[idx, 1]], [l3[idx, 2], l4[idx, 2]], 'k-', linewidth=1.5)

        # Set axis limits, labels, and view angle
        ax_anim.set_xlim([-3, 3])
        ax_anim.set_ylim([-3, 3])
        ax_anim.set_zlim([-3, 3])
        ax_anim.set_xlabel('x', fontsize=12)
        ax_anim.set_ylabel('y', fontsize=12)
        ax_anim.set_zlabel('z', fontsize=12)
        ax_anim.view_init(elev=20, azim=30)

        # Add the time annotation (re-create it each iteration)
        ax_anim.text2D(0.7, 0.9, f'Time: {t[idx]:.2f} s', transform=ax_anim.transAxes, fontsize=18, color='black')

        plt.draw()
        plt.pause(0.0001)

    plt.ioff()
    plt.show()

    # -------------------------------
    # 6. Plot Initial Conditions of the System
    # -------------------------------
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.set_xlim([-10, 10])
    ax3.set_ylim([-10, 15])
    ax3.set_zlim([0, 10])
    ax3.view_init(elev=20, azim=30)
    # Plot leaders as green triangles
    ax3.scatter(il11, il12, il13, marker='^', c='g', s=100)
    ax3.scatter(il21, il22, il23, marker='^', c='g', s=100)
    ax3.scatter(il31, il32, il33, marker='^', c='g', s=100)
    ax3.scatter(il41, il42, il43, marker='^', c='g', s=100)
    # Plot followers as red circles
    ax3.scatter(if11, if12, if13, marker='o', c='r', s=100)
    ax3.scatter(if21, if22, if23, marker='o', c='r', s=100)
    ax3.scatter(if31, if32, if33, marker='o', c='r', s=100)
    ax3.scatter(if41, if42, if43, marker='o', c='r', s=100)
    # Connect leaders to form a tetrahedron
    ax3.plot([il11, il21], [il12, il22], [il13, il23], 'k-', linewidth=1.5)
    ax3.plot([il11, il31], [il12, il32], [il13, il33], 'k-', linewidth=1.5)
    ax3.plot([il11, il41], [il12, il42], [il13, il43], 'k-', linewidth=1.5)
    ax3.plot([il21, il31], [il22, il32], [il23, il33], 'k-', linewidth=1.5)
    ax3.plot([il21, il41], [il22, il42], [il23, il43], 'k-', linewidth=1.5)
    ax3.plot([il31, il41], [il32, il42], [il33, il43], 'k-', linewidth=1.5)
    # Create legend entries using NaN placeholders
    legend_leader, = ax3.plot(np.nan, np.nan, np.nan, '^', c='g', markersize=10)
    legend_follower, = ax3.plot(np.nan, np.nan, np.nan, 'o', c='r', markersize=10)
    ax3.legend([legend_leader, legend_follower], ['Leader', 'Follower'], loc='upper right', fontsize=17)
    ax3.set_xlabel('x', fontsize=18)
    ax3.set_ylabel('y', fontsize=18)
    ax3.set_zlabel('z', fontsize=18)
    plt.title('Initial Conditions', fontsize=20)
    plt.figtext(0.7, 0.9, 'Time: 0 s', fontsize=22, color='black', fontname='Times New Roman')

    plt.savefig("Initial_Conditions.png", dpi=300)

    # -------------------------------
    # 7. Plot Snapshots at Specific Times (all figures remain open simultaneously)
    # -------------------------------
    def plot_snapshot(t_snapshot, title):
        idx = np.argmin(np.abs(t - t_snapshot))
        fig_snap = plt.figure()
        ax_snap = fig_snap.add_subplot(111, projection='3d')
        # Plot leaders
        ax_snap.scatter(l1[idx, 0], l1[idx, 1], l1[idx, 2], marker='^', color='g', s=100)
        ax_snap.scatter(l2[idx, 0], l2[idx, 1], l2[idx, 2], marker='^', color='g', s=100)
        ax_snap.scatter(l3[idx, 0], l3[idx, 1], l3[idx, 2], marker='^', color='g', s=100)
        ax_snap.scatter(l4[idx, 0], l4[idx, 1], l4[idx, 2], marker='^', color='g', s=100)
        # Plot followers
        ax_snap.scatter(x1[idx, 0], x1[idx, 1], x1[idx, 2], marker='o', color='r', s=100)
        ax_snap.scatter(x2[idx, 0], x2[idx, 1], x2[idx, 2], marker='o', color='r', s=100)
        ax_snap.scatter(x3[idx, 0], x3[idx, 1], x3[idx, 2], marker='o', color='r', s=100)
        ax_snap.scatter(x4[idx, 0], x4[idx, 1], x4[idx, 2], marker='o', color='r', s=100)
        ax_snap.set_xlim([-1.8, 1.8])
        ax_snap.set_ylim([-1.8, 1.8])
        ax_snap.set_zlim([-1.8, 1.8])
        ax_snap.set_xlabel('X', fontsize=18)
        ax_snap.set_ylabel('Y', fontsize=18)
        ax_snap.set_zlabel('Z', fontsize=18)
        ax_snap.set_title(f"{title} (t = {t[idx]:.2f} s)", fontsize=20)
        # Connect leaders to form a tetrahedron
        ax_snap.plot([l1[idx, 0], l2[idx, 0]], [l1[idx, 1], l2[idx, 1]], [l1[idx, 2], l2[idx, 2]], 'k-', linewidth=1.5)
        ax_snap.plot([l1[idx, 0], l3[idx, 0]], [l1[idx, 1], l3[idx, 1]], [l1[idx, 2], l3[idx, 2]], 'k-', linewidth=1.5)
        ax_snap.plot([l1[idx, 0], l4[idx, 0]], [l1[idx, 1], l4[idx, 1]], [l1[idx, 2], l4[idx, 2]], 'k-', linewidth=1.5)
        ax_snap.plot([l2[idx, 0], l3[idx, 0]], [l2[idx, 1], l3[idx, 1]], [l2[idx, 2], l3[idx, 2]], 'k-', linewidth=1.5)
        ax_snap.plot([l2[idx, 0], l4[idx, 0]], [l2[idx, 1], l4[idx, 1]], [l2[idx, 2], l4[idx, 2]], 'k-', linewidth=1.5)
        ax_snap.plot([l3[idx, 0], l4[idx, 0]], [l3[idx, 1], l4[idx, 1]], [l3[idx, 2], l4[idx, 2]], 'k-', linewidth=1.5)
        return fig_snap

    fig_snapshot1 = plot_snapshot(2.95, "Snapshot 1")
    plt.savefig("Snapshot1.png", dpi=300)
    fig_snapshot2 = plot_snapshot(3.05, "Snapshot 2")
    plt.savefig("Snapshot2.png", dpi=300)
    fig_snapshot3 = plot_snapshot(4.5, "Snapshot 3")
    plt.savefig("Snapshot3.png", dpi=300)

plt.show(block=True)

if __name__ == '__main__':
    main()

