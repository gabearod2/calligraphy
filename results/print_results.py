import numpy as np
import os

def load_result(mode):
    """Load .npz result file for a given mode."""

    filename = f"results_{mode}.npz"
    if not os.path.exists(filename):
        print(f"[WARNING] File '{filename}' not found.")
        return None

    data = np.load(filename)

    pos_des = data["pos_des"]   # (N,3)
    pos_meas = data["pos_meas"]
    thick_des = data["thick_des"]  # (N,)
    thick_meas = data["thick_meas"]

    # RMSE values (already saved)
    rmse_pos = float(data["rmse_pos"])
    rmse_thick = float(data["rmse_thick"])

    # Compute MAX errors for completeness
    e_pos = pos_meas[:, :2] - pos_des[:, :2]
    e_pos_norm = np.linalg.norm(e_pos, axis=1)
    max_pos_err = float(np.max(e_pos_norm))

    e_thick = thick_meas - thick_des
    max_thick_err = float(np.max(np.abs(e_thick)))

    return {
        "mode": mode,
        "rmse_pos": rmse_pos,
        "rmse_thick": rmse_thick,
        "max_pos": max_pos_err,
        "max_thick": max_thick_err,
    }


def print_table(results):
    """Pretty print results in a table."""
    print("\n=== TRACKING PERFORMANCE COMPARISON ===\n")
    print("{:<10} {:>15} {:>15} {:>15} {:>15}".format(
        "Method", "RMSE_Pos (m)", "Max_Pos (m)", "RMSE_Thick (m)", "Max_Thick (m)"
    ))
    print("-" * 80)

    for res in results:
        if res is None:
            continue
        print("{:<10} {:>15.6f} {:>15.6f} {:>15.6f} {:>15.6f}".format(
            res["mode"],
            res["rmse_pos"],
            res["max_pos"],
            res["rmse_thick"],
            res["max_thick"],
        ))

    print("\n")


if __name__ == "__main__":

    modes = ["ik", "mpc", "mpc_f"]
    results = []

    for mode in modes:
        results.append(load_result(mode))

    print_table(results)
