import numpy as np
import os

def load_result(suffix):
    """Load .npz result file for a given mode."""

    filename = f"results_{suffix}.npz"
    if not os.path.exists(filename):
        print(f"[WARNING] File '{filename}' not found.")
        return None

    data = np.load(filename)

    pos_des = data["pos_des"]   
    pos_meas = data["pos_meas"]
    thick_des = data["thick_des"]  
    thick_meas = data["thick_meas"]
    success = data["success"]
    fname = data["filename"]

    # RMSE values 
    rmse_pos = float(data["rmse_pos"])
    rmse_thick = float(data["rmse_thick"])

    # Compute max errors
    e_pos = pos_meas[:, :2] - pos_des[:, :2]
    e_pos_norm = np.linalg.norm(e_pos, axis=1)
    max_pos_err = float(np.max(e_pos_norm))

    e_thick = thick_meas - thick_des
    max_thick_err = float(np.max(np.abs(e_thick)))

    return {
        "word_mode": suffix,
        "rmse_pos": rmse_pos,
        "rmse_thick": rmse_thick,
        "max_pos": max_pos_err,
        "max_thick": max_thick_err,
        "success": success,
        "fname": fname
    }


def print_table(results):
    """Pretty print results in a table."""
    print("\n=== TRACKING PERFORMANCE, POSE AND THICKNESS ===\n")
    print("{:<20} {:>18} {:>14} {:>16} {:>15} {:>10}".format(
        "Method/Word", "RMSE_Pos (m)", "Max_Pos (m)", "RMSE_Thick (m)", "Max_Thick (m)", "Success",
    ))
    print("-" * 105)

    for res in results:
        if res is None:
            continue
        print("{:<20} {:>15.6f} {:>15.6f} {:>15.6f} {:>15.6f} {:>15.6f}".format(
            res["word_mode"],
            res["rmse_pos"],
            res["max_pos"],
            res["rmse_thick"],
            res["max_thick"],
            res["success"]
        ))

    print("\n")


if __name__ == "__main__":

    print("\nNOTE: Different tuning parameters are usedin this version when compared to papers reusults.")
    print("To replicate: Lower Kp to 0.75e-5 & Lower thickness_scale to 5000.")

    suffixes = [
        "arnav_cursive",
        "arnav_print",
        "g_print",
        "gabriel_print",
        "gabriel_cursive",
        "manipulation_print",
        "manipulation_cursive",
        "mechanics_cursive",
        "mechanics_print",
        "motion_cursive",
        "motion_print",
        "symbol_cursive",
        "symbol_print",
        "talk_cursive",
        "talk_print",
        "virtual_cursive",
        "virtual_print",
    ]
    results = []

    for suffix in suffixes:
        results.append(load_result(suffix))

    print_table(results)
