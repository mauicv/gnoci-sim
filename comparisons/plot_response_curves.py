import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

JOINTS = [
    "head__left_yoke",
    "left_yoke__hip",
    "left_hip__upper_leg",
    "left_upper_leg__lower_leg",
    "left_lower_leg__foot",
    "head__right_yoke",
    "right_yoke__hip",
    "right_hip__upper_leg",
    "right_upper_leg__lower_leg",
    "right_lower_leg__foot",
]

REAL_POS_COLOR = "#2563EB"
SIM_POS_COLOR  = "#EA580C"
REAL_VEL_COLOR = "#7C3AED"
SIM_VEL_COLOR  = "#059669"

def format_joint_name(name):
    a, b = name.split("__")
    return f"{a.replace('_', ' ')}\n{b.replace('_', ' ')}"

def get_entry(data, joint_name, action):
    for item in data:
        if item["joint_name"] == joint_name and item["action"] == action:
            t = np.array(item["time"])
            return t - t[0], np.array(item["angular_pos"]), np.array(item["angular_vel"])
    return None, None, None


with open("comparisons/real_response_data.json") as f:
    real_data = json.load(f)

with open("comparisons/simulated_response_data.json") as f:
    sim_data = json.load(f)


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.spines.top": False,
    "axes.grid": True,
    "grid.color": "#E5E7EB",
    "grid.linewidth": 0.6,
    "axes.labelcolor": "#374151",
    "xtick.color": "#6B7280",
    "ytick.color": "#6B7280",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.titlesize": 7.5,
    "axes.titlecolor": "#111827",
    "axes.titlepad": 6,
    "figure.facecolor": "white",
    "axes.facecolor": "#F9FAFB",
})

fig, axs = plt.subplots(4, 5, figsize=(18, 9), sharex=False)
fig.suptitle("Joint Step Response — Real vs Simulated", fontsize=12, fontweight="bold",
             color="#111827", y=0.995)

k = 0
for i, joint in enumerate(JOINTS):
    for j, action in enumerate([-1, 1]):
        if i >= 5:
            k = 2
        row = j + k
        ax = axs[row, i % 5]
        ax_v = ax.twinx()

        t_real, pos_real, vel_real = get_entry(real_data, joint, action)
        t_sim,  pos_sim,  vel_sim  = get_entry(sim_data,  joint, action)

        if t_real is not None:
            ax.plot(t_real, pos_real, color=REAL_POS_COLOR, linewidth=1.4,
                    label="Real pos", zorder=3)
            ax_v.plot(t_real, vel_real, color=REAL_VEL_COLOR, linewidth=1.0,
                      linestyle=":", label="Real vel", zorder=2)
        if t_sim is not None:
            ax.plot(t_sim, pos_sim, color=SIM_POS_COLOR, linewidth=1.4,
                    linestyle="--", label="Sim pos", zorder=3)
            ax_v.plot(t_sim, vel_sim, color=SIM_VEL_COLOR, linewidth=1.0,
                      linestyle=(0, (3, 1, 1, 1)), label="Sim vel", zorder=2)

        ax.set_title(format_joint_name(joint))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        ax_v.yaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.tick_params(length=2)
        ax_v.tick_params(length=2)

        ax.spines["top"].set_visible(False)
        ax_v.spines["top"].set_visible(False)

        # Only show right-side vel axis on the last column of each group
        if i % 5 == 4:
            ax_v.set_ylabel("vel (×π rad/s)", labelpad=4, color="#6B7280", fontsize=7)
            ax_v.yaxis.label.set_color("#6B7280")
            ax_v.tick_params(colors="#6B7280")
        else:
            ax_v.set_yticks([])
            ax_v.spines["right"].set_visible(False)

        if i % 5 == 0:
            ax.set_ylabel("pos (×π rad)", labelpad=4)
        if row == 3 or (row == 1 and i < 5):
            ax.set_xlabel("time (s)", labelpad=3)

    k = 0

handles = [
    plt.Line2D([0], [0], color=REAL_POS_COLOR, linewidth=1.6,                       label="Real pos"),
    plt.Line2D([0], [0], color=SIM_POS_COLOR,  linewidth=1.6, linestyle="--",       label="Sim pos"),
    plt.Line2D([0], [0], color=REAL_VEL_COLOR, linewidth=1.2, linestyle=":",        label="Real vel"),
    plt.Line2D([0], [0], color=SIM_VEL_COLOR,  linewidth=1.2, linestyle=(0,(3,1,1,1)), label="Sim vel"),
]
fig.legend(handles=handles, loc="lower center", ncol=4,
           frameon=True, framealpha=0.9, edgecolor="#D1D5DB",
           fontsize=9, bbox_to_anchor=(0.5, -0.01))

plt.tight_layout(rect=[0.015, 0.03, 1, 0.99])
plt.savefig("comparisons/response_curves.png", dpi=150, bbox_inches="tight")
plt.show()
