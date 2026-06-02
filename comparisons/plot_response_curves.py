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

REAL_COLOR = "#2563EB"
SIM_COLOR  = "#EA580C"

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
    "axes.spines.right": False,
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

# Row header labels: (row, label)
row_labels = [
    (0, "Left  ·  action = −1"),
    (1, "Left  ·  action = +1"),
    (2, "Right  ·  action = −1"),
    (3, "Right  ·  action = +1"),
]

k = 0
for i, joint in enumerate(JOINTS):
    for j, action in enumerate([-1, 1]):
        if i >= 5:
            k = 2
        row = j + k
        ax = axs[row, i % 5]

        t_real, pos_real, _ = get_entry(real_data, joint, action)
        t_sim,  pos_sim,  _ = get_entry(sim_data,  joint, action)

        if t_real is not None:
            ax.plot(t_real, pos_real, color=REAL_COLOR, linewidth=1.4,
                    label="Real", zorder=3)
        if t_sim is not None:
            ax.plot(t_sim, pos_sim, color=SIM_COLOR, linewidth=1.4,
                    linestyle="--", label="Sim", zorder=2)

        ax.set_title(format_joint_name(joint))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4, integer=False))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.tick_params(length=2)

        if i % 5 == 0:
            ax.set_ylabel("pos (×π rad)", labelpad=4)
        if row == 3 or (row == 1 and i < 5):
            ax.set_xlabel("time (s)", labelpad=3)

    k = 0  # reset after first joint; set again when i >= 5

# Row labels on far-left edge
for row, label in row_labels:
    fig.text(
        0.005, 1 - (row + 0.5) / 4,
        label, va="center", ha="left",
        fontsize=7.5, color="#6B7280",
        rotation=90, transform=fig.transFigure,
    )

# Shared legend in an empty corner area
handles = [
    plt.Line2D([0], [0], color=REAL_COLOR, linewidth=1.6, label="Real"),
    plt.Line2D([0], [0], color=SIM_COLOR,  linewidth=1.6, linestyle="--", label="Simulated"),
]
fig.legend(handles=handles, loc="lower center", ncol=2,
           frameon=True, framealpha=0.9, edgecolor="#D1D5DB",
           fontsize=9, bbox_to_anchor=(0.5, -0.01))

plt.tight_layout(rect=[0.015, 0.03, 1, 0.99])
plt.savefig("comparisons/response_curves.png", dpi=150, bbox_inches="tight")
plt.show()
