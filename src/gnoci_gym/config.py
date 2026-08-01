CONTROL_HZ = 60
# Hardware low-passes the accelerometer and joint velocities before building
# the obs (gnoci-control sensors.py) — these alphas must match its values.
ACC_FILTER_ALPHA       = 0.8
JOINT_VEL_FILTER_ALPHA = 0.8

MAX_JOINT_VEL = 6.0    # max joint angular velocity (rad/s) — scales action deltas
