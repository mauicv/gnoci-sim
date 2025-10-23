import numpy as np

# TODO: These values are not correct, they need to be tuned.
flbt, frbt, brbt, blbt, fltt, frtt, brtt, bltt = [0, 0, 0, 0, 0, 0, 0, 0]


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def compute_posture_reward(state):
    [
        lb_joint1_p,
        lb_joint2_p,
        rb_joint1_p,
        rb_joint2_p,
        rf_joint1_p,
        rf_joint2_p,
        lf_joint1_p,
        lf_joint2_p,
        *_,
        roll,
        pitch
    ] = state

    flbe = 1 - 4 * abs(lf_joint2_p - flbt)
    frbe = 1 - 4 * abs(rf_joint2_p - frbt)
    brbe = 1 - 4 * abs(rb_joint2_p - brbt)
    blbe = 1 - 4 * abs(lb_joint2_p - blbt)
    flte = 1 - 4 * abs(lf_joint1_p - fltt)
    frte = 1 - 4 * abs(rf_joint1_p - frtt)
    brte = 1 - 4 * abs(rb_joint1_p - brtt)
    blte = 1 - 4 * abs(lb_joint1_p - bltt)
    pe = 1 - 4 * abs(pitch)
    re = 1 - 4 * abs(roll)

    posture_reward = 0
    for item in [flbe, frbe, brbe, blbe, flte, frte, brte, blte, pe, re]:
        posture_reward += item

    return np.float32(sigmoid(posture_reward/5))


def compute_reward(state, overturn_flag):
    posture_reward = compute_posture_reward(state)
    if overturn_flag:
        reward = 0
    else:
        reward = posture_reward
    return reward