import math
import time
import numpy as np
from filterpy.kalman import KalmanFilter
import numpy as np

GRAVITY = 9.80665


class ComplementaryFilter:
    def __init__(self, alpha=0.95):
        self.reset()
        self.alpha = alpha

    def reset(self):
        self.rollG = 0
        self.pitchG = 0
        self.rollComp = 0
        self.pitchComp = 0
        self.a_roll = 0
        self.a_pitch = 0
        self.t_last = None

    def update(self, acc_data, gyro_data, dt=None):
        x_accel, y_accel, z_accel = acc_data
        x_gyro, y_gyro, _ = gyro_data

        self.a_roll  = math.atan2(-x_accel, z_accel) * 180 / math.pi
        self.a_pitch = math.atan2( y_accel, z_accel) * 180 / math.pi

        if dt is None:
            t_now = time.process_time()
            dt = (t_now - self.t_last) if self.t_last is not None else 0.0
            self.t_last = t_now

        if dt == 0.0:
            self.rollComp  = self.a_roll
            self.pitchComp = self.a_pitch
            return

        self.rollComp  = self.a_roll  * (1 - self.alpha) + self.alpha * (self.rollComp  + y_gyro * dt)
        self.pitchComp = self.a_pitch * (1 - self.alpha) + self.alpha * (self.pitchComp + x_gyro * dt)

    @property
    def roll(self):
        return self.rollComp / 180

    @property
    def pitch(self):
        return self.pitchComp / 180
        
    @property
    def g_x(self):
        return GRAVITY*math.cos(self.roll*math.pi/180)

    @property
    def g_y(self):
        return GRAVITY*math.sin(self.pitch*math.pi/180)
    
    @property
    def g_xy(self):
        return self.g_x, self.g_y


class Debouncer:
    """Binary-signal debouncer: the output only flips after the raw input has
    held the opposite value for `n` consecutive updates, suppressing shorter
    flickers.  The first sample after a reset seeds the state directly, so an
    episode that starts with feet on the ground reads contact immediately."""

    def __init__(self, n):
        self.n = n
        self.reset()

    def reset(self):
        self.state = None
        self._count = 0

    def update(self, raw):
        raw = bool(raw)
        if self.state is None:
            self.state = raw
        if raw == self.state:
            self._count = 0
        else:
            self._count += 1
            if self._count >= self.n:
                self.state = raw
                self._count = 0
        return self.state


class EMAFilter:
    def __init__(self, alpha=0.4, warm_start=False):
        # warm_start seeds the filter with the first sample after a reset
        # instead of blending it against 0, avoiding a startup transient.
        # Must stay behaviorally identical to gnoci-control's LowPassFilter.
        self.warm_start = warm_start
        self.reset()
        self.alpha = alpha

    def reset(self):
        self.value = None if self.warm_start else 0

    def update(self, value):
        if self.value is None:
            self.value = value
        else:
            self.value = self.alpha * value + (1 - self.alpha) * self.value
        return self.value