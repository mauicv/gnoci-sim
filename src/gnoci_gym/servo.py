from dataclasses import dataclass
from simple_pid import PID
from gnoci_gym.filters import EMAFilter as LowPassFilter
from gnoci_gym.config import CONTROL_HZ, FREQ, MAX_DELTA_V

SERVO_PWM_THRESHOLD_MIN: int = 500
SERVO_PWM_THRESHOLD_MAX: int = 2500
HALF_RANGE = (SERVO_PWM_THRESHOLD_MAX - SERVO_PWM_THRESHOLD_MIN) / 2 # 1000

@dataclass
class Servo:
    name: str
    pin_limits: tuple[float, float]
    init_value: float
    reverse: bool = False
    _value: float = 0.0
    offset: float = 0.0
    freq: int = FREQ
    control_hz: int = CONTROL_HZ
    max_delta_v: float = MAX_DELTA_V
    low_pass_filter: LowPassFilter = None
    low_pass_filter_alpha: float = 0.4


    def __post_init__(self):
        self._value = self.init_value
        self.action_scale = self.max_delta_v / self.control_hz
        self.low_pass_filter = LowPassFilter(alpha=self.low_pass_filter_alpha)
        self.low_pass_filter.reset()
        self.total_ticks = self.freq/self.control_hz
        self.delta_rate = 1/self.total_ticks
        self.current_tick = 0
        self.last_delta = 0.0

    def update_setpoint_delta(self, setpoint_delta: float):
        setpoint_delta = setpoint_delta * self.action_scale
        self.low_pass_filter.update(setpoint_delta)
        self.last_delta = self.low_pass_filter.value
        self.current_tick = 0

    def update_setpoint(self, setpoint: float):
        self._value = setpoint

    def reset(self, value: float = None):
        """Snap the servo to `value` (defaults to init_value), clearing all
        internal state. Used at episode reset, where the sim world jumps to a
        new pose directly rather than slewing there like real hardware would.
        """
        if value is None:
            value = self.init_value
        self._value = value
        self.low_pass_filter.reset()
        self.last_delta = 0.0
        self.current_tick = 0

    @property
    def value(self):
        value = self._value
        if value > self.pin_limits[1]: value = self.pin_limits[1]
        elif value < self.pin_limits[0]: value = self.pin_limits[0]
        value = -value if self.reverse else value
        value += self.offset * (1 if not self.reverse else -1)
        return value

    def _value_to_pwm(self) -> int:
        pwm_val = int(SERVO_PWM_THRESHOLD_MIN + (1 + self.value) * HALF_RANGE)
        if pwm_val > SERVO_PWM_THRESHOLD_MAX: pwm_val = SERVO_PWM_THRESHOLD_MAX
        elif pwm_val < SERVO_PWM_THRESHOLD_MIN: pwm_val = SERVO_PWM_THRESHOLD_MIN
        return pwm_val

    def get_pwm(self):
        if self.current_tick < self.total_ticks:
            self._value += self.delta_rate*self.last_delta
            self.current_tick += 1
        return self._value_to_pwm()

