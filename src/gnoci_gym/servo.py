from dataclasses import dataclass
from simple_pid import PID
from gnoci_gym.filters import EMAFilter as LowPassFilter
from gnoci_gym.config import CONTROL_HZ, KP, KI, KD, FREQ, MAX_DELTA_V

SERVO_PWM_THRESHOLD_MIN: int = 500
SERVO_PWM_THRESHOLD_MAX: int = 2500
HALF_RANGE = (SERVO_PWM_THRESHOLD_MAX - SERVO_PWM_THRESHOLD_MIN) / 2 # 1000


@dataclass
class Servo:
    name: str
    pin_limits: tuple[float, float]
    init_value: float
    reverse: bool = False
    kp: float = KP
    ki: float = KI
    kd: float = KD
    _value: float = 0.0
    _update_value: float = 0.0
    pid_controller: PID = None
    offset: float = 0.0
    freq: int = FREQ
    control_hz: int = CONTROL_HZ
    max_delta_v: float = MAX_DELTA_V
    action_filter_alpha: float = 0.4

    low_pass_filter: LowPassFilter = None

    def __post_init__(self):
        self.pid_controller = PID(
            self.kp, self.ki, self.kd,
            starting_output=0,
            setpoint=self.init_value,
            output_limits=(-0.05, 0.05),  # max 5 units/tick = 5 units/sec at 100Hz
            sample_time=1.0 / self.freq,
        )
        self.action_scale = self.max_delta_v / self.control_hz
        self.low_pass_filter = LowPassFilter(alpha=self.action_filter_alpha)
        self.low_pass_filter.reset()

    def update_setpoint_delta(self, setpoint_delta: float):
        setpoint_delta = setpoint_delta * self.action_scale
        self.low_pass_filter.update(setpoint_delta)
        updated_setpoint = self.pid_controller.setpoint + self.low_pass_filter.value
        if updated_setpoint > self.pin_limits[1]: updated_setpoint = self.pin_limits[1]
        elif updated_setpoint < self.pin_limits[0]: updated_setpoint = self.pin_limits[0]
        self.pid_controller.setpoint = updated_setpoint

    def update_setpoint(self, setpoint: float):
        self.pid_controller.setpoint = setpoint

    def reset(self, value: float = None):
        """Snap the servo to `value` (defaults to init_value), clearing all
        internal state. Used at episode reset, where the sim world jumps to a
        new pose directly rather than slewing there like real hardware would.
        """
        if value is None:
            value = self.init_value
        self._value = value
        self._update_value = 0.0
        self.pid_controller.setpoint = value
        self.pid_controller.reset()
        self.low_pass_filter.reset()

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

    def get_pwm(self, dt=None):
        # dt=None -> simple_pid uses wall-clock time (real hardware). In sim,
        # pass the simulated timestep so the PID is time-correct and
        # deterministic regardless of gains or CPU speed.
        self._update_value = self.pid_controller(self._value, dt=dt)
        self._value += self._update_value
        return self._value_to_pwm()