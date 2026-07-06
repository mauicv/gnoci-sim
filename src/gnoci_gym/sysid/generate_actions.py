from gnoci_gym.filters import EMAFilter

ACTION_DIM = 10

def generate_sine_wave(duration: float, frequency: float, amplitude: float) -> np.ndarray:
    t = np.linspace(0, duration, int(duration * frequency))
    return amplitude * np.sin(2 * np.pi * frequency * t)

