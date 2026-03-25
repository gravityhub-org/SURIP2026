from pycbc.detector import Detector
from pycbc.noise import noise_from_psd
def create_detectors():
    return {
        "H1": Detector("H1"),
        "L1": Detector("L1"),
        "V1": Detector("V1"),
        "K1": Detector("K1")
    }

def compute_arrival_times(detectors, gps_time, ra, dec):
    return {name: det.arrival_time(gps_time, ra, dec) for name, det in detectors.items()}

def create_noise(num_samples, delta_t, psd, epoch, seed):
    noise = noise_from_psd(length=num_samples, delta_t=delta_t, psd=psd, seed=seed)
    noise.start_time = epoch
    return noise


