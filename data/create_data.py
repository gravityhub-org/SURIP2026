''' Create realistic waveform projected onto H1, L1, V1, K1 detectors, with colored Gaussian noise
'''

import numpy as np
import pylab as plt
from miscellaneous import get_gps_time_now
from pycbc.waveform import get_td_waveform
from pycbc.noise import noise_from_psd
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.types import TimeSeries
from pycbc.detector import Detector

def generate_waveform(gps_time, ra, dec, polarization, mass1, mass2, distance, delta_t, f_lower):
    hp, hc = get_td_waveform(
        approximant="NRSur7dq4",
        mass1=mass1,
        mass2=mass2,
        distance=distance,
        delta_t=delta_t,
        f_lower=f_lower
    )
    hp.start_time = gps_time
    hc.start_time = gps_time
    return hp, hc

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

def project_waveforms(detectors, hp, hc, ra, dec, polarization, gps_time):
    return {
        name: det.project_wave(hp, hc, ra, dec, polarization, method='constant', reference_time=gps_time)
        for name, det in detectors.items()
    }

def inject_waveforms(noises, waveforms):
    return {name: noises[name].inject(waveforms[name]) for name in noises}

def plot_detector_data(data, epoch, duration):
    plt.figure(figsize=(12, 8))
    for idx, (name, d) in enumerate(data.items(), 1):
        plt.subplot(4, 1, idx)
        plt.plot(d.sample_times, d.data, label=name)
        plt.xlim(epoch, epoch + duration)
        plt.ylabel('Strain')
        plt.title(name)
        if idx == 4:
            plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

def main():
    # Set parameters
    gps_time = 1458431355.3721366
    ra, dec = 1.375, -1.2108
    polarization = 0.5 * np.pi
    mass1 = 30
    mass2 = 30
    distance = 1000
    delta_t = 1.0 / 4096
    f_lower = 20
    duration = 120
    delta_f = 1.0 / duration
    epoch = gps_time - 60
    num_samples = int(duration / delta_t)

    hp, hc = generate_waveform(gps_time, ra, dec, polarization, mass1, mass2, distance, delta_t, f_lower)
    detectors = create_detectors()
    compute_arrival_times(detectors, gps_time, ra, dec)  # Not used, but kept for completeness

    psd = aLIGOZeroDetHighPower(length=num_samples, delta_f=delta_f, low_freq_cutoff=f_lower)
    noises = {
        name: create_noise(num_samples, delta_t, psd, epoch, seed)
        for seed, name in enumerate(detectors)
    }
    waveforms = project_waveforms(detectors, hp, hc, ra, dec, polarization, gps_time)
    data = inject_waveforms(noises, waveforms)
    plot_detector_data(data, epoch, duration)

if __name__ == "__main__":
    main()
