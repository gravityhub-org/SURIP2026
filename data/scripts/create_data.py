''' Create realistic waveform projected onto H1, L1, V1, K1 detectors, with colored Gaussian noise
'''

import numpy as np
import pylab as plt
from miscellaneous import get_gps_time_now
from pycbc.psd import aLIGOZeroDetHighPower
# Import TimeSeries
from pycbc.types import timeseries
# Default files
from waveform import project_waveforms, generate_waveform, inject_waveforms
from detector import create_detectors, compute_arrival_times, create_noise
from plot import plot_detector_data

if __name__ == "__main__":
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
    noise_data = {
        name: create_noise(num_samples, delta_t, psd, epoch, seed)
        for seed, name in enumerate(detectors)
    }
    waveforms = project_waveforms(detectors, hp, hc, ra, dec, polarization, gps_time)
    data = inject_waveforms(noise_data, waveforms)
    # Save the data:
    for name, dataset in data.items():
        filename = f"{name}_data.txt"
        dataset.save(filename)
    # Load the dataset:
    data = {}
    for name in detectors:
        filename = f"{name}_data.txt"
        data[name] = timeseries.load_timeseries(filename)
    fig = plot_detector_data(data, waveforms, epoch, duration)
    # Create 'plots' directory if it doesn't exist
    import os
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/detector_data.pdf", bbox_inches='tight')
    plt.close()


