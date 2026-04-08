from pycbc.waveform import get_td_waveform

def project_waveforms(detectors, hp, hc, ra, dec, polarization, gps_time):
    return {
        name: det.project_wave(hp, hc, ra, dec, polarization, method='constant', reference_time=gps_time)
        for name, det in detectors.items()
    }

def generate_waveform(gps_time, ra, dec, polarization, mass1, mass2, distance, delta_t, f_lower):
    hp, hc = get_td_waveform(
        approximant="IMRPhenomXPHM", # NRsur7dq4 should be used for mergers with mainly merger visible for higher accuracy
        mass1=mass1,
        mass2=mass2,
        distance=distance,
        delta_t=delta_t,
        f_lower=f_lower
    )
    hp.start_time = gps_time
    hc.start_time = gps_time
    return hp, hc

def inject_waveforms(noises, waveforms):
    return {name: noises[name].inject(waveforms[name]) for name in noises}

