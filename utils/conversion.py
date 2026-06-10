import numpy as np
from scipy.interpolate import interp1d
from astropy.cosmology import wCDM, z_at_value
import astropy.units as u

__all__ = [
    'add_mass_parameters',
    'add_spin_parameters',
    'cogwheel_to_bilby'
]

# print("Pre-computing cosmology interpolation...")
# This cosmology is taken from FIGARO, not exactly the same as Planck18 from astropy
h, om, ol, w0 = 0.674, 0.315, 0.685, -1
w1, w2 = 0, 0
Planck18 = wCDM(H0=h*100, Om0=om, Ode0=ol, w0=w0)

# Pre-compute and interpolate dL-z relation for faster conversion
dLs = np.geomspace(0.5, 4e5, 10000)
zs = z_at_value(Planck18.luminosity_distance, dLs * u.Mpc).value
dL2z = interp1d(dLs, zs, bounds_error=True)


def add_mass_parameters(dataframe):
    # Add mass parameters
    m1, m2 = dataframe['mass_1'], dataframe['mass_2']
    dataframe['total_mass'] = m1 + m2
    if 'chirp_mass' not in dataframe.keys():
        dataframe['chirp_mass'] = ((m1 * m2)**3 / (m1 + m2))**(1/5)
    if 'mass_ratio' not in dataframe.keys():
        dataframe['mass_ratio'] = m2 / m1
    assert np.all(dataframe['mass_ratio'] <= 1), "All m2 <= m1 should always be true."

    # Add source-frame parameters
    redshift = dL2z(dataframe['luminosity_distance'])
    dataframe['redshift'] = redshift
    dataframe['mass_1_source'] = m1 / (1 + redshift)
    dataframe['mass_2_source'] = m2 / (1 + redshift)
    dataframe['chirp_mass_source'] = (dataframe['chirp_mass'] / (1 + redshift))
    dataframe['total_mass_source'] = (dataframe['total_mass'] / (1 + redshift))
    return dataframe


def rotate_cogwheel_spins(s1x_n, s1y_n, s2x_n, s2y_n, phi_ref):
    # Rotate spins from cogwheel frame to standard PE frame
    # The rotation is around the z-axis by the reference phase
    # Follows Eq. B1 from https://arxiv.org/pdf/2207.03508
    cos_phi = np.cos(phi_ref)
    sin_phi = np.sin(phi_ref)

    rot_mat = np.array([[cos_phi, sin_phi], [-sin_phi, cos_phi]])
    spins_in = np.array([[s1x_n, s1y_n], [s2x_n, s2y_n]])
    spins_out = np.einsum('ij...,kj...->ki...', rot_mat, spins_in)

    return spins_out[0], spins_out[1]


def add_spin_parameters(injection_dict):
    mass_ratio = injection_dict['mass_ratio']
    # Rotate spins
    spin_1xy, spin_2xy = rotate_cogwheel_spins(
        injection_dict.pop('s1x_n'), injection_dict.pop('s1y_n'),
        injection_dict.pop('s2x_n'), injection_dict.pop('s2y_n'),
        injection_dict.get('phi_ref')
    )
    injection_dict['spin_1x'] = spin_1xy[0]
    injection_dict['spin_1y'] = spin_1xy[1]
    injection_dict['spin_2x'] = spin_2xy[0]
    injection_dict['spin_2y'] = spin_2xy[1]
    injection_dict['phase'] = injection_dict.pop('phi_ref')

    injection_dict['chi_1_in_plane'] = spin_1_inpln = np.sqrt(np.sum(spin_1xy**2, axis=0))
    injection_dict['chi_2_in_plane'] = spin_2_inpln = np.sqrt(np.sum(spin_2xy**2, axis=0))

    # The spin magnitudes and polar angles (tilts)
    injection_dict['a_1'] = np.sqrt(injection_dict['spin_1z']**2 + spin_1_inpln**2)
    injection_dict['a_2'] = np.sqrt(injection_dict['spin_2z']**2 + spin_2_inpln**2)
    injection_dict['tilt_1'] = np.arccos(injection_dict['spin_1z'])
    injection_dict['tilt_2'] = np.arccos(injection_dict['spin_2z'])

    # Effective spin
    injection_dict['chi_eff'] = (injection_dict['spin_1z'] + mass_ratio * injection_dict['spin_2z']) / (1 + mass_ratio)
    # Effective precession spin
    chi_p_factor = (4 + 3 * mass_ratio) / (4 + 3 / mass_ratio)
    injection_dict['chi_p'] = np.max([spin_1_inpln, spin_2_inpln * chi_p_factor])
    return injection_dict


cogwheel_to_bilby_mapping = {
    'm1': 'mass_1',
    'm2': 'mass_2',
    'd_luminosity': 'luminosity_distance',
    'f_ref': 'reference_frequency',
    's1z': 'spin_1z',
    's2z': 'spin_2z',
    'l1': 'lambda_1',
    'l2': 'lambda_2',
    'SNR_net': 'network_optimal_snr',
    't_geocenter': 'geocent_time',
}


def cogwheel_to_bilby(injection_dict):
    for old_key, new_key in cogwheel_to_bilby_mapping.items():
        try:
            injection_dict[new_key] = injection_dict.pop(old_key)
        except KeyError:
            print(f'Warning: {old_key} not found, skipping.')
            continue

    # Add the mass parameters
    injection_dict = add_mass_parameters(injection_dict)
    injection_dict = add_spin_parameters(injection_dict)

    return injection_dict
