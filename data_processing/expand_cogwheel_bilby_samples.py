from pathlib import Path
import numpy as np
import pandas as pd
import json
from multiprocessing import Pool
from tqdm import tqdm

# This part is for importing local utilities
import sys
sys.path.append("../utils")
from files_io import h5store

main_dir = Path("/users/shared.user/Summer2026/cogwheel_O4_gaussian_injections/PE/IntrinsicLVCPrior/")
all_dirs = list(main_dir.glob("*/MD_*/run_0"))
total = len(all_dirs)

cogwheel_style = 'samples.feather'
bilby_style = 'bilby_style_samples.feather'

print("Pre-computing cosmology interpolation...")
# This cosmology is taken from FIGARO, not exactly the same as Planck18 from astropy
from astropy.cosmology import wCDM
h, om, ol, w0 = 0.674, 0.315, 0.685, -1
w1, w2 = 0, 0
Planck18 = wCDM(H0=h*100, Om0=om, Ode0=ol, w0=w0)

# Pre-compute and interpolate dL-z relation for faster conversion
from astropy.cosmology import z_at_value
import astropy.units as u
from scipy.interpolate import interp1d
dLs = np.geomspace(0.5, 4e5, 10000)
zs = z_at_value(Planck18.luminosity_distance, dLs * u.Mpc).value
dL2z = interp1d(dLs, zs, bounds_error=True)


def worker(dir):
    event_name = dir.parent.name
    outdir = dir.parent.parent / "processed_samples"
    outdir.mkdir(exist_ok=True)
    inj_outdir = dir.parent.parent / "injections_values"
    inj_outdir.mkdir(exist_ok=True)

    try:
        orig_cogwheel_samples = pd.read_feather(dir / cogwheel_style)
        bilby_samples = pd.read_feather(dir / bilby_style)
    except IOError:
        # print(f'{event_name} does not have samples!')
        return (event_name, 2)

    bilby_samples['network_optimal_snr'] = np.sqrt(orig_cogwheel_samples['h_h'])
    bilby_samples['log_likelihood'] = orig_cogwheel_samples['lnl']

    # Add mass parameters
    m1, m2 = bilby_samples['mass_1'], bilby_samples['mass_2']
    bilby_samples['total_mass'] = m1 + m2
    assert np.all(bilby_samples['mass_ratio'] <= 1), "All m2 <= m1 should always be true."

    # Add source-frame parameters
    redshift = dL2z(bilby_samples['luminosity_distance'].values)
    bilby_samples['redshift'] = redshift
    bilby_samples['mass_1_source'] = m1 / (1 + redshift)
    bilby_samples['mass_2_source'] = m2 / (1 + redshift)
    bilby_samples['chirp_mass_source'] = (bilby_samples['chirp_mass'] / (1 + redshift))
    bilby_samples['total_mass_source'] = (bilby_samples['total_mass'] / (1 + redshift))

    # Load injection values
    npz_file = np.load(dir / f'{event_name}.npz')
    injection_meta = json.loads(npz_file['injection'].tobytes())
    inj_dict = injection_meta.pop('par_dic')
    vals = list(inj_dict.values())
    dtypes = [(k, 'f8') for k in inj_dict.keys()]

    # Add metadata
    bilby_samples.attrs['event_name'] = event_name
    bilby_samples.attrs['injection_values'] = inj_dict
    bilby_samples.attrs['cogwheel_samples_path'] = (dir / cogwheel_style).as_posix()
    h5store(outdir / f"{event_name}_bilby_style_samples.h5", bilby_samples)
    # bilby_samples.to_feather(outdir / f"{event_name}_bilby_style_samples.feather")

    for key, val in injection_meta.items():
        vals.append(val)
        if key == 'approximant':
            dtypes.append((key, 'U20'))
        else:
            dtypes.append((key, 'f8', 3))

    np_struct_arr = np.array(tuple(vals), dtype=dtypes)
    np.save(inj_outdir / f"{event_name}_injection_values.npy", np_struct_arr)

    return (event_name, 0)


if __name__ == "__main__":
    print("Starting multi-processing...")
    with Pool(20) as pool:
        output = list(tqdm(pool.imap(worker, all_dirs), total=total))

    statuses = np.array(output, dtype=[('event_name', 'U30'), ('status', 'i4')])
    # Mark events with (at least) one failed image as 3
    failed_events_idx = statuses['status'] == 2
    for event in statuses['event_name'][failed_events_idx]:
        name_parts = event.split('_')
        if len(name_parts) < 5:  # No other images.
            continue
        event_name = '_'.join(name_parts[:4])
        same_event_idx = np.char.startswith(statuses['event_name'], event_name + '_')
        statuses['status'] = np.where(same_event_idx, 3, statuses['status'])
    statuses['status'] = np.where(failed_events_idx, 2, statuses['status'])

    np.save(main_dir / "event_preprocessing_status.npy", statuses)
    print('Done')
