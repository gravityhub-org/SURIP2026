#!/cvmfs/software.igwn.org/conda/envs/igwn-py311/bin/python
from pathlib import Path
import numpy as np
import pandas as pd
import json
from multiprocessing import Pool
from tqdm import tqdm

# This part is for importing local utilities
import sys
sys.path.append("..")
from utils import h5store, cogwheel_to_bilby, add_mass_parameters

main_dir = Path("/users/shared.user/Summer2026/cogwheel_O4_gaussian_injections/PE/IntrinsicLVCPrior/")
# main_dir = Path.home() / "tmp/ankur_cogwheel_injections/PE/IntrinsicLVCPrior/"
all_dirs = list(main_dir.glob("*/MD_*/run_0"))
total = len(all_dirs)

cogwheel_style = 'samples.feather'
bilby_style = 'bilby_style_samples.feather'


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

    bilby_samples = add_mass_parameters(bilby_samples)

    # Load injection values
    npz_file = np.load(dir / f'{event_name}.npz')
    injection_meta = json.loads(npz_file['injection'].tobytes())
    inj_dict = cogwheel_to_bilby(injection_meta.pop('par_dic'))
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
