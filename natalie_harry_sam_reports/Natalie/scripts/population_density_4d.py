import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from figaro.mixture import DPGMM, HDPGMM
from figaro.utils import get_priors
from figaro.plot import plot_multidim
from tqdm import tqdm

import scienceplots
plt.style.use(['science', 'ieee', 'bright'])

print("Loading events...", flush=True)

data_dir = Path(
    "/users/shared.user/Summer2026/"
    "cogwheel_O4_gaussian_injections/"
    "PE/IntrinsicLVCPrior/"
    "unlensed_samples/processed_samples"
)
posterior_samples_list = []
n_events = 1486  # Change this to 1486 later

for i in range(n_events):
    event_id = f"MD_SDSS_u_{i}"
    file_path = data_dir / f"{event_id}_bilby_style_samples.h5"
    if not file_path.exists():
        print(f"Skipping {event_id} (file not found)", flush=True)
        continue
    # posterior_samples = pd.read_hdf(file_path)
    posterior_samples = (pd.read_hdf(file_path).sample(n=100, random_state=42))
    samples = posterior_samples[["mass_1", "mass_2", "chi_eff", "redshift"]].to_numpy(dtype=float)
    samples = samples[np.all(np.isfinite(samples), axis=1)]
    posterior_samples_list.append(samples)

np.save("intermediates/posterior_samples_list.npy", posterior_samples_list)

posteriors = []
n_draws = 10  # for testing, can be increased for final version

print("Building DPGMMs...", flush=True)

for samples in tqdm(posterior_samples_list, desc="DPGMMs"):
    lower = np.min(samples, axis=0) - 0.5
    upper = np.max(samples, axis=0) + 0.5
    bounds = list(zip(lower, upper))
    prior_pars = get_priors(bounds=bounds, samples=samples, probit=False)
    mix = DPGMM(bounds=bounds, prior_pars=prior_pars, probit=False)

    draws = []
    for _ in tqdm(range(n_draws)):
        draws.append(mix.density_from_samples(samples))
    posteriors.append(draws)

np.save("intermediates/posterior_densities_4d.npy", posteriors)

global_lower = []
global_higher = []
for i in range(4):  # Assuming 4D samples (mass_1, mass_2, chi_eff, redshift)
    global_lower.append(min(np.min(samples[:, i]) for samples in posterior_samples_list) - 0.5)
    global_higher.append(max(np.max(samples[:, i]) for samples in posterior_samples_list) + 0.5)
global_bounds = list(zip(global_lower, global_higher))

print("Building HDPGMM...", flush=True)

hier_mix = HDPGMM(global_bounds, prior_pars = get_priors(global_bounds, samples = posterior_samples_list, 
                                                            hierarchical = True, probit = False), probit = False)

n_draws_hier = 100
hier_draws = []

for _ in tqdm(range(n_draws_hier)):
    hier_draws.append(hier_mix.density_from_samples(posteriors))

np.save("intermediates/population_densities_4d.npy", hier_draws)

print("Loading figure...", flush=True)

fig = plot_multidim(hier_draws, samples = np.concatenate(posterior_samples_list), labels = [r'm_1', r'm_2', r'\chi_\mathrm{eff}', r'\mathrm{redshift}'],show = False, median_label=r'\mathrm{HDPGMM}')

axs = np.array(fig.axes).reshape(4, 4)

fig.suptitle(r"$\mathrm{Masses,\ Effective\ Spin,\ and\ Redshift\ Population}$", fontsize=30)

# Set font sizes for axes labels and ticks
for ax in fig.axes:
    ax.xaxis.label.set_size(25)
    ax.yaxis.label.set_size(25)
    ax.tick_params(axis='both', labelsize=20)

# Adjust the limits of the diagonal and off-diagonal plots based on the data
for i in range(4):
    all_samples = np.concatenate([s[:, i] for s in posterior_samples_list])

    xmin = np.percentile(all_samples, 1)
    xmax = np.percentile(all_samples, 99)
    axs[i, i].set_xlim(xmin, xmax)
    ymax = max(line.get_ydata().max() for line in axs[i, i].lines)
    axs[i, i].set_ylim(0, 1.05 * ymax)

    for j in range(i + 1, 4):
        axs[j, i].set_xlim(xmin, xmax)
    for j in range(i):
        axs[i, j].set_ylim(xmin, xmax)

# Set font sizes for legend texts
for ax in fig.axes:
    if ax.legend_ is not None:
        for text in ax.legend_.get_texts():
            text.set_fontsize(25)

fig.savefig("results/unlensed_4d.pdf", dpi=300)

print("Done! Figure saved to Projects/results")
