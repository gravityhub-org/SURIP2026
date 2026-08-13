import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from scipy.integrate import quad
from figaro.utils import rvs_median
from ler import LeR
from ler.gw_source_population import CBCSourceRedshiftDistribution
from pathlib import Path
ler = LeR(spin_precession=True)

np.random.seed(42)

population_densities_2d = np.load("/Users/natalieyyyyy/Projects/intermediates/population_densities_2d.npy", allow_pickle=True)
cbc = CBCSourceRedshiftDistribution(z_min=0.001, z_max=10)

def generate_samples(size, dict_name):

    # draw masses from FIGARO population
    samples_2d = rvs_median(population_densities_2d, size=size)

    # draw redshift from merger-rate density
    zs_samples = cbc.merger_rate_density_based_source_redshift.rvs(size=size)

    # luminosity distance
    luminosity_distance = ler.luminosity_distance.function(zs_samples)

    # other parameters
    T = 365 * 24 * 3600
    t_start = 0

    geocent_time = np.random.uniform(t_start, t_start + T, size)
    ra = np.random.uniform(0, 2*np.pi, size)
    phase = np.random.uniform(0, 2*np.pi, size)
    psi = np.random.uniform(0, np.pi, size)

    phi_12 = np.random.uniform(0, 2*np.pi, size)
    phi_jl = np.random.uniform(0, 2*np.pi, size)

    a_1 = np.random.uniform(0, 0.99, size)
    a_2 = np.random.uniform(0, 0.99, size)

    # isotropic distributions
    theta_jn = np.arccos(np.random.uniform(-1, 1, size))
    tilt_1 = np.arccos(np.random.uniform(-1, 1, size))
    tilt_2 = np.arccos(np.random.uniform(-1, 1, size))
    dec = np.arcsin(np.random.uniform(-1, 1, size))

    # construct dictionary
    dict_name = {
        "zs": zs_samples,
        "geocent_time": geocent_time,
        "ra": ra,
        "dec": dec,
        "phase": phase,
        "luminosity_distance": luminosity_distance,
        "psi": psi,
        "theta_jn": theta_jn,
        "a_1": a_1,
        "a_2": a_2,
        "mass_1": samples_2d[:, 0],
        "mass_2": samples_2d[:, 1],
        "tilt_1": tilt_1,
        "tilt_2": tilt_2,
        "phi_12": phi_12,
        "phi_jl": phi_jl,
    }

    # finite values
    mask = np.ones(size, dtype=bool)

    for value in dict_name.values():
        mask &= np.isfinite(value)

    # physical constraints
    mask &= dict_name["zs"] >= 0
    mask &= dict_name["mass_1"] >= 0
    mask &= dict_name["mass_2"] >= 0
    mask &= dict_name["luminosity_distance"] >= 0
    mask &= dict_name["mass_1"] >= dict_name["mass_2"]

    # filter
    dict_name = {
        key: value[mask]
        for key, value in dict_name.items()
    }

    return dict_name

print("Drawing samples from the population distribution...", flush=True)

unlensed_params = generate_samples(
    size=10000000,
    dict_name="unlensed_params"
)

# z_hor as a function of chirp mass

n_realizations = 1000

chirp_mass_samples = (unlensed_params["mass_1"] * unlensed_params["mass_2"]) ** (3/5) / (unlensed_params["mass_1"] + unlensed_params["mass_2"]) ** (1/5)
Mc1 = np.linspace(chirp_mass_samples.min(), 100, 20)
Mc2 = np.linspace(100, chirp_mass_samples.max(), 30)
chirp_mass_grid = np.unique(np.concatenate([Mc1, Mc2]))
z_hor_grid_all = []
z_values = np.linspace(0.001, 10.0, 200)  # example search range
q_samples = unlensed_params["mass_2"] / unlensed_params["mass_1"]

for i in tqdm(range(n_realizations)):
    z_hor_grid = []

    gw_param_dict = {key: values[np.random.randint(0, len(unlensed_params["zs"]))] for key, values in unlensed_params.items()}

    q = float(np.random.choice(q_samples))

    for M_c in chirp_mass_grid:
        last_detectable = np.nan
        gw_param_dict["mass_1"] = M_c * (1+q)**(1/5) * q**(-3/5)
        gw_param_dict["mass_2"] = M_c * (1+q)**(1/5) * q**(2/5)

        for z in z_values:
            gw_param_dict["zs"] = z
            gw_param_dict["luminosity_distance"] = float(ler.luminosity_distance.function(np.array([z], dtype=np.float64)))

            pdet = ler.pdet_finder(gw_param_dict=gw_param_dict)["pdet_net"]
            
            if np.any(pdet == 1):
                last_detectable = z
            else:
                break

        z_hor_grid.append(last_detectable)

    z_hor_grid_all.append(z_hor_grid)

z_hor_grid_mean = np.nanmean(np.array(z_hor_grid_all), axis=0)

z_hor_interp = interp1d(
    chirp_mass_grid,
    z_hor_grid_mean,
    kind="linear",
    fill_value="extrapolate"
)

z_hor_samples = z_hor_interp(chirp_mass_samples)

plt.plot(chirp_mass_grid, z_hor_grid_mean)
plt.xlabel(r"Chirp Mass, $\mathcal{M}$ [$M_{\odot}$]", fontsize=12)
plt.ylabel(r"Horizon Redshift, $z_\text{hor}$", fontsize=12)
plt.savefig("/Users/natalieyyyyy/Projects/results/z_hor_vs_m_c", dpi=300)

# compute integral (N_obs) and hence the weights for each sample

def R_GW(z):
    [a_1, a_2, a_3, a_4] = [6600, 1.6, 2.1, 30]
    return a_1 * math.e ** (a_2*z) / (math.e ** (a_3*z) + a_4) # merger rate density as a function of redshift

dvc_dz = ler.differential_comoving_volume

def integrand(z):
    return R_GW(z) * dvc_dz(z) / (1 + z)

weight_grid = 1.0 / np.array([
    quad(integrand, 0, z)[0]
    for z in z_hor_grid_mean
])
weight_grid /= weight_grid.sum()

plt.figure(figsize=(7, 5))

plt.plot(
    chirp_mass_grid,
    weight_grid,
    linewidth=2
)

plt.xlabel(r"Chirp Mass, $\mathcal{M}_c$ [$M_\odot$]", fontsize=14)
plt.ylabel(r"Normalized weight $w$", fontsize=14)
plt.yscale("log")
plt.tight_layout()
plt.savefig("/Users/natalieyyyyy/Projects/results/w_vs_m_c", dpi=300)

# rejection sampling in during generation of samples from FIGARO population density

good = (
    np.isfinite(chirp_mass_grid)
    & np.isfinite(z_hor_grid_mean)
    & (z_hor_grid_mean > 0)
)

chirp_mass_grid = chirp_mass_grid[good]
z_hor_grid_mean = z_hor_grid_mean[good]

n_det_grid = np.array([
    quad(integrand, 0.0, float(z_hor))[0]
    for z_hor in z_hor_grid_mean
])

# inverse-detection weight
w_grid = 1.0 / np.maximum(n_det_grid, 1e-300)

w_func = interp1d(
    chirp_mass_grid,
    w_grid,
    kind="linear",
    fill_value="extrapolate",
    bounds_error=False
)

# global upper bound for the weight
chirp_mass_grid_dense = np.linspace(chirp_mass_grid.min(), chirp_mass_grid.max(), 10000)
w_max = 1.05 * np.nanmax(w_func(chirp_mass_grid_dense))

print("w_max =", w_max)

batch_n = 10000
target_n = 10000
accepted = []
n_acc_total = 0

while n_acc_total < target_n:

    cand = generate_samples(
        size=batch_n,
        dict_name="cand"
    )

    if len(cand["zs"]) == 0:
        continue

    # chirp mass
    mc = (
        (cand["mass_1"] * cand["mass_2"]) ** (3/5)
        / (cand["mass_1"] + cand["mass_2"]) ** (1/5)
    )

    # only use masses within the range of the weight function
    valid_mc = (
        np.isfinite(mc)
        & (mc >= chirp_mass_grid.min())
        & (mc <= chirp_mass_grid.max())
    )

    if not np.any(valid_mc):
        continue

    # normalized acceptance probability
    p_acc = np.asarray(
        w_func(mc[valid_mc]) / w_max,
        dtype=float
    )

    if np.any(p_acc < 0) or np.any(p_acc > 1):
        raise RuntimeError("Invalid acceptance probability.")

    # draw uniformly between 0 and the batch maximum
    u = np.random.uniform(0, 1, len(p_acc))

    # accept valid events
    keep_valid = u < p_acc

    # map back to the full candidate array
    keep = np.zeros(len(mc), dtype=bool)
    keep[valid_mc] = keep_valid

    accepted_batch = {
        key: value[keep]
        for key, value in cand.items()
    }

    n_acc = len(accepted_batch["zs"])

    if n_acc > 0:
        accepted.append(accepted_batch)
        n_acc_total += n_acc

    print(
        f"Accepted: {n_acc}; "
        f"total: {n_acc_total}/{target_n}"
    )

# combine accepted batches
new_params = {}

for key in accepted[0]:
    new_params[key] = np.concatenate(
        [batch[key] for batch in accepted]
    )

# exactly target_n events
for key in new_params:
    new_params[key] = new_params[key][:target_n]

print("final catalogue size:", len(new_params["zs"]))

print("Generating lens parameters...", flush=True)

size = len(new_params['zs'])
zl = ler.lens_redshift_sl.rvs(size, new_params['zs'])
sigma, q, phi, gamma, gamma1, gamma2 = ler.cross_section_based_sampler(
    size, new_params['zs'], zl
)
lens_params = {
    "zl": zl,
    "zs": new_params['zs'],
    "sigma": sigma,
    "theta_E": ler.compute_einstein_radii(sigma, zl, new_params['zs']),
    "q": q,
    "phi": phi,
    "gamma": gamma,
    "gamma1": gamma1,
    "gamma2": gamma2,
}
lens_params.update(new_params)
image_params = ler.image_properties_function(lens_params.copy())
image_params = ler.recover_redundant_parameters(image_params)

# unlensed rate
print("Calculating unlensed rate...", flush=True)
pdet_net = ler.pdet_finder(gw_param_dict=new_params)
new_params.update(pdet_net)
unlensed_rate, unlensed_param_detectable = ler.unlensed_rate(new_params)

# lensed rate
print("Calculating lensed rate...", flush=True)
pdet, lensed_params = ler.get_lensed_snrs(
            lensed_param=image_params.copy(),
            pdet_finder=ler.pdet_finder,
        )
lensed_params.update(pdet)
lensed_params["n_images"] = np.count_nonzero(lensed_params["image_type"], axis=1) # add n_images back
lensed_rate, lensed_param_detectable = ler.lensed_rate(lensed_params)

# rate ratio
ratio = lensed_rate / unlensed_rate

# prior odds
n_signals = 2
prior_odds = math.factorial(n_signals) * lensed_rate / unlensed_rate ** n_signals

# results
print("Results:")
print(f"Total unlensed rate: {unlensed_rate:.2f} detectable events per year")
print(f"Total lensed rate: {lensed_rate:.2f} detectable events per year")
print(f"Rate ratio (lensed/unlensed): {ratio:.2g}")
print(f"Prior odds: {prior_odds:.2g}")
