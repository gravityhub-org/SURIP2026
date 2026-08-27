# ==========================================================
#set up and imports
# ==========================================================
import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=12")
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
print("JAX devices:", jax.devices())

import sys
from pathlib import Path

repo_root = Path.cwd().resolve()
for parent in [repo_root, *repo_root.parents]:
    src = parent / "src"
    if (src / "gwemfish").is_dir():
        sys.path.insert(0, str(src))
        break

import matplotlib.pyplot as plt

from gwemfish.simple_pipeline import (
    _deep_merge_dict,
    make_default_cfg,
    plot_system_observation,
    setup_em_observation,
    setup_gw_observation,
    run_inference, 
    plot_posterior, 
    to_source_plane_samples, 
    plot_source_posterior,
)
from gwemfish.config import DEFAULT_KWARGS_NUMERICS, SOLVER_PARAMS
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
import copy
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.number):
            return obj.item()
        return super().default(obj)

from lenstronomy.Util import param_util
from lenstronomy.SimulationAPI.mag_amp_conversion import MagAmpConversion

#Same config file for GWEMFISH
sample_cfg = {
    "em": {
        "pixel_grid_kwargs": {"npix": int(40), "pix_scl": 0.1},
        "psf_kwargs": {"psf_type": "GAUSSIAN", "fwhm": 0.4, "pixel_size": 0.1},#0.067
        "noise_simu_kwargs": {"npix": int(40), "background_rms": 5e-2, "exposure_time": 2200},
        "noise_inf_kwargs": {"npix": int(40), "background_rms": None, "exposure_time": 2200},
        "kwargs_numerics": DEFAULT_KWARGS_NUMERICS,
        "exposure_time": 2200,
        "seed": 87651,
        "source_pos": (0.0, 0.0), #Is this the position of the source of the GW or the position of the center of the source galaxy?
        "kwargs_source": [
            {
                "amp": 250,
                "R_sersic": 0.4,
                "n_sersic": 4.0,
                "e1": 0,
                "e2": 0,
                "center_x": 0.0, #same question for these center_x and y values
                "center_y": 0.0,
            }
        ],
        "kwargs_lens_light": [
            {
                "amp": 50.0,
                "R_sersic": 2.0,
                "n_sersic": 4.0,
                "e1": 0.0,
                "e2": 0.0,
                "center_x": 0.0,
                "center_y": 0.0,
            }
        ],
    },
    "lens": {
        "lens_model_list": ["EPL", "SHEAR"],
        "kwargs_lens": [
            {
                "theta_E": 1.2,
                "e1": 0.0,
                "e2": 0.1,
                "gamma": 2.0,
                "center_x": 0.0,
                "center_y": 0.0,
            },
            {"gamma1": 0.1, "gamma2": 0.0, "ra_0": 0.0, "dec_0": 0.0},
        ],
        "zl": 0.7,
        "zs": 1.5,
    },
    "gw": {
        "enabled": True,
        "n_images": 2,
        "source_pos": (0.0, 0.0),
        "solver_params": SOLVER_PARAMS,
        "image_box_half_width": 10.6,
        "error_scales": {
            "sigma_td": 0.05,
            "sigma_dL_eff": 0.2,
            "epsilon": 0.005,
        },
    },
    "plot": {"plot_mode": "groupwise", "save_path": None, "save_tag": None, "hist_kwargs": {"density": True}},
    "source_plane": {"filter_std": None, "use_filtered": False},
    "output": {
        "output_dir": OUTPUT_DIR,
        "save_samples_path": None,
        "save_truths_path": None,
        "save_source_samples_path": None,
        "save_system_plot_path": None,
        "json_path": None,
    },
}

#Conversion between Qiuhan's catalog and GWEMFish config file
def row_to_cfg(row, sample_cfg, gw_enabled):
    """
    Convert one row of the Qiuhan's sky catalog into a GWEMFISH configuration.

    Parameters
    ----------
    row : pandas.Series
        One row of the lens catalog.

    sample_cfg : dict
        Reference GWEMFISH config.

    Returns
    -------
    cfg : dict
        GWEMFISH configuration for this lens system.
    """

    cfg = copy.deepcopy(sample_cfg)

    # ==========================================================
    # Convert axis ratio + PA -> ellipticity

    # --- 1. PREPARE LENS (DEFLECTOR) LIGHT PARAMETERS ---
    # Convert q and pa to lenstronomy ellipticity (e1, e2)
    lens_e1, lens_e2 = param_util.phi_q2_ellipticity(phi=row['deflector_pa'], q=row['deflector_q'])

    kwargs_lens_light_mag = [{
    'magnitude': row['deflector_app_mag_VIS'],
    'R_sersic': row['deflector_Re'],
    'n_sersic': 4.0,  # Standard assumption for elliptical lens galaxies
    'e1': lens_e1, 'e2': lens_e2,
    'center_x': 0, 'center_y': 0
    }]

    # --- 2. PREPARE SOURCE LIGHT PARAMETERS ---
    # Convert q and pa to lenstronomy ellipticity (e1, e2)
    source_e1, source_e2 = param_util.phi_q2_ellipticity(phi=row['source_pa'], q=row['source_q'])
    
    # Choose your band (e.g., VIS)
    kwargs_source_mag = [{
        'magnitude': row['source_app_mag_VIS'], 
        'R_sersic': row['source_Re'],
        'n_sersic': row['source_sersic_index'],
        'e1': source_e1, 'e2': source_e2,
        'center_x': row['source_relative_x'], 
        'center_y': row['source_relative_y']
    }]
    
    # --- 3. CONVERT BOTH TO AMP ---
    # Define your model profile types
    kwargs_model = {
        'lens_light_model_list': ['SERSIC_ELLIPSE'],
        'source_light_model_list': ['SERSIC_ELLIPSE']
    }
    
    # Initialize the converter with your survey zero-point
    mag_converter = MagAmpConversion(kwargs_model=kwargs_model, magnitude_zero_point=25.9)
    
    # Get the final dictionaries containing the calculated 'amp' keys
    lens_light_amp = mag_converter.magnitude2amplitude(kwargs_lens_light_mag=kwargs_lens_light_mag)
    source_amp = mag_converter.magnitude2amplitude(kwargs_source_mag=kwargs_source_mag)
    

    lens_pos = (float(row['deflector_ra']), float(row['deflector_dec']))
    
    source_pos = (float(row["source_relative_x"]), float(row["source_relative_y"]))

    # ==========================================================
    # Lens geometry
    cfg["lens"]["zl"] = float(row["deflector_z"])
    cfg["lens"]["zs"] = float(row["source_z"])

    # ==========================================================
    # Lens mass model (EPL)
    cfg["lens"]["kwargs_lens"][0]["theta_E"] = float(row["deflector_thetaE"])
    cfg["lens"]["kwargs_lens"][0]["gamma"] = float(row["deflector_slope"])

    cfg["lens"]["kwargs_lens"][0]["e1"] = float(lens_e1)
    cfg["lens"]["kwargs_lens"][0]["e2"] = float(lens_e2)

    cfg["lens"]["kwargs_lens"][0]["center_x"] = 0.00
    cfg["lens"]["kwargs_lens"][0]["center_y"] = 0.00

    # ==========================================================
    # External shear
    cfg["lens"]["kwargs_lens"][1]["gamma1"] = float(row["deflector_shear1"])
    cfg["lens"]["kwargs_lens"][1]["gamma2"] = float(row["deflector_shear2"])

    cfg["lens"]["kwargs_lens"][1]["ra_0"] = 0.00
    cfg["lens"]["kwargs_lens"][1]["dec_0"] = 0.00

    # ==========================================================
    # Source light
    cfg["em"]["kwargs_source"][0]["R_sersic"] = float(row["source_Re"])
    cfg["em"]["kwargs_source"][0]["n_sersic"] = float(row["source_sersic_index"])

    cfg["em"]["kwargs_source"][0]["e1"] = float(source_e1)
    cfg["em"]["kwargs_source"][0]["e2"] = float(source_e2)

    cfg["em"]["kwargs_source"][0]["center_x"] = source_pos[0]
    cfg["em"]["kwargs_source"][0]["center_y"] = source_pos[1]

    cfg["em"]["kwargs_source"][0]["amp"] = float(source_amp[1][0]['amp'])

    # ==========================================================
    # Lens light
    cfg["em"]["kwargs_lens_light"][0]["R_sersic"] = float(row["deflector_Re"])
    cfg["em"]["kwargs_lens_light"][0]["n_sersic"] = float(4)

    cfg["em"]["kwargs_lens_light"][0]["e1"] = float(lens_e1)
    cfg["em"]["kwargs_lens_light"][0]["e2"] = float(lens_e2)

    cfg["em"]["kwargs_lens_light"][0]["center_x"] = 0.00
    cfg["em"]["kwargs_lens_light"][0]["center_y"] = 0.00

    cfg["em"]["kwargs_lens_light"][0]["amp"] = float(lens_light_amp[0][0]['amp'])

    # ==========================================================
    # GW source position
    cfg["em"]["source_pos"] = source_pos
    
    if gw_enabled is True:
        cfg["gw"]["source_pos"] = (source_pos[0]+0.005, source_pos[1]-0.005)

    if gw_enabled is False:
        cfg["gw"] = {"enabled": False}

    return cfg

# ==========================================================
#Import Qiuhan's catalog
# ==========================================================
import pandas as pd
df = pd.read_csv('../../data_catalog/Qiuhan/Euclid_cats/filtered_lens_catalog_PL_IC_gt_70.csv')


# ==========================================================
#Searching for 4 image candidates (roughly)
# ==========================================================
import time
start_time = time.time()

accepted_rows = []
for i in range(200, 1000):

    one_galaxy = df.iloc[i:i+1]

    if one_galaxy.iloc[0]['deflector_thetaE'] < 0.6:
        continue

    row = one_galaxy.iloc[0]
    cfg = row_to_cfg(row, sample_cfg, False)
    cfg["use_parameter_layout"] = True
    ctx = setup_em_observation(cfg=cfg)
    tp_pair = ctx['truth_params']

    if len(tp_pair['x_image_true_em']) < 4:
        continue
    
    #print(f'System {i} is close enough to the GW source and has at least four images. Proceed with manual inspection')
    accepted_rows.append(one_galaxy.iloc[0])

candidates = pd.DataFrame(accepted_rows, columns=df.columns)
print("--- %s seconds ---" % (time.time() - start_time))


# ==========================================================
#preparing to compute the posterior overlap and EM, GW only evidnces
# ==========================================================

from scipy.stats import gaussian_kde
shared_params = ['lens0_e1', 'lens0_e2', 'lens0_gamma', 'lens0_theta_E', 'lens1_gamma1', 'lens1_gamma2']
import numpyro.distributions as dist
from gwemfish import prune_gw_images
from scipy.stats import norm
from scipy.special import logsumexp
start = 2
finish = 4
this_run = candidates.iloc[start:finish]


# ==========================================================
#run GWEMFISH inference and compute posterior overlap. Then, save as jsonl
# ==========================================================
for idx, row in this_run.iterrows():

    print(f'----------------------------------------Starting {idx}-th iteration---------------------------------------------------')
    
    #row = candidates.iloc[i]
    cfg_org = row_to_cfg(row, sample_cfg, True)
    cfg_org['em']['pixel_grid_kwargs']['npix'] = 60
    cfg_org['em']['noise_simu_kwargs']['npix'] = 60
    cfg_org['em']['noise_inf_kwargs']['npix'] = 60
    cfg_org['em']['psf_kwargs']['fwhm'] = 0.18
    
    cfg_org["use_parameter_layout"] = True

    ctx_post_em = setup_em_observation(cfg=cfg_org)
    ctx = setup_gw_observation(ctx_post_em, cfg=cfg_org)
    
    
    if len(ctx["x_img_gw"]) < 4:
        print('-----------END: LESS THAN 4 IMAGES----------')
        continue
        
    if len(ctx["x_img_gw"]) > 4:
        ctx = prune_gw_images(ctx, n_keep=4)
    
    #save true values
    tp = ctx['truth_params']
    fig = plot_system_observation(
        ctx,
        cfg={"output": {"save_system_plot_path": os.path.join(OUTPUT_DIR, f"system_observation_{idx}.png")}},
    )

    ctx["cfg"]["priors"] = {
        "T_star": float(tp["T_star"]),
        "dL": float(tp["dL"]),
        'lens0_gamma': dist.TruncatedNormal(tp['lens0_gamma'], 0.1, low=0.001, high=5.0),
        'lens0_theta_E': dist.TruncatedNormal(tp['lens0_theta_E'], 0.1, low=0.001, high=5.0),
         #'lens0_e1': float(tp["lens0_e1"]), 
         #'lens0_e2': float(tp['lens0_e2']),
         'lens0_e1': dist.Normal(tp['lens0_e1'], 0.01),
         'lens0_e2': dist.Normal(tp['lens0_e2'], 0.01),
         'lens0_center_x': 0.0,
         'lens0_center_y': 0.0,
         #'lens1_gamma1': float(tp["lens1_gamma1"]),
         #'lens1_gamma2': float(tp["lens1_gamma2"]),
         'lens1_gamma1': dist.Normal(tp['lens1_gamma1'], 0.01),
         'lens1_gamma2': dist.Normal(tp['lens1_gamma2'], 0.01),
         'lens1_ra_0': 0.0,
         'lens1_dec_0': 0.0,
          'light0_R_sersic': float(tp["light0_R_sersic"]),
          'light0_n_sersic': float(tp["light0_n_sersic"]),
          'light0_amp': float(tp['light0_amp']),
         'light0_e1': float(tp["light0_e1"]),
         'light0_e2': float(tp["light0_e2"]),
         'light0_center_x': float(tp["light0_center_x"]),
         'light0_center_y': float(tp["light0_center_y"]),
          "noise_sigma_bkg": tp["noise_sigma_bkg"],  
        } 

    ctx_em = copy.deepcopy(ctx)
    ctx_gw = copy.deepcopy(ctx)
    
    
    #perform the parameter estimation in EM-only mode
    method = 'deriv-approx'
    
    samples_em, truths_em = run_inference(
    ctx_em,
    mode="EM-only",#"EM-only",#"GW-only",
    method=method,
    cfg={
        "output": {"json_tag": method},
        **({"inference": {"informed": True, "regularize": False,"num_chains": 8, "num_samples": 14500, "num_warmup": 9500}} if method == "deriv-approx" else {}),
        # "inference": {"num_chains": 12, "num_samples": 8000, "num_warmup": 8000, "n_fisher_samples": 10000},
    },
    )

    #Param est for GW-only mode
    
    samples_gw, truths_gw = run_inference(
    ctx_gw,
    mode="GW-only",#"EM-only",#"GW-only",
    method=method,
    cfg={
        "output": {"json_tag": method},
        **({"inference": {"informed": True, "regularize": False,"num_chains": 8, "num_samples": 14500, "num_warmup": 9500}} if method == "deriv-approx" else {}),
        # "inference": {"num_chains": 12, "num_samples": 8000, "num_warmup": 8000, "n_fisher_samples": 10000},
    },
    )

    #param sigmas and check if NaN values exists
    has_any_nan_em = any(np.isnan(arr).any() for arr in samples_em.values())
                
    has_any_nan_gw = any(np.isnan(arr).any() for arr in samples_gw.values())
    
    if has_any_nan_gw or has_any_nan_em:
        print("--------------------UH OH! Parameter estimation exploded!!!! Contains NaN values-----------------")
        break

    print(f'-------------------Finished parameter estimation -------------------------')
    
    #compute the em only evidence
    logp0_em = float(ctx_em["fisher"]["logp0"]) #true point, 
    H_em = np.asarray(ctx_em["fisher"]["H0"]) # also around true point
    
    k_em = H_em.shape[0]
    
    sign_em, logdet_em = np.linalg.slogdet(-H_em) #Compute the sign and (natural) logarithm of the determinant of an array.
    
    logZ_em = (
        logp0_em
        + 0.5 * k_em * np.log(2*np.pi)
        - 0.5 * logdet_em
    )


    samples_matrix_GW = np.vstack([samples_gw[p] for p in shared_params])
    #defining the gaussian kde
    kde_gw = gaussian_kde(samples_matrix_GW)
    
    print(f'-------------------Finished computing logZ_em-------------------------')
    
    #compute the joint evidence with monte carlo integration
    samples_matrix_EM = np.vstack([samples_em[p] for p in shared_params])
    
    p_gw = kde_gw(samples_matrix_EM)
    
    gamma_samples = np.asarray(samples_em["lens0_gamma"])
    thetaE_samples = np.asarray(samples_em["lens0_theta_E"])
    e1_samples = np.asarray(samples_em["lens0_e1"])
    e2_samples = np.asarray(samples_em["lens0_e2"])
    gamma1_samples = np.asarray(samples_em["lens1_gamma1"])
    gamma2_samples = np.asarray(samples_em["lens1_gamma2"])
    
    log_prior = (
        norm.logpdf(gamma_samples,tp['lens0_gamma'],0.1)
        + norm.logpdf(thetaE_samples,tp['lens0_theta_E'],0.1)
        + norm.logpdf(e1_samples,tp['lens0_e1'],0.01)
        + norm.logpdf(e2_samples,tp['lens0_e2'],0.01)
        + norm.logpdf(gamma1_samples,tp['lens1_gamma1'],0.01)
        + norm.logpdf(gamma2_samples,tp['lens1_gamma2'],0.01)
    )
    
    log_integrand = np.log(p_gw) - log_prior
    
    log_I = logsumexp(log_integrand) - np.log(len(log_integrand))
    
    I = np.exp(log_I)

    print(f'-------------------Finished evaluating the integral -------------------------')

    result_dict =  {'index': idx,
                    'truths_em': truths_em, 
                    'logZ_em': logZ_em,
                    'I': I,
                    'logZ_em+log_I': (logZ_em + log_I)}
    
    #sample_truth_pairs.append(result)
    
    # 2. Append directly to the file immediately
    with open('progress_results.jsonl', 'a') as f:
        json_line = json.dumps(result_dict, cls=NumpyEncoder)
        f.write(json_line + '\n')

    print(f'----------------------------------------Finished {idx}-th iteration---------------------------------------------------')

# ==========================================================
#Load the results and plot the EM improvements
# ==========================================================

import json

# Initialize an empty list to store the recovered data
loaded_run = []

# Open and read the file line by line
with open('progress_results.jsonl', 'r') as f:
    for line in f:
        if line.strip():  # This skips any accidental empty lines
            loaded_run.append(json.loads(line))

print(f"Successfully loaded {len(loaded_run)} items from the file!")

EM_contributions = []
for i in range(len(loaded_run)):
    print(loaded_run[i]['I'])
    EM_contributions.append(loaded_run[i]['I'])


plt.hist(EM_contributions)
plt.show()
