import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from itertools import combinations

labels = [f"{i}_{j}" for i in range(9) for j in range(2)] 
pairs = list(combinations(labels, 2))
results = []

for lab_0, lab_1 in pairs:
    f_0 = f"../data/processed_samples/MD_SDSS_l2_{lab_0}_bilby_style_samples.h5"
    df_0 = pd.read_hdf(f_0, key='df')
    d_0 = np.vstack([df_0['chirp_mass'], df_0['ra'], df_0['dec']])
    m_0 = df_0['chirp_mass']
    kde_m_0 = gaussian_kde(m_0)
    kde_0 = gaussian_kde(d_0)

    f_1 = f"../data/processed_samples/MD_SDSS_l2_{lab_1}_bilby_style_samples.h5"
    df_1 = pd.read_hdf(f_1, key='df')
    d_1 = np.vstack([df_1['chirp_mass'], df_1['ra'], df_1['dec']])
    m_1 = df_1['chirp_mass']
    kde_m_1 = gaussian_kde(m_1)
    kde_1 = gaussian_kde(d_1)

    m_grid = np.linspace(min(df_0['chirp_mass'].min(), df_1['chirp_mass'].min()), max(df_0['chirp_mass'].max(), df_1['chirp_mass'].max()), 100)
    pdf_m_0 = kde_m_0.evaluate(m_grid)
    pdf_m_1 = kde_m_1.evaluate(m_grid)
    prior_m = 1 / (max(df_0['chirp_mass'].max(), df_1['chirp_mass'].max()) - min(df_0['chirp_mass'].min(), df_1['chirp_mass'].min()))   #need to change
    B_m = np.trapezoid(pdf_m_0 * pdf_m_1 / prior_m, m_grid)

    M_chirp_grid = np.linspace(min(df_0['chirp_mass'].min(), df_1['chirp_mass'].min()), max(df_0['chirp_mass'].max(), df_1['chirp_mass'].max()), 40)
    RA_grid = np.linspace(min(df_0['ra'].min(), df_1['ra'].min()), max(df_0['ra'].max(), df_1['ra'].max()), 40)
    DEC_grid = np.linspace(min(df_0['dec'].min(), df_1['dec'].min()), max(df_0['dec'].max(), df_1['dec'].max()), 40)
    M_chirp, RA, DEC = np.meshgrid(M_chirp_grid, RA_grid, DEC_grid, indexing='ij')
    grid = np.vstack([M_chirp.ravel(), RA.ravel(), DEC.ravel()])
    pdf_0 = kde_0.evaluate(grid).reshape(M_chirp.shape)
    pdf_1 = kde_1.evaluate(grid).reshape(M_chirp.shape)
    prior_sky = np.cos(DEC) / (4 * np.pi)
    prior_3d = prior_m * prior_sky
    B = np.trapezoid(np.trapezoid(np.trapezoid(pdf_0 * pdf_1 / prior_3d, DEC_grid, axis=2), RA_grid, axis=1), M_chirp_grid, axis=0)

    print(f"Processing: {lab_0} vs {lab_1}")

    results.append({
        "Pair_0": lab_0,
        "Pair_1": lab_1,
        "log_B_m": np.log10(B_m),
        "log_B": np.log10(B)
    })

df_results = pd.DataFrame(results)
df_clean = df_results[np.isfinite(df_results['log_B'])]
df_sorted = df_clean.sort_values(by='log_B', ascending=False)
df_sorted.to_csv("../plots/bayes_factor_results.csv", index=False)
print(f"Cleaned results saved to ../plots/bayes_factor_results.csv")
print(df_sorted.to_string(index=False))