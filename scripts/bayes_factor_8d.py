import pandas as pd
import numpy as np
from figaro.mixture import DPGMM
from scipy.special import logsumexp
from itertools import combinations
import os

params_1d = ['chirp_mass']
params_4d = ['chirp_mass', 'mass_ratio', 'ra', 'dec']
# 8D 包含: Mc, q, ra, dec, dL, cos_iota, psi, chi_eff
params_8d = ['chirp_mass', 'mass_ratio', 'ra', 'dec', 'luminosity_distance', 'cos_theta_jn', 'psi', 'chi_eff']

labels = [f"{i}_{j}" for i in range(10) for j in range(2)] 
pairs = list(combinations(labels, 2))
results = []

def get_union_info(df0, df1, col):
    v_min = min(df0[col].min(), df1[col].min())
    v_max = max(df0[col].max(), df1[col].max())
    v_range = v_max - v_min
    pad = v_range * 0.05 if v_range > 0 else 0.1
    return v_range, [v_min - pad, v_max + pad]

def prepare_8d_data(df):
    """計算 8D 分析所需的衍生參數"""
    df = df.copy()
    df['cos_theta_jn'] = np.cos(df['theta_jn'])
    df['chi_eff'] = (df['spin_1z'] + df['mass_ratio'] * df['spin_2z']) / (1 + df['mass_ratio'])
    return df

def calculate_bf_figaro(df0, df1, p_list, prior_dens):
    """使用 FIGARO 計算貝葉斯因子"""
    d0 = df0[p_list].values.astype(np.float64).copy()
    d1 = df1[p_list].values.astype(np.float64).copy()
    
    b_list = []
    for p in p_list:
        if p == 'ra': b_list.append([0.0, 2*np.pi])
        elif p == 'dec': b_list.append([-np.pi/2, np.pi/2])
        elif p == 'cos_theta_jn': b_list.append([-1.0, 1.0]) # cos(iota) 範圍
        elif p == 'psi': b_list.append([0.0, np.pi])
        elif p == 'chi_eff': b_list.append([-1.0, 1.0])    # chi_eff 範圍
        else:
            _, limits = get_union_info(df0, df1, p)
            b_list.append(limits)
    
    model_0 = DPGMM(bounds=np.array(b_list), prior_pars=None)
    model_1 = DPGMM(bounds=np.array(b_list), prior_pars=None)

    mix0 = model_0.density_from_samples(d0)
    mix1 = model_1.density_from_samples(d1)
    
    try:
        pts = mix0.rvs(5000)
    except AttributeError:
        pts = mix0.sample(5000)
    
    log_probs = None
    for method in ['log_prob', 'log_pdf', 'evaluate_log_pdf']:
        if hasattr(mix1, method):
            log_probs = getattr(mix1, method)(pts)
            break
            
    if log_probs is None:
        if hasattr(mix1, 'pdf'):
            log_probs = np.log(mix1.pdf(pts) + 1e-300)
        else:
            raise AttributeError("FIGARO mixture object 無法評估密度")
    
    log_e_overlap = logsumexp(log_probs) - np.log(len(log_probs))
    return (log_e_overlap - np.log(prior_dens)) / np.log(10)

print("Starting 8D Bayes factor calculations...")

for lab_0, lab_1 in pairs:
    try:
        f_0 = f"../data/processed_samples/MD_SDSS_l2_{lab_0}_bilby_style_samples.h5"
        f_1 = f"../data/processed_samples/MD_SDSS_l2_{lab_1}_bilby_style_samples.h5"
        
        if not (os.path.exists(f_0) and os.path.exists(f_1)):
            continue
        
        df_0 = prepare_8d_data(pd.read_hdf(f_0, key='df').dropna())
        df_1 = prepare_8d_data(pd.read_hdf(f_1, key='df').dropna())

        r_m, _ = get_union_info(df_0, df_1, 'chirp_mass')
        r_q, _ = get_union_info(df_0, df_1, 'mass_ratio')
        r_dL, _ = get_union_info(df_0, df_1, 'luminosity_distance')
        r_chi, _ = get_union_info(df_0, df_1, 'chi_eff')
        avg_cos = np.mean(np.cos(np.concatenate([df_0['dec'], df_1['dec']])))
        
        p_sky = avg_cos / (4 * np.pi)
        prior_1d = 1.0 / r_m
        prior_4d = prior_1d * (1.0 / r_q) * p_sky
        prior_8d = prior_4d * (1.0 / r_dL) * (1.0 / 2.0) * (1.0 / np.pi) * (1.0 / r_chi)

        log10_B_1d = calculate_bf_figaro(df_0, df_1, params_1d, prior_1d)
        log10_B_4d = calculate_bf_figaro(df_0, df_1, params_4d, prior_4d)
        log10_B_8d = calculate_bf_figaro(df_0, df_1, params_8d, prior_8d)

        print(f"✅ 對應: {lab_0}-{lab_1} | 1D: {log10_B_1d:.2f} | 4D: {log10_B_4d:.2f} | 8D: {log10_B_8d:.2f}")

        results.append({
            "Signal_0": lab_0, "Signal_1": lab_1,
            "logB (1d)": log10_B_1d,
            "logB (4d)": log10_B_4d,
            "logB (8d)": log10_B_8d
        })
        
    except Exception as e:
        print(f"❌ 處理 {lab_0} vs {lab_1} 時出錯: {e}")

if results:
    df_results = pd.DataFrame(results)
    os.makedirs("../plots", exist_ok=True)
    df_results.sort_values(by='logB (8d)', ascending=False).to_csv("../plots/bayes_factor_8d_results.csv", index=False)
    print(f"🎉 結果已儲存至 ../plots/bayes_factor_8d_results.csv")