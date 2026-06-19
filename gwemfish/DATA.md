# Dataset structure

## Layout

```
configs/
  batch_sim_config.yaml   # simulation settings
  priors.yaml             # prior spec (reference)

sims/
  batch_summary.json
  sim_{id:03d}/           # id = 0 … 4
```

---

## `configs/batch_sim_config.yaml`

- **`base`** — shared settings: lens model list, redshifts, EM grid/PSF/noise, lens light, GW noise scales
- **`simulations`** — list of 5 entries; each has `id`, `seed`, `source_pos`, `kwargs_lens`, `kwargs_source`

## `configs/priors.yaml`

- **`global`** — priors keyed by parameter name (`lens0_theta_E`, `source0_amp`, …)
- **`sims`** — optional per-id overrides (empty in provided file)

---

## `sims/batch_summary.json`

List of objects, one per simulation:

| Field | Type |
|-------|------|
| `sim_id` | int |
| `sim_dir` | str |
| `seed` | int |
| `source_pos` | `[x, y]` arcsec |
| `theta_E`, `kappa`, `e2`, `gamma` | float |
| `n_gw_images` | int (4) |

---

## `sims/sim_{id:03d}/`

| File | Format | Shape / structure |
|------|--------|-------------------|
| `em_image.npy` | float64 ndarray | `(20, 20)` — noisy image, counts/pixel |
| `gw_observables.json` | JSON | see below |
| `truth_params.json` | JSON | flat key→value map of all true parameters |
| `sim_params.json` | JSON | `{"base": {...}, "sim": {...}}` |
| `system_observation.png` | PNG | EM image + image positions |

### `em_image.npy`

- Shape: `(npix, npix)` = `(20, 20)`
- Pixel scale: `0.2` arcsec/pixel → 4×4 arcsec field, centred on `(0, 0)`
- Row 0 = bottom (`y` min); use `origin="lower"`, extent `[-2, 2, -2, 2]`

### `gw_observables.json`

| Key | Type | Length | Units |
|-----|------|--------|-------|
| `time_delays` | list of float | 3 | seconds (images 2–4 vs image 1) |
| `dL_eff` | list of float | 4 | Mpc |
| `image_x1` … `image_x4` | float | — | arcsec |
| `image_y1` … `image_y4` | float | — | arcsec |

### `truth_params.json`

Flat JSON. Main key groups:

| Prefix | Contents |
|--------|----------|
| `lens0_*` | EPL mass (6 params) |
| `lens1_*` | convergence sheet (3 params) |
| `source0_*` | source Sersic (7 params) |
| `light0_*` | lens Sersic light (7 params) |
| `image_x{i}`, `image_y{i}` | GW image positions, i=1…4 |
| `x_image_true_em`, `y_image_true_em` | lists, len 4 — EM image positions |
| `T_star`, `dL` | GW reference values |
| `noise_sigma_bkg` | float (= 0.01) |

Duplicate aliases exist without `0` suffix (e.g. `source_amp` = `source0_amp`).

### `sim_params.json`

```json
{
  "base": { ... },   // same structure as batch_sim_config.yaml → base
  "sim":  { ... }    // one entry from batch_sim_config.yaml → simulations
}
```

---

## Shared constants (all sims)

| Quantity | Value |
|----------|-------|
| `zl`, `zs` | 0.7, 1.5 |
| `npix` | 20 |
| `pix_scl` | 0.2 arcsec |
| `psf_fwhm` | 0.4 arcsec |
| `background_rms` | 0.01 |
| `exposure_time` | 2200 |
| Lens model | EPL + CONVERGENCE |
| Source model | 1× Sersic |
| Lens light | 1× Sersic (shared params in `base.em.kwargs_lens_light`) |
| GW images | 4 |
