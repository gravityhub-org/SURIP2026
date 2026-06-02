#####################################################################
# This script extends the data-loading function of FIGARO           #
# accepting also the .feather format used for the cogwheel samples. #
#####################################################################
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from figaro.exceptions import FIGAROException
from figaro.load import GW_par, supported_extensions, supported_waveforms, _prior_gw

supported_extensions += 'feather'


def load_feather_data(dir_path, seed=False, par=None, n_samples=-1,
                      cosmology='Planck18', volume=False, waveform='combined',
                      snr_threshold=None, far_threshold=None,
                      verbose=True, likelihood=False, keep_dVdz=False):
    '''
    Loads the data from .txt files (for simulations) or .h5/.hdf5/.dat files (posteriors from GWTC-x).
    Default cosmological parameters from Planck Collaboration (2021) in a flat Universe (https://www.aanda.org/articles/aa/pdf/2020/09/aa33910-18.pdf)
    Not all GW parameters are implemented: run figaro.load.available_gw_pars() for a list of available parameters.

    Arguments:
        str or Path path:     folder with data files
        bool seed:            fixes the seed to a default value (1) for reproducibility
        list-of-str par:      list with parameter(s) to extract from GW posteriors
        int n_samples:        number of samples for (random) downsampling. Default -1: all samples
        str cosmology:        set of cosmological parameters (Planck18 or Planck15)
        str waveform:         waveform family to be used ('combined', 'seob', 'imr')
        double snr_threhsold: SNR threshold for event filtering. For injection analysis only.
        double far_threshold: FAR threshold for event filtering. For injection analysis only.
        bool verbose:         show progress bar
        bool likelihood:      resample to get likelihood samples
        bool keep_dVdz:       do not remove dV/dz from the likelihood (only if the population model do NOT include it)

    Returns:
        np.ndarray: samples
        np.ndarray: names
    '''
    folder      = Path(dir_path).resolve()
    event_files = [Path(folder,f) for f in folder.glob('[!.]*')] # Ignores hidden files
    events      = []
    names       = []
    n_events    = len(event_files)
    removed_snr = False
    if volume:
        par = ['ra', 'dec', 'luminosity_distance']
    if n_events == 0:
        raise FIGAROException("Empty folder")
    # Check that a list of parameters is passed
    if par is None:
        raise TypeError("Please provide a list of parameter names you want to load (e.g. ['m1']).")
    # Check that all the parametes are loadable
    unknown_pars = set(par).difference(set(GW_par.keys()))
    if not unknown_pars == set():
        raise FIGAROException("The following parameters are not implemented: "+", ".join(unknown_pars)+". Run figaro.load.available_gw_pars() for a list of available parameters.")

    for event in tqdm(event_files, desc='Loading events', disable=not(verbose)):
        rdstate = np.random.RandomState(seed=1 if seed else None)
        ext = event.suffix[1:]

        # This function only accept `feather` as extension
        if ext != 'feather':
            if ext in supported_extensions:
                raise FIGAROException("Not supported extension for this function, recommend using original `load_data` function")
            else:
                raise TypeError(f"File {event.name} is not supported")

        # If everything is ok, load the samples
        sample_df = pd.read_feather(event)
        out = _unpack_cogwheel_posterior(
            sample_df, par=par, n_samples=n_samples, cosmology=cosmology,
            rdstate=rdstate, waveform=waveform,
            snr_threshold=snr_threshold, far_threshold=far_threshold,
            likelihood=likelihood, keep_dVdz=keep_dVdz)
        if out is not None:
            if out.shape[-1] == len(par):
                events.append(out)
                names.append(sample_df.attrs['event_name'])
            elif 'snr' in par:
                removed_snr = True

    if removed_snr:
        warnings.warn("At least one event does not have SNR samples. These events cannot be loaded for this parameter choices.")
    return (events, np.array(names))


def _unpack_cogwheel_posterior(
        dataframe, par, cosmology, rdstate, n_samples=-1, waveform='combined',
        snr_threshold=None, far_threshold=None, likelihood=False, keep_dVdz=False
    ):
        '''
        Reads data from .h5/.hdf5 GW posterior files.
        For GWTC-3 data release, it uses by default the Mixed posterior samples.
        Not all parameters are implemented: run figaro.load.available_gw_pars() for a list of available parameters.
        The waveform argument allows the user to select a waveform family. The default value, 'combined' uses samples from both imr and seob waveforms.
        For SEOB waveforms, the following waveform models are used (in descending priority order):
            * SEOBNRv4PHM
            * SEOBNRv4P
            * SEOBNRv4
        For IMR waveforms, in descending order:
            * IMRPhenomXPHM
            * IMRPhenomPv2
            * IMRPhenomPv3HM

        Arguments:
            str event:                 file to read
            list-of-str par:           parameter to extract
            str cosmology:             set of cosmological parameters to use.
            np.random.rdstate rdstate: state for random number generation
            int n_samples:             number of samples for (random) downsampling. Default -1: all samples
            str waveform:              waveform family to be used ('combined', 'imr', 'seob')
            double snr_threhsold:      SNR threshold for event filtering. For injection analysis only.
            double far_threshold:      FAR threshold for event filtering. For injection analysis only.
            bool likelihood:           resample to get likelihood samples
            bool keep_dVdz:       do not remove dV/dz from the likelihood (only if the population model do NOT include it)

        Returns:
            np.ndarray: samples
        '''
        if waveform not in supported_waveforms:
            raise FIGAROException("Unknown waveform: please use 'combined' (default), 'imr' or 'seob'")

        _far_thres_set = far_threshold is not None
        _snr_thres_set = snr_threshold is not None

        if _far_thres_set and _snr_thres_set:
            warnings.warn("Both FAR and SNR threshold provided. FAR will be used.")
            snr_threshold = None
            _snr_thres_set = False

        if _far_thres_set and ('far' not in par):
            par = np.append(par, 'far')
        elif _snr_thres_set and ('snr' not in par):
            par = np.append(par, 'snr')

        samples     = []
        loaded_pars = []
        flag_filter = False
        MDC_flag = True

        snr = np.array([])
        far = np.array([])

        for name, lab in GW_par.items():
            if name not in par:
                continue
            if name == 'far' and _far_thres_set:
                try:
                    far = np.array(dataframe[lab])
                    flag_filter = True
                except KeyError:
                    warnings.warn("FAR filter is not available with this data set.")
                else:
                    loaded_pars.append(name)
                    continue
            if name == 'snr':
                try:
                    if MDC_flag or waveform != 'combined':
                        snr = np.array(dataframe[lab])
                        samples.append(dataframe[lab])
                    if _snr_thres_set:
                        flag_filter = True
                except KeyError:
                    warnings.warn("SNR filter is not available with this data set.")
                else:
                    loaded_pars.append(name)
                    continue
            try:
                samples.append(dataframe[lab])
            except KeyError:
                if name == 's1':
                    samples.append(np.sqrt(dataframe['spin_1x']**2+dataframe['spin_1y']**2+dataframe['spin_1z']**2))
                elif name == 's2':
                    samples.append(np.sqrt(dataframe['spin_2x']**2+dataframe['spin_2y']**2+dataframe['spin_2z']**2))
                elif name == 'luminosity_distance':
                    samples.append(np.exp(dataframe['logdistance']))
                elif name == 'log_z':
                    samples.append(np.log(dataframe['redshift']))
                else:
                    raise FIGAROException(f"Cannot process parameter {name} from the data set.")
            else:
                loaded_pars.append(name)

        if len(par) == 1:
            samples = np.atleast_2d(samples).T
        else:
            par = np.array(par)
            loaded_pars = np.array(loaded_pars)
            samples_loaded = np.array(samples)
            samples = []
            for pi in par:
                if not (pi == 'far' or (pi == 'snr' and flag_filter)):
                    samples.append(samples_loaded[(loaded_pars == pi)].flatten())
            samples = np.array(samples)
            if flag_filter:
                if _far_thres_set:
                    samples = samples[:, (far < far_threshold) & (far > 0)]
                elif _snr_thres_set:
                    samples = samples[:, (snr > snr_threshold)]
            samples = samples.T

        # Resample to equal-weight in likelihood
        if likelihood:
            inv_prior = 1./_prior_gw(par, dataframe, cosmology, keep_dVdz=keep_dVdz)
            h         = np.random.uniform(0, 1, len(samples))
            samples   = samples[h < inv_prior / np.max(inv_prior)]

        # Resample to desire number of samples
        if n_samples > -1:
            ns = int(min([n_samples, len(samples)]))
            return samples[rdstate.choice(np.arange(len(samples)), size=ns, replace=False)]
        else:
            return samples
