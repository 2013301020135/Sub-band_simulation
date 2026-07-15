#!/usr/bin/env python
"""Run simulate_ToAs, do MCMC fit, and draw plots for a series of parameter sets in parallel.
    Written by Yang Liu (liuyang@shao.ac.cn)."""

import argparse
import os
import glob
import subprocess
import shutil
import time

import numpy as np
import pandas as pd
# import libstempo as lt
# import libstempo.toasim as ltt
import logging
import traceback
import sys
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
# from datetime import datetime

parser = argparse.ArgumentParser(description='Pipeline for observational strategy research.'
                                             'Do the following three things for a series test with a parameter varied.' 
                                             'Generate simulated ToAs for a sample of pulsars with simulate_ToAs.'
                                             'Run MCMC fit with enterprise (IPTA).'
                                             'Draw corner plots and chain plots.'
                                             'All functions are enabled for running in parallel.'
                                             'Other parameters are keep fixed in all tests.'
                                             'Parameters allowed to vary in different tests are:'
                                             'refgamma, rnc, rnamp, rngam, dmnc, dmnamp, dmngam, gwbnpts, gwbamp, gwbgam.'
                                             'Written by Yang Liu (liuyang@shao.ac.cn).')
# parser.add_argument('-p', '--parfile', type=str, default=[], nargs='+', help='Parameter files for pulsars used in simulation')
parser.add_argument('-d', '--datadir', type=str, default=None, help='Path to the directory containing the par files')
parser.add_argument('--cad', '--observation-cadence', type=float, default=14,
                    help='The number of days between observations')
parser.add_argument('--nobs', '--no-of-observation', type=int, default=1,
                    help='The number of observations on a given day')
parser.add_argument('--maxha', '--hour-angle', type=float, default=8, help='The maximum absolute hour angle allowed')
parser.add_argument('--rha', '--random-hour-angle', action='store_true',
                    help='Use random hour angle coverage if called, otherwise use regular hour angle')
parser.add_argument('--mjds', '--initial-mjd', type=int, default=50000, help='The initial MJD for the simulated TOAs')
parser.add_argument('--mjde', '--final-mjd', type=int, default=60000, help='The final MJD for the simulated TOAs')
parser.add_argument('--nuhfb', '--num-uhfband', type=int, default=0, help='Number of arrays in UHF-Band')
parser.add_argument('--nlb', '--num-lband', type=int, default=32, help='Number of arrays in L-Band')
parser.add_argument('--nsb', '--num-sband', type=int, default=32, help='Number of arrays in S-Band')
parser.add_argument('--nsbuhf', '--num-uhf-subband', type=int, default=8, help='Number of sub-bands in UHF-Band')
parser.add_argument('--nsbl', '--num-l-subband', type=int, default=8, help='Number of sub-bands in L-Band')
parser.add_argument('--nsbs', '--num-s-subband', type=int, default=8, help='Number of sub-bands in S-Band')
parser.add_argument('--cfrequhf', '--central-frequency-uhf', type=float, default=810,
                    help='Central frequency of UHF-Band in MHz')
parser.add_argument('--cfreql', '--central-frequency-l', type=float, default=1280,
                    help='Central frequency of L-Band in MHz')
parser.add_argument('--cfreqs', '--central-frequency-s', type=float, default=2600,
                    help='Central frequency of S-Band in MHz')
parser.add_argument('--bwuhf', '--bandwidth-uhf', type=float, default=500, help='Bandwidth of UHF-Band in MHz')
parser.add_argument('--bwl', '--bandwidth-l', type=float, default=800, help='Bandwidth of L-Band in MHz')
parser.add_argument('--bws', '--bandwidth-s', type=float, default=800, help='Bandwidth of S-Band in MHz')
parser.add_argument('--narray', '--num-array', type=int, default=64, help='Number of total arrays')
parser.add_argument('--randnum', '--random-number-seed', type=int, default=None, help='Specify random number seed')
parser.add_argument('--tel', type=str, default="meerkat", help='The name of the telescope')
parser.add_argument('--refsig', '--reference-sigma', type=float, default=1,
                    help='The rms of Gaussian noise in micro-second when all telescope are in reference frequency')
parser.add_argument('--reffreq', '--reference-frequency', type=float, default=1300,
                    help='The reference frequency in MHz')
parser.add_argument('--refflux', '--reference-flux', type=float, default=1,
                    help='The reference flux in micro-Jy at reference frequency')  # Add coefficient
parser.add_argument('--refgamma', '--reference-gamma', type=float, default=1.6, help='The reference spectral index')
parser.add_argument('--rn', '--red-noise', action='store_true',
                    help='Inject red noise if called')
parser.add_argument('--rnamp', '--red-noise-amplitude', type=float, default=1e-12,
                    help='The red noise amplitude')
parser.add_argument('--rngamma', '--red-noise-gamma', type=float, default=3,
                    help='The red noise spectral slope (gamma, positive)')
parser.add_argument('--rnc', '--red-noise-component', type=int, default=100,
                    help='The number of red noise component')
parser.add_argument('--rntspan', '--red-noise-tspan', type=float, default=None,
                    help='The time span used for red noise injection')
parser.add_argument('--dmn', '--dm-noise', action='store_true',
                    help='Inject DM noise if called')
parser.add_argument('--dmnamp', '--dm-noise-amplitude', type=float, default=1e-12,
                    help='The DM noise amplitude')
parser.add_argument('--dmngamma', '--dm-noise-gamma', type=float, default=3,
                    help='The DM noise spectral slope (gamma, positive)')
parser.add_argument('--dmnc', '--dm-noise-component', type=int, default=100,
                    help='The number of DM noise component')
parser.add_argument('--gwb', '--gw-background', action='store_true',
                    help='Inject gravitational wave background if called')
parser.add_argument('--gwbamp', '--gwb-amplitude', type=float, default=1e-14,
                    help='The gravitational wave background amplitude')
parser.add_argument('--gwbgam', '--gwb-gamma', type=float, default=4,
                    help='The gravitational wave background spectral slope (gamma, positive)')
parser.add_argument('--nocorr', '--gwb-no-corr', action='store_true', help='Add red noise with no correlation')
parser.add_argument('--lmax', '--gwb-lmax', type=int, default=0, help='The maximum multipole of GW power decomposition')
parser.add_argument('--turnover', '--gwb-turnover', action='store_true',
                    help='Produce spectrum with turnover at frequency f0')
parser.add_argument('--gwbf0', '--gwb-f0', type=float, default=1e-9,
                    help='The frequency of spectrum turnover')
parser.add_argument('--gwbbeta', '--gwb-beta', type=float, default=1,
                    help='The spectral index of power spectrum for f<<f0')
parser.add_argument('--gwbpower', '--gwb-power', type=float, default=1,
                    help='The fudge factor for flatness of spectrum turnover')
parser.add_argument('--gwbnpts', '--gwb-npts', type=int, default=600,
                    help='The number of points used in interpolation')
parser.add_argument('--gwbhowml', '--gwb-howml', type=float, default=10,
                    help='The lowest frequency is 1/(howml * T)')
parser.add_argument('--timd', '--tim-dir', type=str, default='tims/',
                    help='The relative path to the directory for output simulation tim files')
parser.add_argument('--resd', '--result-dir', type=str, default='results/',
                    help='The relative path to the directory for results')
parser.add_argument('--chaind', '--chains-dir', type=str, default='chains/',
                    help='The relative path to the directory for chains')
parser.add_argument('--noised', '--noises-dir', type=str, default='noisefiles/',
                    help='The relative path to the directory for noisefiles')
parser.add_argument('--testd', '--test-dir', type=str, default="test",
                    help='The directory for saving all files in a series test of a specific parameters')
parser.add_argument('--comtd', '--comment-dir', type=str, default="comment",
                    help='The subdirectory for saving all files in a series test')
parser.add_argument('--reald', '--realization-dir', type=str, default="",
                    help='The directory for saving all files in one run')
parser.add_argument('--vary', type=str, required=True,
                    choices=['refgamma', 'rnc', 'rnamp', 'rngam', 'dmnc', 'dmnamp', 'dmngam', 'gwbnpts', 'gwbamp', 'gwbgam'],
                    help='Specify which parameter to vary in this series test')
parser.add_argument('--values', '--vary-values', type=float, nargs='+', required=True,
                    help='Values for the varying parameter')
parser.add_argument('--workd', '--work-dir', type=str, default='/cluster/home/liuyang/Sub-Array/Sub-band_simulation/',
                    help='The absolute path of this simulation package')
parser.add_argument('--entd', '--enterprise-dir', type=str, default='/cluster/home/liuyang/Software/scripts/',
                    help='The absolute path of enterprise package')
parser.add_argument('--ncore', '--num-core', type=int, default=16,
                    help='The number of tasks to run in srun')
parser.add_argument('--maxobs', '--max-observation', type=int, default=100000,
                    help='The maximum number of observation allowed in MCMC fitting')
parser.add_argument('--samp', '--sampler', type=str, default='ptmcmc',
                    help='The sampler to use in MCMC fit, options are: ptmcmc, dynesty, mc3, nessai')
parser.add_argument('--niter', '--num-iteration', type=int, default=10000,
                    help='The number of iteration in MCMC fitting')
parser.add_argument('--rnnb', '--red-noise-num-bin', type=int, default=100,
                    help='The number of bins used to fit red noise')
parser.add_argument('--dmnnb', '--dm-noise-num-bin', type=int, default=100,
                    help='The number of bins used to fit DM noise')
parser.add_argument('--gwbnb', '--gwb-num-bin', type=int, default=100,
                    help='The number of bins used to fit gravitational wave background')
parser.add_argument('-b', '--burn', type=float, default=0.5, help='Fraction of chains to be burned')
parser.add_argument('--refit', '--rerun-fitting', type=str, default=None,
                    help='Rerun MCMC with existing simulation data in given test folder + old comment name')
# /cluster/home/liuyang/Sub-Array
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.FileHandler(f'{args.datadir}pipeline_{args.vary}_{args.reald}.log', mode='a'),
                              logging.StreamHandler()])

VARY_PARAM_MAP = {
    'refgamma': {'args_field': 'refgamma', 'type': float, 'sim_args_idx': 17, 'in_log': False, 'valid_range': (0., 2.)},
    'rnc': {'args_field': 'rnc', 'type': int, 'sim_args_idx': 26, 'in_log': False, 'valid_range': (10, 1000)},
    'rnamp': {'args_field': 'rnamp', 'type': float, 'sim_args_idx': 27, 'in_log': True, 'valid_range': (-18, -10)},
    'rngamma': {'args_field': 'rngamma', 'type': float, 'sim_args_idx': 28, 'in_log': False, 'valid_range': (0, 7)},
    'dmnc': {'args_field': 'dmnc', 'type': int, 'sim_args_idx': 29, 'in_log': False, 'valid_range': (10, 1000)},
    'dmnamp': {'args_field': 'dmnamp', 'type': float, 'sim_args_idx': 30, 'in_log': True, 'valid_range': (-18, -10)},
    'dmngamma': {'args_field': 'dmngamma', 'type': float, 'sim_args_idx': 31, 'in_log': False, 'valid_range': (0, 7)},
    'gwbnpts': {'args_field': 'gwbnpts', 'type': int, 'sim_args_idx': 32, 'in_log': False, 'valid_range': (10, 1000)},
    'gwbamp': {'args_field': 'gwbamp', 'type': float, 'sim_args_idx': 33, 'in_log': True, 'valid_range': (-18, -10)},
    'gwbgam': {'args_field': 'gwbgam', 'type': float, 'sim_args_idx': 34, 'in_log': False, 'valid_range': (0, 7)},
}


def describe_injection(rnamp, rngamma, dmnamp, dmngamma, gwbamp, gwbgam, rn=False, dmn=False, gwb=False):
    """Describe the injected signals with their amplitudes and spectral indices."""
    describe = ""
    if rn is True:
        describe += "_RN%A{}#G{}".format(int(np.log10(rnamp)), int(rngamma))
    if dmn is True:
        describe += "_DM%A{}#G{}".format(int(np.log10(dmnamp)), int(dmngamma))
    if gwb is True:
        describe += "_GWB%A{}#G{}".format(int(np.log10(gwbamp)), int(gwbgam))
    return describe


def validate_parameter(config, value):
    """Verify that the varying parameter's values are in its valid range."""
    if 'valid_range' in config:
        min_val, max_val = config['valid_range']
        if not (min_val <= value <= max_val):
            raise ValueError(f"The value {value} is out of the allowed range for parameter "
                             f"{config['args_field']} ({min_val} - {max_val}).")


def build_task_params(arg, vary_param, values):
    """Construct parameter list with validating its values."""
    params_list = []
    config = VARY_PARAM_MAP[vary_param]
    base_params = [arg.datadir, arg.workd, arg.timd, arg.resd, arg.chaind, arg.noised, arg.testd, arg.comtd, arg.reald,
                   arg.cad, arg.nobs, arg.maxha, arg.mjds, arg.mjde, arg.reffreq, arg.refsig, arg.refflux, arg.refgamma,
                   arg.tel, arg.narray, arg.nuhfb, arg.nlb, arg.nsb, arg.nsbuhf, arg.nsbl, arg.nsbs, arg.rnc,
                   arg.rnamp, arg.rngamma, arg.dmnc, arg.dmnamp, arg.dmngamma, arg.gwbnpts, arg.gwbamp, arg.gwbgam,
                   arg.rn, arg.dmn, arg.gwb, arg.rha, arg.randnum, arg.cfrequhf, arg.cfreql, arg.cfreqs,
                   arg.bwuhf, arg.bwl, arg.bws, arg.lmax, arg.gwbf0, arg.gwbbeta, arg.gwbpower, arg.gwbhowml,
                   arg.rntspan, arg.nocorr, arg.turnover]
    # The original parameter list used in generate_toa
    for val in values:
        # Convert type of parameters and validate its values
        try:
            typed_val = config['type'](val)
            validate_parameter(config, typed_val)
        except ValueError as e:
            logging.error(f"Parameters validation failed: {str(e)}")
            raise
        # Clone and modify parameter values from input values
        toa_params = list(base_params)
        toa_params[config['sim_args_idx']] = typed_val
        toa_params[6] = f"{arg.testd}_{vary_param}"  # Name the test directory with testd and the varying parameter
        describe = describe_injection(toa_params[27], toa_params[28], toa_params[30], toa_params[31], toa_params[33],
                                      toa_params[34], rn=toa_params[35], dmn=toa_params[36], gwb=toa_params[37])
        toa_params[7] = f"{arg.comtd}{describe}"  # Name the comment directory with comtd and the injected signals
        toa_params[8] = f"realization_{vary_param}%{typed_val}#{arg.reald}"
        # Name the realization directory with varying parameter and its value, plus the No. of realization
        params_list.append(toa_params)
        logging.info(f"Parameters created: {vary_param} = {typed_val}")
    return params_list


def monitored_parallel_execute(func, params, dscrb):
    """Monitor the execution of parallel tasks with tqdm progress bar."""
    results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(func, p): i for i, p in enumerate(params)}
        with tqdm(total=len(params), desc=dscrb) as pbar:
            for future in as_completed(futures):
                try:
                    res = future.result()
                    results.append((futures[future], res))
                except Exception as e:
                    logging.error(f"Task failed: {str(e)}")
                    raise
                finally:
                    pbar.update(1)
    # Order results in original order
    return [res for _, res in sorted(results, key=lambda x: x[0])]


def make_directories(datadir, testdir, comtdir, realdir):
    """Make necessary folders inside data directory."""
    extradir = ""
    if testdir is not None:
        # if not os.path.exists(datadir + extradir + testdir):
        #    os.makedirs(datadir + extradir + testdir)
        extradir = extradir + testdir + "/"
    if comtdir is not None:
        if not os.path.exists(datadir + extradir + comtdir):
            os.makedirs(datadir + extradir + comtdir)
        extradir = extradir + comtdir + "/"
    if realdir is not None:
        if not os.path.exists(datadir + extradir + realdir):
            os.makedirs(datadir + extradir + realdir)
        extradir = extradir + realdir + "/"
    return extradir


def copy_folder(src, dst):
    """Copy all contents in source folder to destination folder."""
    try:
        shutil.copytree(src, dst, dirs_exist_ok=True)
    except Exception as e:
        print(f"Folder copy failed: {str(e)}")


def subband_strategy(strtgy=None, uhf_ratio=1/3, l_ratio=1/3, s_ratio=1/3):
    """Find the corresponding subband settings and strategies."""
    if strtgy is None:
        return strtgy
    elif strtgy == "":
        return strtgy  # strtgy, uhf, l, s


def load_posteriors(posts):
    """Load the posteriors for Common Red Signals from posteriors.txt."""
    p = {}
    with open(posts, "r") as f:
        for line in f:
            if line.startswith("crs_log10_A"):
                e = line.split()
                p["GWB log10amp max-like"], p["GWB log10amp mean"], p["GWB log10amp std"] = e[1], e[2], e[3]
                p["GWB log10amp 03"], p["GWB log10amp 16"], p["GWB log10amp 50"] = e[4], e[5], e[6]
                p["GWB log10amp 84"], p["GWB log10amp 97"] = e[7], e[8]
            elif line.startswith("crs_gamma"):
                e = line.split()
                p["GWB gam max-like"], p["GWB gam mean"], p["GWB gam std"] = e[1], e[2], e[3]
                p["GWB gam 03"], p["GWB gam 16"], p["GWB gam 50"] = e[4], e[5], e[6]
                p["GWB gam 84"], p["GWB gam 97"] = e[7], e[8]
    return p


def extract_results(arg, paras):
    """Extract input and output parameters and information to a csv table with pandas dataframe."""
    entry = []
    source_idx = ["Pulsars list", "No. of sources", "Telescope", "No. of arrays", "Strategy"]
    observe_idx = ["MJD start", "MJD end", "Cadence", "No. of observation", "Max hour angle", "Random hour angle"]
    band_idx = ["UHF arrays", "UHF subbands", "UHF central freq", "UHF bandwidth",
                "L arrays", "L subbands", "L central freq", "L bandwidth",
                "S arrays", "S subbands", "S central freq", "S bandwidth"]
    reference_idx = ["Reference freq", "Reference sigma", "Reference flux", "Reference gamma"]
    noise_idx = ["Red Noise", "RN log10amp", "RN gam", "RN coefficient", "RN timespan",
                 "Dispersion Measure", "DM log10amp", "DM gam", "DM coefficient"]
    signal_idx = ["Gravitational Wave Background", "GWB log10amp", "GWB gam", "GWB points num",
                  "GWB no corr", "GWB lmax", "GWB turnover freq", "GWB beta", "GWB power", "GWB lowest freq"]
    fit_idx = ["RN bin num", "DM bin num", "GWB bin num", "No. of iter", "Burn fraction"]
    crs_idx = ["GWB log10amp max-like", "GWB log10amp mean", "GWB log10amp std",
               "GWB log10amp 03", "GWB log10amp 16", "GWB log10amp 50", "GWB log10amp 84", "GWB log10amp 97",
               "GWB gam max-like", "GWB gam mean", "GWB gam std",
               "GWB gam 03", "GWB gam 16", "GWB gam 50", "GWB gam 84", "GWB gam 97"]
    sum_idx = source_idx + observe_idx + band_idx + reference_idx + noise_idx + signal_idx + fit_idx + crs_idx
    fit = {"RN bin num": arg.rnnb, "DM bin num": arg.dmnnb, "GWB bin num": arg.gwbnb, 
           "No. of iter": arg.niter, "Burn fraction": arg.burn}
    for p in paras:
        s = {}
        pl = ""
        psrs = [pf.split("/")[-1].split(".", 1)[0] for pf in sorted(glob.glob(os.path.join(p[0], "*.par")))]
        for psrn in psrs:
            pl += f"{psrn} "
        s["Pulsars list"], s["No. of sources"], s["Telescope"], s["No. of arrays"] = pl, len(psrs), p[18], p[19]
        s["Strategy"] = subband_strategy(uhf_ratio=p[20]/p[19], l_ratio=p[21]/p[19], s_ratio=p[22]/p[19])
        s["MJD start"], s["MJD end"], s["Cadence"], s["No. of observation"] = p[12], p[13], p[9], p[10]
        s["Max hour angle"], s["Random hour angle"] = p[11], p[38]
        s["UHF arrays"], s["UHF subbands"], s["UHF central freq"], s["UHF bandwidth"] = p[20], p[23], p[40], p[43]
        s["L arrays"], s["L subbands"], s["L central freq"], s["L bandwidth"] = p[21], p[24], p[41], p[44]
        s["S arrays"], s["S subbands"], s["S central freq"], s["S bandwidth"] = p[22], p[25], p[42], p[45]
        s["Reference freq"], s["Reference sigma"], s["Reference flux"], s["Reference gamma"] = p[14:18]
        s["Red Noise"], s["Dispersion Measure"], s["Gravitational Wave Background"] = p[35], p[36], p[37]
        if p[35] is True:
            s["RN log10amp"], s["RN gam"], s["RN coefficient"], s["RN timespan"] = np.log10(p[27]), p[28], p[26], p[51]
        else:
            s["RN log10amp"], s["RN gam"], s["RN coefficient"], s["RN timespan"] = None, None, None, None
        if p[36] is True:
            s["DM log10amp"], s["DM gam"], s["DM coefficient"] = np.log10(p[30]), p[31], p[29]
        else:
            s["DM log10amp"], s["DM gam"], s["DM coefficient"] = None, None, None
        s["GWB turnover freq"], s["GWB beta"], s["GWB power"] = None, None, None
        if p[37] is True:
            s["GWB log10amp"], s["GWB gam"], s["GWB points num"] = np.log10(p[33]), p[34], p[32]
            s["GWB no corr"], s["GWB lmax"] = p[52], p[46]
            if p[53] is True:
                s["GWB turnover freq"], s["GWB beta"], s["GWB power"], s["GWB lowest freq"] = p[47], p[48], p[49], p[50]
        else:
            s["GWB log10amp"], s["GWB gam"], s["GWB points num"] = None, None, None
            s["GWB no corr"], s["GWB lmax"] = None, None
        # desc = describe_injection(p[27], p[28], p[30], p[31], p[33], p[34], rn=p[35], dmn=p[36], gwb=p[37])
        # extd = make_directories(p[0], p[6], p[7], p[8])
        posts = f"{p[0]}{p[6]}/{p[7]}/{p[8]}/{p[4]}/posterior.txt"
        sum_slt = {**s, **fit, **load_posteriors(posts)}
        series_slt = pd.Series(sum_slt, index=sum_idx)
        entry.append(series_slt)
    # fr_name = ["Pulsars list", "No. of sources"]
    # fr_id = pd.MultiIndex.from_tuples([(self.psrn, i + 1) for i in range(self.max_glitch)], names=fr_name)
    frame_slt = pd.DataFrame(entry)
    fnlrslt = "summary.csv"
    if os.path.exists(fnlrslt):
        logging.info("File exists: appending results")
        frame_slt.to_csv(fnlrslt, mode='a', index=False, header=False)
        logging.info("Pipeline results extraction success")
    else:
        logging.info("Create new files: writing results")
        frame_slt.to_csv(fnlrslt, mode='w', index=False, header=True)
        logging.info("Pipeline results extraction success")


def generate_toa_wrapper(params):
    """An wrapper for generate_toa."""
    return generate_toa(*params)


def mcmc_fit_wrapper(params):
    """An wrapper for mcmc_fit."""
    return mcmc_fit(*params)


def draw_plots_wrapper(params):
    """An wrapper for draw_plots."""
    return draw_plots(*params)


def generate_toa(datadir, workdir, timdir, resdir, chaindir, noisedir, testdir, comtdir, realdir,
                 obs_cad, no_obs, maxabs_ha, mjd_start, mjd_end, reffreq, refsig, refflux, refgam,
                 telescope, narray, num_uhfb, num_lb, num_sb, num_sbuhf, num_sbl, num_sbs,
                 rnc, rnamp, rngamma, dmnc, dmnamp, dmngamma, gwbnpts, gwbamp, gwbgam,
                 rn=False, dmn=False, gwb=False, randha=False, randnum=None,
                 cfrequhf=810, cfreql=1280, cfreqs=2600, bwuhf=500, bwl=800, bws=800,
                 lmax=0, gwbf0=1e-9, gwbbeta=1, gwbpower=1, gwbhowml=10, rntspan=None, nocorr=False, turnover=False):
    """Generate simulation ToAs in a realization with given settings and parameters."""
    start = time.time()
    command = ["python", workdir+"simulate_ToAs.py", "-d", datadir, "--timd", timdir, "--resd", resdir,
               "--chaind", chaindir, "--noised", noisedir, "--testd", testdir, "--comtd", comtdir, "--reald", realdir,
               "--cad", str(obs_cad), "--nobs", str(no_obs), "--mjds", str(mjd_start), "--mjde", str(mjd_end),
               "--maxha", str(maxabs_ha), "--tel", telescope, "--narray", str(narray),
               "--reffreq", str(reffreq), "--refsig", str(refsig), "--refflux", str(refflux), "--refgamma", str(refgam),
               "--nuhfb", str(num_uhfb), "--nlb", str(num_lb), "--nsb", str(num_sb),
               "--nsbuhf", str(num_sbuhf), "--nsbl", str(num_sbl), "--nsbs", str(num_sbs),
               "--cfrequhf", str(cfrequhf), "--cfreql", str(cfreql), "--cfreqs", str(cfreqs),
               "--bwuhf", str(bwuhf), "--bwl", str(bwl), "--bws", str(bws)]
    if randha is True:
        command.extend(["--rha"])
    if randnum is not None:
        command.extend(["--randnum", str(randnum)])
    if rn is True:
        command.extend(["--rn", "--rnc", str(rnc), "--rnamp", str(rnamp), "--rngamma", str(rngamma)])
        if rntspan is not None:
            command.extend(["--rntspan", str(rntspan)])
    if dmn is True:
        command.extend(["--dmn", "--dmnc", str(dmnc), "--dmnamp", str(dmnamp), "--dmngamma", str(dmngamma)])
    if gwb is True:
        command.extend(["--gwb", "--gwbnpts", str(gwbnpts), "--gwbamp", str(gwbamp), "--gwbgam", str(gwbgam)])
        command.extend(["--lmax", str(lmax), "--gwbf0", str(gwbf0), "--gwbbeta", str(gwbbeta),
                        "--gwbpower", str(gwbpower), "--gwbhowml", str(gwbhowml)])
        if nocorr is True:
            command.extend(["--nocorr"])
        if turnover is True:
            command.extend(["--turnover"])
    describe = describe_injection(rnamp, rngamma, dmnamp, dmngamma, gwbamp, gwbgam, rn=rn, dmn=dmn, gwb=gwb)
    cmd = ""
    for cmdi in command:
        cmd += str(cmdi) + " "
    extradir = make_directories(datadir, testdir, comtdir, realdir)
    with open(f"{datadir}{extradir}simulate_command.txt", 'w') as file:
        file.writelines(f'{cmd}\n')
    print("######")
    print("Start simulate ToA generation")
    print("Parameter files directory is:", datadir)
    print("Realization directory is:", extradir)
    print("### Simulation Command is:\n", cmd)
    print("######")
    with open(f'{datadir}{extradir}simulate_out.txt', 'w') as outfile:
        simulatetoa = subprocess.run(command, check=True, stdout=outfile, stderr=subprocess.STDOUT)
    logging.info(f"Task: {cmd} \nTime usage is: {time.time() - start:.2f}s")
    return simulatetoa.returncode, extradir, describe, resdir, chaindir, noisedir


def mcmc_fit(datadir, entdir, extradir, describe, resdir, chaindir, noisedir, numcore, maxobs, sampler, niter,
             rnnb=None, dmnnb=None, gwbnb=None):
    """Run MCMC fit to recover the injected signals for corresponding simulation ToAs"""
    start = time.time()
    # "nohup", "nice", "srun", "-n", str(numcore), "--exclusive", "--ntasks-per-node=16", "--mem=64G", "--partition=q3",
    command = ["python", entdir+"enterprise_bayesian_analysis.py",
               "-o", datadir+extradir+chaindir, "-d", datadir+extradir+resdir, "--noisedirs",
               datadir+extradir+chaindir+noisedir, "--maxobs", str(maxobs), "--sampler", sampler+",Niter="+str(niter)]
    model = "TM/WN,fix"
    if rnnb is not None:
        model += f"/RN,nb={rnnb}"
    if dmnnb is not None:
        model += f"/RN,idx=2,nb={dmnnb}"
    command.extend(["-m", model])
    if gwbnb is not None:
        command.extend(["-c", f"CRS,nb={gwbnb}"])
    # command.extend([str(2), ">", datadir+extradir+"chains/out.0", ">", datadir+extradir+"chains/err.0", "&"])
    cmd = ""
    for cmdi in command:
        cmd += str(cmdi) + " "
    with open(f"{datadir}{extradir}mcmc_command.txt", 'w') as file:
        file.writelines(f'{cmd}\n')
    print("######")
    print("Start MCMC fitting")
    print("Parameter files directory is:", datadir)
    print("Realization directory is:", extradir)
    print("### MCMC Command is:\n", cmd)
    print("######")
    with open(f'{datadir}{extradir}mcmc_out.txt', 'w') as outfile:
        mcmcfit = subprocess.run(command, check=True, stdout=outfile, stderr=subprocess.STDOUT)
    logging.info(f"Task: {cmd} \nTime usage is: {time.time() - start:.2f}s")
    return mcmcfit.returncode, extradir, describe, chaindir


def draw_plots(datadir, workdir, entdir, extradir, describe, chaindir, burn):
    """Draw corner plot and chain histogram plot for corresponding MCMC fits"""
    start = time.time()
    comm_co = ["python", entdir+"/post_sampling_analysis/plot_corner.py", "-d", datadir+extradir+chaindir]
    if 'GWB' in describe:
        comm_co.extend(["-o", datadir+extradir+chaindir+f"/corner_crs{describe}.pdf", "-p", "crs_log10_A", "crs_gamma"])
    else:
        comm_co.extend(["-o", datadir+extradir+chaindir+f"/corner{describe}.pdf"])
    comm_ch = ["python", entdir+"/post_sampling_analysis/plot_chain_hist.py", "-d", datadir+extradir+chaindir,
               "-o", datadir+extradir+chaindir+f"/chain{describe}.pdf"]
    comm_po = ["python", workdir+"/posteriors.py", "-c", datadir+extradir+chaindir+"/chain_1.txt", "-b", str(burn)]
    cmd_corner, cmd_chain, cmd_posterior = "", "", ""
    for cmdi, cmdj, cmdk in zip(comm_co, comm_ch, comm_po):
        cmd_corner += str(cmdi) + " "
        cmd_chain += str(cmdj) + " "
        cmd_posterior += str(cmdk) + " "
    with open(f"{datadir}{extradir}plot_commands.txt", 'w') as file:
        file.writelines(f'{cmd_corner}\n')
        file.writelines(f'{cmd_chain}\n')
        file.writelines(f'{cmd_posterior}\n')
    plot_corner = subprocess.run(comm_co, check=True)
    plot_chain = subprocess.run(comm_ch, check=True)
    posterior = subprocess.run(comm_po, check=True)
    logging.info(f"Task: {cmd_corner} \n, task: {cmd_chain} \n, and task: {cmd_posterior} \n, "
                 f"Time usage is: {time.time() - start:.2f}s")
    return plot_corner.returncode, plot_chain.returncode, posterior.returncode, extradir, describe


def execute_pipeline(arg):
    """Execute standard pipeline with reinforced systematic log"""
    logging.info("===== Begin pipeline execution =====")
    logging.info(f"Varying parameter: {args.vary}")
    logging.info(f"Values for varying parameter: {args.values}")
    try:
        if arg.refit is None:
            # Step 1: ToA simulation
            logging.info("===Begin simulated ToA generation===")
            toa_params = build_task_params(arg, arg.vary, arg.values)
            logging.info(f"Successfully constructed parameters lists for {len(toa_params)} tasks")
            # Construct parameters lists and record
            generate_results = monitored_parallel_execute(generate_toa_wrapper, toa_params, "ToA progress")
            # Check results
            success_generate = [res for res in generate_results if res[0] == 0]
            if len(success_generate) != len(toa_params):
                lost = len(toa_params) - len(success_generate)
                logging.error(f"Number of failed tasks in simulated ToA generation: {lost}")
                raise RuntimeError("Exist failed tasks in simulated ToA generation!")
        else:
            logging.info("===Skip ToA generation===")
            generate_results = []
            toa_params = build_task_params(arg, arg.vary, arg.values)
            for tp in toa_params:
                desc = describe_injection(tp[27], tp[28], tp[30], tp[31], tp[33], tp[34],
                                          rn=tp[35], dmn=tp[36], gwb=tp[37])
                extd = make_directories(tp[0], tp[6], tp[7], tp[8])
                if not os.path.exists(tp[0] + extd + tp[4]):
                    os.makedirs(tp[0] + extd + tp[4])
                if os.path.relpath(arg.refit + desc + f"/{tp[8]}/" + tp[3]) != os.path.relpath(tp[0] + extd + tp[3]):
                    copy_folder(arg.refit + desc + f"/{tp[8]}/" + tp[3], tp[0] + extd + tp[3])
                    copy_folder(arg.refit + desc + f"/{tp[8]}/" + tp[4] + tp[5], tp[0] + extd + tp[4] + tp[5])
                    shutil.copy(arg.refit + desc + f"/{tp[8]}/paras_info.txt", tp[0] + extd + "/paras_info.txt")
                    shutil.copy(arg.refit + desc + f"/{tp[8]}/simulate_command.txt",
                                tp[0] + extd + "/simulate_command.txt")
                    shutil.copy(arg.refit + desc + f"/{tp[8]}/simulate_out.txt", tp[0] + extd + "/simulate_out.txt")
                generate_results.append([0, extd, desc, tp[3], tp[4], tp[5]])
            success_generate = [res for res in generate_results if res[0] == 0]
            logging.info(f"Successfully loaded simulation ToAs for {len(toa_params)} tasks")
            if len(success_generate) != len(toa_params):
                lost = len(toa_params) - len(success_generate)
                logging.error(f"Number of failed tasks in reload simulated ToA: {lost}")
                raise RuntimeError("Exist failed tasks in reload simulated ToA!")
        # Step 2: MCMC fit
        logging.info("===Begin MCMC fitting===")
        mcmc_params = [(arg.datadir, arg.entd, extd, desc, res, cha, noi, arg.ncore, arg.maxobs, arg.samp, arg.niter,
                        arg.rnnb, arg.dmnnb, arg.gwbnb) for _, extd, desc, res, cha, noi in success_generate]
        logging.info(f"Successfully constructed parameters lists for {len(mcmc_params)} tasks")
        # The original input parameters for MCMC
        mcmc_results = monitored_parallel_execute(mcmc_fit_wrapper, mcmc_params, "MCMC progress")
        # Check results
        success_mcmc = [res for res in mcmc_results if res[0] == 0]
        if len(success_mcmc) != len(mcmc_params):
            lost = len(mcmc_params) - len(success_mcmc)
            logging.error(f"Number of failed tasks in MCMC fitting: {lost}")
            raise RuntimeError("Exist failed tasks in MCMC fitting!")
        # Step 3: Draw plots
        logging.info("===Begin figures plotting===")
        plot_params = [(arg.datadir, arg.workd, arg.entd, extd, desc, chain, arg.burn)
                       for _, extd, desc, chain in success_mcmc]
        logging.info(f"Successfully constructed parameters lists for {len(plot_params)} tasks")
        # The original input parameters for plotting
        plot_results = monitored_parallel_execute(draw_plots_wrapper, plot_params, "Plot progress")
        # Check results
        success_plot = [res for res in plot_results if (res[0] == 0 and res[1] == 0 and res[2] == 0)]
        if len(success_plot) != len(plot_params):
            lost = len(plot_params) - len(success_plot)
            logging.error(f"Number of failed tasks in figures plotting: {lost}")
            raise RuntimeError("Exist failed tasks in figures plotting!")
        extract_results(arg, toa_params)
        # Final check
        logging.info("===== Pipeline execution complete =====")
        return {'generate': generate_results, 'mcmc': mcmc_results, 'plot': plot_results}
    except Exception as e:
        logging.error(f"Pipeline execution error: {str(e)}")
        logging.error(traceback.format_exc())
        raise


if not args.rn:
    args.rnnb = None
if not args.dmn:
    args.dmnnb = None
if not args.gwb:
    args.gwbnb = None

try:
    # Make the test directory for this test
    if not os.path.exists(args.datadir + f"{args.testd}_{args.vary}"):
        os.makedirs(args.datadir + f"{args.testd}_{args.vary}")
    all_results = execute_pipeline(args)
    # Final state report
    logging.info("Execution results summary:")
    logging.info(f"TOA simulation: {sum(1 for c in all_results['generate'] if c[0] == 0)}"
                 f" in {len(all_results['generate'])} success")
    logging.info(f"MCMC fit: {sum(1 for c in all_results['mcmc'] if c[0] == 0)}"
                 f" in {len(all_results['mcmc'])} success")
    logging.info(f"Draw plots: {sum(1 for c in all_results['plot'] if (c[0] == 0 and c[1] == 0))}"
                 f" in {len(all_results['plot'])} success")
except Exception as err:
    logging.error(f"Pipeline execution exit: {str(err)}")
    sys.exit(1)


'''
    # Preprocess parameters
    # if not hasattr(args, 'refgamma'):
    #     args.refgamma = [1.6]  # ???
    # args.refgamma = [args.refgamma] if not isinstance(args.refgamma, list) else args.refgamma
'''