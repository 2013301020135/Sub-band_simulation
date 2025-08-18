#!/usr/bin/env python
"""Run simulate_ToAs, do MCMC fit, and draw plots for a series of parameter sets in parallel.
    Written by Yang Liu (liuyang@shao.ac.cn)."""

import argparse
import copy
import os
# import glob
import subprocess
import shutil
import time
import math
import logging
import traceback
import sys
# import inspect
import numpy as np
import pandas as pd
# import libstempo as lt
# import libstempo.toasim as ltt
import simulate_ToAs as sT
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
# from datetime import datetime


def validate_parameter(config, value):
    """Verify that the varying parameter's values are in its valid range.

    :param config: the configuration dictionary of parameter
    :param value: the value for the parameter
    """
    if 'valid_range' in config:
        min_val, max_val = config['valid_range']
        if not (min_val <= value <= max_val):
            raise ValueError(f"The value {value} is out of the allowed range for parameter "
                             f"{config['args_field']} ({min_val} - {max_val}).")


def build_task_params(**kwargs):
    """Construct parameter list with validating its values.

    :return: params_list: the list containing the parameter dictionaries for all realizations
    """
    params_list = []
    config = VARY_PARAM_MAP[kwargs['vary']]
    # The original parameter list used in generate_toa
    for val in kwargs['values']:
        # Convert type of parameters and validate its values
        try:
            typed_val = config['type'](val)
            validate_parameter(config, typed_val)
        except ValueError as e:
            logging.error(f"Parameters validation failed: {str(e)}")
            raise
        # Clone and modify parameter values from input values
        toa_params = copy.deepcopy(kwargs)
        describe = sT.describe_name(**toa_params)
        toa_params[config['args_field']] = typed_val
        if config['in_log']:
            toa_params[config['args_field']] = np.power(10, typed_val)
        if toa_params['rlzno'] is not None:
            toa_params['reald'] = str(toa_params['rlzno'])
        toa_params['testd'] = f"{kwargs['testd']}_{kwargs['vary']}"
        # Name the test directory with testd and the varying parameter
        toa_params['comtd'] = f"{kwargs['comtd']}{describe}"
        # Name the comment directory with comtd and the injected signals
        toa_params['reald'] = f"realization_{kwargs['vary']}%{typed_val}#{kwargs['reald']}"
        # Name the realization directory with varying parameter and its value, plus the No. of realization
        sT.make_directories(toa_params['datadir'], toa_params['testd'], toa_params['comtd'], toa_params['reald'], True)
        params_list.append(toa_params)
        logging.info(f"Parameters created: {kwargs['vary']} = {typed_val}")
    return params_list


def monitored_parallel_execute(func, params, dscrb):
    """Monitor the execution of parallel tasks with tqdm progress bar.

    :param func: the function to run in parallel
    :param params: the list containing the parameter dictionaries for all realizations
    :param dscrb: the description for the progress bar
    """
    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(func, **p): i for i, p in enumerate(params)}
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


def copy_folder(src, dst):
    """Copy all contents in source folder to destination folder.

    :param src: the source folder to copy
    :param dst: the destination folder to paste
    """
    try:
        shutil.copytree(src, dst, dirs_exist_ok=True)
    except Exception as e:
        print(f"Folder copy failed: {str(e)}")


def load_posteriors(posts):
    """Load the posteriors for Common Red Signals from posteriors.txt.

    :param posts: the text files containing the posteriors

    :return: p: the dictionary containing the posteriors
    """
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


def extract_results(paras, **kwargs):
    """Extract input and output parameters and information to a csv table with pandas dataframe.

    :param paras: the list containing the parameter dictionaries for all realizations
    """
    entry = []
    source_idx = ['Pulsars list', 'No. of sources', 'Telescope', 'No. of arrays', 'Strategy', 'No. of realization']
    observe_idx = ['MJD start', 'MJD end', 'Cadence', 'No. of observation', 'Max hour angle', 'Random hour angle']
    band_idx = ['UHF ratios', 'UHF arrays', 'UHF subbands', 'UHF central freq', 'UHF bandwidth',
                'L ratios', 'L arrays', 'L subbands', 'L central freq', 'L bandwidth',
                'S ratios', 'S arrays', 'S subbands', 'S central freq', 'S bandwidth']
    reference_idx = ['Reference freq', 'Reference sigma', 'Reference flux', 'Reference gamma']
    noise_idx = ['WN from par', 'RN from par', 'Red Noise', 'RN log10amp', 'RN gam', 'RN coefficient', 'RN timespan',
                 'Dispersion Measure', 'DM log10amp', 'DM gam', 'DM coefficient']
    signal_idx = ['Gravitational Wave Background', 'GWB log10amp', 'GWB gam', 'GWB points num',
                  'GWB no corr', 'GWB lmax', 'GWB turnover freq', 'GWB beta', 'GWB power', 'GWB lowest freq']
    fit_idx = ['RN bin num', 'DM bin num', 'GWB bin num', 'No. of iter', 'Burn fraction']
    crs_idx = ['GWB log10amp max-like', 'GWB log10amp mean', 'GWB log10amp std',
               'GWB log10amp 03', 'GWB log10amp 16', 'GWB log10amp 50', 'GWB log10amp 84', 'GWB log10amp 97',
               'GWB gam max-like', 'GWB gam mean', 'GWB gam std',
               'GWB gam 03', 'GWB gam 16', 'GWB gam 50', 'GWB gam 84', 'GWB gam 97']
    sum_idx = source_idx + observe_idx + band_idx + reference_idx + noise_idx + signal_idx + fit_idx + crs_idx
    fit = {'RN bin num': kwargs['rnnb'], 'DM bin num': kwargs['dmnnb'], 'GWB bin num': kwargs['gwbnb'],
           'No. of iter': kwargs['niter'], 'Burn fraction': kwargs['burn']}
    for p in paras:
        s = {}
        p = sT.subband_strategy(**p)
        datad, pfiles, psrs = sT.psr_list(p['datadir'])
        s['Pulsars list'] = " ".join(str(psrn) for psrn in psrs)
        s['No. of sources'], s['Telescope'], s['No. of arrays'] = len(psrs), p['tel'], p['narray']
        s['MJD start'], s['MJD end'], s['Cadence'], s['No. of observation'] = p['mjds'], p['mjde'], p['cad'], p['nobs']
        s['Max hour angle'], s['Random hour angle'] = p['maxha'], p['rha']
        s['Strategy'], s['No. of realization'] = p['strategy'], p['rlzno']
        s['WN from par'], s['RN from par'] = p['wfp'], p['nfp']
        s['UHF ratios'], s['L ratios'], s['S ratios'] = p['ruhfb'], p['rlb'], p['rsb']
        s['UHF arrays'], s['L arrays'], s['S arrays'] = p['nuhfb'], p['nlb'], p['nsb']
        s['UHF subbands'], s['UHF central freq'], s['UHF bandwidth'] = p['nsbuhf'], p['cfrequhf'], p['bwuhf']
        s['L subbands'], s['L central freq'], s['L bandwidth'] = p['nsbl'], p['cfreql'], p['bwl']
        s['S subbands'], s['S central freq'], s['S bandwidth'] = p['nsbs'], p['cfreqs'], p['bws']
        s['Reference freq'], s['Reference sigma'] = p['reffreq'], p['refsig']
        s['Reference flux'], s['Reference gamma'] = p['refflux'], p['refgamma']
        s['Red Noise'], s['Dispersion Measure'], s['Gravitational Wave Background'] = p['rn'], p['dmn'], p['gwb']
        if p['nfp'] or (p['rn'] is False):
            s['RN log10amp'], s['RN gam'], s['RN coefficient'], s['RN timespan'] = None, None, None, None
        else:
            s['RN log10amp'], s['RN gam'] = np.log10(p['rnamp']), p['rngamma']
            s['RN coefficient'], s['RN timespan'] = p['rnc'], p['rntspan']
        if p['nfp'] or (p['dmn'] is False):
            s['DM log10amp'], s['DM gam'], s['DM coefficient'] = None, None, None
        else:
            s['DM log10amp'], s['DM gam'], s['DM coefficient'] = np.log10(p['dmnamp']), p['dmngamma'], p['dmnc']
        s['GWB turnover freq'], s['GWB beta'], s['GWB power'] = None, None, None
        if p['gwb']:
            s['GWB log10amp'], s['GWB gam'], s['GWB points num'] = np.log10(p['gwbamp']), p['gwbgamma'], p['gwbnpts']
            s['GWB no corr'], s['GWB lmax'], s['GWB lowest freq'] = p['nocorr'], p['lmax'], p['gwbhowml']
            if p['turnover']:
                s['GWB turnover freq'], s['GWB beta'], s['GWB power'] = p['gwbf0'], p['gwbbeta'], p['gwbpower']
        else:
            s['GWB log10amp'], s['GWB gam'], s['GWB points num'] = None, None, None
            s['GWB no corr'], s['GWB lmax'], s['GWB lowest freq'] = None, None, None
        extd = sT.make_directories(p['datadir'], p['testd'], p['comtd'], p['reald'])
        posts = f"{p['datadir']}{extd}{p['chaind']}/posterior.txt"
        sum_slt = {**s, **fit, **load_posteriors(posts)}
        series_slt = pd.Series(sum_slt, index=sum_idx)
        entry.append(series_slt)
    # fr_name = ['Pulsars list', 'No. of sources']
    # fr_id = pd.MultiIndex.from_tuples([(self.psrn, i + 1) for i in range(self.max_glitch)], names=fr_name)
    frame_slt = pd.DataFrame(entry)
    fnlrslt = kwargs['sumnam'] + ".csv"
    if os.path.exists(fnlrslt):
        logging.info("File exists: appending results")
        frame_slt.to_csv(fnlrslt, mode='a', index=False, header=False)
        logging.info("Pipeline results extraction success")
    else:
        logging.info("Create new files: writing results")
        frame_slt.to_csv(fnlrslt, mode='w', index=False, header=True)
        logging.info("Pipeline results extraction success")


def generate_toa(**kwargs):
    """Generate simulation ToAs in a realization with given settings and parameters.

    :return: simulatetoa.returncode: the status of the task
    """
    start = time.time()
    extradir = sT.make_directories(kwargs['datadir'], kwargs['testd'], kwargs['comtd'], kwargs['reald'], make=True)
    command = ["python", kwargs['workd']+"simulate_ToAs.py", "-d", kwargs['datadir'], "--timd", kwargs['timd'],
               "--resd", kwargs['resd'], "--chaind", kwargs['chaind'], "--noised", kwargs['noised'],
               "--testd", kwargs['testd'], "--comtd", kwargs['comtd'], "--reald", kwargs['reald'],
               "--tel", kwargs['tel'], "--narray", str(kwargs['narray']), "--cad", str(kwargs['cad']),
               "--nobs", str(kwargs['nobs']), "--mjds", str(kwargs['mjds']), "--mjde", str(kwargs['mjde']),
               "--maxha", str(kwargs['maxha']), "--reffreq", str(kwargs['reffreq']), "--refsig", str(kwargs['refsig']),
               "--refflux", str(kwargs['refflux']), "--refgamma", str(kwargs['refgamma']),
               "--ruhfb", str(kwargs['ruhfb']), "--rlb", str(kwargs['rlb']), "--rsb", str(kwargs['rsb']),
               "--nuhfb", str(kwargs['nuhfb']), "--nlb", str(kwargs['nlb']), "--nsb", str(kwargs['nsb']),
               "--nsbuhf", str(kwargs['nsbuhf']), "--nsbl", str(kwargs['nsbl']), "--nsbs", str(kwargs['nsbs']),
               "--cfrequhf", str(kwargs['cfrequhf']), "--cfreql", str(kwargs['cfreql']),
               "--cfreqs", str(kwargs['cfreqs']), "--bwuhf", str(kwargs['bwuhf']),
               "--bwl", str(kwargs['bwl']), "--bws", str(kwargs['bws'])]
    if kwargs['rlzno'] is not None:
        command.extend(["--rlzno", str(kwargs['rlzno'])])
    if kwargs['rha']:
        command.extend(["--rha"])
    if kwargs['randnum'] is not None:
        command.extend(["--randnum", str(kwargs['randnum'])])
    if kwargs['wfp']:
        command.extend(["--wfp"])
    if kwargs['nfp']:
        command.extend(["--nfp"])
    if kwargs['rn']:
        command.extend(["--rn", "--rnamp", str(kwargs['rnamp']), "--rngamma", str(kwargs['rngamma']),
                        "--rnc", str(kwargs['rnc'])])
        if kwargs['rntspan'] is not None:
            command.extend(["--rntspan", str(kwargs['rntspan'])])
    if kwargs['dmn']:
        command.extend(["--dmn", "--dmnamp", str(kwargs['dmnamp']), "--dmngamma", str(kwargs['dmngamma']),
                        "--dmnc", str(kwargs['dmnc'])])
    if kwargs['gwb']:
        command.extend(["--gwb", "--gwbamp", str(kwargs['gwbamp']), "--gwbgamma", str(kwargs['gwbgamma']),
                        "--gwbnpts", str(kwargs['gwbnpts'])])
        command.extend(["--lmax", str(kwargs['lmax']), "--gwbhowml", str(kwargs['gwbhowml'])])
        if kwargs['nocorr']:
            command.extend(["--nocorr"])
        if kwargs['turnover']:
            command.extend(["--turnover", "--gwbf0", str(kwargs['gwbf0']), "--gwbbeta", str(kwargs['gwbbeta']),
                            "--gwbpower", str(kwargs['gwbpower'])])
    cmd = ""
    for cmdi in command:
        cmd += str(cmdi) + " "
    with open(f"{kwargs['datadir']}{extradir}simulate_command.txt", 'w') as file:
        file.writelines(f"{cmd}\n")
    print("######")
    print("Start simulate ToA generation")
    print("Parameter files directory is:", kwargs['datadir'])
    print("Realization directory is:", extradir)
    print("### Simulation Command is:\n", cmd)
    print("######")
    with open(f"{kwargs['datadir']}{extradir}simulate_out.txt", 'w') as outfile:
        simulatetoa = subprocess.run(command, check=True, stdout=outfile, stderr=subprocess.STDOUT)
    logging.info(f"Task: {cmd} \nTime usage is: {time.time() - start:.2f}s")
    return simulatetoa.returncode


def mcmc_fit(**kwargs):
    """Run MCMC fit to recover the injected signals for corresponding simulation ToAs.

    :return: mcmcfit.returncode: the status of the task
    """
    start = time.time()
    extradir = sT.make_directories(kwargs['datadir'], kwargs['testd'], kwargs['comtd'], kwargs['reald'])
    # "nohup", "nice", "srun", "-n", str(numcore), "--exclusive", "--ntasks-per-node=16", "--mem=64G", "--partition=q3",
    command = ["python", kwargs['entd']+"enterprise_bayesian_analysis.py", "-o",
               kwargs['datadir']+extradir+kwargs['chaind'], "-d", kwargs['datadir']+extradir+kwargs['resd'],
               "--noisedirs", kwargs['datadir']+extradir+kwargs['chaind']+kwargs['noised'], "--maxobs",
               str(kwargs['maxobs']), "--sampler", kwargs['samp']+",Niter="+str(kwargs['niter'])]
    model = "TM/WN,fix"
    if kwargs['rnnb'] is not None:
        model += f"/RN,nb={kwargs['rnnb']}"
    if kwargs['dmnnb'] is not None:
        model += f"/RN,idx=2,nb={kwargs['dmnnb']}"
    command.extend(["-m", model])
    if kwargs['gwbnb'] is not None:
        command.extend(["-c", f"CRS,nb={kwargs['gwbnb']}"])
    # command.extend([str(2), ">", datadir+extradir+"chains/out.0", ">", datadir+extradir+"chains/err.0", "&"])
    cmd = ""
    for cmdi in command:
        cmd += str(cmdi) + " "
    with open(f"{kwargs['datadir']}{extradir}mcmc_command.txt", 'w') as file:
        file.writelines(f'{cmd}\n')
    print("######")
    print("Start MCMC fitting")
    print("Parameter files directory is:", kwargs['datadir'])
    print("Realization directory is:", extradir)
    print("### MCMC Command is:\n", cmd)
    print("######")
    with open(f"{kwargs['datadir']}{extradir}mcmc_out.txt", 'w') as outfile:
        mcmcfit = subprocess.run(command, check=True, stdout=outfile, stderr=subprocess.STDOUT)
    logging.info(f"Task: {cmd} \nTime usage is: {time.time() - start:.2f}s")
    return mcmcfit.returncode


def draw_plots(**kwargs):
    """Draw corner plot and chain histogram plot for corresponding MCMC fits.

    :return: plot_corner.returncode: the status of plot corner task
    :return: plot_chain.returncode: the status of plot chain task
    :return: posterior.returncode: the status of posterior task
    """
    start = time.time()
    extradir = sT.make_directories(kwargs['datadir'], kwargs['testd'], kwargs['comtd'], kwargs['reald'])
    describe = sT.describe_name(**kwargs)
    comm_co = ["python", kwargs['entd']+"/post_sampling_analysis/plot_corner.py", "--burn", str(kwargs['burn']),
               "-d", kwargs['datadir']+extradir+kwargs['chaind']]
    if 'GWB' in describe:
        comm_co.extend(["-o", kwargs['datadir']+extradir+kwargs['chaind']+f"/corner_crs{describe}.pdf",
                        "-p", "crs_log10_A", "crs_gamma"])
    else:
        comm_co.extend(["-o", kwargs['datadir']+extradir+kwargs['chaind']+f"/corner{describe}.pdf"])
    comm_ch = ["python", kwargs['entd']+"/post_sampling_analysis/plot_chain_hist.py", "--plot_chains",
               "--burn", str(kwargs['burn']), "-d", kwargs['datadir']+extradir+kwargs['chaind'],
               "-o", kwargs['datadir']+extradir+kwargs['chaind']+f"/chain{describe}.pdf"]
    comm_po = ["python", kwargs['workd']+"/posteriors.py",
               "-c", kwargs['datadir']+extradir+kwargs['chaind']+"/chain_1.txt", "-b", str(kwargs['burn'])]
    cmd_corner, cmd_chain, cmd_posterior = "", "", ""
    for cmdi in comm_co:
        cmd_corner += str(cmdi) + " "
    for cmdj in comm_ch:
        cmd_chain += str(cmdj) + " "
    for cmdk in comm_po:
        cmd_posterior += str(cmdk) + " "
    with open(f"{kwargs['datadir']}{extradir}plot_commands.txt", 'w') as file:
        file.writelines(f'{cmd_corner}\n')
        file.writelines(f'{cmd_chain}\n')
        file.writelines(f'{cmd_posterior}\n')
    plot_corner = subprocess.run(comm_co, check=True)
    plot_chain = subprocess.run(comm_ch, check=True)
    posterior = subprocess.run(comm_po, check=True)
    logging.info(f"Task: {cmd_corner} \n, task: {cmd_chain} \n, and task: {cmd_posterior} \n, "
                 f"Time usage is: {time.time() - start:.2f}s")
    return plot_corner.returncode, plot_chain.returncode, posterior.returncode


def execute_pipeline(**kwargs):
    """Execute standard pipeline with reinforced systematic log."""
    logging.info("===== Begin pipeline execution =====")
    logging.info(f"Varying parameter: {kwargs['vary']}")
    logging.info(f"Values for varying parameter: {kwargs['values']}")
    try:
        if kwargs['refit'] is None:
            # Step 1: ToA simulation
            logging.info("===Begin simulated ToA generation===")
            toa_params = build_task_params(**kwargs)
            logging.info(f"Successfully constructed parameters lists for {len(toa_params)} tasks")
            # Construct parameters lists and record
            generate_results = monitored_parallel_execute(generate_toa, toa_params, "ToA progress")
            # Check results
            success_generate = [res for res in generate_results if res == 0]
            if len(success_generate) != len(toa_params):
                lost = len(toa_params) - len(success_generate)
                logging.error(f"Number of failed tasks in simulated ToA generation: {lost}")
                raise RuntimeError("Exist failed tasks in simulated ToA generation!")
        else:
            logging.info("===Skip ToA generation===")
            generate_results = []
            toa_params = build_task_params(**kwargs)
            for tp in toa_params:
                desc = sT.describe_name(**tp)
                extd = sT.make_directories(tp['datadir'], tp['testd'], tp['comtd'], tp['reald'])
                sT.check_directory(tp['datadir']+extd+tp['chaind'])
                if os.path.relpath(kwargs['refit']+desc+f"/{tp['reald']}/") != os.path.relpath(tp['datadir']+extd):
                    copy_folder(kwargs['refit'] + desc + f"/{tp['reald']}/" + tp['resd'],
                                tp['datadir'] + extd + tp['resd'])
                    copy_folder(kwargs['refit'] + desc + f"/{tp['reald']}/" + tp['chaind'] + tp['noised'],
                                tp['datadir'] + extd + tp['chaind'] + tp['noised'])
                    shutil.copy(kwargs['refit'] + desc + f"/{tp['reald']}/paras_info.txt",
                                tp['datadir'] + extd + "/paras_info.txt")
                    shutil.copy(kwargs['refit'] + desc + f"/{tp['reald']}/simulate_command.txt",
                                tp['datadir'] + extd + "/simulate_command.txt")
                    shutil.copy(kwargs['refit'] + desc + f"/{tp['reald']}/simulate_out.txt",
                                tp['datadir'] + extd + "/simulate_out.txt")
                generate_results.append(0)
            success_generate = [res for res in generate_results if res == 0]
            logging.info(f"Successfully loaded simulation ToAs for {len(toa_params)} tasks")
            if len(success_generate) != len(toa_params):
                lost = len(toa_params) - len(success_generate)
                logging.error(f"Number of failed tasks in reload simulated ToA: {lost}")
                raise RuntimeError("Exist failed tasks in reload simulated ToA!")
        # Step 2: MCMC fit
        logging.info("===Begin MCMC fitting===")
        mcmc_params = toa_params[:len(success_generate)]
        logging.info(f"Successfully constructed parameters lists for {len(mcmc_params)} tasks")
        # The original input parameters for MCMC
        mcmc_results = monitored_parallel_execute(mcmc_fit, mcmc_params, "MCMC progress")
        # Check results
        success_mcmc = [res for res in mcmc_results if res == 0]
        if len(success_mcmc) != len(mcmc_params):
            lost = len(mcmc_params) - len(success_mcmc)
            logging.error(f"Number of failed tasks in MCMC fitting: {lost}")
            raise RuntimeError("Exist failed tasks in MCMC fitting!")
        # Step 3: Draw plots
        logging.info("===Begin figures plotting===")
        plot_params = mcmc_params[:len(success_mcmc)]
        logging.info(f"Successfully constructed parameters lists for {len(plot_params)} tasks")
        # The original input parameters for plotting
        plot_results = monitored_parallel_execute(draw_plots, plot_params, "Plot progress")
        # Check results
        success_plot = [res for res in plot_results if (res[0] == res[1] == res[2] == 0)]
        if len(success_plot) != len(plot_params):
            lost = len(plot_params) - len(success_plot)
            logging.error(f"Number of failed tasks in figures plotting: {lost}")
            raise RuntimeError("Exist failed tasks in figures plotting!")
        extract_results(toa_params, **kwargs)
        # Final check
        logging.info("===== Pipeline execution complete =====")
        return {'generate': generate_results, 'mcmc': mcmc_results, 'plot': plot_results}
    except Exception as e:
        logging.error(f"Pipeline execution error: {str(e)}")
        logging.error(traceback.format_exc())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline for observational strategy research.'
                                                 'Do the following things for a series test with a parameter varied.'
                                                 'Generate simulated ToAs for a sample of pulsars with simulate_ToAs.'
                                                 'Run MCMC fit with enterprise (IPTA).'
                                                 'Draw corner plots, chain plots, and extract results.'
                                                 'All functions are enabled for running in parallel.'
                                                 'Other parameters are keep fixed in all tests.'
                                                 'Parameters allowed to vary in different tests are:'
                                                 'refgamma, rnc, rnamp, rngam, dmnc, dmnamp, dmngam,'
                                                 'gwbnpts, gwbamp, gwbgam.'
                                                 'Written by Yang Liu (liuyang@shao.ac.cn).')
    parser.add_argument('-p', '--parfile', type=str, default=[], nargs='+',
                        help='Parameter files for pulsars used in simulation')
    parser.add_argument('-d', '--datadir', type=str, default=None,
                        help='Path to the directory containing the par files')
    parser.add_argument('--cad', '--observation-cadence', type=float, default=14,
                        help='The number of days between observations')
    parser.add_argument('--nobs', '--no-of-observation', type=int, default=1,
                        help='The number of observations on a given day')
    parser.add_argument('--maxha', '--hour-angle', type=float, default=8,
                        help='The maximum absolute hour angle allowed')
    parser.add_argument('--rha', '--random-hour-angle', action='store_true',
                        help='Use random hour angle coverage if called, otherwise use regular hour angle')
    parser.add_argument('--mjds', '--initial-mjd', type=int, default=50000,
                        help='The initial MJD for the simulated TOAs')
    parser.add_argument('--mjde', '--final-mjd', type=int, default=60000, help='The final MJD for the simulated TOAs')
    parser.add_argument('--ruhfb', '--ratio-uhfband', type=float, default=None, help='Ratio of arrays in UHF-Band')
    parser.add_argument('--rlb', '--ratio-lband', type=float, default=None, help='Ratio of arrays in L-Band')
    parser.add_argument('--rsb', '--ratio-sband', type=float, default=None, help='Ratio of arrays in S-Band')
    parser.add_argument('--nuhfb', '--num-uhfband', type=int, default=16, help='Number of arrays in UHF-Band')
    parser.add_argument('--nlb', '--num-lband', type=int, default=16, help='Number of arrays in L-Band')
    parser.add_argument('--nsb', '--num-sband', type=int, default=16, help='Number of arrays in S-Band')
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
    parser.add_argument('--tel', type=str, default='meerkat', help='The name of the telescope')
    parser.add_argument('--refsig', '--reference-sigma', type=float, default=1,
                        help='The rms of Gaussian noise in micro-second when all telescope are in reference frequency')
    parser.add_argument('--reffreq', '--reference-frequency', type=float, default=1300,
                        help='The reference frequency in MHz')
    parser.add_argument('--refflux', '--reference-flux', type=float, default=1,
                        help='The reference flux in micro-Jy at reference frequency')  # Add coefficient
    parser.add_argument('--refgamma', '--reference-gamma', type=float, default=1.6, help='The reference spectral index')
    parser.add_argument('--wfp', '--white-from-par', action='store_true',
                        help='Use the white noise parameters in par file for writing json noise files if called')
    parser.add_argument('--nfp', '--noise-from-par', action='store_true',
                        help='Use the parameters in par file for injecting red noise and DM noise if called')
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
    parser.add_argument('--gwbgamma', '--gwb-gamma', type=float, default=4,
                        help='The gravitational wave background spectral slope (gamma, positive)')
    parser.add_argument('--nocorr', '--gwb-no-corr', action='store_true', help='Add red noise with no correlation')
    parser.add_argument('--lmax', '--gwb-lmax', type=int, default=0,
                        help='The maximum multipole of GW power decomposition')
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
    parser.add_argument('--testd', '--test-dir', type=str, default='test',
                        help='The directory for saving all files in a series test of a specific parameters')
    parser.add_argument('--comtd', '--comment-dir', type=str, default='comment',
                        help='The subdirectory for saving all files in a series test')
    parser.add_argument('--reald', '--realization-dir', type=str, default='',
                        help='The directory for saving all files in one run')
    parser.add_argument('--rlzno', '--realization-number', type=int, default=None,
                        help='The No. of realization for a given setting')
    parser.add_argument('--vary', type=str, required=True,
                        choices=['rlzno', 'refsig', 'refgamma', 'rnamp', 'rngamma', 'rnc', 'dmnamp', 'dmngamma', 'dmnc',
                                 'gwbamp', 'gwbgamma', 'gwbnpts'],
                        help='Specify which parameter to vary in this series test')
    parser.add_argument('--values', '--vary-values', type=float, nargs='+', required=True,
                        help='Values for the varying parameter')
    parser.add_argument('--workd', '--work-dir', type=str,
                        default='/cluster/home/liuyang/Sub-Array/Sub-band_simulation/',
                        help='The absolute path of this simulation package')
    parser.add_argument('--entd', '--enterprise-dir', type=str, default='/cluster/home/liuyang/Software/scripts/',
                        help='The absolute path of enterprise package')
    parser.add_argument('--ncore', '--num-core', type=int, default=16,
                        help='The number of tasks to run in srun')
    parser.add_argument('--maxobs', '--max-observation', type=int, default=100000,
                        help='The maximum number of observation allowed in MCMC fitting')
    parser.add_argument('--samp', '--sampler', type=str, default='ptmcmc',
                        choices=['ptmcmc', 'dynesty', 'mc3', 'nessai'], help='The sampler to use in MCMC fit')
    parser.add_argument('--niter', '--num-iteration', type=int, default=100000,
                        help='The number of iteration in MCMC fitting')
    parser.add_argument('--rnnb', '--red-noise-num-bin', type=int, default=100,
                        help='The number of bins used to fit red noise')
    parser.add_argument('--dmnnb', '--dm-noise-num-bin', type=int, default=100,
                        help='The number of bins used to fit DM noise')
    parser.add_argument('--gwbnb', '--gwb-num-bin', type=int, default=100,
                        help='The number of bins used to fit gravitational wave background')
    parser.add_argument('-b', '--burn', type=float, default=0.3, help='Fraction of chains to be burned')
    parser.add_argument('--refit', '--rerun-fitting', type=str, default=None,
                        help='Rerun MCMC with existing simulation data in given test folder + old comment name')
    parser.add_argument('--strategy', '--observing-strategy', type=str, default=None, help='The subband strategy used')
    parser.add_argument('--sumnam', '--summary-name', type=str, default='summary', help='The name for the csv table')
    # /cluster/home/liuyang/Sub-Array
    args = parser.parse_args()
    args_keys = ['parfile', 'datadir', 'timd', 'resd', 'chaind', 'noised', 'testd', 'comtd', 'reald', 'maxha', 'rha',
                 'randnum', 'cad', 'nobs', 'mjds', 'mjde', 'tel', 'narray', 'reffreq', 'refsig', 'refflux', 'refgamma',
                 'ruhfb', 'rlb', 'rsb', 'nuhfb', 'nlb', 'nsb', 'nsbuhf', 'nsbl', 'nsbs', 'cfrequhf', 'cfreql', 'cfreqs',
                 'bwuhf', 'bwl', 'bws', 'wfp', 'nfp', 'rn', 'rnamp', 'rngamma', 'rnc', 'rntspan', 'rnnb',
                 'dmn', 'dmnamp', 'dmngamma', 'dmnc', 'dmnnb', 'gwb', 'gwbamp', 'gwbgamma', 'gwbnpts', 'gwbnb', 'lmax',
                 'turnover', 'gwbf0', 'gwbbeta', 'gwbpower', 'gwbhowml', 'nocorr', 'vary', 'values', 'workd', 'entd',
                 'ncore', 'maxobs', 'samp', 'niter', 'burn', 'refit', 'rlzno', 'strategy', 'sumnam']
    kw_args = {key: getattr(args, key) for key in args_keys if hasattr(args, key)}
    datadir, par_files, psrnlist = sT.psr_list(datad=kw_args['datadir'], pf=kw_args['parfile'])
    kw_args['datadir'], kw_args['parfiles'], kw_args['psrnlist'] = datadir, par_files, psrnlist
    if args.rlzno is not None:
        kw_args['reald'] = str(args.rlzno)
    if not args.rn:
        kw_args['rnnb'] = None
    if not args.dmn:
        kw_args['dmnnb'] = None
    if not args.gwb:
        kw_args['gwbnb'] = None
    kw_args = sT.subband_strategy(**kw_args)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(f"{kw_args['datadir']}pipeline_{kw_args['vary']}_"
                                                      f"{kw_args['reald']}.log", mode='a'), logging.StreamHandler()])

    VARY_PARAM_MAP = {
        'rlzno': {'args_field': 'rlzno', 'type': int, 'in_log': False, 'valid_range': (1, 10)},
        'refsig': {'args_field': 'refsig', 'type': float, 'in_log': False, 'valid_range': (1e-4, 10)},
        'refgamma': {'args_field': 'refgamma', 'type': float, 'in_log': False, 'valid_range': (-5., 5.)},
        'rnamp': {'args_field': 'rnamp', 'type': float, 'in_log': True, 'valid_range': (-18, -10)},
        'rngamma': {'args_field': 'rngamma', 'type': float, 'in_log': False, 'valid_range': (0, 7)},
        'rnc': {'args_field': 'rnc', 'type': int, 'in_log': False, 'valid_range': (10, 1000)},
        'dmnamp': {'args_field': 'dmnamp', 'type': float, 'in_log': True, 'valid_range': (-18, -10)},
        'dmngamma': {'args_field': 'dmngamma', 'type': float, 'in_log': False, 'valid_range': (0, 7)},
        'dmnc': {'args_field': 'dmnc', 'type': int, 'in_log': False, 'valid_range': (10, 1000)},
        'gwbamp': {'args_field': 'gwbamp', 'type': float, 'in_log': True, 'valid_range': (-18, -10)},
        'gwbgamma': {'args_field': 'gwbgamma', 'type': float, 'in_log': False, 'valid_range': (0, 7)},
        'gwbnpts': {'args_field': 'gwbnpts', 'type': int, 'in_log': False, 'valid_range': (10, 1000)},
    }

    try:
        # Make the test directory for this test
        sT.check_directory(kw_args['datadir'] + f"{kw_args['testd']}_{kw_args['vary']}")
        all_results = execute_pipeline(**kw_args)
        # Final state report
        logging.info("Execution results summary:")
        logging.info(f"TOA simulation: {sum(1 for c in all_results['generate'] if c == 0)}"
                     f" in {len(all_results['generate'])} success")
        logging.info(f"MCMC fit: {sum(1 for c in all_results['mcmc'] if c == 0)}"
                     f" in {len(all_results['mcmc'])} success")
        logging.info(f"Draw plots: {sum(1 for c in all_results['plot'] if (c[0] == c[1] == c[2] == 0))}"
                     f" in {len(all_results['plot'])} success")
    except Exception as err:
        logging.error(f"Pipeline execution exit: {str(err)}")
        sys.exit(1)
