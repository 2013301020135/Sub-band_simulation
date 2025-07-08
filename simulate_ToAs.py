#!/usr/bin/env python
"""ToA simulate wrapper for generating simulated ToAs with tempo2 fake plugin and injecting RN, DM, GWB with libstempo.
    Written by Yang Liu (liuyang@shao.ac.cn)."""

import argparse
import os
import glob
import subprocess
import shutil
import psutil
# import inspect
import numpy as np
import libstempo as lt
import libstempo.toasim as ltt


class Band:
    def __init__(self, freq, bw, ratio_tel, num_tel, num_sub):
        """Initialize the Band class with given central frequency and bandwidth,
        ratios of arrays observed at this band, and number of sub-bands in this band."""
        self.freq = freq
        self.bandwidth = bw
        self.ratio_tel = ratio_tel
        self.num_tel = num_tel
        self.num_sub = num_sub
        self.subbw = self.bandwidth/self.num_sub
        self.subfreq = np.linspace(self.freq-0.5*(self.bandwidth-self.subbw),
                                   self.freq+0.5*(self.bandwidth-self.subbw), self.num_sub)
        self.rms = None

    def calculate_rms(self, reffreq=1300, refsig=1, refgamma=1.6):
        """Calculate the corresponding rms of ToA at given frequency based on reference rms at reference frequency.
        Default spectral index is 1.6."""
        rms = refsig/self.ratio_tel*np.sqrt(self.num_sub) * (self.subfreq/reffreq)**refgamma
        self.rms = rms
        return self.rms


def psrn_from_parfile(pf):
    """Get the name of pulsar from the parameter file. 
    
    :param pf: the parameter file of pulsar

    :return: psrn: the name of pulsar
    """
    psrname = pf.split("/")[-1].split(".", 1)[0]
    return psrname


def check_directory(path, make=True):
    """Check if directory exists and create if doesn't.

    :param path: the directory to check
    :param make: create directories if True

    :return: path: the directory made
    """
    if make is True and not os.path.exists(path):
        os.makedirs(path)
    return path


def make_directories(datad, testdir, comtdir, realdir, make=False):
    """Make necessary folders inside data directory. 
    
    :param datad: the directory containing the par files
    :param testdir: the directory for saving all files in a series test of a specific parameters
    :param comtdir: the subdirectory for saving all files in a series test
    :param realdir: the subdirectory for saving all files in one run
    :param make: create directories if True

    :return: extrad: the name of the directories (testdir/comtdir/realdir)
    """
    extrad = ""
    if testdir is not None:
        check_directory(datad + extrad + testdir, make)
        extrad = extrad + testdir + "/"
    if comtdir is not None:
        check_directory(datad + extrad + comtdir, make)
        extrad = extrad + comtdir + "/"
    if realdir is not None:
        check_directory(datad + extrad + realdir, make)
        extrad = extrad + realdir + "/"
    return extrad


def tempo2_fake_simulate(pf, rmssub, subfreq, subbw, **kwargs):
    """Simulate ToA for given pulsar based on its par file.
    Input: cadence, number of observation each time, regular/random hour angle, the maximum absolute hour angle allowed,
    simulation start date in MJD, simulation end date in MJD, observation telescope, rms in ToA,
    observation frequency, bandwidth, and random number seed. 

    :param pf: the parameter file of pulsar for ToA simulation
    :param rmssub: the root-mean-square of ToA at the specific subband
    :param subfreq: the central frequency of the specific subband to simulate ToA at
    :param subbw: the bandwidth of the specific subband to simulate ToA at
    """
    extrad = make_directories(kwargs['datadir'], kwargs['testd'], kwargs['comtd'], kwargs['reald'])
    outdir = f"{kwargs['datadir']}/{extrad}{psrn_from_parfile(pf)}"
    command = ["tempo2", "-gr", "fake", "-f", pf, "-ndobs", str(kwargs['cad']), "-nobsd", str(kwargs['nobs']),
               "-ha", str(kwargs['maxha']), "-randha", kwargs['rha'], "-tel", kwargs['tel'], "-withpn", "-setref",
               "-start", str(kwargs['mjds']), "-end", str(kwargs['mjde']), "-rms", str(1e-3 * rmssub),
               "-freq", str(subfreq), "-bw", str(subbw), "-o", str(outdir+".simulate")]
    if kwargs['randnum'] is not None:
        command.extend(["-idum", str(kwargs['randnum'])])
    subprocess.run(command, check=True)


def psr_list(datad=None, pf=None):
    """Find the list of pulsars parameter files from given directory or file. 
    
    :param datad: the directory containing the par files
    :param pf: the parameter file of pulsar for ToA simulation
    
    :return: datad: the directory containing the par files
    :return: pf: the name of parameter files
    :return: psrns: the name of pulsars
    """
    if datad is not None:
        pf = sorted(glob.glob(os.path.join(datad, "*.par")))
    elif len(pf) > 0:
        datad = pf.rsplit("/", 1)[0] + "/"
    psrns = [psrn_from_parfile(p) for p in pf]
    return datad, pf, psrns


def describe_name(**kwargs):
    """The name to describe the simulation data based on the signal injected. 
    Format: %RN #Amplitude #Gamma #C + %DM #Amplitude #Gamma #C + %GWB #Amplitude #Gamma #Npts . 
    
    :return: desc: the name describing the injected noises and signals
    """
    desc = '_'
    if kwargs['rn']:
        desc += f"%RN#A{int(np.log10(kwargs['rnamp']))}#G{int(kwargs['rngamma'])}#C{int(kwargs['rnc'])}"
    if kwargs['dmn']:
        desc += f"%DM#A{int(np.log10(kwargs['dmnamp']))}#G{int(kwargs['dmngamma'])}#C{int(kwargs['dmnc'])}"
    if kwargs['gwb']:
        desc += f"%GWB#A{int(np.log10(kwargs['gwbamp']))}#G{int(kwargs['gwbgamma'])}#N{int(kwargs['gwbnpts'])}"
    return '' if desc == '_' else desc


def save_paras(**kwargs):
    """Export the directory and parameter values used in this simulation to a txt file for recording. 
    
    :return: infotxt: the txt file with all the information for this simulation
    """
    datad, pfs, psrns = psr_list(datad=kwargs['datadir'], pf=kwargs['parfile'])
    extrad = make_directories(datad, kwargs['testd'], kwargs['comtd'], kwargs['reald'], make=False)
    infotxt = f"{datad}/{extrad}/paras_info.txt"
    with open(infotxt, "w") as f:
        f.writelines("Parameter information:\n")
        f.writelines(f"Relative path to the directory containing the par files: {kwargs['datadir']}\n")
        f.writelines(f"{len(psrns)} pulsars are used in simulation: \n")
        f.writelines(f"{pf} " for pf in psrns)
        f.writelines("\n")
        f.writelines(f"Directory for output simulation tim files: {kwargs['timd']}\n")
        f.writelines(f"Directory for result par and tim files: {kwargs['resd']}\n")
        f.writelines(f"Directory for MCMC chains: {kwargs['chaind']}\n")
        f.writelines(f"Directory for noise files: {kwargs['noised']}\n")
        f.writelines(f"Directory for test of a specific parameters: {kwargs['testd']}/\n")
        f.writelines(f"Directory for a test series: {kwargs['comtd']}/\n")
        f.writelines(f"Directory for a realization of given value: {kwargs['reald']}/\n")
        f.writelines(f"No. of realization of given value: {kwargs['rlzno']}\n")
        f.writelines(f"MJD range: {kwargs['mjds']} - {kwargs['mjde']}\n")
        f.writelines(f"Observation cadence: {kwargs['cad']} days, {kwargs['nobs']} observations each time\n")
        f.writelines(f"Telescope: {kwargs['tel']}, {kwargs['narray']} arrays in total\n")
        f.writelines(f"Strategy: {kwargs['strategy']}\n")
        f.writelines(f"UHF Band: central frequency - {kwargs['cfrequhf']} MHz, bandwidth - {kwargs['bwuhf']} MHz, "
                     f"Ratio: {kwargs['ruhfb']} - {kwargs['nuhfb']} arrays, {kwargs['nsbuhf']} sub-bands\n")
        f.writelines(f"L Band: central frequency - {kwargs['cfreql']} MHz, bandwidth - {kwargs['bwl']} MHz, "
                     f"Ratio: {kwargs['rlb']} - {kwargs['nlb']} arrays, {kwargs['nsbl']} sub-bands\n")
        f.writelines(f"S Band: central frequency - {kwargs['cfreqs']} MHz, bandwidth - {kwargs['bws']} MHz, "
                     f"Ratio: {kwargs['rsb']} - {kwargs['nsb']} arrays, {kwargs['nsbs']} sub-bands\n")
        f.writelines(f"Reference: frequency - {kwargs['reffreq']} MHz, sigma - {kwargs['refsig']} μs, "
                     f"flux - {kwargs['refflux']} μJy, spectral index - {kwargs['refgamma']}\n")
        f.writelines(f"Maximum absolute hour angle allowed: {kwargs['maxha']}, ")
        f.writelines("random hour angle\n" if kwargs['rha'] == 'y' else "regular hour angle\n")
        f.writelines(f"Random number seed: {kwargs['randnum']}\n" if kwargs['randnum'] is not None else "")
        if kwargs['rn']:
            f.writelines(f"Red Noise injection: amplitude - {kwargs['rnamp']}, gamma - {kwargs['rngamma']}, "
                         f"components - {kwargs['rnc']}, time span - {kwargs['rntspan']}\n")
        if kwargs['dmn']:
            f.writelines(f"Dispersion Measure Noise injection: amplitude - {kwargs['dmnamp']}, "
                         f"gamma - {kwargs['dmngamma']}, components - {kwargs['dmnc']}\n")
        if kwargs['gwb']:
            f.writelines(f"Gravitational Wave Background injection: amplitude - {kwargs['gwbamp']}, "
                         f"gamma - {kwargs['gwbgamma']}, No. of points used in interpolation - {kwargs['gwbnpts']}\n")
            f.writelines(f"Maximum multipole of GW power decomposition - {kwargs['lmax']}, "
                         f"lowest frequency 1/(howml * T) - {kwargs['gwbhowml']}, "
                         f"Add red noise with no correlation\n" if kwargs['nocorr'] else "\n")
            if kwargs['turnover']:
                f.writelines(f"Produce spectrum with turnover at frequency f0 - {kwargs['gwbf0']} Hz, "
                             f"Spectral index of power spectrum for f<<f0 - {kwargs['gwbbeta']}, "
                             f"Fudge factor for flatness of spectrum turnover - {kwargs['gwbpower']}\n")
    return infotxt


def ratio_number(ratio, number, total):
    """Set the ratio or number of arrays based on each other and total arrays.

    :param ratio: the ratio of arrays for given band
    :param number: the number of arrays for given band
    :param total: the total number of arrays

    :return: ratio: the new ratio of arrays for given band
    :return number: the new number of arrays for given band
    """
    if ratio is not None:
        number = int(total * ratio)
    elif number is not None:
        ratio = number/total
    return ratio, number


def subband_strategy(**kwargs):
    """Find the corresponding subband settings and strategies."""
    if kwargs['strategy'] is None:
        kwargs['ruhfb'], kwargs['nuhfb'] = ratio_number(kwargs['ruhfb'], kwargs['nuhfb'], kwargs['narray'])
        kwargs['rlb'], kwargs['nlb'] = ratio_number(kwargs['rlb'], kwargs['nlb'], kwargs['narray'])
        kwargs['rsb'], kwargs['nsb'] = ratio_number(kwargs['rsb'], kwargs['nsb'], kwargs['narray'])
        strategy = ""
        if kwargs['nuhfb'] > 0 and kwargs['nsbuhf'] > 0:
            strategy += f"+{kwargs['nuhfb']}UHF{kwargs['nsbuhf']}"
        if kwargs['nlb'] > 0 and kwargs['nsbl'] > 0:
            strategy += f"+{kwargs['nlb']}L{kwargs['nsbl']}"
        if kwargs['nsb'] > 0 and kwargs['nsbs'] > 0:
            strategy += f"+{kwargs['nsb']}S{kwargs['nsbs']}"
        kwargs['strategy'] = strategy.split("+", 1)[-1]
    else:
        for bandset in kwargs['strategy'].split("+"):
            if "UHF" in bandset:
                kwargs['nuhfb'], kwargs['nsbuhf'] = int(bandset.split("UHF")[0]), int(bandset.split("UHF")[1])
            elif "L" in bandset:
                kwargs['nlb'], kwargs['nsbl'] = int(bandset.split("L")[0]), int(bandset.split("L")[1])
            elif "S" in bandset:
                kwargs['nsb'], kwargs['nsbs'] = int(bandset.split("S")[0]), int(bandset.split("S")[1])
        kwargs['ruhfb'] = ratio_number(None, kwargs['nuhfb'], kwargs['narray'])[0]
        kwargs['rlb'] = ratio_number(None, kwargs['nlb'], kwargs['narray'])[0]
        kwargs['rsb'] = ratio_number(None, kwargs['nsb'], kwargs['narray'])[0]
    return kwargs


def log_memory(stage):
    """Print the memory usage at certain stage.

    :param stage: the stage in the program
    """
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    with open("memory_log.txt", "a") as f:
        f.write(f"Stage: {stage}\n RSS: {mem.rss/1024**2:.2f} MB\n VMS: {mem.vms/1024**2:.2f} MB\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ToA simulate wrapper.'
                                                 'Simulated ToAs are generated with tempo2 fake plugin.'
                                                 'Red noise, DM noise, GWB are added with libstempo.'
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
    parser.add_argument('--testd', '--test-dir', type=str, default=None,
                        help='The directory for saving all files in a series test of a specific parameters')
    parser.add_argument('--comtd', '--comment-dir', type=str, default=None,
                        help='The subdirectory for saving all files in a series test')
    parser.add_argument('--reald', '--realization-dir', type=str, default=None,
                        help='The subdirectory for saving all files in one run')
    parser.add_argument('--rlzno', '--realization-number', type=int, default=None,
                        help='The No. of realization for a given setting')
    parser.add_argument('--strategy', '--observing-strategy', type=str, default=None, help='The subband strategy used')
    # /cluster/home/liuyang/Sub-Array

    args = parser.parse_args()
    args_keys = ['parfile', 'datadir', 'testd', 'comtd', 'reald', 'timd', 'resd', 'chaind', 'noised', 'maxha', 'rha',
                 'randnum', 'cad', 'nobs', 'mjds', 'mjde', 'tel', 'narray', 'reffreq', 'refsig', 'refflux', 'refgamma',
                 'ruhfb', 'rlb', 'rsb', 'nuhfb', 'nlb', 'nsb', 'nsbuhf', 'nsbl', 'nsbs', 'cfrequhf', 'cfreql', 'cfreqs',
                 'bwuhf', 'bwl', 'bws', 'rn', 'rnamp', 'rngamma', 'rnc', 'rntspan', 'dmn', 'dmnamp', 'dmngamma', 'dmnc',
                 'nocorr', 'gwb', 'gwbamp', 'gwbgamma', 'gwbnpts', 'turnover', 'gwbf0', 'gwbbeta', 'gwbpower',
                 'gwbhowml', 'lmax', 'rlzno', 'strategy']
    kw_args = {key: getattr(args, key) for key in args_keys if hasattr(args, key)}

    if args.rha:
        kw_args['rha'] = 'y'
    else:
        kw_args['rha'] = 'n'
    if (args.reald is None) and (args.rlzno is not None):
        kw_args['reald'] = str(args.rlzno)
    kw_args = subband_strategy(**kw_args)

    datadir, par_files, psrnlist = psr_list(datad=kw_args['datadir'], pf=kw_args['parfile'])
    extradir = make_directories(datadir, kw_args['testd'], kw_args['comtd'], kw_args['reald'], make=True)
    check_directory(datadir + extradir + kw_args['timd'])
    describe = describe_name(**kw_args)
    info = save_paras(**kw_args)

    tempopulsar = []
    for parfile, psrn in zip(par_files, psrnlist):
        timlines = ["FORMAT 1 \n"]

        if kw_args['nuhfb'] != 0:
            U_Band = Band(kw_args['cfrequhf'], kw_args['bwuhf'], kw_args['ruhfb'], kw_args['nuhfb'], kw_args['nsbuhf'])
            rms_uhf = U_Band.calculate_rms(kw_args['reffreq'], kw_args['refsig'], kw_args['refgamma'])
            for j, rms_sub in enumerate(rms_uhf):
                tempo2_fake_simulate(parfile, rms_sub, U_Band.subfreq[j], U_Band.subbw, **kw_args)
                target = f"{psrn}_UHF_{j + 1}.tim"
                os.rename(datadir + extradir + f"{psrn}.simulate", datadir + extradir + kw_args['timd'] + target)
                timlines.append(f"INCLUDE {target} \n")

        if kw_args['nlb'] != 0:
            L_Band = Band(kw_args['cfreql'], kw_args['bwl'], kw_args['rlb'], kw_args['nlb'], kw_args['nsbl'])
            rms_l = L_Band.calculate_rms(kw_args['reffreq'], kw_args['refsig'], kw_args['refgamma'])
            for j, rms_sub in enumerate(rms_l):
                tempo2_fake_simulate(parfile, rms_sub, L_Band.subfreq[j], L_Band.subbw, **kw_args)
                target = f"{psrn}_L_{j + 1}.tim"
                os.rename(datadir + extradir + f"{psrn}.simulate", datadir + extradir + kw_args['timd'] + target)
                timlines.append(f"INCLUDE {target} \n")

        if kw_args['nsb'] != 0:
            S_Band = Band(kw_args['cfreqs'], kw_args['bws'], kw_args['rsb'], kw_args['nsb'], kw_args['nsbs'])
            rms_s = S_Band.calculate_rms(kw_args['reffreq'], kw_args['refsig'], kw_args['refgamma'])
            for j, rms_sub in enumerate(rms_s):
                tempo2_fake_simulate(parfile, rms_sub, S_Band.subfreq[j], S_Band.subbw, **kw_args)
                target = f"{psrn}_S_{j + 1}.tim"
                os.rename(datadir + extradir + f"{psrn}.simulate", datadir + extradir + kw_args['timd'] + target)
                timlines.append(f"INCLUDE {target} \n")

        tp = datadir + extradir + kw_args['timd'] + psrn
        with open(f"{tp}.tim", "w") as newf:
            newf.writelines(timlines)

        psr = lt.tempopulsar(parfile=parfile, timfile=f"{tp}.tim")
        ltt.make_ideal(psr)
        tempopulsar.append(psr)

    for psrn, psr in zip(psrnlist, tempopulsar):
        if kw_args['rn']:
            ltt.add_rednoise(psr, kw_args['rnamp'], kw_args['rngamma'],
                             components=kw_args['rnc'], tspan=kw_args['rntspan'], seed=kw_args['randnum'])
        if kw_args['dmn']:
            # ltt.add_dm(psr, kw_args['dmnamp'], kw_args['dmngamma'],
            #            components=kw_args['dmnc'], seed=kw_args['randnum'])
            ltt.add_dm(psr, kw_args['dmnamp'] * 1400 * 1400 * 2.41e-4, kw_args['dmngamma'],
                       components=kw_args['dmnc'], seed=kw_args['randnum'])
            # Convert the enterprise DM amplitude to the libstempo DM amplitude
            # ltt.add_dm(psr, kw_args['dmnamp'] * np.sqrt(12) * np.pi, kw_args['dmngamma'],
            #            components=kw_args['dmnc'], seed=kw_args['randnum'])
            # Convert the run_enterprise DM amplitude to the libstempo DM amplitude

        tp = datadir + extradir + kw_args['timd'] + psrn
        descr = describe_name(**kw_args).split("%GWB")[0]
        rndescribe = '' if descr == '_' else descr
        psr.savetim(f"{tp}{rndescribe}.tim")
        lt.purgetim(f"{tp}{rndescribe}.tim")
        # Delete all lines with reference
        lines = filter(lambda l: 'reference' not in l, open(f"{tp}{rndescribe}.tim").readlines())
        with open(f"{tp}_injected.tim", 'w') as file:
            file.writelines(lines)

    check_directory(datadir + extradir + kw_args['resd'])

    if kw_args['gwb']:
        psrobject = [lt.tempopulsar(parfile=f"{datadir + psrn}.par",
                                    timfile=f"{datadir + extradir + kw_args['timd'] + psrn}_injected.tim")
                     for psrn in psrnlist]
        ltt.createGWB(psrobject, kw_args['gwbamp'], kw_args['gwbgamma'], npts=kw_args['gwbnpts'], lmax=kw_args['lmax'],
                      noCorr=kw_args['nocorr'], howml=kw_args['gwbhowml'], turnover=kw_args['turnover'],
                      f0=kw_args['gwbf0'], beta=kw_args['gwbbeta'], power=kw_args['gwbpower'], seed=kw_args['randnum'])
        gwbdescribe = "_%GWB" + describe_name(**kw_args).split("%GWB")[-1]
        for k, psr in enumerate(psrobject):
            psrn = psrnlist[k]
            rp = datadir + extradir + kw_args['resd'] + psrn
            check_directory(rp)
            psr.savetim(f"{rp}/{psrn + gwbdescribe}.tim")
            lt.purgetim(f"{rp}/{psrn + gwbdescribe}.tim")
            lines = filter(lambda l: 'reference' not in l, open(f"{rp}/{psrn + gwbdescribe}.tim").readlines())
            open(f"{rp}/{psrn}_all.tim", 'w').writelines(lines)
            # Move par and tim files to the right directory and rename them according to enterprise standard
            shutil.copy(par_files[k], f"{rp}/{psrn}.par")
            # shutil.copy(f"{rp}/{psrn + gwbdescribe}.tim", f"{rp}/{psrn}_all.tim")
    else:
        for parfile, psrn in zip(par_files, psrnlist):
            rp = datadir + extradir + kw_args['resd'] + psrn
            check_directory(rp)
            # Move par and tim files to the right directory and rename them according to enterprise standard
            shutil.copy(parfile, f"{rp}/{psrn}.par")
            lines = filter(lambda l: 'reference' not in l,
                           open(f"{datadir + extradir + kw_args['timd'] + psrn}_injected.tim").readlines())
            open(f"{rp}/{psrn}_all.tim", 'w').writelines(lines)

    check_directory(datadir + extradir + kw_args['chaind'])
    check_directory(datadir + extradir + kw_args['chaind'] + kw_args['noised'])

    for psrn in psrnlist:
        with open(f"{datadir + extradir + kw_args['chaind'] + kw_args['noised'] + psrn}.json", 'w') as file:
            file.writelines('{ \n')
            file.writelines(f'    "{psrn}_efac": 1,\n')
            file.writelines(f'    "{psrn}_log10_tnequad": -10\n')
            file.writelines('} \n')
