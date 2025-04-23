#!/usr/bin/env python
"""ToA simulate wrapper for generating simulated ToAs with tempo2 fake plugin and injecting RN, DM, GWB with libstempo.
    Written by Yang Liu (liuyang@shao.ac.cn)."""

import argparse
import os
import glob
import subprocess
import shutil
import numpy as np
import libstempo as lt
import libstempo.toasim as ltt

parser = argparse.ArgumentParser(description='ToA simulate wrapper.'
                                             'Simulated ToAs are generated with tempo2 fake plugin.'
                                             'Red noise, DM noise, GWB are added with libstempo.'
                                             'Written by Yang Liu (liuyang@shao.ac.cn).')
parser.add_argument('-p', '--parfile', type=str, default=[], nargs='+',
                    help='Parameter files for pulsars used in simulation')
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
parser.add_argument('--testd', '--test-dir', type=str, default=None,
                    help='The directory for saving all files in a serie test of a specific parameters')
parser.add_argument('--comtd', '--comment-dir', type=str, default=None,
                    help='The subdirectory for saving all files in a serie test')
parser.add_argument('--reald', '--realization-dir', type=str, default=None,
                    help='The directory for saving all files in one run')
# /cluster/home/liuyang/Sub-Array


class Band:
    def __init__(self, freq, bw, num_tel, num_sub):
        """Initialize the Band class with given central frequency and bandwidth,
        number of arrays observed at this band, and number of sub-bands in this band."""
        self.freq = freq
        self.bandwidth = bw
        self.num_tel = num_tel
        self.ratio_tel = self.num_tel/args.narray
        self.num_sub = num_sub
        self.subbw = self.bandwidth/self.num_sub
        self.subfreq = np.linspace(self.freq-0.5*(self.bandwidth-self.subbw),
                                   self.freq+0.5*(self.bandwidth-self.subbw), self.num_sub)
        self.rms = None

    def calculate_rms(self, gamma_p=1.6):
        """Calculate the corresponding rms of ToA at given frequency based on reference rms at reference frequency.
        Default spectral index is 1.6."""
        rms = args.refsig/self.ratio_tel*np.sqrt(self.num_sub) * (self.subfreq/args.reffreq)**gamma_p
        self.rms = rms
        return self.rms


def tempo2_fake_simulate(parfile, obs_cad, no_obs, maxabs_ha, rha, mjd_start, mjd_end, rmssub, telescope,
                         subfreq, subbw, randnum=None, outdir=None):
    """Simulate ToA for given pulsar based on its par file. Input:
    cadence, number of observation each time, regular/random hour angle, the maximum absolute hour angle allowed,
    simulation start date in MJD, simulation end date in MJD, observation telescope, rms in ToA,
    observation frequency, bandwidth, and random number seed."""
    command = ["tempo2", "-gr", "fake", "-f", parfile, "-ndobs", str(obs_cad), "-nobsd", str(no_obs), "-ha",
               str(maxabs_ha), "-randha", rha, "-start", str(mjd_start), "-end", str(mjd_end), "-rms",
               str(1e-3 * rmssub), "-tel", telescope, "-freq", str(subfreq), "-bw", str(subbw), "-withpn", "-setref"]
    if randnum is not None:
        command.extend(["-idum", str(randnum)])
    if outdir is not None:
        command.extend(["-o", str(outdir+".simulate")])
    subprocess.run(command, check=True)


def describe_name(rnamp, rngamma, dmnamp, dmngamma, gwbamp, gwbgam, rn=False, dmn=False, gwb=False):
    """The name to describe the simulation data based on the signal injected. Format:
    RN %Amplitude #Gamma + DM %Amplitude #Gamma + GWB %Amplitude #Gamma ."""
    desc = ""
    if rn:
        desc += f"_RN%A{int(np.log10(rnamp))}#G{int(rngamma)}"
    if dmn:
        desc += f"_DM%A{int(np.log10(dmnamp))}#G{int(dmngamma)}"
    if gwb:
        desc += f"_GWB%A{int(np.log10(gwbamp))}#G{int(gwbgam)}"
    return desc


def save_paras(arg, exd, plist):
    """Export the directory and parameter values used in this simulation to a txt file for recording."""
    with open(f"{arg.datadir+exd}paras_info.txt", "w") as paraf:
        paraf.writelines("Parameter information:\n")
        paraf.writelines(f"Relative path to the directory containing the par files: {arg.datadir}\n")
        paraf.writelines(f"{len(plist)} pulsars are used in simulation: \n")
        paraf.writelines(f"{pf}, " for pf in plist)
        paraf.writelines("\n")
        paraf.writelines(f"Directory for output simulation tim files: {arg.timd}\n")
        paraf.writelines(f"Directory for result par and tim files: {arg.resd}\n")
        paraf.writelines(f"Directory for MCMC chains: {arg.chaind}\n")
        paraf.writelines(f"Directory for noise files: {arg.noised}\n")
        paraf.writelines(f"Directory for test of a specific parameters: {arg.testd}/\n")
        paraf.writelines(f"Directory for a test series: {arg.comtd}/\n")
        paraf.writelines(f"Directory for a realization of given value: {arg.reald}/\n")
        paraf.writelines(f"MJD range: {arg.mjds} - {arg.mjde}\n")
        paraf.writelines(f"Observation cadence: {arg.cad} days, {arg.nobs} observations each time\n")
        paraf.writelines(f"Telescope: {arg.tel}, {arg.narray} arrays in total\n")
        paraf.writelines(f"UHF Band: central frequency - {arg.cfrequhf} MHz, bandwidth - {arg.bwuhf} MHz, "
                         f"{arg.nuhfb} arrays, {arg.nsbuhf} sub-bands\n")
        paraf.writelines(f"L Band: central frequency - {arg.cfreql} MHz, bandwidth - {arg.bwl} MHz, "
                         f"{arg.nlb} arrays, {arg.nsbl} sub-bands\n")
        paraf.writelines(f"S Band: central frequency - {arg.cfreqs} MHz, bandwidth - {arg.bws} MHz, "
                         f"{arg.nsb} arrays, {arg.nsbs} sub-bands\n")
        paraf.writelines(f"Reference: frequency - {arg.reffreq} MHz, sigma - {arg.refsig} μs, "
                         f"flux - {arg.refflux} μJy, spectral index - {arg.refgamma}\n")
        paraf.writelines(f"Maximum absolute hour angle allowed: {arg.maxha}, ")
        paraf.writelines("random hour angle\n" if arg.rha else "regular hour angle\n")
        paraf.writelines(f"Random number seed: {arg.randnum}\n" if arg.randnum is not None else "")
        if arg.rn:
            paraf.writelines(f"Red Noise injection: amplitude - {arg.rnamp}, gamma - {arg.rngamma}, "
                             f"components - {arg.rnc}, time span - {arg.rntspan}\n")
        if arg.dmn:
            paraf.writelines(f"Dispersion Measure Noise injection: amplitude - {arg.dmnamp}, gamma - {arg.dmngamma}, "
                             f"components - {arg.dmnc}\n")
        if arg.gwb:
            paraf.writelines(f"Gravitational Wave Background injection: amplitude - {arg.gwbamp}, "
                             f"gamma - {arg.gwbgam}, number of points used in interpolation - {arg.gwbnpts}\n")
            paraf.writelines(f"Maximum multipole of GW power decomposition - {arg.lmax}, "
                             f"lowest frequency 1/(howml * T) - {arg.gwbhowml}, "
                             f"Add red noise with no correlation\n" if arg.nocorr else "\n")
            if arg.turnover:
                paraf.writelines(f"Produce spectrum with turnover at frequency f0 - {arg.gwbf0} Hz, "
                                 f"Spectral index of power spectrum for f<<f0 - {arg.gwbbeta}, "
                                 f"Fudge factor for flatness of spectrum turnover - {arg.gwbpower}\n")


args = parser.parse_args()
if args.rha:
    randha = "y"
else:
    randha = "n"
if args.datadir is not None:
    par_files = sorted(glob.glob(os.path.join(args.datadir, "*.par")))
else:
    par_files = args.parfile
    args.datadir = args.parfile[0].rsplit("/", 1)[0]+"/"

datadir = args.datadir
psrnlist = [parfile.split("/")[-1].split(".", 1)[0] for parfile in par_files]

extradir = ""
if args.testd is not None:
    if not os.path.exists(datadir + extradir + args.testd):
        os.makedirs(datadir + extradir + args.testd)
    extradir = extradir + args.testd + "/"
if args.comtd is not None:
    if not os.path.exists(datadir + extradir + args.comtd):
        os.makedirs(datadir + extradir + args.comtd)
    extradir = extradir + args.comtd + "/"
if args.reald is not None:
    if not os.path.exists(datadir + extradir + args.reald):
        os.makedirs(datadir + extradir + args.reald)
    extradir = extradir + args.reald + "/"
if not os.path.exists(datadir + extradir + args.timd):
    os.makedirs(datadir + extradir + args.timd)

describe = describe_name(args.rnamp, args.rngamma, args.dmnamp, args.dmngamma, args.gwbamp, args.gwbgam,
                         rn=args.rn, dmn=args.dmn, gwb=args.gwb)
save_paras(args, extradir, psrnlist)

for i, parfile in enumerate(par_files):
    psrn = parfile.split("/")[-1].split(".", 1)[0]
    timlines = ["FORMAT 1 \n"]

    if args.nuhfb != 0:
        UHF_Band = Band(args.cfrequhf, args.bwuhf, args.nuhfb, args.nsbuhf)
        rms_uhf = UHF_Band.calculate_rms(gamma_p=args.refgamma)
        for j, rms_sub in enumerate(rms_uhf):
            tempo2_fake_simulate(parfile, args.cad, args.nobs, args.maxha, randha, args.mjds, args.mjde, rms_sub,
                                 args.tel, UHF_Band.subfreq[j], UHF_Band.subbw, args.randnum, datadir+extradir+psrn)
            target = f"{psrnlist[i]}_UHF_{j+1}.tim"
            os.rename(datadir+extradir+f"{psrnlist[i]}.simulate", datadir+extradir+args.timd+target)
            timlines.append(f"INCLUDE {target} \n")

    if args.nlb != 0:
        L_Band = Band(args.cfreql, args.bwl, args.nlb, args.nsbl)
        rms_l = L_Band.calculate_rms(gamma_p=args.refgamma)
        for j, rms_sub in enumerate(rms_l):
            tempo2_fake_simulate(parfile, args.cad, args.nobs, args.maxha, randha, args.mjds, args.mjde, rms_sub,
                                 args.tel, L_Band.subfreq[j], L_Band.subbw, args.randnum, datadir+extradir+psrn)
            target = f"{psrnlist[i]}_L_{j + 1}.tim"
            os.rename(datadir+extradir+f"{psrnlist[i]}.simulate", datadir+extradir+args.timd+target)
            timlines.append(f"INCLUDE {target} \n")

    if args.nsb != 0:
        S_Band = Band(args.cfreqs, args.bws, args.nsb, args.nsbs)
        rms_s = S_Band.calculate_rms(gamma_p=args.refgamma)
        for j, rms_sub in enumerate(rms_s):
            tempo2_fake_simulate(parfile, args.cad, args.nobs, args.maxha, randha, args.mjds, args.mjde, rms_sub,
                                 args.tel, S_Band.subfreq[j], S_Band.subbw, args.randnum, datadir+extradir+psrn)
            target = f"{psrnlist[i]}_S_{j + 1}.tim"
            os.rename(datadir+extradir+f"{psrnlist[i]}.simulate", datadir+extradir+args.timd+target)
            timlines.append(f"INCLUDE {target} \n")

    with open(f"{datadir+extradir+args.timd+psrn}.tim", "w") as newf:
        newf.writelines(timlines)

    psr = lt.tempopulsar(parfile=parfile, timfile=f"{datadir+extradir+args.timd+psrn}.tim")
    ltt.make_ideal(psr)
    rndescribe = ""

    if args.rn:
        ltt.add_rednoise(psr, args.rnamp, args.rngamma, components=args.rnc, tspan=args.rntspan, seed=args.randnum)
        rndescribe += "_RN%A{}#G{}".format(int(np.log10(args.rnamp)), int(args.rngamma))
    if args.dmn:
        # ltt.add_dm(psr, args.dmnamp, args.dmngamma, components=args.dmnc, seed=args.randnum)
        ltt.add_dm(psr, args.dmnamp*1400*1400*2.41e-4, args.dmngamma, components=args.dmnc, seed=args.randnum)
        # Convert the enterprise DM amplitude to the libstempo DM amplitude
        # ltt.add_dm(psr, args.dmnamp*np.sqrt(12)*np.pi, args.dmngamma, components=args.dmnc, seed=args.randnum)
        # Convert the run_enterprise DM amplitude to the libstempo DM amplitude
        rndescribe += "_DM%A{}#G{}".format(int(np.log10(args.dmnamp)), int(args.dmngamma))

    psr.savetim(f"{datadir+extradir+args.timd+psrn}_injected.tim")
    lt.purgetim(f"{datadir+extradir+args.timd+psrn}_injected.tim")
    # Delete all lines with reference
    lines = filter(lambda l: 'reference' not in l, open(f"{datadir+extradir+args.timd+psrn}_injected.tim").readlines())
    with open(f"{datadir+extradir+args.timd+psrn}{rndescribe}.tim", 'w') as file:
        file.writelines(lines)

if not os.path.exists(datadir+extradir+args.resd):
    os.makedirs(datadir+extradir+args.resd)

if args.gwb:
    psrobject = [lt.tempopulsar(parfile=f"{datadir+psrn}.par",
                                timfile=f"{datadir+extradir+args.timd+psrn}_injected.tim")
                 for psrn in psrnlist]
    ltt.createGWB(psrobject, args.gwbamp, args.gwbgam, noCorr=args.nocorr, seed=args.randnum, lmax=args.lmax,
                  turnover=args.turnover, f0=args.gwbf0, beta=args.gwbbeta, power=args.gwbpower, npts=args.gwbnpts,
                  howml=args.gwbhowml)
    gwbdescribe = "_GWB%A{}#G{}".format(int(np.log10(args.gwbamp)), int(args.gwbgam))
    for k, psr in enumerate(psrobject):
        psrn = psrnlist[k]
        if not os.path.exists(datadir+extradir+args.resd+psrn):
            os.makedirs(datadir+extradir+args.resd+psrn)
        psr.savetim(f"{datadir+extradir+args.resd+psrn}/{psrn+gwbdescribe}.tim")
        lt.purgetim(f"{datadir+extradir+args.resd+psrn}/{psrn+gwbdescribe}.tim")
        lines = filter(lambda l: 'reference' not in l,
                       open(f"{datadir+extradir+args.resd+psrn}/{psrn+gwbdescribe}.tim").readlines())
        open(f"{datadir+extradir+args.resd+psrn}/{psrn+gwbdescribe}.tim", 'w').writelines(lines)
        # Move par and tim files to the right directory and rename them according to enterprise standard
        shutil.copy(par_files[k], datadir+extradir+args.resd+psrn+f"/{psrn}.par")
        shutil.copy(datadir+extradir+args.resd+psrn+f"/{psrn+gwbdescribe}.tim",
                    datadir+extradir+args.resd+psrn+f"/{psrn}_all.tim")
else:
    for k, psrn in enumerate(psrnlist):
        if not os.path.exists(datadir+extradir+args.resd+psrn):
            os.makedirs(datadir+extradir+args.resd+psrn)
        # Move par and tim files to the right directory and rename them according to enterprise standard
        shutil.copy(par_files[k], datadir+extradir+args.resd+psrn+f"/{psrn}.par")
        lines = filter(lambda l: 'reference' not in l,
                       open(f"{datadir+extradir+args.timd+psrn}_injected.tim").readlines())
        with open(f"{datadir+extradir+args.resd+psrn}/{psrn}_all.tim", 'w') as file:
            file.writelines(lines)

if not os.path.exists(datadir+extradir+args.chaind):
    os.makedirs(datadir+extradir+args.chaind)

if not os.path.exists(datadir+extradir+args.chaind+args.noised):
    os.makedirs(datadir+extradir+args.chaind+args.noised)

for psrn in psrnlist:
    with open(f"{datadir+extradir+args.chaind+args.noised+psrn}.json", 'w') as file:
        file.writelines('{ \n')
        file.writelines(f'    "{psrn}_efac": 1,\n')
        file.writelines(f'    "{psrn}_log10_tnequad": -10\n')
        file.writelines('} \n')
