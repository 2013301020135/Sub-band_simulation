#!/usr/bin/env python
"""ToA simulate wrapper for generating simulated ToAs with tempo2 fake plugin and injecting RN, DM, GWB with libstempo.
    Written by Yang Liu (liuyang@shao.ac.cn)."""

import time
import argparse
import sys
import os
import subprocess
import shutil
import bisect
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import numpy as np
import numpy.ma as ma
import pandas as pd
# from __future__ import print_function
# from uncertainties import ufloat
from copy import deepcopy
import libstempo as lt
import libstempo.plot as ltp, libstempo.toasim as ltt

parser = argparse.ArgumentParser(description='ToA simulate wrapper for generating simulated ToAs with tempo2 fake plugin. Red noise, DM noise, GWB are added with libstempo. Written by Yang Liu (liuyang@shao.ac.cn).')
parser.add_argument('-p', '--parfile', type=str, default=[], nargs='+', help='Parameter files for pulsars used in simulation', required=True)
# parser.add_argument('-t', '--timfile', type=str, help='TOA file', required=True)
# parser.add_argument('--sub', '--sub-band', action='store_false', help='Deactivate sub-band mode if called')
parser.add_argument('--cad', '--observation-cadence', type=float, default=14, help='The number of days between observations')
parser.add_argument('--nobs', '--no-of-observation', type=int, default=1, help='The number of observations on a given day')
parser.add_argument('--maxha', '--hour-angle', type=float, default=8, help='The maximum absolute hour angle allowed')
parser.add_argument('--rha', '--random-hour-angle', action='store_true', help='Use random hour angle coverage if called, otherwise use regular hour angle')
parser.add_argument('--mjds', '--initial-mjd', type=int, default=50000, help='The initial MJD for the simulated TOAs')
parser.add_argument('--mjde', '--final-mjd', type=int, default=60000, help='The final MJD for the simulated TOAs')
parser.add_argument('--nuhfb', '--num-uhfband', type=int, default=0, help='Number of arrays in UHF-Band')
parser.add_argument('--nlb', '--num-lband', type=int, default=32, help='Number of arrays in L-Band')
parser.add_argument('--nsb', '--num-sband', type=int, default=32, help='Number of arrays in S-Band')
parser.add_argument('--nsbuhf', '--num-uhf-subband', type=int, default=8, help='Number of sub-bands in UHF-Band')
parser.add_argument('--nsbl', '--num-l-subband', type=int, default=8, help='Number of sub-bands in L-Band')
parser.add_argument('--nsbs', '--num-s-subband', type=int, default=8, help='Number of sub-bands in S-Band')
parser.add_argument('--cfrequhf', '--central-frequency-uhf', type=float, default=810, help='Central frequency of UHF-Band in MHz')
parser.add_argument('--cfreql', '--central-frequency-l', type=float, default=1280, help='Central frequency of L-Band in MHz')
parser.add_argument('--cfreqs', '--central-frequency-s', type=float, default=2600, help='Central frequency of S-Band in MHz')
parser.add_argument('--bwuhf', '--bandwidth-uhf', type=float, default=500, help='Bandwidth of UHF-Band in MHz')
parser.add_argument('--bwl', '--bandwidth-l', type=float, default=800, help='Bandwidth of L-Band in MHz')
parser.add_argument('--bws', '--bandwidth-s', type=float, default=800, help='Bandwidth of S-Band in MHz')
parser.add_argument('--narray', '--num-array', type=int, default=64, help='Number of total arrays')
parser.add_argument('--randnum', '--random-number-seed', type=int, default=None, help='Specify random number seed')
parser.add_argument('--tel', type=str, default="meerkat", help='The name of the telescope')
parser.add_argument('--refsig', '--reference-sigma', type=float, default=1, help='The rms of Gaussian noise in micro-second when all telescope are in reference frequency')
parser.add_argument('--reffreq', '--reference-frequency', type=float, default=1300, help='The reference frequency in MHz')
parser.add_argument('--refflux', '--reference-flux', type=float, default=1, help='The reference flux in micro-Jy at reference frequency')
parser.add_argument('--rn', '--red-noise', action='store_true', help='Inject red noise if called')
parser.add_argument('--rnamp', '--red-noise-amplitude', type=float, default=1e-12, help='The red noise amplitude')
parser.add_argument('--rngamma', '--red-noise-gamma', type=float, default=3, help='The red noise spectral slope (gamma, positive)')
parser.add_argument('--rnc', '--red-noise-component', type=int, default=100, help='The number of red noise component')
parser.add_argument('--rntspan', '--red-noise-tspan', type=float, default=None, help='The time span used for red noise injection')
parser.add_argument('--dmn', '--dm-noise', action='store_true', help='Inject DM noise if called')
parser.add_argument('--dmnamp', '--dm-noise-amplitude', type=float, default=1e-12, help='The DM noise amplitude')
parser.add_argument('--dmngamma', '--dm-noise-gamma', type=float, default=3, help='The DM noise spectral slope (gamma, positive)')
parser.add_argument('--dmnc', '--dm-noise-component', type=int, default=100, help='The number of DM noise component')
parser.add_argument('--gwb', '--gw-background', action='store_true', help='Inject gravitational wave background if called')
# parser.add_argument('--psrdist', '--pulsar-distance', type=float, default=1, help='The pulsar distance in kpc')
# parser.add_argument('--ngw', '--num-binaries', type=int, default=1000, help='The number of binaries used when inject GWB')
# parser.add_argument('--logspace', '--gwb-log-spacing', action='store_false', help='Use linear spacing for GWB if called')
# parser.add_argument('--gwbflow', '--gwb-frequency-low', type=float, default=1e-8, help='The low frequency end in Hz')
# parser.add_argument('--gwbfhigh', '--gwb-frequency-high', type=float, default=1e-5, help='The high frequency end in Hz')
# parser.add_argument('--gwbalpha', '--gwb-alpha', type=float, default=0.66, help='The gravitational wave background exponent')
parser.add_argument('--gwbamp', '--gwb-amplitude', type=float, default=1e-14, help='The gravitational wave background amplitude')
parser.add_argument('--gwbgam', '--gwb-gamma', type=float, default=4, help='The gravitational wave background spectral slope (gamma, positive)')
parser.add_argument('--nocorr', '--gwb-no-corr', action='store_true', help='Add red noise with no correlation')
parser.add_argument('--lmax', '--gwb-lmax', type=int, default=0, help='The maximum multipole of GW power decomposition')
parser.add_argument('--turnover', '--gwb-turnover', action='store_true', help='Produce spectrum with turnover at frequency f0')
parser.add_argument('--gwbf0', '--gwb-f0', type=float, default=1e-9, help='The frequency of spectrum turnover')
parser.add_argument('--gwbbeta', '--gwb-beta', type=float, default=1, help='The spectral index of power spectrum for f<<f0')
parser.add_argument('--gwbpower', '--gwb-power', type=float, default=1, help='The fudge factor for flatness of spectrum turnover')
parser.add_argument('--gwbnpts', '--gwb-npts', type=int, default=600, help='The number of points used in interpolation')
parser.add_argument('--gwbhowml', '--gwb-howml', type=float, default=10, help='The lowest frequency is 1/(howml * T)')
parser.add_argument('--dir', '--tim-dir', type=str, default='tims/', help='The relative path to the simulate directory')
# /cluster/home/liuyang/Sub-Array

class Band:
    def __init__(self, freq, bw):
        self.freq = freq
        self.bandwidth = bw

    def telescope_ratio(self, num_tel):
        self.num_tel = num_tel
        self.ratio_tel = self.num_tel/args.narray

    def sub_band(self, num_sub):
        self.num_sub = num_sub
        self.subbw = self.bandwidth/self.num_sub
        self.subfreq = self.freq + 0.5*(self.subbw-self.bandwidth) + self.subbw*np.arange(self.num_sub)

    def calculate_rms(self, gamma_p=1.6):
        rms = args.refsig/self.ratio_tel*self.num_sub * (self.subfreq/args.reffreq)**gamma_p
        self.rms = rms
        return rms

args = parser.parse_args()
obs_cad = args.cad
no_obs = args.nobs
maxabs_ha = args.maxha
mjd_start = args.mjds
mjd_end = args.mjde
telescope = args.tel

if args.rha:
    randha = "y"
else:
    randha = "n"

if not os.path.exists(args.dir):
    os.makedirs(args.dir)

parlist = []
psrnlist = []
psrobject = []

for i, parfile in enumerate(args.parfile):
    par = parfile.split(".", 1)[0]   # The name of par file without suffix
    parlist.append(par)
    psrn = par.split("_")[-1]   # Get the name of pulsar
    psrnlist.append(psrn)

    timlines = []
    timlines.append("FORMAT 1 \n")

    UHF_Band = Band(args.cfrequhf, args.bwuhf)
    L_Band = Band(args.cfreql, args.bwl)
    S_Band = Band(args.cfreqs, args.bws)

    UHF_Band.telescope_ratio(args.nuhfb)
    L_Band.telescope_ratio(args.nlb)
    S_Band.telescope_ratio(args.nsb)

    UHF_Band.sub_band(args.nsbuhf)
    L_Band.sub_band(args.nsbl)
    S_Band.sub_band(args.nsbs)

    if UHF_Band.num_tel != 0:
        rms_uhf = UHF_Band.calculate_rms()
        for i, rms_sub in enumerate(rms_uhf):
            if args.randnum is None:
                subprocess.call(["tempo2", "-gr", "fake", "-f", parfile, "-ndobs", str(obs_cad), "-nobsd", str(no_obs),
                                 "-ha", str(maxabs_ha), "-randha", randha, "-start", str(mjd_start), "-end", str(mjd_end),
                                 "-rms", str(1e-3*rms_sub), "-tel", telescope, "-freq", str(UHF_Band.subfreq[i]),
                                 "-bw", str(UHF_Band.subbw), "-withpn", "-setref"])
            else:
                subprocess.call(["tempo2", "-gr", "fake", "-f", parfile, "-ndobs", str(obs_cad), "-nobsd", str(no_obs),
                                 "-ha", str(maxabs_ha), "-randha", randha, "-start", str(mjd_start), "-end", str(mjd_end),
                                 "-rms", str(1e-3*rms_sub), "-tel", telescope, "-idum", str(args.randnum),
                                 "-freq", str(UHF_Band.subfreq[i]), "-bw", str(UHF_Band.subbw), "-withpn", "-setref"])
            target = os.path.join(args.dir, f"{par}_UHF_{i+1}.tim")
            os.rename(f"{par}.simulate", target)
            timlines.append(f"INCLUDE {target} \n")

    if L_Band.num_tel != 0:
        rms_l = L_Band.calculate_rms()
        for i, rms_sub in enumerate(rms_l):
            if args.randnum is None:
                subprocess.call(["tempo2", "-gr", "fake", "-f", parfile, "-ndobs", str(obs_cad), "-nobsd", str(no_obs),
                                 "-ha", str(maxabs_ha), "-randha", randha, "-start", str(mjd_start), "-end", str(mjd_end),
                                 "-rms", str(1e-3*rms_sub), "-tel", telescope, "-freq", str(L_Band.subfreq[i]),
                                 "-bw", str(L_Band.subbw), "-withpn", "-setref"])
            else:
                subprocess.call(["tempo2", "-gr", "fake", "-f", parfile, "-ndobs", str(obs_cad), "-nobsd", str(no_obs),
                                 "-ha", str(maxabs_ha), "-randha", randha, "-start", str(mjd_start), "-end", str(mjd_end),
                                 "-rms", str(1e-3*rms_sub), "-tel", telescope, "-idum", str(args.randnum),
                                 "-freq", str(L_Band.subfreq[i]), "-bw", str(L_Band.subbw), "-withpn", "-setref"])
            target = os.path.join(args.dir, f"{par}_L_{i+1}.tim")
            os.rename(f"{par}.simulate", target)
            timlines.append(f"INCLUDE {target} \n")

    if S_Band.num_tel != 0:
        rms_s = S_Band.calculate_rms()
        for i, rms_sub in enumerate(rms_s):
            if args.randnum is None:
                subprocess.call(["tempo2", "-gr", "fake", "-f", parfile, "-ndobs", str(obs_cad), "-nobsd", str(no_obs),
                                 "-ha", str(maxabs_ha), "-randha", randha, "-start", str(mjd_start), "-end", str(mjd_end),
                                 "-rms", str(1e-3*rms_sub), "-tel", telescope, "-freq", str(S_Band.subfreq[i]),
                                 "-bw", str(S_Band.subbw), "-withpn", "-setref"])
            else:
                subprocess.call(["tempo2", "-gr", "fake", "-f", parfile, "-ndobs", str(obs_cad), "-nobsd", str(no_obs),
                                 "-ha", str(maxabs_ha), "-randha", randha, "-start", str(mjd_start), "-end", str(mjd_end),
                                 "-rms", str(1e-3*rms_sub), "-tel", telescope, "-idum", str(args.randnum),
                                 "-freq", str(S_Band.subfreq[i]), "-bw", str(S_Band.subbw), "-withpn", "-setref"])
            target = os.path.join(args.dir, f"{par}_S_{i+1}.tim")
            os.rename(f"{par}.simulate", target)
            timlines.append(f"INCLUDE {target} \n")

    with open(f"{par}.tim", "w") as newf:
        newf.writelines(timlines)
        newf.close()

    psr = lt.tempopulsar(parfile=parfile, timfile=f"{par}.tim")
    ltt.make_ideal(psr)
    describe = ""

    if args.rn:
        ltt.add_rednoise(psr, args.rnamp, args.rngamma, components=args.rnc, tspan=args.rntspan, seed=args.randnum)
        describe += "_RN%A{}+#{}".format(int(np.log10(args.rnamp)), int(args.rngamma))
    if args.dmn:
        # ltt.add_dm(psr, args.dmnamp, args.dmngamma, components=args.dmnc, seed=args.randnum)
        ltt.add_dm(psr, args.dmnamp*np.sqrt(12)*np.pi, args.dmngamma, components=args.dmnc, seed=args.randnum)
        # Convert the enterprise DM amplitude to the libstempo DM amplitude
        describe += "_DM%A{}#G{}".format(int(np.log10(args.dmnamp)), int(args.dmngamma))

    psr.savetim(f"{par}_injected.tim")
    lt.purgetim(f"{par}_injected.tim")
    # Delete all lines with reference
    lines = filter(lambda l: 'reference' not in l, open(f"{par}_injected.tim").readlines())
    open(f"{par}{describe}.tim", 'w').writelines(lines)
    psrobject.append(psr)

if args.gwb:
    # ltt.add_gwb(psr, dist=args.psrdist, ngw=args.ngw, seed=args.randnum, flow=args.gwbflow, fhigh=args.gwbfhigh,
    #            gwAmp=args.gwbamp, alpha=-args.gwbalpha, logspacing=args.logspace)
    ltt.createGWB(psrobject, args.gwbamp, args.gwbgam, noCorr=args.nocorr, seed=args.randnum, lmax=args.lmax, turnover=args.turnover,
                  f0=args.gwbf0, beta=args.gwbbeta, power=args.gwbpower, npts=args.gwbnpts, howml=args.gwbhowml)
    gwbdescribe = "_GWB%A{}#G{}".format(int(np.log10(args.gwbamp)), int(args.gwbgam))
    for i, psr in enumerate(psrobject):
        par = parlist[i]
        psrn = psrnlist[i]
        psr.savetim(f"{par}{gwbdescribe}.tim")
        lt.purgetim(f"{par}{gwbdescribe}.tim")
        lines = filter(lambda l: 'reference' not in l, open(f"{par}{gwbdescribe}.tim"))
        open(f"{par}{gwbdescribe}.tim", 'w').writelines(lines)
        if not os.path.exists(f"gwb_test/{psrn}/"):
            os.makedirs(f"gwb_test/{psrn}/")
        shutil.copy(args.parfile[i], f"gwb_test/{psrn}/")
        shutil.copy(f"{par}{gwbdescribe}.tim", f"gwb_test/{psrn}/{psrn}_all.tim")
        shutil.copy(f"gwb_test/{psrn}/{psrn}_all.tim", f"gwb_test/{psrn}/{par}{gwbdescribe}.tim")
        os.remove(f"{par}{gwbdescribe}.tim")
        os.rename(f"{psrn}.tim", f"gwb_test/{psrn}/{psrn}.tim")
        os.rename(f"{psrn}_injected.tim", f"gwb_test/{psrn}/{psrn}_injected.tim")