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

parser = argparse.ArgumentParser(description='ToA simulate wrapper for generating simulated ToAs with tempo2 fake plugin. Red noise, DM noise, GWB are added with libstempo. Written by Yang Liu (liuyang@shao.ac.cn).')
parser.add_argument('-p', '--parfile', type=str, default=[], nargs='+', help='Parameter files for pulsars used in simulation')
parser.add_argument('-d', '--datadir', type=str, default=None, help='Path to the directory containing the par files.')
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
parser.add_argument('--refflux', '--reference-flux', type=float, default=1, help='The reference flux in micro-Jy at reference frequency') # Add coefficient
parser.add_argument('--refgamma', '--reference-gamma', type=float, default=1.6, help='The reference spectral index')
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
parser.add_argument('--timd', '--tim-dir', type=str, default='tims/', help='The relative path to the directory for output simulation tim files')
parser.add_argument('--resd', '--result-dir', type=str, default='results/', help='The relative path to the directory for results')
# /cluster/home/liuyang/Sub-Array

class Band:
    def __init__(self, freq, bw, num_tel, num_sub):
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
        rms = args.refsig/self.ratio_tel*np.sqrt(self.num_sub) * (self.subfreq/args.reffreq)**gamma_p
        self.rms = rms
        return self.rms

def tempo2_fake_simulate(parfile, obs_cad, no_obs, maxabs_ha, randha, mjd_start, mjd_end, rms_sub, telescope,
                         subfreq, subbw, randnum=None):
    command = ["tempo2", "-gr", "fake", "-f", parfile, "-ndobs", str(obs_cad), "-nobsd", str(no_obs), "-ha",
               str(maxabs_ha), "-randha", randha, "-start", str(mjd_start), "-end", str(mjd_end), "-rms",
               str(1e-3 * rms_sub), "-tel", telescope, "-freq", str(subfreq), "-bw", str(subbw), "-withpn", "-setref"]
    if randnum is not None:
        command.extend(["-idum", str(randnum)])
    subprocess.run(command, check=True)

args = parser.parse_args()
if args.rha:
    randha = "y"
else:
    randha = "n"
if args.datadir is not None:
    par_files = sorted(glob.glob(os.path.join(args.datadir, "*.par")))
    datadir = args.datadir
else:
    par_files = args.parfile
    datadir = ""

parlist = [parfile.split(".", 1)[0].split("/")[-1] for parfile in par_files]

if not os.path.exists(datadir+args.timd):
    os.makedirs(datadir+args.timd)

for i, parfile in enumerate(par_files):
    dirpar = parfile.split(".", 1)[0]
    timlines = ["FORMAT 1 \n"]

    if args.nuhfb != 0:
        UHF_Band = Band(args.cfrequhf, args.bwuhf, args.nuhfb, args.nsbuhf)
        rms_uhf = UHF_Band.calculate_rms(gamma_p=args.refgamma)
        for j, rms_sub in enumerate(rms_uhf):
            tempo2_fake_simulate(parfile, args.cad, args.nobs, args.maxha, randha, args.mjds, args.mjde, rms_sub,
                                 args.tel, UHF_Band.subfreq[j], UHF_Band.subbw, args.randnum)
            target = os.path.join(args.timd, f"{parlist[i]}_UHF_{j+1}.tim")
            os.rename(datadir+f"{parlist[i]}.simulate", datadir+target)
            timlines.append(f"INCLUDE {target} \n")

    if args.nlb != 0:
        L_Band = Band(args.cfreql, args.bwl, args.nlb, args.nsbl)
        rms_l = L_Band.calculate_rms(gamma_p=args.refgamma)
        for j, rms_sub in enumerate(rms_l):
            tempo2_fake_simulate(parfile, args.cad, args.nobs, args.maxha, randha, args.mjds, args.mjde, rms_sub,
                                 args.tel, L_Band.subfreq[j], L_Band.subbw, args.randnum)
            target = os.path.join(args.timd, f"{parlist[i]}_L_{j+1}.tim")
            os.rename(datadir+f"{parlist[i]}.simulate", datadir+target)
            timlines.append(f"INCLUDE {target} \n")

    if args.nsb != 0:
        S_Band = Band(args.cfreqs, args.bws, args.nsb, args.nsbs)
        rms_s = S_Band.calculate_rms(gamma_p=args.refgamma)
        for j, rms_sub in enumerate(rms_s):
            tempo2_fake_simulate(parfile, args.cad, args.nobs, args.maxha, randha, args.mjds, args.mjde, rms_sub,
                                 args.tel, S_Band.subfreq[j], S_Band.subbw, args.randnum)
            target = os.path.join(args.timd, f"{parlist[i]}_S_{j+1}.tim")
            os.rename(datadir+f"{parlist[i]}.simulate", datadir+target)
            timlines.append(f"INCLUDE {target} \n")

    with open(f"{dirpar}.tim", "w") as newf:
        newf.writelines(timlines)

    psr = lt.tempopulsar(parfile=parfile, timfile=f"{dirpar}.tim")
    ltt.make_ideal(psr)
    describe = ""

    if args.rn:
        ltt.add_rednoise(psr, args.rnamp, args.rngamma, components=args.rnc, tspan=args.rntspan, seed=args.randnum)
        describe += "_RN%A{}#G{}".format(int(np.log10(args.rnamp)), int(args.rngamma))
    if args.dmn:
        # ltt.add_dm(psr, args.dmnamp, args.dmngamma, components=args.dmnc, seed=args.randnum)
        ltt.add_dm(psr, args.dmnamp*1400*1400*2.41e-4, args.dmngamma, components=args.dmnc, seed=args.randnum)
        # Convert the enterprise DM amplitude to the libstempo DM amplitude
        # ltt.add_dm(psr, args.dmnamp*np.sqrt(12)*np.pi, args.dmngamma, components=args.dmnc, seed=args.randnum)
        # Convert the run_enterprise DM amplitude to the libstempo DM amplitude
        describe += "_DM%A{}#G{}".format(int(np.log10(args.dmnamp)), int(args.dmngamma))

    psr.savetim(f"{dirpar}_injected.tim")
    lt.purgetim(f"{dirpar}_injected.tim")
    # Delete all lines with reference
    lines = filter(lambda l: 'reference' not in l, open(f"{dirpar}_injected.tim").readlines())
    with open(f"{dirpar}{describe}.tim", 'w') as file:
        file.writelines(lines)

if args.gwb:
    psrobject = [lt.tempopulsar(parfile=pf, timfile="{}_injected.tim".format(pf.split(".", 1)[0])) for pf in par_files]
    ltt.createGWB(psrobject, args.gwbamp, args.gwbgam, noCorr=args.nocorr, seed=args.randnum, lmax=args.lmax,
                  turnover=args.turnover, f0=args.gwbf0, beta=args.gwbbeta, power=args.gwbpower, npts=args.gwbnpts,
                  howml=args.gwbhowml)
    gwbdescribe = "_GWB%A{}#G{}".format(int(np.log10(args.gwbamp)), int(args.gwbgam))
    if not os.path.exists(datadir+f"{args.resd}/"):
        os.makedirs(datadir+f"{args.resd}/")
    for k, psr in enumerate(psrobject):
        par = parlist[k]
        psrn = par.split("_")[-1]
        psr.savetim(f"{dirpar}{gwbdescribe}.tim")
        lt.purgetim(f"{dirpar}{gwbdescribe}.tim")
        lines = filter(lambda l: 'reference' not in l, open(f"{dirpar}{gwbdescribe}.tim").readlines())
        open(f"{dirpar}{gwbdescribe}.tim", 'w').writelines(lines)
        if not os.path.exists(datadir+f"{args.resd}/{psrn}/"):
            os.makedirs(datadir+f"{args.resd}/{psrn}/")
        # Move par and tim files to the right directory and rename them according to enterprise standard
        shutil.copy(par_files[k], datadir+f"{args.resd}/{psrn}/")
        shutil.copy(f"{dirpar}{gwbdescribe}.tim", datadir+f"{args.resd}/{psrn}/{psrn}_all.tim")
        shutil.move(f"{dirpar}{gwbdescribe}.tim", datadir+f"{args.resd}/{psrn}/{par}{gwbdescribe}.tim")
        os.rename(datadir+f"{psrn}.tim", datadir+f"{args.resd}/{psrn}/{psrn}.tim")
        os.rename(datadir+f"{psrn}_injected.tim", datadir+f"{args.resd}/{psrn}/{psrn}_injected.tim")

if not os.path.exists(datadir+"chains/"):
    os.makedirs(datadir+"chains/")

for name in parlist:
    with open(f"{datadir}/chains/noisefiles/{name}.json", 'w') as file:
        file.writelines('{ \n')
        file.writelines(f'    "{name}_efac": 1,\n')
        file.writelines(f'    "{name}_log10_tnequad": -10\n')
        file.writelines('}')