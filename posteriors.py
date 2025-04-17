#!/usr/bin/env python
"""Script for calculating posteriors of parameters in the MCMC fit. Written by Yang Liu (liuyang@shao.ac.cn)."""

import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Script for calculating posteriors of parameters in the MCMC fit.'
                                             'Written by Yang Liu (liuyang@shao.ac.cn).')
parser.add_argument('-c', '--chain', type=str, required=True, help='Chain files from MCMC fit')
parser.add_argument('-b', '--burn', type=float, default=0.5, help='Fraction of chains to be burned')
args = parser.parse_args()


def get_stats(samps):
    ul68 = np.percentile(samps, 84.1, axis=0)
    ll68 = np.percentile(samps, 15.9, axis=0)
    ul = np.percentile(samps, 97.5, axis=0)
    ll = np.percentile(samps, 2.5, axis=0)
    ulm = np.max(samps, axis=0)
    llm = np.min(samps, axis=0)
    means = np.mean(samps, axis=0)
    stds = np.std(samps, axis=0)
    medians = np.median(samps, axis=0)
    return means, stds, llm, ll, ll68, medians, ul68, ul, ulm


def get_posteriors(chain, pars, burn, posterior):
    data = np.loadtxt(chain)
    if burn < 0:
        burn = 0
    elif burn > 1:
        burn = 1
    params = open(pars, 'r').readlines()
    with open(posterior, "w") as resf:
        form = "{:20s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s}"
        s = form.format("Parameters", "max-like", "mean", "std", "2.5%", "15.9%", "50%", "84.1%", "97.5%")
        resf.write(s + "\n")
        burned = data[int(len(data)*burn):, :]
        means, stds, llm, ll97, ll68, medians, ul68, ul97, ulm = get_stats(burned[:, :-4])
        imax = np.argmax(burned[:, -4])
        pmax = burned[imax, :-4]
        for p, v, mean, std, l, ll, median, ul, u in zip(params, pmax, means, stds, ll97, ll68, medians, ul68, ul97):
            form = "{:20s} {:< 10g} {:< 10g} {:< 10g} {:< 10g} {:< 10g} {:< 10g} {:< 10g} {:< 10g}"
            s = form.format(p, v, mean, std, l, ll, median, ul, u)
            resf.write(s + "\n")


chaindir = args.chain.rsplit("/", 1)[0]
pars = chaindir+"/pars.txt"
posterior = chaindir+"/posterior.txt"
get_posteriors(args.chain, pars, args.burn, posterior)