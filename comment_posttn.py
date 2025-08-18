#!/usr/bin/env python
"""Tidy up posttn par files by commenting out all JUMP and TN parameters. Written by Yang Liu (liuyang@shao.ac.cn)."""

import argparse
import os
import glob

parser = argparse.ArgumentParser(description='Tidy up posttn par files by commenting out all JUMP and TN parameters.'
                                             'Written by Yang Liu (liuyang@shao.ac.cn).')
parser.add_argument('-p', '--parfile', type=str, default=[], nargs='+',
                    help='Parameter files for pulsars used in simulation')
parser.add_argument('-d', '--datadir', type=str, default=None, help='Path to the directory containing the par files')
parser.add_argument('-j', '--jump', action='store_true', help='Comment out the JUMP parameters if called')
parser.add_argument('-w', '--white', action='store_true', help='Comment out the WN parameters if called')
parser.add_argument('-r', '--red', action='store_true', help='Comment out the RN/DM parameters if called')


def comment_posttn(parfile, jump=True, wn=True, rndm=True):
    """Comment out the JUMP and TN parameters in the posttn parfiles."""
    if "-posttn" in parfile:
        parlines = []
        with open(parfile) as f:
            for line in f:
                e = line.split()
                if jump and "JUMP" in line:
                    parlines.append("# "+line)
                elif wn and e[0].startswith("TNE"):  # ("TNEF" in e[0] or "TNEQ" in e[0]):
                    parlines.append("# " + line)
                elif rndm and (e[0].startswith("TNRed") or e[0].startswith("TNDM")):
                    parlines.append("# " + line)
                elif e[0] == "ECC" and len(e) > 3 and float(e[2]) == 1:
                    newline = e[0] + "            " + e[1] + "   " + e[3] + "\n"
                    parlines.append(newline)
                elif e[0] == "ECC" and len(e) == 3 and float(e[2]) == 1:
                    newline = e[0] + "            " + e[1] + "\n"
                    parlines.append(newline)
                else:
                    parlines.append(line)
        newpar = parfile.split("-posttn")[0]+".par"
        with open(newpar, "w") as newf:
            newf.writelines(parlines)
        os.remove(parfile)
        return newpar
    else:
        return parfile


args = parser.parse_args()
if args.datadir is not None:
    posttn_files = sorted(glob.glob(os.path.join(args.datadir, "*.par")))
    par_files = sorted(comment_posttn(pfs, args.jump, args.white, args.red) for pfs in posttn_files)
else:
    par_files = sorted(comment_posttn(pfs, args.jump, args.white, args.red) for pfs in args.parfile)
