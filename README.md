# Sub-band_simulation
Simulate ToAs in different sub-bands and inject RN, DM, GWB, etc.


## simulate_ToAs.py
usage: simulate_ToAs.py [-h] -p PARFILE [PARFILE ...] [--cad CAD] [--nobs NOBS] [--maxha MAXHA]
                        [--rha] [--mjds MJDS] [--mjde MJDE] [--nuhfb NUHFB] [--nlb NLB] [--nsb NSB]
                        [--nsbuhf NSBUHF] [--nsbl NSBL] [--nsbs NSBS] [--cfrequhf CFREQUHF]
                        [--cfreql CFREQL] [--cfreqs CFREQS] [--bwuhf BWUHF] [--bwl BWL] [--bws BWS]
                        [--narray NARRAY] [--randnum RANDNUM] [--tel TEL] [--refsig REFSIG]
                        [--reffreq REFFREQ] [--refflux REFFLUX] [--rn] [--rnamp RNAMP]
                        [--rngamma RNGAMMA] [--rnc RNC] [--rntspan RNTSPAN] [--dmn]
                        [--dmnamp DMNAMP] [--dmngamma DMNGAMMA] [--dmnc DMNC] [--gwb]
                        [--gwbamp GWBAMP] [--gwbgam GWBGAM] [--nocorr] [--lmax LMAX] [--turnover]
                        [--gwbf0 GWBF0] [--gwbbeta GWBBETA] [--gwbpower GWBPOWER]
                        [--gwbnpts GWBNPTS] [--gwbhowml GWBHOWML] [--dir DIR]

ToA simulate wrapper for generating simulated ToAs with tempo2 fake plugin. Red noise, DM noise,
GWB are added with libstempo. Written by Yang Liu (liuyang@shao.ac.cn).


## Example:
  python ../Sub-band_simulation/simulate_ToAs.py -p J0023+0923.par J0030+0451.par --cad 15 --nobs 2 --maxha 6 --mjds 54000 --mjde 59000 --nuhfb 21 --nlb 22 --nsb 21 --nsbuhf 2 --nsbl 2 --nsbs 2 --narray 64 --refsig 0.01 --reffreq 1300 --refflux 1 --rnc 100 --rnamp 1e-14 --rngamma 4 --dmnc 100 --dmnamp 1e-11 --dmngamma 2 --gwb --gwbamp 1e-14 --gwbgam 4 --tel meerkat


## optional arguments:
  -h, --help            show this help message and exit
  
  -p PARFILE [PARFILE ...], --parfile PARFILE [PARFILE ...]
                        Parameter files for pulsars used in simulation
                        
  --cad CAD, --observation-cadence CAD
                        The number of days between observations
                        
  --nobs NOBS, --no-of-observation NOBS
                        The number of observations on a given day
                        
  --maxha MAXHA, --hour-angle MAXHA
                        The maximum absolute hour angle allowed
                        
  --rha, --random-hour-angle
                        Use random hour angle coverage if called, otherwise use regular hour angle
                        
  --mjds MJDS, --initial-mjd MJDS
                        The initial MJD for the simulated TOAs
                        
  --mjde MJDE, --final-mjd MJDE
                        The final MJD for the simulated TOAs
                        
  --nuhfb NUHFB, --num-uhfband NUHFB
                        Number of arrays in UHF-Band
                        
  --nlb NLB, --num-lband NLB
                        Number of arrays in L-Band
                        
  --nsb NSB, --num-sband NSB
                        Number of arrays in S-Band
                        
  --nsbuhf NSBUHF, --num-uhf-subband NSBUHF
                        Number of sub-bands in UHF-Band
                        
  --nsbl NSBL, --num-l-subband NSBL
                        Number of sub-bands in L-Band
                        
  --nsbs NSBS, --num-s-subband NSBS
                        Number of sub-bands in S-Band
                        
  --cfrequhf CFREQUHF, --central-frequency-uhf CFREQUHF
                        Central frequency of UHF-Band in MHz
                        
  --cfreql CFREQL, --central-frequency-l CFREQL
                        Central frequency of L-Band in MHz
                        
  --cfreqs CFREQS, --central-frequency-s CFREQS
                        Central frequency of S-Band in MHz
                        
  --bwuhf BWUHF, --bandwidth-uhf BWUHF
                        Bandwidth of UHF-Band in MHz
                        
  --bwl BWL, --bandwidth-l BWL
                        Bandwidth of L-Band in MHz
                        
  --bws BWS, --bandwidth-s BWS
                        Bandwidth of S-Band in MHz
                        
  --narray NARRAY, --num-array NARRAY
                        Number of total arrays
                        
  --randnum RANDNUM, --random-number-seed RANDNUM
                        Specify random number seed
                        
  --tel TEL             The name of the telescope
  
  --refsig REFSIG, --reference-sigma REFSIG
                        The rms of Gaussian noise in micro-second when all telescope are in
                        reference frequency
                        
  --reffreq REFFREQ, --reference-frequency REFFREQ
                        The reference frequency in MHz
                        
  --refflux REFFLUX, --reference-flux REFFLUX
                        The reference flux in micro-Jy at reference frequency
                        
  --rn, --red-noise     Inject red noise if called
  
  --rnamp RNAMP, --red-noise-amplitude RNAMP
                        The red noise amplitude
                        
  --rngamma RNGAMMA, --red-noise-gamma RNGAMMA
                        The red noise spectral slope (gamma, positive)
                        
  --rnc RNC, --red-noise-component RNC
                        The number of red noise component
                        
  --rntspan RNTSPAN, --red-noise-tspan RNTSPAN
                        The time span used for red noise injection
                        
  --dmn, --dm-noise     Inject DM noise if called
  
  --dmnamp DMNAMP, --dm-noise-amplitude DMNAMP
                        The DM noise amplitude
                        
  --dmngamma DMNGAMMA, --dm-noise-gamma DMNGAMMA
                        The DM noise spectral slope (gamma, positive)
                        
  --dmnc DMNC, --dm-noise-component DMNC
                        The number of DM noise component
                        
  --gwb, --gw-background
                        Inject gravitational wave background if called
                        
  --gwbamp GWBAMP, --gwb-amplitude GWBAMP
                        The gravitational wave background amplitude
                        
  --gwbgam GWBGAM, --gwb-gamma GWBGAM
                        The gravitational wave background spectral slope (gamma, positive)
                        
  --nocorr, --gwb-no-corr
                        Add red noise with no correlation
                        
  --lmax LMAX, --gwb-lmax LMAX
                        The maximum multipole of GW power decomposition
                        
  --turnover, --gwb-turnover
                        Produce spectrum with turnover at frequency f0
                        
  --gwbf0 GWBF0, --gwb-f0 GWBF0
                        The frequency of spectrum turnover
                        
  --gwbbeta GWBBETA, --gwb-beta GWBBETA
                        The spectral index of power spectrum for f<<f0
                        
  --gwbpower GWBPOWER, --gwb-power GWBPOWER
                        The fudge factor for flatness of spectrum turnover
                        
  --gwbnpts GWBNPTS, --gwb-npts GWBNPTS
                        The number of points used in interpolation
                        
  --gwbhowml GWBHOWML, --gwb-howml GWBHOWML
                        The lowest frequency is 1/(howml * T)
                        
  --dir DIR, --tim-dir DIR
                        The relative path to the simulate directory
