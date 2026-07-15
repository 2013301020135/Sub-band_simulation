import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

chain1_file = "/cluster/home/liuyang/Sub-Array/TN_parfile/maskbin16-17/source20/NEBw112_UHF4_dmnamp/gwbnb_40_%DM#A-14#G2#C100%GWB#A-14#G4#N600/realization_dmnamp%-13.5#1/chains/chain_1.txt"
chain1_data = pd.read_csv(chain1_file, sep='\s+', header=None, comment='#')

chain2_file = "/cluster/home/liuyang/Sub-Array/TN_parfile/maskbin16-17/source20/NEBw112_UHF4L4_dmnamp/gwbnb_40_%DM#A-14#G2#C100%GWB#A-14#G4#N600/realization_dmnamp%-13.5#1/chains/chain_1.txt"
chain2_data = pd.read_csv(chain2_file, sep='\s+', header=None, comment='#')

rho1_samples = chain1_data.iloc[:, -44:-4].values
rho1_linear = 10**rho1_samples
rho2_samples = chain2_data.iloc[:, -44:-4].values
rho2_linear = 10**rho2_samples

T_obs = 6000 * 86400
n_freqs = 40
frequencies = np.arange(1, n_freqs+1) / T_obs
h_c1_samples = np.zeros_like(rho1_samples)
h_c2_samples = np.zeros_like(rho2_samples)

for i in range(len(frequencies)):
    f = frequencies[i]
    #h_c1_samples[:, i] = rho1_linear[:, i] * np.sqrt(f * T_obs)
    #h_c2_samples[:, i] = rho2_linear[:, i] * np.sqrt(f * T_obs)
    h_c1_samples[:, i] = rho1_linear[:, i] * np.sqrt(12 * np.pi**2 * f**3 * T_obs)
    h_c2_samples[:, i] = rho2_linear[:, i] * np.sqrt(12 * np.pi**2 * f**3 * T_obs)
    #h_c1_samples[:, i] = rho1_samples[:, i] * f**(-2/3)
    #h_c2_samples[:, i] = rho2_samples[:, i] * f**(-2/3)

#h_c1_samples = rho1_linear 
h_c1_median = np.median(h_c1_samples, axis=0)
h_c1_low = np.percentile(h_c1_samples, 16, axis=0)
h_c1_high = np.percentile(h_c1_samples, 84, axis=0)
#h_c2_samples = rho2_linear 
h_c2_median = np.median(h_c2_samples, axis=0)
h_c2_low = np.percentile(h_c2_samples, 16, axis=0)
h_c2_high = np.percentile(h_c2_samples, 84, axis=0)

psd1_samples = (h_c1_samples**2) / (12 * np.pi**2 * frequencies**3)
#psd1_samples = rho1_linear**2
psd1_median = np.median(psd1_samples, axis=0)
psd1_low = np.percentile(psd1_samples, 16, axis=0)
psd1_high = np.percentile(psd1_samples, 84, axis=0)
psd2_samples = (h_c2_samples**2) / (12 * np.pi**2 * frequencies**3)
#psd2_samples = rho2_linear**2
psd2_median = np.median(psd2_samples, axis=0)
psd2_low = np.percentile(psd2_samples, 16, axis=0)
psd2_high = np.percentile(psd2_samples, 84, axis=0)

plt.figure(figsize=(12, 9))
plt.loglog(frequencies, h_c1_median, 'o-', color='g', linewidth=2, markersize=4, label='1ns Median')
plt.fill_between(frequencies, h_c1_low, h_c1_high, alpha=0.3, color='g', label='1ns 68% CI')
plt.loglog(frequencies, h_c2_median, 'o-', color='b', linewidth=2, markersize=4, label='3ns Median')
plt.fill_between(frequencies, h_c2_low, h_c2_high, alpha=0.3, color='b', label='3ns 68% CI')
plt.xlabel('Frequency [Hz]', fontsize=16)
plt.ylabel('Characteristic Strain', fontsize=16)
plt.title('GWB Free Spectrum', fontsize=24)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('gwb_free_spectrum_fs.pdf', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 9))
plt.loglog(frequencies, psd1_median, 'o-', color='y', linewidth=2, markersize=4, label='1ns Median')
plt.fill_between(frequencies, psd1_low, psd1_high, alpha=0.3, color='y', label='1ns 68% CI')
plt.loglog(frequencies, psd2_median, 'o-', color='r', linewidth=2, markersize=4, label='3ns Median')
plt.fill_between(frequencies, psd2_low, psd2_high, alpha=0.3, color='r', label='3ns 68% CI')
plt.xlabel('Frequency [Hz]', fontsize=16)
plt.ylabel('Power Spectral Density', fontsize=16)
plt.title('GWB Power Spectral Density', fontsize=24)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('gwb_psd_fs.pdf', dpi=300, bbox_inches='tight')
plt.show()

print(f"Frequence range: {frequencies[0]:.2e} Hz to {frequencies[-1]:.2e} Hz")
print(f"Characteristic strain range 1: {np.min(h_c1_median):.2e} to {np.max(h_c1_median):.2e}")
print(f"Characteristic strain range 2: {np.min(h_c2_median):.2e} to {np.max(h_c2_median):.2e}")
