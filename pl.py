import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

chain1_file = "/cluster/home/liuyang/Sub-Array/TN_parfile/maskbin16-17/source20/NEBw112_UHF4_dmnamp/gwbnb_40_%DM#A-14#G2#C100%GWB#A-14#G4#N600/realization_dmnamp%-13.5#1/chains/chain_1.txt"
chain1_data = pd.read_csv(chain1_file, sep='\s+', header=None, comment='#')
chain2_file = "/cluster/home/liuyang/Sub-Array/TN_parfile/maskbin16-17/source20/NEBw112_UHF4L4_dmnamp/gwbnb_40_%DM#A-14#G2#C100%GWB#A-14#G4#N600/realization_dmnamp%-13.5#1/chains/chain_1.txt"
chain2_data = pd.read_csv(chain2_file, sep='\s+', header=None, comment='#')

log10_A1 = chain1_data.iloc[:, -5].values
gamma1 = chain1_data.iloc[:, -6].values
log10_A2 = chain2_data.iloc[:, -5].values
gamma2 = chain2_data.iloc[:, -6].values

log10A1_median = np.median(log10_A1)
log10A1_low = np.percentile(log10_A1, 16)
log10A1_high = np.percentile(log10_A1, 84)
gamma1_median = np.median(gamma1)
gamma1_low = np.percentile(gamma1, 16)
gamma1_high = np.percentile(gamma1, 84)

A1_linear = 10**log10_A1
A1_median = np.median(A1_linear)
A1_low = np.percentile(A1_linear, 16)
A1_high = np.percentile(A1_linear, 84)

print(f"log10A = {log10A1_median:.3f} [{log10A1_low:.3f}, {log10A1_high:.3f}]")
print(f"gamma = {gamma1_median:.3f} [{gamma1_low:.3f}, {gamma1_high:.3f}]")
print(f"Number of samples: {len(log10_A1)}")

T_obs = 6000 * 86400
n_freqs = 40
f_hz = np.arange(1, n_freqs+1) / T_obs  
seconds_per_year = 365.25 * 86400
f_year = f_hz * seconds_per_year
f_ref = 1.0

h_c1 = np.zeros((len(log10_A1), len(f_year)))
h_c2 = np.zeros((len(log10_A2), len(f_year)))

for i in range(len(log10_A1)):
    A1 = 10**log10_A1[i]
    h_c1[i, :] = A1 * (f_year / f_ref)**((3 - gamma1[i])/2)

for i in range(len(log10_A2)):
    A2 = 10**log10_A2[i]
    h_c2[i, :] = A2 * (f_year / f_ref)**((3 - gamma2[i])/2)

h_c1_median = np.median(h_c1, axis=0)
h_c1_low = np.percentile(h_c1, 16, axis=0)
h_c1_high = np.percentile(h_c1, 84, axis=0)
h_c2_median = np.median(h_c2, axis=0)
h_c2_low = np.percentile(h_c2, 16, axis=0)
h_c2_high = np.percentile(h_c2, 84, axis=0)

plt.figure(figsize=(12, 9))
plt.loglog(f_hz, h_c1_median, color='g', label='Power-Law')
plt.fill_between(f_hz, h_c1_low, h_c1_high, color='g', alpha=0.3, label='Power-Law 68% CI')
#plt.loglog(f_hz, h_c2_median, color='b', label='3ns PL')
#plt.fill_between(f_hz, h_c2_low, h_c2_high, color='b', alpha=0.3, label='3ns PL 68% CI')
plt.xlabel('Frequency [Hz]', fontsize=16)
plt.ylabel('Characteristic Strain', fontsize=16)
plt.title('GWB Power-Law', fontsize=24)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('gwb_powerlaw_pl.pdf', dpi=300, bbox_inches='tight')
plt.show()

print(f"Frequence range: {f_hz[0]:.2e} Hz to {f_hz[-1]:.2e} Hz")
print(f"Frequence range in years: {f_year[0]:.2e} 1/year to {f_year[-1]:.2e} 1/year")
print(f"Characteristic strain range 1: {np.min(h_c1_median):.2e} to {np.max(h_c1_median):.2e}")
#print(f"Characteristic strain range 2: {np.min(h_c2_median):.2e} to {np.max(h_c2_median):.2e}")
