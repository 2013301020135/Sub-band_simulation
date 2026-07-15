import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ==================== 用户配置 ====================
T_obs = 6000 * 86400                     # 观测时长 [秒]
n_freqs = 15
freqs = np.arange(1, n_freqs + 1) / T_obs   # 频率 [Hz]
# =================================================

# ========== 读取数据 ==========
chain_file = "/cluster/home/liuyang/Sub-Array/TN_parfile/source50/nRNfitCdT_UHF4L4_rlzno/gwbnb_50_%RN#A-14#G4#C100%DM#A-13#G2#C100%GWB#A-14#G4#N600/realization_rlzno%1#3/fs/chain_1.txt"
#chain = pd.read_csv(chain_file, sep='\s+', header=None, comment='#')
chain = np.loadtxt(chain_file)

fobs = freqs
hc_bg = 10**np.transpose(chain[:,:15])
#log10hc_grid = log10hc_gridog10rho + (0.5 * np.log10(12 * np.pi ** 2 * fbin_NG15**3 * 16.03 * YR))[:,np.newaxis]
#print(hc_bg.shape)
#sample_idx = reals['chain_row_idx']
xx = fobs / 1e-9
fig = plt.figure()
ax = fig.add_subplot(111, rasterized=True)
med = np.median(hc_bg, axis=-1)
# plot violins
v1 = ax.violinplot(list(hc_bg), positions=xx, widths=0.6, showextrema=False) #quantiles=[[0.25, 0.75]] * len(xx)
# label the HD spectrum
plt.plot([], [], color='C0', linestyle='solid', 
         label='Hellings-Downs Spectrum', alpha=0.25)

# Make the violins look good B)
for pc in v1['bodies']:
    pc.set_facecolor('grey')
    pc.set_edgecolor('C0')
    pc.set_linestyle('solid')
    pc.set_alpha(0.5)
    pc.set_linewidth(0.5)


yy = med[0] * np.power(xx/xx[0], -2.0/3.0)
ax.plot(xx, yy, 'k--', alpha=0.25, lw=2.0)
ax.plot(xx, med, 'k-', alpha=0.5)

#ax.set_xlim(-8.8, -7.68)
ax.grid(which='both',alpha=0.1)
# plot the median power law
#ax.plot(xx, 0.5*np.log10(pl_med), color='C1', alpha=0.5, lw=2, label='Median Varied Gamma PL')

#ax.minorticks_on()
#ax.tick_params(which='both', direction='in', tick2On=True)
#ax.set_ylim(3e-16, 5e-14)
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12, rotation=0)
plt.legend()
plt.yscale('log')
plt.xscale('log')
#ax.set_ylabel(r'$\log_{10}$(Excess timing delay [s])')
#ax.set_xlabel(r'$\log_{10}$(Frequency [Hz])')
ax.set_xlabel(r'GW Frequency $f_\mathrm{obs}$ [nHz]')
#ax.set_ylabel('Characteristic Strain $h_c$')
ax.set_ylabel(r'$\rho$')

#plt.title(f'sample_idx={sample_idx}')
plt.tight_layout()
plt.savefig("/cluster/home/liuyang/Sub-Array/TN_parfile/source50/nRNfitCdT_UHF4L4_rlzno/gwbnb_50_%RN#A-14#G4#C100%DM#A-13#G2#C100%GWB#A-14#G4#N600/realization_rlzno%1#3/fs/violin.pdf")
plt.show()


print(f"Frequency range: {freqs[0]:.2e} to {freqs[-1]:.2e} Hz")
print(f"hc median range: {np.min(med):.2e} to {np.max(med):.2e}")
