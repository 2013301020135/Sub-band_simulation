import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

chain_file = "/cluster/home/liuyang/Sub-Array/TN_parfile/source50/nRNCdT_UHF4L4_rlzno/gwbnb_50_%DM#A-13#G2#C100%GWB#A-14#G4#N600/realization_rlzno%12#3/fs/chain_1.txt"
chain_data = pd.read_csv(chain_file, sep='\s+', header=None, comment='#')

rho_samples = chain_data.iloc[:, -54:-4].values
rho_linear = 10**rho_samples

T_obs = 6000 * 86400
n_freqs = 50
frequencies = np.arange(1, n_freqs+1) / T_obs
seconds_per_year = 365.25 * 86400
f_year = frequencies * seconds_per_year

h_c_samples = np.zeros_like(rho_samples)

for i in range(len(frequencies)):
    f = frequencies[i]
    h_c_samples[:, i] = rho_linear[:, i] * np.sqrt(12 * np.pi**2 * f**3 * T_obs)

h_c_median = np.median(h_c_samples, axis=0)
h_c_low = np.percentile(h_c_samples, 16, axis=0)
h_c_high = np.percentile(h_c_samples, 84, axis=0)

def power_law(f, log10A, gamma):
    """Power-law model: h_c(f) = 10^log10A * (f/f_ref)^((3-gamma)/2)
    where f_ref = 1/year"""
    A = 10**log10A
    return A * (f)**((3 - gamma)/2)


def fit_power_law(f_data, hc_data):
    """Fit power-law and return parameters and errors"""
    f_ref = 1.0  # 1/year
    # Exclude points around 1-year frequency [1/1.2, 1/0.8] year^{-1}
    # mask_year = (f_data < 1/1.2) | (f_data > 1/0.8)
    mask_high_freq = np.ones_like(f_data, dtype=bool)
    #mask_high_freq[15], mask_high_freq[16] = False, False
    #mask_high_freq[-1] = False  # Remove the last point (highest frequency)
    mask = mask_high_freq
    
    print("Excluding points around 1-year frequency and the highest frequency point")
    print(f"  Original data points: {len(f_data)}")
    print(f"  Data points after exclusion: {np.sum(mask)}")
    
    f_fit = f_data[mask]
    hc_fit = hc_data[mask]   
    log10A_guess = np.log10(np.median(hc_fit))
    gamma_guess = 13/3

    try:
        popt, pcov = curve_fit(power_law, f_fit, hc_fit, p0=[log10A_guess, gamma_guess], maxfev=5000)
        perr = np.sqrt(np.diag(pcov))
        hc_pred = power_law(f_fit, *popt)
        residuals = hc_fit - hc_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((hc_fit - np.mean(hc_fit))**2)
        r_squared = 1 - (ss_res / ss_tot)
        print("  Fitting results:")
        print(f"    log10A = {popt[0]:.3f} ± {perr[0]:.3f}")
        print(f"    gamma = {popt[1]:.3f} ± {perr[1]:.3f}")
        print(f"    R² = {r_squared:.3f}")
        return popt, perr, pcov, r_squared, mask, f_fit, hc_pred
    except Exception as e:
        print(f"  Fitting failed: {e}")
        return None, None, None, None, mask, None, None


def generate_powerlaw_confidence_interval(f_values, popt, pcov, n_samples=1000):
    """Generate confidence intervals for power-law fit using Monte Carlo sampling"""
    # Generate parameter samples from multivariate normal distribution
    param_samples = np.random.multivariate_normal(popt, pcov, n_samples)   
    # Calculate power-law values for all samples and frequencies
    powerlaw_samples = np.zeros((n_samples, len(f_values)))
    for i, params in enumerate(param_samples):
        powerlaw_samples[i, :] = power_law(f_values, *params)

    powerlaw_median = np.median(powerlaw_samples, axis=0)
    powerlaw_low = np.percentile(powerlaw_samples, 16, axis=0)
    powerlaw_high = np.percentile(powerlaw_samples, 84, axis=0)
    return powerlaw_median, powerlaw_low, powerlaw_high


popt, perr, pcov, r_squared, mask, f_fit, hc_pred = fit_power_law(f_year, h_c_median)

plt.figure(figsize=(12, 9))
plt.loglog(frequencies, h_c_median, 'o-', color='b', markersize=4, label='Free Spectrum')
plt.fill_between(frequencies, h_c_low, h_c_high, color='b', alpha=0.3, label='Free Spectrum 68% CI')
plt.loglog(frequencies[mask], h_c_median[mask], 's', color='r', markersize=4, label='Data used for fitting')

if popt is not None:
    f_plot_year = np.logspace(np.log10(f_year[mask].min()), np.log10(f_year[mask].max()), 100)
    f_plot_hz = f_plot_year / seconds_per_year  # Convert to Hz for plotting
    powerlaw_median, powerlaw_low, powerlaw_high = generate_powerlaw_confidence_interval(f_plot_year, popt, pcov)
    plt.loglog(f_plot_hz, powerlaw_median, '--', color='r', linewidth=2, label=f'Power-law fit: log10A={popt[0]:.2f}, γ={popt[1]:.2f}')
    plt.fill_between(f_plot_hz, powerlaw_low, powerlaw_high, color='r', alpha=0.3, label='Power-law 68% CI')

# plt.axvspan(1/1.2, 1/0.8, alpha=0.2, color='gray', label='Excluded year frequency region')
# plt.axvline(f_year[-1], color='red', linestyle=':', alpha=0.7, label='Excluded highest frequency')
plt.xlabel('Frequency [Hz]', fontsize=16)
plt.ylabel('Characteristic Strain', fontsize=16)
plt.title('Power-law Fit to Free Spectrum', fontsize=24)
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig('/cluster/home/liuyang/Sub-Array/TN_parfile/source50/nRNCdT_UHF4L4_rlzno/gwbnb_50_%DM#A-13#G2#C100%GWB#A-14#G4#N600/realization_rlzno%12#3/fs/powerlaw_fit_to_free_spectrum.pdf', dpi=300, bbox_inches='tight')
plt.show()
