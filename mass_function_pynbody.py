import pynbody
from matplotlib import pyplot as plt
import numpy as np
import h5py
s = pynbody.load('output/snapshot_019.hdf5'); s.physical_units()
s1 = pynbody.load('/home/franklin/Documents/SIM2024/DM-L136-N256/output/snapshot_019.hdf5'); s1.physical_units()
#s2 = pynbody.load('../DM-L50-N128/output/snapshot_019.hdf5'); s2.physical_units()
s3 = pynbody.load('/home/franklin/Documents/SIM2024/DM-L136-N512/output/snapshot_019.hdf5'); s3.physical_units()
print(s.properties)
print(s1.properties)

#s.properties['sigma8'] = 0.8288
#s1.properties
my_cosmology = pynbody.analysis.hmf.PowerSpectrumCAMB(s, filename='CAMB_WMAP7')
my_cosmology1 = pynbody.analysis.hmf.PowerSpectrumCAMB(s1, filename='CAMB_WMAP7')

m, sig, dn_dlogm = pynbody.analysis.hmf.halo_mass_function(s, log_M_min=10, log_M_max=15, delta_log_M=0.1, kern="ST", pspec=my_cosmology)
m1, sig1, dn_dlogm1 = pynbody.analysis.hmf.halo_mass_function(s1, log_M_min=10, log_M_max=15, delta_log_M=0.1, kern="ST", pspec=my_cosmology1)

bin_center, bin_counts, err = pynbody.analysis.hmf.simulation_halo_mass_function(s, log_M_min=10, log_M_max=15, delta_log_M=0.1)
bin_center1, bin_counts1, err1 = pynbody.analysis.hmf.simulation_halo_mass_function(s1, log_M_min=10, log_M_max=15, delta_log_M=0.1)
#bin_center2, bin_counts2, err2 = pynbody.analysis.hmf.simulation_halo_mass_function(s2, log_M_min=10, log_M_max=15, delta_log_M=0.1)
bin_center3, bin_counts3, err3 = pynbody.analysis.hmf.simulation_halo_mass_function(s3, log_M_min=10, log_M_max=15, delta_log_M=0.1)
boxSizeMpc = 1000.0
boxSizeMpc_c = 400.0

minm = np.log10(20*1.51e9)
minm_c = np.log10(20*9.631e7)

ff = h5py.File("mdpl_full.h5", "r")
fc = h5py.File("smdpl_full.h5", "r")

tag = "sussing_125.z0.000"
bins = ff[tag+'/bins'][:]
tot = ff[tag+'/hist'][:]
cen = ff[tag+'/hist_cen'][:]
sat = ff[tag+'/hist_sat'][:]

tag = "sussing_116.z0.000"
bins_c = fc[tag+'/bins'][:]
tot_c = fc[tag+'/hist'][:]
cen_c = fc[tag+'/hist_cen'][:]
sat_c = fc[tag+'/hist_sat'][:]

fac = (boxSizeMpc**3)*(bins[1]-bins[0])
tot = tot.astype(float)/fac
cen = cen.astype(float)/fac
sat = sat.astype(float)/fac
x = (bins[1:]+bins[:-1])/2.

fac_c = (boxSizeMpc_c**3)*(bins_c[1]-bins_c[0])
tot_c = tot_c.astype(float)/fac_c
cen_c = cen_c.astype(float)/fac_c
sat_c = sat_c.astype(float)/fac_c
x_c = (bins_c[1:]+bins_c[:-1])/2.


plt.clf()

#plt.errorbar(bin_center, bin_counts, yerr=err, fmt='.', capthick=2, elinewidth=1, color='darkgoldenrod', label='DM-L50-N128-MUSIC')
plt.errorbar(bin_center1, bin_counts1, yerr=err1, fmt='.', capthick=2, elinewidth=1, color='blue', label='TGG-L200-N256')
#plt.errorbar(bin_center2, bin_counts2, yerr=err2, fmt='.', capthick=2, elinewidth=1, color='red', label='DM-L50-N128-NGENIC')
plt.errorbar(bin_center3, bin_counts3, yerr=err3, fmt='.', capthick=2, elinewidth=1, color='red', label='TGG-L200-N512')
plt.plot(10**x, tot, '-', color='orange', lw=2, label='MDPL2')
plt.plot(10**x_c, tot_c, '--', color='purple', lw=2, label='SMDPL')
plt.plot(m, dn_dlogm, color='darkmagenta', linewidth=1)
#plt.plot(m1, dn_dlogm1, color='darkmagenta', linewidth=1)

plt.ylabel(r'$\frac{dN}{d\; logM}$ ($h^{3}Mpc^{-3}$)')

#plt.ylabel(r'$\frac{dN}{d\logM}$ ($h^{3}Mpc^{-3}$)')

plt.xlabel('Mass ($h^{-1} M_{\odot}$)')
plt.legend()
plt.xlim((1E10, 1E15))
plt.yscale('log'); plt.xscale('log')

plt.savefig("Mass_function_pynbody.png", dpi=300)
