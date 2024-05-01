import h5py 
import matplotlib
import numpy as np 
from matplotlib import pyplot as plt 
import functions_X_rays as FN
import scipy 
import csv 
import pandas as pd
import glob
matplotlib.rcParams.update({'font.size': 18})

h=0.6774
filenames=sorted(glob.glob("/data01/faldas/DM-L136-N512-Halo59/output/snapshot_*.hdf5"))
print(filenames)
def plot_halo(snap, rel_pos, snap_num, ax1,ax2, params,  h, HaloPos, ranges, ax1_bins, ax2_bins):
    x1=rel_pos[:,ax1]
    x2=rel_pos[:,ax2]
    statistic, x_edge, y_edge, binnumber=scipy.stats.binned_statistic_2d(x1, x2, np.ones_like(x1)*params["MassTable"][1]*1E10, statistic='sum', bins=(ax1_bins,ax2_bins), range=ranges)
    fig, ax=plt.subplots(figsize=(10,12))
    plt.style.use('dark_background')
    mesh=plt.pcolormesh(statistic.T, norm=matplotlib.colors.LogNorm(vmin=1E8, vmax=3.1E11), cmap="magma")
    plt.text(20,ax2_bins+30,"z= "+str(round(params["Redshift"], 2)), fontsize=20 )
    plt.axis("off")
    ax.hlines(y=ax2_bins, xmin=600, xmax=800, linewidth=6, color='w')
    lsize=(ranges[0][1]-ranges[0][0])/800*200
    plt.text(610,ax2_bins+20, str(int(lsize))+" ckpc/h", fontsize=20 )

    cbar=fig.colorbar(mesh, pad=0.0, location="bottom")
    cbar.set_label(r"Projected Particle Mass ($M_{\odot}$)")
    plt.savefig("ProjCM_"+str(ax1)+str(ax2)+"/snap_"+snap_num+".png", dpi=300)
    plt.close()
    snap.clear()
    del fig, ax, x1, x2, statistic, x_edge, y_edge, binnumber

def plot_snaps(filenames):
    ax0_min, ax1_min,ax2_min, ax0_max, ax1_max, ax2_max =compute_limits(filenames)
    rangesxy=[[ax0_min, ax0_max],[ax1_min, ax1_max]]
    rangesxz=[[ax0_min, ax0_max],[ax2_min, ax2_max]]
    rangesyz=[[ax1_min, ax1_max],[ax2_min, ax2_max]]
    ax0_ax1_rate=(ax0_max-ax0_min)/(ax1_max-ax1_min)
    ax0_ax2_rate=(ax0_max-ax0_min)/(ax2_max-ax2_min)
    ax1_ax2_rate=(ax1_max-ax1_min)/(ax2_max-ax2_min)
    
    ax1_bins=800
    ax01_bins=round(ax0_ax1_rate*ax1_bins,0)
    ax02_bins=round(ax0_ax2_rate*ax1_bins,0)
    ax12_bins=round(ax1_ax2_rate*ax1_bins,0)

    for filename in filenames:
        snap=h5py.File(filename, "r")
        HaloPos=np.mean(snap["PartType1"]["Coordinates"][:], axis=0, dtype=np.float64).flatten()
        snap_num=filename.split(".")[0][-3:]
        print(f"The snap number is {snap_num}")
        params=snap["Header"].attrs
        distances, rel_pos=FN.distances_from_center(snap['PartType1']['Coordinates'][:], HaloPos, params["BoxSize"])
        h=0.6774
        plot_halo(snap, rel_pos, snap_num, 0,1, params, h, HaloPos, rangesxy, ax1_bins, ax01_bins)
        plot_halo(snap, rel_pos, snap_num, 1,2, params, h, HaloPos, rangesyz, ax1_bins, ax12_bins)
        plot_halo(snap, rel_pos, snap_num, 0,2, params, h, HaloPos, rangesxz, ax1_bins, ax02_bins)
        
        snap.clear()
        del snap, rel_pos

def compute_limits(filenames):
    isnap=h5py.File(filenames[0], "r")
    params=isnap["Header"].attrs
    iHaloPos=np.mean(isnap["PartType1"]["Coordinates"][:], axis=0, dtype=np.float64).flatten()
    idistances, irel_pos=FN.distances_from_center(isnap['PartType1']['Coordinates'][:], iHaloPos, params["BoxSize"])
    fsnap=h5py.File(filenames[-1], "r")
    fHaloPos=np.mean(fsnap["PartType1"]["Coordinates"][:], axis=0, dtype=np.float64).flatten()
    fdistances, frel_pos=FN.distances_from_center(fsnap['PartType1']['Coordinates'][:], fHaloPos, params["BoxSize"])
    ix0=irel_pos[:,0]
    ix1=irel_pos[:,1]
    ix2=irel_pos[:,2]
    fx0=frel_pos[:,0]
    fx1=frel_pos[:,1]
    fx2=frel_pos[:,2]
    ax0_max  = np.maximum(np.max(ix0), np.max(fx0), dtype=np.float64)
    ax0_min  = np.minimum(np.min(ix0), np.min(fx0), dtype=np.float64)
    ax1_max  = np.maximum(np.max(ix1), np.max(fx1), dtype=np.float64)
    ax1_min  = np.minimum(np.min(ix1), np.min(fx1), dtype=np.float64)
    ax2_max  = np.maximum(np.max(ix2), np.max(fx2), dtype=np.float64)
    ax2_min  = np.minimum(np.min(ix2), np.min(fx2), dtype=np.float64)
    del  isnap, fsnap, fdistances, frel_pos, idistances, irel_pos
    return ax0_min, ax1_min, ax2_min, ax0_max, ax1_max, ax2_max

plot_snaps(filenames)


