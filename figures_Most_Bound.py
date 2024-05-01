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
def plot_halo(snap,snap_num, ax1,ax2, params,  h, HaloPos, ranges=None, ax1_bins=800, ax2_bins=None):
    distances, rel_pos=FN.distances_from_center(snap['PartType1']['Coordinates'][:], HaloPos, params["BoxSize"])
    #x1=snap["PartType1"]["Coordinates"][:,ax1]
    #x2=snap["PartType1"]["Coordinates"][:,ax2]
    x1=rel_pos[:,ax1]
    x2=rel_pos[:,ax2]
    ax1_max  = np.max(x1)
    ax1_min  = np.min(x1)
    ax2_max  = np.max(x2)
    ax2_min  = np.min(x2)
    print(f"Computed bounds: {[[ax1_min, ax1_max], [ax2_min, ax2_max]]}")
    ax1_ax2_rate=(ax1_max-ax1_min)/(ax2_max-ax2_min)
    if ax2_bins==None:
        ax2_bins=round(ax1_ax2_rate*800,0)
    statistic, x_edge, y_edge, binnumber=scipy.stats.binned_statistic_2d(x1, x2, np.ones_like(x1)*params["MassTable"][1]*1E10, statistic='sum', bins=(ax1_bins,ax2_bins), range=ranges)
    #print("Start to plot")
    fig, ax=plt.subplots(figsize=(10,12))
    plt.style.use('dark_background')
    #ax.set_facecolor("k")
    mesh=plt.pcolormesh(statistic.T, norm=matplotlib.colors.LogNorm(vmin=1E8, vmax=3.1E11), cmap="magma")
    #xpos=800/(np.max(x_edge)-np.min(x_edge))*(MOST_MASSIVE[ax1]-np.min(x_edge))
    #ypos=800/(np.max(y_edge)-np.min(y_edge))*(MOST_MASSIVE[ax2]-np.min(y_edge))
    #plt.plot(xpos, ypos, "o:b", ms=10)
    plt.text(20,ax2_bins+30,"z= "+str(round(params["Redshift"], 2)), fontsize=20 )
    plt.axis("off")
    ax.hlines(y=ax2_bins, xmin=600, xmax=800, linewidth=6, color='w')
    if ranges:
        lsize=(ranges[0][1]-ranges[0][0])/800*200
    else:
        lsize=(ax1_max-ax1_min)/800*200
    plt.text(610,ax2_bins+20, str(int(lsize))+" ckpc/h", fontsize=20 )

    #plt.axhline(y=830, color='k')
    #plt.text(360, 860,"200 Mpc" )
    cbar=fig.colorbar(mesh, pad=0.0, location="bottom")
    cbar.set_label(r"Projected Particle Mass ($M_{\odot}$)")
    plt.savefig("Proj_"+str(ax1)+str(ax2)+"/snap_"+snap_num+".png", dpi=300)
    plt.close()
    snap.clear()
    del fig, ax, x1, x2, statistic, x_edge, y_edge, binnumber
    return [[ax1_min, ax1_max], [ax2_min, ax2_max]], ax1_bins,ax2_bins

def plot_snaps(filenames):
    FOF=h5py.File("/data01/faldas/DM-L136-N512-Halo59/output/fof_subhalo_tab_267.hdf5", "r")
    BCGID=FOF['Group']['GroupFirstSub'][0]
    IDMostBound=FOF["Subhalo"]['SubhaloIDMostbound'][BCGID]
    for filename in filenames[:-1]:
        snap=h5py.File(filename, "r")
        HaloPos=snap["PartType1"]['Coordinates'][snap["PartType1"]["ParticleIDs"][:]==IDMostBound].flatten()
        snap_num=filename.split(".")[0][-3:]
        print(f"The snap number is {snap_num}")
        params=snap["Header"].attrs
        #HaloPos=np.array([0.5, 0.5, 0.5])*params["BoxSize"]
        h=0.6774
        if snap_num=="000":
            rangesxy, ax1_bins, ax2_bins=plot_halo(snap, snap_num,  0,1, params, h, HaloPos)
            print(rangesxy)
            rangesyz, ax1_bins, ax2_bins=plot_halo(snap, snap_num, 1,2, params, h, HaloPos)
            rangesxz, ax1_bins, ax2_bins=plot_halo(snap, snap_num, 0,2, params, h, HaloPos)

        else:
            print(rangesxy)
            plot_halo(snap, snap_num, 0,1, params, h, HaloPos, rangesxy, ax1_bins, ax2_bins)
            plot_halo(snap, snap_num, 1,2, params, h, HaloPos, rangesyz, ax1_bins, ax2_bins)
            plot_halo(snap, snap_num, 0,2, params, h, HaloPos, rangesxz, ax1_bins, ax2_bins)
        snap.clear()
        del snap


plot_snaps(filenames)


