import h5py 
import matplotlib
import numpy as np 
from matplotlib import pyplot as plt 
import functions_X_rays as FN
import scipy 
import csv 
import pandas as pd
import glob
###### reading_mode_depends if the snapshot is written in a single or multiple files. 0 (single), 1 (chuncks)
matplotlib.rcParams.update({'font.size': 18})

h=0.6774
basePath="/home/faldas/SIM/gadget4/examples/DM-L50-N128/output"
filenames=sorted(glob.glob(str(basePath+"/snapshot_*.hdf5")))
reading_mode=0
if np.size(filenames==0):
    filenames=glob.glob(str(basePath+"/snapdir_*"))
    reading_mode=1


def read_snapshot_coordinates(filename, reading_mode=0):
    if reading_mode==0:
        snap=h5py.File(filename, "r")
        params=snap["Header"].attrs
        Coordinates=snap["PartType1"]["Coordinates"][:]
    if reading_mode==1:
        chunks=sorted(glob.glob(str(filename+"/snapshot_*.hdf5")))
        Npart=int(0)
        for chunk in chunks:
            ch=h5py.File(chunk, "r")
            params=ch["Header"].attrs
            NumPart_ThisFile=int(params["NumPart_ThisFile"][1])
            #Coordinates[Npart:Npart+NumPart_ThisFile]= np.array(ch["PartType1"]["Coordinates"][:])
            if Npart==0:
                Coordinates= np.array(ch["PartType1"]["Coordinates"][:])
            else:
                Coordinates= np.vstack((Coordinates, np.array(ch["PartType1"]["Coordinates"][:])))
            Npart +=NumPart_ThisFile
            #print(chunk, np.shape(Coordinates))
            #print(Coordinates)
    return Coordinates, params



        
print(filenames)
def plot_halo( rel_pos, snap_num, ax1,ax2, params,  h, HaloPos, ranges, ax1_bins, ax2_bins):
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
    #snap.clear()
    del fig, ax, x1, x2, statistic, x_edge, y_edge, binnumber

def plot_snaps(filenames, reading_mode):
    ax0_min, ax1_min,ax2_min, ax0_max, ax1_max, ax2_max =compute_limits(filenames, reading_mode)
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
        Coordinates, params= read_snapshot_coordinates(filename, reading_mode)
        #snap=h5py.File(filename, "r")
        HaloPos=np.mean(Coordinates, axis=0, dtype=np.float64).flatten()
        if reading_mode==0:
            snap_num=filename.split(".")[0][-3:]
        if reading_mode==1:
            snap_num=filename[-3:]

        print(f"The snap number is {snap_num}")
        #params=snap["Header"].attrs
        distances, rel_pos=FN.distances_from_center(Coordinates, HaloPos, params["BoxSize"])
        h=0.6774
        plot_halo( rel_pos, snap_num, 0,1, params, h, HaloPos, rangesxy, ax1_bins, ax01_bins)
        plot_halo( rel_pos, snap_num, 1,2, params, h, HaloPos, rangesyz, ax1_bins, ax12_bins)
        plot_halo( rel_pos, snap_num, 0,2, params, h, HaloPos, rangesxz, ax1_bins, ax02_bins)
        
        #Coordinates.clear()
        del Coordinates, rel_pos

def compute_limits(filenames, reading_mode):
    iCoordinates, params=read_snapshot_coordinates(filenames[0], reading_mode)
    #isnap=h5py.File(filenames[0], "r")
    #params=isnap["Header"].attrs
    iHaloPos=np.mean(iCoordinates, axis=0, dtype=np.float64).flatten()
    #print(iCoordinates)
    #print(iHaloPos)
    idistances, irel_pos=FN.distances_from_center(iCoordinates, iHaloPos, params["BoxSize"])
    fCoordinates, params=read_snapshot_coordinates(filenames[-1], reading_mode)
    #fsnap=h5py.File(filenames[-1], "r")
    fHaloPos=np.mean(fCoordinates, axis=0, dtype=np.float64).flatten()
    fdistances, frel_pos=FN.distances_from_center(fCoordinates, fHaloPos, params["BoxSize"])
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
    del  iCoordinates, fCoordinates, fdistances, frel_pos, idistances, irel_pos
    return ax0_min, ax1_min, ax2_min, ax0_max, ax1_max, ax2_max

plot_snaps(filenames, reading_mode)


