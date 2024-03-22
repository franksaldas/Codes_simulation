import h5py 
from matplotlib import pyplot as plt 
import numpy as np 
import functions_X_rays as FN
import scipy as sp
import pandas as pd
import matplotlib

HubbleParam=0.6774


def plot_halo(rel_pos, IDS_PART, params, snap_num, ax1, ax2):
    statistic, x_edge, y_edge, binnumber=sp.stats.binned_statistic_2d(rel_pos[IDS_PART,ax1], rel_pos[IDS_PART,ax2], np.ones_like(rel_pos[IDS_PART,ax1])*params["MassTable"][1]*1E10, statistic='sum', bins=(800,800))
    print(f"Start to plot snapshot {snap_num}")
    fig=plt.figure(figsize=(10,12))
    mesh=plt.pcolormesh(statistic.T, norm=matplotlib.colors.LogNorm(), cmap="magma")
    plt.axis("off")
    #plt.axhline(y=830, color='k')
    #plt.text(360, 860,"200 Mpc" )
    cbar=fig.colorbar(mesh, pad=0.0, location="bottom")
    cbar.set_label(r"Projected Particle Mass ($M_{\odot}$)")
    plt.savefig("NFigures/Halo_"+str(Nhalo)+"snap_"+str(snap_num)+"_pj_"+str(ax1)+"_"+str(ax2)+".png", dpi=300)




FOF_filename="../output/fof_subhalo_tab_019.hdf5"
FOF=h5py.File(FOF_filename, "r")

halos_number=[4,8,33,34,74,69,65]
Nhalo=69
fsnap_name="../output/snapshot_019.hdf5"
fsnap=h5py.File(fsnap_name, "r")
params=fsnap["Header"].attrs
print(params.keys())
print(FOF["Header"].attrs.keys())
HaloPos=FOF["Group"]["GroupPos"][Nhalo, :]
Radius=np.maximum(4*FOF["Group"]["Group_R_Crit200"][Nhalo], 4000*HubbleParam)
distances, rel_pos=FN.distances_from_center(fsnap['PartType1']['Coordinates'][:], HaloPos, params["BoxSize"])
IDS_PART=distances<Radius
IDS_sim=fsnap['PartType1']['ParticleIDs'][IDS_PART]
selected=np.in1d(fsnap["PartType1"]["ParticleIDs"][:], IDS_sim)
print(np.array_equal(IDS_PART, selected))
plot_halo(rel_pos, selected, params, 19, 0,1)
plot_halo(rel_pos, selected, params, 19, 0,2)
plot_halo(rel_pos, selected, params, 19, 1,2)

BCGID=FOF['Group']['GroupFirstSub'][Nhalo]
IDMostBound=FOF["Subhalo"]['SubhaloIDMostbound'][BCGID]
#snaps=np.arange(9,0, -1)
snaps=[18]
for i in snaps:
    snap_name="../output/snapshot_0"+str(i)+".hdf5"
    snap=h5py.File(snap_name, "r")
    params_s=snap["Header"].attrs
    selected=np.in1d(snap["PartType1"]["ParticleIDs"][:], IDS_sim)
    HaloPos_s=snap["PartType1"]['Coordinates'][snap["PartType1"]["ParticleIDs"][:]==IDMostBound].flatten()
    distances, rel_pos=FN.distances_from_center(snap['PartType1']['Coordinates'][:], HaloPos_s, params["BoxSize"])
    #plot_halo(rel_pos, selected, params_s, i, 0,1)
    plot_halo(snap["PartType1"]["Coordinates"][:], selected, params_s, i,0,1)
    plot_halo(snap["PartType1"]["Coordinates"][:], selected, params_s, i,0,2)
    plot_halo(snap["PartType1"]["Coordinates"][:], selected, params_s, i,1,2)

INIT_name="../IC-DM-L136-N512.hdf5"
INIT=h5py.File(INIT_name, "r")
params_INIT=INIT["Header"].attrs
selected=np.in1d(INIT["PartType1"]["ParticleIDs"][:], IDS_sim)
HaloPos_INIT=INIT["PartType1"]['Coordinates'][INIT["PartType1"]["ParticleIDs"][:]==IDMostBound].flatten()
plot_halo(INIT["PartType1"]["Coordinates"][:], selected, params_s, "IC",0,1)
plot_halo(INIT["PartType1"]["Coordinates"][:], selected, params_s, "IC",0,2)
plot_halo(INIT["PartType1"]["Coordinates"][:], selected, params_s, "IC",1,2)
POS_INIT=INIT["PartType1"]["Coordinates"][selected]
PCOOR=pd.DataFrame(POS_INIT/INIT["Header"].attrs["BoxSize"])
PCOOR.to_csv("POS_PART/POS_"+str(Nhalo)+".txt", sep=' ', index=False, header=False)


