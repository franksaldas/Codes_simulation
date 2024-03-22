import h5py 
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import functions_X_rays as FN
import scipy
import numpy_indexed as npi
import csv
import pandas as pd
HubbleParam=0.6774
LogMinMass=13.8
LogMaxMass=14.2
LogMinMassSat=10
basePath="../output/"
filename="fof_subhalo_tab_019.hdf5"
filename_first="fof_subhalo_tab_001.hdf5"
INIT_file="../IC-DM-L136-N256.hdf5"
file_path=basePath+filename
file_path_first=basePath+filename_first
FOF=h5py.File(file_path, "r")
FOF_first=h5py.File(file_path_first, "r")
LogM200=np.log10(FOF["Group"]["Group_M_Crit200"][:]*1E10/HubbleParam)
CIDS=np.logical_and(LogM200>LogMinMass, LogM200<LogMaxMass)
print(f"We found {CIDS.sum()} halos with masses between {LogMinMass}, and {LogMaxMass}")
#halos_list=[20,22,29,32,33,38,39,42,44,45]
halos_list=np.where(CIDS)[0]
###Check that the second halo have at least the double of mass of any other satellite, and check the number of subhalos with masses 
def check_sat_masses(FOF, CIDS, LogMinMassSat, HubbleParam):
    SubMasses=FOF["Subhalo"]["SubhaloMass"][:]*1E10/HubbleParam
    LogSubMasses=np.log10(SubMasses)
    Catalogue={}
    Catalogue["GroupNumb"]=np.where(CIDS)[0]
    print(Catalogue["GroupNumb"])
    keys_create=["NumSat", "BCGMass", "TotSatMass", "FirstSatMass", "Flag", "IDMostBound" ]
    for key in keys_create:
        Catalogue[key]=np.empty_like(Catalogue["GroupNumb"])
    for i in range(np.size(Catalogue["GroupNumb"])):
        SubCand=np.where(np.logical_and(FOF["Subhalo"]['SubhaloGroupNr'][:]==Catalogue['GroupNumb'][i], LogSubMasses>LogMinMassSat))[0]
        BCGID=FOF['Group']['GroupFirstSub'][Catalogue['GroupNumb'][i]]
        Catalogue['NumSat'][i]=np.size(SubCand)-1
        Catalogue['BCGMass'][i]=SubMasses[BCGID]
        Catalogue['TotSatMass'][i]=SubMasses[SubCand].sum()-SubMasses[BCGID]
        Catalogue['FirstSatMass'][i]= SubMasses[BCGID+1]
        Catalogue['Flag'][i]= SubMasses[BCGID]/SubMasses[BCGID+1]
        Catalogue['IDMostBound'][i]=FOF["Subhalo"]['SubhaloIDMostbound'][BCGID]
        print(f"Checking consistency: {BCGID==SubCand[0]}")
    return Catalogue


Catalogue= check_sat_masses(FOF,CIDS,LogMinMassSat, HubbleParam)
Catalogue["Group_M_Crit200"]=FOF['Group']["Group_M_Crit200"][CIDS]*1E10/HubbleParam
Catalogue["Group_R_Crit200"]=FOF['Group']["Group_R_Crit200"][CIDS]/HubbleParam
Catalogue["GroupPos"]=FOF['Group']["GroupPos"][CIDS,:]
Catalogue["Group_R_Crit200C"]=FOF['Group']["Group_R_Crit200"][CIDS]
fig=plt.figure(figsize=(8,8))
plt.scatter(Catalogue['FirstSatMass']/Catalogue['BCGMass'], Catalogue['NumSat'])
plt.xlabel("MassFirstSat/MassBCG")
plt.ylabel("Number of satellites")
plt.xscale('log')
plt.yscale('log')
plt.savefig("Figures/MassFirstsateMBCG_vs_NumSat.png", dpi=300)
#plt.text(350,900, f"M200={Catalogue['Group_M_Crit200'][]}")
fig2=plt.figure(figsize=(8,8))
plt.scatter(Catalogue["Group_M_Crit200"], Catalogue["Group_R_Crit200"])
plt.xlabel("M200[M_odot]")
plt.ylabel("R200")
plt.yscale('log')
plt.xscale('log')
plt.savefig("Figures/M200vsR200.png", dpi=300)

snap_name="snapshot_019.hdf5"
snap_path=basePath+snap_name
snap=h5py.File(snap_path, "r")


def compute_center(positions_list):
    center=(np.min(positions_list, axis=0)+np.max(positions_list, axis=0))/2
    range_halo=(np.max(positions_list, axis=0)+np.min(positions_list, axis=0))
    return center, range_halo

def plot_halo(snap, HaloNum, HaloPos, plot_radius, HaloMass, SatNum, BCGvsSat, IDMostBound, R200):
    print("Start to compute relative positions")
    distances, rel_pos=FN.distances_from_center(snap['PartType1']['Coordinates'][:], HaloPos, snap['Header'].attrs["BoxSize"])
    print("Start to select particles")
    IDS_PART=distances<plot_radius
    IDS_sim=snap['PartType1']['ParticleIDs'][IDS_PART]
    font = {'size'   : 18}
    params=snap["Header"].attrs
    print("Start to compute binned statistics")
    statistic, x_edge, y_edge, binnumber=scipy.stats.binned_statistic_2d(rel_pos[IDS_PART,0], rel_pos[IDS_PART,1], np.ones_like(rel_pos[IDS_PART,0])*params["MassTable"][1]*1E10/HubbleParam, statistic='sum', bins=(800,800))
    print("Start to plot")
    fig=plt.figure(figsize=(10,12))
    mesh=plt.pcolormesh(statistic, norm=matplotlib.colors.LogNorm(), cmap="magma")
    plt.axis("off")
    #plt.axhline(y=830, color='k')
    #plt.text(360, 860,"200 Mpc" )
    cbar=fig.colorbar(mesh, pad=0.0, location="bottom")
    cbar.set_label(r"Projected Particle Mass ($M_{\odot}$)")
    plt.text(30,900, f"LogM200={HaloMass:.2f}", fontsize=18)
    plt.text(30,800, f"NSat={np.round(SatNum,2)}", fontsize=18)
    plt.text(30,850, f"R200={np.round(R200,2)} kpc", fontsize=18)
    plt.text(500,900, f"MBCG/MFirstSat={np.round(BCGvsSat,2)}", fontsize=18)
    plt.text(500, 850, f"NP={IDS_PART.sum()}", fontsize=18)
    plt.text(500, 800, f"HR Part={IDS_PART.sum()*8}", fontsize=18)
    plt.text(500, 700, f"Tot. Part 1HR={round(512**3+(2**3-1)*IDS_PART.sum(),2)}", fontsize=18)
    plt.text(500, 650, f"Tot. Part 2HR={round(512**3+(2**6-1)*IDS_PART.sum(),2)}", fontsize=18)
    plt.text(500, 600, f"Tot. Part 3HR={round(512**3+(2**9-1)*IDS_PART.sum(),2)}", fontsize=18)
    plt.savefig("Figures/Halo_"+str(HaloNum)+".png", dpi=300)
    print("Plot saved")
    print("Start to find the particles in the initial conditions")
    #snap_name2="snapshot_011.hdf5"
    #INIT_file=basePath+snap_name2
    INIT=h5py.File(INIT_file, "r")
    #ind_INIT=npi.indices(INIT["PartType1"]["ParticleIDs"], IDS_sim)
    #print(f"We find {np.size(ind_INIT)} particles")
    #IDS_sorted=np.sort(ind_INIT)
    #print(f"Array sorted")
    IDS_in1d=np.in1d(INIT["PartType1"]["ParticleIDs"], IDS_sim)
    print("IDS found using in1d")
    COOR_INIT=INIT["PartType1"]["Coordinates"][:]
    #POS_INIT=INIT["PartType1"]["Coordinates"][np.in1d(INIT["PartType1"]["ParticleIDs"], IDS_sim), :]
    POS_INIT=COOR_INIT[IDS_in1d]
    PCOOR=pd.DataFrame(POS_INIT/snap["Header"].attrs["BoxSize"])
    PCOOR.to_csv("POSITIONS_LB/POS_"+str(HaloNum)+".txt", sep=' ', index=False, header=False)
    print("Initial positions loaded...")
    #HALO_POS_INIT, range_halo=compute_center(POS_INIT)
    HALO_POS_INIT=INIT["PartType1"]['Coordinates'][INIT["PartType1"]["ParticleIDs"][:]==IDMostBound].flatten()
    print(f"{HALO_POS_INIT}")
    print("MOst bound position found...")
    distances_INIT, rel_pos_INIT=FN.distances_from_center(POS_INIT, HALO_POS_INIT, snap['Header'].attrs["BoxSize"])
    print(f"The maximum distance is {np.max(distances_INIT/plot_radius*3)} in units of R200 at redshift 1.")
    statistic_INIT, x_edge_INIT, y_edge_INIT, binnumber=scipy.stats.binned_statistic_2d(rel_pos_INIT[:,0], rel_pos_INIT[:,1], np.ones_like(rel_pos_INIT[:,0])*params["MassTable"][1]*1E10, statistic='sum', bins=(800,800))
    fig=plt.figure(figsize=(10,12))
    mesh=plt.pcolormesh(statistic_INIT, norm=matplotlib.colors.LogNorm(), cmap="magma")
    plt.axis("off")
    #plt.axhline(y=830, color='k')
    #plt.text(360, 860,"200 Mpc" )
    cbar=fig.colorbar(mesh, pad=0.0, location="bottom")
    cbar.set_label(r"Projected Particle Mass ($M_{\odot}$)")
    BoxSize=snap['Header'].attrs["BoxSize"]
    plt.text(30,900, f"Range={(np.max(rel_pos_INIT,axis=0)-np.min(rel_pos_INIT, axis=0))/BoxSize}", fontsize=18)
    plt.text(30, 800, f"The lowest right coordinates are: {(np.min(rel_pos_INIT)+HALO_POS_INIT)/BoxSize}", fontsize=18)
    #plt.text(30,700, f"NSat={SatNum}", fontsize=18)
    #plt.text(600,900, f"MBCG/MFirstSat={BCGvsSat}", fontsize=18)
    plt.savefig("Figures/Halo_INIT"+str(HaloNum)+".png", dpi=300)
    #print(f"The range is {range_halo} in units of kpc/h")

for l in range(np.size(halos_list)):
    k=np.where(Catalogue['GroupNumb']==halos_list[l])[0][0] 
    print(f"Processing halo {k}")
    print(4000.0*HubbleParam)
    print(4.0*Catalogue['Group_R_Crit200C'][k])
    Lagrange_box_size=np.maximum(4000.0*HubbleParam, 4.0*Catalogue['Group_R_Crit200C'][k])

    plot_halo(snap, Catalogue['GroupNumb'][k],Catalogue["GroupPos"][k,:], Lagrange_box_size, np.log10(Catalogue['Group_M_Crit200'][k]), Catalogue['NumSat'][k], Catalogue['BCGMass'][k]/Catalogue['FirstSatMass'][k], Catalogue['IDMostBound'][k], Catalogue["Group_R_Crit200C"][k]/HubbleParam)


