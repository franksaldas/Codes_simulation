import h5py 
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import functions_X_rays as FN
import scipy
HubbleParam=0.6774
LogMinMass=13.8
LogMaxMass=14.2
LogMinMassSat=10
basePath="../output/"
filename="fof_subhalo_tab_019.hdf5"
file_path=basePath+filename
FOF=h5py.File(file_path, "r")
LogM200=np.log10(FOF["Group"]["Group_M_Crit200"][:]*1E10/HubbleParam)
CIDS=np.logical_and(LogM200>LogMinMass, LogM200<LogMaxMass)
print(f"We found {CIDS.sum()} halos with masses between {LogMinMass}, and {LogMaxMass}")
###Check that the second halo have at least the double of mass of any other satellite, and check the number of subhalos with masses 
def check_sat_masses(FOF, CIDS, LogMinMassSat, HubbleParam):
    SubMasses=FOF["Subhalo"]["SubhaloMass"][:]*1E10/HubbleParam
    LogSubMasses=np.log10(SubMasses)
    Catalogue={}
    Catalogue["GroupNumb"]=np.where(CIDS)[0]
    print(Catalogue["GroupNumb"])
    keys_create=["NumSat", "BCGMass", "TotSatMass", "FirstSatMass", "Flag" ]
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


def plot_halo(snap, HaloNum, HaloPos, plot_radius, HaloMass, SatNum, BCGvsSat):
    print("Start to compute relative positions")
    distances, rel_pos=FN.distances_from_center(snap['PartType1']['Coordinates'][:], HaloPos, snap['Header'].attrs["BoxSize"])
    print("Start to select particles")
    IDS_PART=distances<plot_radius
    font = {'size'   : 18}
    params=snap["Header"].attrs
    print("Start to compute binned statistics")
    statistic, x_edge, y_edge, binnumber=scipy.stats.binned_statistic_2d(rel_pos[IDS_PART,0], rel_pos[IDS_PART,1], np.ones_like(rel_pos[IDS_PART,0])*params["MassTable"][1]*1E10, statistic='sum', bins=(800,800))
    print("Start to plot")
    fig=plt.figure(figsize=(10,12))
    mesh=plt.pcolormesh(statistic, norm=matplotlib.colors.LogNorm(), cmap="magma")
    plt.axis("off")
    #plt.axhline(y=830, color='k')
    #plt.text(360, 860,"200 Mpc" )
    cbar=fig.colorbar(mesh, pad=0.0, location="bottom")
    cbar.set_label(r"Projected Particle Mass ($M_{\odot}$)")
    plt.text(30,900, f"LogM200={round(HaloMass, 2)}", fontsize=18)
    plt.text(30,700, f"NSat={SatNum}", fontsize=18)
    plt.text(600,900, f"MBCG/MFirstSat={BCGvsSat}", fontsize=18)
    plt.savefig("Figures/Halo"+str(HaloNum)+".png", dpi=300)
    print("Plot saved")

for k in range(np.size(Catalogue['GroupNumb'])):
    print("Processing halo {k}")
    plot_halo(snap, Catalogue['GroupNumb'][k],Catalogue["GroupPos"][k,:], 3*Catalogue["Group_R_Crit200C"][k], np.log10(Catalogue['Group_M_Crit200'][k]), Catalogue['NumSat'][k], Catalogue['BCGMass'][k]/Catalogue['FirstSatMass'][k] )


