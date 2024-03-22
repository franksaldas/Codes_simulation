import time 
import numpy as np
#import pyatomdb
from numba import jit, prange
#from spec_params import DParam as CF
from scipy.stats import binned_statistic_2d
from scipy.stats import binned_statistic_dd
from matplotlib import pyplot as plt
#import cv2



@jit(nopython=True, parallel=True)
def distances_from_center(positions, center, L):
    N = positions.shape[0]
    distances = np.empty(N)
    rel_pos = np.empty((N,3))
    
    for i in prange(N):
        # Compute the absolute distance between the current position and the center
        diff = positions[i] - center

        # Apply periodic boundary conditions
        for j in range(3):
            if diff[j] > L / 2:
                diff[j] -= L
            elif diff[j] < -L / 2:
                diff[j] += L
            rel_pos[i,j]=diff[j]

        # Compute the squared distance
        squared_distance = np.sum(diff**2)
        
        # Store the square root of the squared distance in the distances array
        distances[i] = np.sqrt(squared_distance)
    
    return distances, rel_pos

"""


def create_lum_matrix(sess):
    #Traced elements: H, He, C, N, O, Ne, Mg, Si, Fe
    el_list = (np.array([1,2,6,7,8,10,12,14,26])-1).astype(int)
    mbins=(sess.ebins_out[:-1]+sess.ebins_out[1:])/2
    idtu=np.logical_and(mbins<=10.0, mbins>=0.1)
    #A_el = np.array([1.0078, 4.0026, 12.011, 14.007, 15.999, 20.180, 24.305, 28.086, 55.845 ])
    matr_emiss= np.zeros((np.size(CF['LogTemp']), np.size(el_list)), dtype=float)
    spectrum=np.zeros((np.size(CF['LogTemp']), np.size(el_list), np.size(sess.ebins_out)-1), dtype=float)
    uma = 1.6605E-24 #in grams units
    el = range(30)
    em_model=0
    for i in range(np.size(el_list)):
        abund = np.zeros(30)
        abund[el_list[i]] = 1
        sess.set_abund(el, abund)
        for k in range(np.size(CF['LogTemp'])):
            aux_spec=sess.return_spectrum(10**CF['LogTemp'][k], teunit='K')*mbins
            spectrum[k, i, :]=aux_spec
            matr_emiss[k,i]=aux_spec[idtu].sum()
    return mbins, matr_emiss, spectrum 


@jit(nopython=True, parallel=True)
def c_temperature(InternalEnergy, ElectronAbundance):
    XH=0.76
    gamma=5/3
    kB=1.38065E-16 #erg/K
    proton_mass=1.6726E-24 ##grams
    mu= (4*proton_mass)/(1+3*XH+4*XH*ElectronAbundance)
    temperature= (gamma-1) *mu/kB*InternalEnergy*1E10
    return temperature

@jit(nopython=True, parallel=True)
def c_abundances(AG89, at_weight, GFM_Metals):
    abund=np.ones(np.shape(GFM_Metals))
    n_at_H=GFM_Metals[:,0]/at_weight[0]
    for i in range(9):
        abund[:,i]=GFM_Metals[:,i]/at_weight[i]/n_at_H/AG89[i]
    abund[:,0]=GFM_Metals[:,0]/0.70683 ## Porcentage of abundance in mass
    return abund
        
def c_abundances_metallicity(AG89, at_weight,  GFM_Metallicity):
    "This part needs to be checked if this function works well to estimate the abundances of each elemant"
    facZ= np.divide(GFM_Metallicity,0.01886) #To convert to solar metallicity see IllustrisTNG webpage
    facX=np.divide(1-GFM_Metallicity-0.85*0.27431,0.70683)
    abund=np.transpose(np.array([facX, 0.85*np.ones(np.size(GFM_Metallicity)), facZ,  facZ, facZ, facZ, facZ, facZ, facZ ]))
    return abund



@jit(nopython=True, parallel=True)
def c_emi_model(temperatures, abundances, m_emiss, LogTempLIM):    
    LogT=np.log10(temperatures)
    emi_model=np.zeros(temperatures.size)
    for i in range(LogTempLIM.size-1):
        IDS_PB=np.logical_and(LogT>LogTempLIM[i], LogT<=LogTempLIM[i+1])
        emi_model[IDS_PB]= np.sum(abundances[IDS_PB, :]*m_emiss[i,:], axis=1)
    return emi_model

@jit(nopython=True, parallel=True)
def c_emissivity_measure(GFM_Metals, masses, density, ElectronAbundance, uma, UnitLenght, UnitMass, at_weight):
    UnitVolume=UnitLenght**3
    UnitDensity=UnitMass/(UnitVolume)
    n_H=GFM_Metals[:,0]/(at_weight[0]*uma)*density*UnitDensity
    n_e=n_H*ElectronAbundance
    Vol=masses/density*UnitVolume
    emiss_measure=n_e*n_H*Vol
    return emiss_measure

@jit(nopython=True, parallel=True)
def compute_metals(Metallicity):
    Wiersma2009=np.array([0.7065, 0.2806, 2.07E-3, 8.36E-4, 5.49E-3, 1.41E-3, 5.91E-4, 6.83E-4, 1.10E-3])
    theta=Metallicity/ 0.0127
    alpha=(1-Metallicity)/(1- 0.0127)
    facX=(Metallicity+0.2449-1)/(-0.7065)
    fracH=facX*Wiersma2009[0]
    fracHe=np.ones_like(Metallicity)*0.2449
    fraction=[fracH, fracHe, theta*Wiersma2009[2],  theta*Wiersma2009[3], theta*Wiersma2009[4], theta*Wiersma2009[5], theta*Wiersma2009[6],  theta*Wiersma2009[7], theta*Wiersma2009[8]]
    return fraction



def plot_emission_map(emiss_measure, emi_model, CF, temperatures, StarFormationRate, distances, rel_pos_R500, HaloNumber, R500, Header, Velocities, GroupVel, InternalEnergy):
    particlesIDS=np.logical_and(temperatures>=1E5, StarFormationRate==0)
    emissivity= emiss_measure[particlesIDS]*emi_model[particlesIDS]*CF['keV2erg']
    emissivity_map, edges, binnumber=binned_statistic_dd(rel_pos_R500[particlesIDS], emissivity, statistic='sum',range=[[-1.5,1.5], [-1.5,1.5], [-1.5,1.5]], bins=[CF.nbins,CF.nbins, CF.nbins], expand_binnumbers=True) #####Uncomment this line for plots
    mid_edges= np.transpose(np.array([(edges[k][:-1]+edges[k][ 1:])/2 for k in range(3)]))
    #print(mid_edges)
    emission_map=emissivity_map.sum(axis=2)#Make projection along the z axis
    #print(emission_map)
    #LogGridGAS=emission_map
    LogGridGAS=np.log10(emission_map)
    #print(np.min(emission_map), np.max(emission_map))
    filename=str("/freya/ptmp/mpa/fald/FiguresNew/XR_BCG"+str(HaloNumber)+".pdf")
    aDM=1.5
    extentDM=np.array([-aDM, aDM,-aDM, aDM]).flatten()
    min_s=np.min(LogGridGAS[LogGridGAS != np.NINF])
    #print(f"Computed moments are: {M00, M10, M01}")
    #print(f"The computed means are: {xm, ym} ")
    ###########Compute of the concentration index
    X,Y=np.meshgrid(mid_edges[:,0], mid_edges[:,1])
    PIX_dist=np.square(X**2+ Y**2)
    ### Start computing the dynamical state parameters
    mod_emission_map=emission_map.copy()
    #print(np.size(emission_map!=0))
    PIXR500=PIX_dist<=1
    M00=np.sum(emission_map[PIXR500])
    M10=np.sum(X[PIXR500]*emission_map[PIXR500])
    M01=np.sum(Y[PIXR500]*emission_map[PIXR500])
    xm=M10/M00
    ym=M01/M00
    mod_emission_map[~PIXR500]=0.0
    Mo = cv2.moments(mod_emission_map)
    #print(Mo['m00'], Mo['m01'], Mo['m10'])
    centroid_x = (Mo['m10'] / Mo['m00'])*3/CF['nbins']-1.5
    centroid_y = (Mo['m01'] / Mo['m00'])*3/CF['nbins']-1.5
    #print(f"Centroids in X are: {xm}, and {centroid_x} ")
    #print(f"Centroids in Y are: {ym}, and {centroid_y} ")
    PIXCORE=PIX_dist<0.1
    PIXR500=PIX_dist<1
    #emiss=emiss_measure*emi_model*CF.kev2erg
    Score=np.sum(emission_map[PIXCORE])
    Stot=np.sum(emission_map[PIXR500])
    #print(Score, Stot)
    CI=Score/Stot
    #print(distances[particlesIDS])
    IDSCore= distances[particlesIDS]<0.1*R500
    IDSR500= distances[particlesIDS]<1*R500
    Score3D=np.sum(emissivity[IDSCore])
    Stot3D=np.sum(emissivity[IDSR500])
    #print(Score3D, Stot3D)
    CI_3D=Score3D/Stot3D
    #print(f"The concentration index is: {CI}, and the 3D is {CI_3D} ")
    ###### Compute P3/P0, first considering images inside R500
    MDIST=PIX_dist*R500
    PIXIT=PIX_dist<1
    theta= np.arctan(Y/X)
    a0=np.sum(emission_map[PIXIT])
    a3=np.sum(emission_map[PIXIT]*MDIST[PIXIT]**3*np.cos(3*theta[PIXIT]))
    b3=np.sum(emission_map[PIXIT]*MDIST[PIXIT]**3*np.sin(3*theta[PIXIT]))
    P0=(a0*np.log(R500))**2
    P3=1/(2*3**2*(R500)**(2*3))*(a3**2+b3**2)
    P3P0=P3/P0

    rit=np.arange(1, 0.05, -0.05)
    delta=[]
    for k in rit:
        PIXTU= PIX_dist<k
        M00=np.sum(emission_map[PIXTU])
        M10=np.sum(X[PIXTU]*emission_map[PIXTU])
        M01=np.sum(Y[PIXTU]*emission_map[PIXTU])
        xm=M10/M00
        ym=M01/M00
        delta=np.append(delta, np.sqrt(xm**2+ym**2))
    delta_med=np.average(delta)
    #print(delta)
    omega=np.sqrt(1/(np.size(rit)-1)*sum((delta-delta_med)**2))
    #print(f"The value of omega is {omega}")
    IDSR2= np.logical_and(distances<1*R500,particlesIDS )
    Kin= (0.5*np.linalg.norm(Velocities[IDSR2]-GroupVel, axis=1)**2).sum() #In units of (km/s)**2 per unit mass
    Ther=InternalEnergy[IDSR2].sum()
    EnergyRatio= Kin/Ther
    return CI, CI_3D, P3P0, omega, EnergyRatio
"""
