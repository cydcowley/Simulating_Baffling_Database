from operator import index
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
import json
from natsort import natsorted
import matplotlib.image as m
import imageio as io
from scipy import interpolate
from scipy.integrate import quad,trapz, cumtrapz, odeint, solve_ivp
from PIL import Image
import cv2
import glob
import matplotlib as mpl
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
from parse import parse
sys.path.append('D:\\my stuff\\PhD\\Theoretical_Detachment_Control_Scripts')
sys.path.append('D:\\my stuff\\PhD\\MastEnrichmentAnalysis')
from LipschultzDLS import ChInt,ChIntThermalForce,averageB
from UnpackSOLPS import unpackSOLPS,SOLring
from AnalyticCoolingCurves import LfuncN
from scipy.optimize import curve_fit
from SharedFunctions import return2d, ImportGridue, plotWALL
import re
from scipy.optimize import curve_fit
plt.rcParams["font.family"] = "serif"
params = {'legend.fontsize': 'small',
         'font.size': '16',
        #  'figure.figsize': (4,3.2),
         }


sys.path.append('D:\\my stuff\\PhD\\Theoretical_Detachment_Control_Scripts')
from LipschultzDLS import ChInt,ChIntThermalForce,averageB
plt.rcParams.update(params)

customOrder = 0

colors = ["#356288","#fe1100","#aacfdd","#fe875d"]
# customOrder = ["fi200E-3","fi250E-3","fi260E-3","fi262E-3","fi264E-3","fi270E-3","fi262E-3Backward","fi250E-3Backward","fi245E-3Backward","fi240E-3Backward","fi235E-3Backward","fi230E-3Backward","fi225E-3Backward","fi220E-3Backward"]
gridcolors = ["#53ba83","#059b9a","#095169","#0c0636","000000"]
def determineC0(Spar,C):
    for k in range(len(C)):
        if Spar[k] > Spar[-1]/100:

            return k-1

image_list = []
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def readWallflx(fname,targtype):
    # paramnames include: pdena for atom dens, pdenm for mol dens, volume for triangle volume 
    fortFile = fname+"//ld_tg_"+targtype+".dat"
    rootgrp = Dataset(fname+"//balance.nc", "r", format="NETCDF4") 

    rnew = np.array(rootgrp['crx'][0]+rootgrp['crx'][1]+rootgrp['crx'][2]+rootgrp['crx'][3])/4
    A = []
    for i in range(1,len(rootgrp['crx'][1])-1):
        A.append(np.sqrt((rootgrp['crx'][1][i][-2]-rootgrp['crx'][3][i][-2])**2+(rootgrp['cry'][1][i][-2]-rootgrp['cry'][3][i][-2])**2))

    A = np.array(A)
    rnew = rnew[1:-1,-2]
    dataFort= open(fortFile)
    tline = dataFort.readlines(1)
    linenum = 0
    while True:
        tline = dataFort.readlines(1)

        
        if  "#           x" in tline[0]:
            break
    # tline = dataFort.readlines(1)
    nonradflx = []
    totalflx = []
    recombflx =[]
    x = []

    r = []
    while True:
        # print(tline)
        tline = dataFort.readlines(1)

        if len(tline)==0:
            break
        co = re.findall("\D+\d+\.\d+\D+\d+",tline[0])
        nonradflx.append(float(co[4])-float(co[6]))
        totalflx.append(float(co[4]))
        recombflx.append(float(co[5]))
        x.append(float(co[0]))
        r.append(float(co[14]))
    x = np.array(x)
    totalflx = np.array(totalflx)
    nonradflx = np.array(nonradflx)
    recombflx = np.array(recombflx)
    r = np.array(r)
    total = np.sum(nonradflx*2*np.pi*r*A)

    peak = np.max(totalflx)
    peakind = np.argmax(totalflx)
    print("recomb is",recombflx[peakind]*1E-6)
    print(peak*1E-6)
            # data.append(float(co[i]))
    # print(r)
    return peak,total




def expFunc(x,A,B):
    return A*np.exp((-x)/B)

def pressureBalance(rootgrp):

    fmox_flua = np.sum(rootgrp['fmo_flua'],axis=0)[0]
    b2mndr_hz = np.array(rootgrp["b2mndr_hz"])

    pe = np.array(rootgrp["ne"])*np.array(rootgrp["te"])
    dv = np.array(rootgrp["vol"])
    gs = np.array(rootgrp["gs"])
    hz = (1-b2mndr_hz)+b2mndr_hz*(dv/gs[2])
    hx = rootgrp["hx"]
    B = rootgrp['bb']
    leftix = np.array(rootgrp['leftix'])
    leftiy = np.array(rootgrp['leftiy'])
    rightix = np.array(rootgrp['rightix'])
    rightiy = np.array(rootgrp['rightiy'])
    topix = np.array(rootgrp['topix'])
    topiy = np.array(rootgrp['topiy'])
    apll = rootgrp["vol"]*hz/hx*abs(rootgrp['bb'][0]/rootgrp['bb'][3])
    ny = len(rootgrp["vol"])
    nx = len(rootgrp["vol"][0])
    apllx = np.ones((ny,nx))*1000
    apllc = np.ones((ny,nx))*1000
    for i in range(ny):
        for j in range(nx):
            if leftix[i,j]<1:
                continue
            apllx[i,j] = (apll[leftiy[i,j],leftix[i,j]]*dv[i,j]+\
                            apll[i,j]*dv[leftiy[i,j],leftix[i,j]])/ \
                       (dv[i,j]+dv[leftiy[i,j],leftix[i,j]])
            
    for i in range(ny):
        for j in range(nx):
            if rightix[i,j]>nx:
                continue
            apllc[i,j] = 0.5*(apllx[i,j]+apllx[rightiy[i,j],rightix[i,j]])
    za = np.array(rootgrp["za"])
    ion_indices = np.argwhere(za>0).flatten()

    raddiv_flu = np.zeros((ny,nx))+1
    fmoy_flua = np.sum(np.array(rootgrp["fmo_flua"]),axis=0)[1]+np.sum(np.array(rootgrp["fmo_cvsa"]),axis=0)[1]
    for i in range(ny):
        for j in range(nx):
            if topiy[i,j]>ny:
                continue
        
            raddiv_flu[i,j] = fmoy_flua[i,j]-fmoy_flua[topiy[i,j],topix[i,j]]

    #b2 neutral sinks
    b2stel_smq_ion_bal=np.sum(np.array(rootgrp['b2stel_smq_ion_bal'])[ion_indices],axis=0)
    b2stel_smq_rec_bal=np.sum(np.array(rootgrp['b2stel_smq_rec_bal'])[ion_indices],axis=0)
    b2stel_smq_rec_bal=np.sum(np.array(rootgrp['b2stel_smq_rec_bal'])[ion_indices],axis=0)

    b2sink = b2stel_smq_ion_bal+b2stel_smq_rec_bal+b2stel_smq_rec_bal
    # b2sink = b2sink/apllc
    # eirene sinks
    eirene_mc_mapl_smo = np.sum(np.array(rootgrp['eirene_mc_mapl_smo_bal']),axis=0)[1]
    eirene_mc_mapl_smo = eirene_mc_mapl_smo/apllc

    eirene_mc_mmpl_smo_bal = np.sum(np.array(rootgrp['eirene_mc_mmpl_smo_bal']),axis=0)[1]
    eirene_mc_mmpl_smo_bal = eirene_mc_mmpl_smo_bal/apllc

    eirene_mc_cppv_smo_bal = np.sum(np.array(rootgrp['eirene_mc_cppv_smo_bal']),axis=0)[1]
    eirene_mc_cppv_smo_bal = eirene_mc_cppv_smo_bal/apllc

    eirene_mc_mipl_smo_bal = np.sum(np.array(rootgrp['eirene_mc_mipl_smo_bal']),axis=0)[1]
    eirene_mc_mipl_smo_bal = eirene_mc_mipl_smo_bal/apllc


    eireneSink = eirene_mc_mapl_smo+eirene_mc_mmpl_smo_bal+eirene_mc_mipl_smo_bal+eirene_mc_cppv_smo_bal
    fmox_flua = fmox_flua/apllx
    raddiv_flu = raddiv_flu/apllc
    pi = np.sum(np.array(rootgrp["na"][ion_indices]),axis=0)*np.array(rootgrp["ti"])
    
    # pi = np.sum(rootgrp["b2sigp_smogpi_bal"],axis=0)
    return pe,pi,fmox_flua,eireneSink, raddiv_flu


def plot2d(rootgrp,plotquantity,grid,ilim,norm,cmap,axs,
plotmode = "real"):

    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    # PLOT RELEVANT QUANTITY ON GRID
    for j in range(ilim, len(rootgrp['crx'][0][0])):  # len(R[0])-1):
        for i in range(0, len(rootgrp['crx'][0])):
            if plotmode == "real": 
                x = [rootgrp['crx'][0, i, j], rootgrp['crx'][2, i, j], rootgrp['crx']
                    [3, i, j], rootgrp['crx'][1, i, j], rootgrp['crx'][0, i, j]]
                y = [rootgrp['cry'][0, i, j], rootgrp['cry'][2, i, j], rootgrp['cry']
                    [3, i, j], rootgrp['cry'][1, i, j], rootgrp['cry'][0, i, j]]
            else:
                x = [j, j+1, j+1, j, j]
                y = [i, i, i+1, i+1, i]
            
            color1 = m.to_rgba(plotquantity[i, j])
            axs.fill(x, y, color=color1, linewidth=0.01)

    # PLOT PFCs IN BLACK
    if plotmode == "real": 
        if grid=="Tight":
            plotWALL("balFiles\MAST_Tight//input.dat", axs)
        else:

            plotWALL("balFiles\MAST_Open//input.dat", axs)



def CalculateSinks(rootgrp):
    ion_heat={"sinks":[],"sum_sinks":[],"labels":[]}

    
    ion_heat["sinks"].append(np.array(rootgrp['b2stbr_phys_shi_bal']))
    ion_heat["labels"].append("b2stbr_phys_shi")

    ion_heat["sinks"].append(np.array(rootgrp['b2stbr_bas_shi_bal']))
    ion_heat["labels"].append("b2stbr_bas_shi_bal")

    ion_heat["sinks"].append(np.array(rootgrp['b2stbr_first_flight_shi_bal']))
    ion_heat["labels"].append("b2stbr_first_flight_shi_bal")

    # ion_heat["sinks"].append(np.array(rootgrp['b2stbc_shi_bal']))
    # ion_heat["labels"].append("b2stbc_shi_bal")

    ion_heat["sinks"].append(np.array(rootgrp['b2stbm_shi_bal']))
    ion_heat["labels"].append("b2stbm_shi_bal")

    ion_heat["sinks"].append(np.array(rootgrp['ext_shi_bal']))
    ion_heat["labels"].append("ext_shi_bal")

    ion_heat["sinks"].append(np.array(rootgrp['b2stel_shi_ion_bal']))
    ion_heat["labels"].append("b2stel_shi_ion_bal")

    ion_heat["sinks"].append(np.array(rootgrp['b2stel_shi_rec_bal']))
    ion_heat["labels"].append("b2stel_shi_rec_bal")

    ion_heat["sinks"].append(np.array(rootgrp['b2stcx_shi_bal']))
    ion_heat["labels"].append("b2stcx_shi_bal")

    ion_heat["sinks"].append(np.array(rootgrp['b2srsm_shi_bal']))
    ion_heat["labels"].append("b2srsm_shi_bal")

    ion_heat["sinks"].append(np.array(rootgrp['b2srdt_shi_bal']))
    ion_heat["labels"].append("b2srdt_shi_bal")

    ion_heat["sinks"].append(np.array(rootgrp['b2srst_shi_bal']))
    ion_heat["labels"].append("b2srst_shi_bal")

    ion_heat["sinks"].append(np.array(rootgrp['b2sihs_diaa_bal']))
    ion_heat["labels"].append("b2sihs_diaa_bal")

    ion_heat["sinks"].append(np.array(rootgrp['b2sihs_divua_bal']))
    ion_heat["labels"].append("b2sihs_divua_bal")

    ion_heat["sinks"].append(np.array(rootgrp['b2sihs_exba_bal']))
    ion_heat["labels"].append("b2sihs_exba_bal")

    ion_heat["sinks"].append(np.array(rootgrp['b2sihs_visa_bal']))
    ion_heat["labels"].append("b2sihs_visa_bal")

    ion_heat["sinks"].append(np.array(rootgrp['b2sihs_fraa_bal']))
    ion_heat["labels"].append("b2sihs_fraa_bal")


    ion_heat["sinks"].append(np.sum(np.array(rootgrp['eirene_mc_eapl_shi_bal']),axis=0))
    ion_heat["labels"].append("eirene_mc_eapl_shi")

    ion_heat["sinks"].append(np.sum(np.array(rootgrp['eirene_mc_empl_shi_bal']),axis=0))
    ion_heat["labels"].append("eirene_mc_empl_shi_bal")

    ion_heat["sinks"].append(np.sum(np.array(rootgrp['eirene_mc_eipl_shi_bal']),axis=0))
    ion_heat["labels"].append("eirene_mc_eipl_shi_bal")

    ion_heat["sinks"].append(np.sum(np.array(rootgrp['eirene_mc_eppl_shi_bal']),axis=0))
    ion_heat["labels"].append("eirene_mc_eppl_shi_bal")
    for si in range(len(ion_heat["sinks"])):
        ion_heat["sum_sinks"].append(np.sum(ion_heat["sinks"][si]))

    

    elec_heat={"sinks":[],"sum_sinks":[],"labels":[]}

    elec_heat["sinks"].append(np.array(rootgrp['b2stbr_phys_she_bal']))
    elec_heat["labels"].append("b2stbr_phys_she_bal")

    elec_heat["sinks"].append(np.array(rootgrp['b2stbr_bas_she_bal']))
    elec_heat["labels"].append("b2stbr_bas_she_bal")

    elec_heat["sinks"].append(np.array(rootgrp['b2stbr_first_flight_she_bal']))
    elec_heat["labels"].append("b2stbr_first_flight_she_bal")

    # elec_heat["sinks"].append(np.array(rootgrp['b2stbc_she_bal'])) # boundary sources
    # elec_heat["labels"].append("b2stbc_she_bal")

    elec_heat["sinks"].append(np.array(rootgrp['b2stbm_she_bal']))
    elec_heat["labels"].append("b2stbm_she_bal")

    elec_heat["sinks"].append(np.array(rootgrp['ext_she_bal']))
    elec_heat["labels"].append("ext_she_bal")

    elec_heat["sinks"].append(np.sum(np.array(rootgrp['b2stel_she_bal']),axis=0))
    elec_heat["labels"].append("b2stel_she_bal")

    elec_heat["sinks"].append(np.array(rootgrp['b2srsm_she_bal']))
    elec_heat["labels"].append("b2srsm_she_bal")

    elec_heat["sinks"].append(np.array(rootgrp['b2srdt_she_bal']))
    elec_heat["labels"].append("b2srdt_she_bal")

    elec_heat["sinks"].append(np.array(rootgrp['b2srst_she_bal']))
    elec_heat["labels"].append("b2srst_she_bal")

    elec_heat["sinks"].append(np.array(rootgrp['b2sihs_diae_bal']))
    elec_heat["labels"].append("b2sihs_diae_bal")

    elec_heat["sinks"].append(np.array(rootgrp['b2sihs_divue_bal']))
    elec_heat["labels"].append("b2sihs_divue_bal")

    elec_heat["sinks"].append(np.array(rootgrp['b2sihs_exbe_bal']))
    elec_heat["labels"].append("b2sihs_exbe_bal")

    elec_heat["sinks"].append(np.array(rootgrp['b2sihs_joule_bal']))
    elec_heat["labels"].append("b2sihs_joule_bal")

    elec_heat["sinks"].append(np.sum(np.array(rootgrp['eirene_mc_eael_she_bal']),axis=0))
    elec_heat["labels"].append("eirene_mc_eael_she_bal")

    elec_heat["sinks"].append(np.sum(np.array(rootgrp['eirene_mc_emel_she_bal']),axis=0))
    elec_heat["labels"].append("eirene_mc_emel_she_bal")

    elec_heat["sinks"].append(np.sum(np.array(rootgrp['eirene_mc_eiel_she_bal']),axis=0))
    elec_heat["labels"].append("eirene_mc_eiel_she_bal")

    elec_heat["sinks"].append(np.sum(np.array(rootgrp['eirene_mc_epel_she_bal']),axis=0))
    elec_heat["labels"].append("eirene_mc_epel_she_bal")

    for si in range(len(elec_heat["sinks"])):
        elec_heat["sum_sinks"].append(np.sum(elec_heat["sinks"][si]))

    
    return ion_heat,elec_heat

def heatBalance(fname,rootgrp,quantities2d,XPTs,heatmode,heat_counts):

    radsource = np.sum(quantities2d["radHeate"][1,:])+np.sum(quantities2d["radHeati"][1,:])
    
    imprad = np.sum(np.array(rootgrp["b2stel_she_bal"][1]))

    b2stbc_shi_bal= np.array(rootgrp['b2stbc_shi_bal']) # boundary sources
    b2stbc_she_bal= np.array(rootgrp['b2stbc_she_bal']) # boundary sources

    innertarHeat = np.sum(b2stbc_she_bal[1:-1,0])+ np.sum(b2stbc_shi_bal[1:-1,0])+\
    np.sum(b2stbc_she_bal[1:-1,XPTs[2]-1])+ np.sum(b2stbc_shi_bal[1:-1,XPTs[2]-1])

    outertarHeat = np.sum(b2stbc_she_bal[:,XPTs[2]])+ np.sum(b2stbc_shi_bal[:,XPTs[2]])+\
    np.sum(b2stbc_she_bal[:,-1])+ np.sum(b2stbc_shi_bal[:,-1])

    wallLoss = np.sum(b2stbc_she_bal[-1,1:-1])+np.sum(b2stbc_shi_bal[-1,1:-1])+\
        np.sum(np.where(b2stbc_she_bal[0]<=0,b2stbc_she_bal[0],0))+np.sum(np.where(b2stbc_she_bal[0]<=0,b2stbc_shi_bal[0],0)) #wall loss

    fluidRad = np.sum(np.array(rootgrp["b2stel_she_bal"][0]))

    
    ion_heat,elec_heat = CalculateSinks(rootgrp)






    # calculate hydrogen radiation
    atomrad = readParam44(fname,"eneutrad")

    molrad = readParam44(fname,"emolrad")

    eionrad = readParam44(fname,"eionrad")

    Hrad = np.array(atomrad)+np.array(molrad)+np.array(eionrad)
    peak,totalol = readWallflx(fname,"o")
    peak,totalou = readWallflx(fname,"ou")
    totalo = totalol+totalou

    peak,totalil = readWallflx(fname,"i")
    peak,totaliu = readWallflx(fname,"iu")
    totali = totalil+totaliu

    Hrad = np.reshape(Hrad,(len(b2stbc_shi_bal)-2,len(b2stbc_shi_bal[0])-2))
    tot = np.abs(imprad)+np.abs(totalo)+np.abs(totali)+np.abs(np.sum(atomrad)+np.sum(molrad)+np.sum(eionrad))
    wallLoss = radsource-tot
  

    totalsink = np.sum(rootgrp["b2stel_she_bal"],axis=0)[1:-1,1:-1]+Hrad
    totalsink_Core = np.sum(totalsink[:rootgrp["jsep"][0]+2,XPTs[0]:XPTs[1]])+np.sum(totalsink[:rootgrp["jsep"][0]+2,XPTs[3]:XPTs[4]])
    totalsink_SOL = np.sum(totalsink[rootgrp["jsep"][0]+2:,XPTs[0]:XPTs[1]])+np.sum(totalsink[rootgrp["jsep"][0]+2:,XPTs[3]:XPTs[4]])
    totalsink_Div = np.sum(totalsink[:,:XPTs[0]])+np.sum(totalsink[:,XPTs[1]:XPTs[3]])+np.sum(totalsink[:,XPTs[4]:])
    # print("ratio is",totalsink_Div/(totalsink_SOL))
    if heatmode == "source_type":

        heat_counts["nitrogen"].append(np.abs(imprad)/radsource)
        heat_counts["hydrogen"].append(np.abs(np.sum(atomrad)+np.sum(molrad)+np.sum(eionrad))/radsource)
        heat_counts["non-radiative outer"].append(np.abs(totalo)/radsource)
        heat_counts["non-radiative inner"].append(np.abs(totali)/radsource)
        heat_counts["non-radiative wall"].append(np.abs(wallLoss)/radsource)

    else:
        
        heat_counts["core radiation"].append(np.abs(totalsink_Core)/radsource)
        heat_counts["main chamber SOL radiation"].append(np.abs(totalsink_SOL)/radsource)
        heat_counts["divertor radiation"].append(np.abs(totalsink_Div)/radsource)
        heat_counts["non-radiative targets"].append((np.abs(totali)+np.abs(totalo))/radsource)
        heat_counts["non-radiative wall"].append(np.abs(wallLoss)/radsource)
        print("ratio is",np.abs(totalsink_Div)/np.abs(totalsink_SOL))
    # heat_counts["other"].append(np.abs(other))
    return heat_counts

def perform_Analysis(power,simFiles,collisionality=False):



    folderList = [
    "balfiles\MAST_Tight\\"+power,
    "balfiles\MAST_Open\\"+power,
    # "balfiles\MAST_Open\\12MWpump"
    ]

    plt.rcParams["axes.labelsize"] = "Large"
    fig0,axs0 = plt.subplots(1,1,figsize=(6,5.2))
    fig1, axs1 = plt.subplots(1,1,figsize=(6,5.2))
    figpress,axspress = plt.subplots(1,1,figsize=(6,5.2))
    ax12 = axs1.twinx() 
    fig2, axs2 = plt.subplots(1,1,figsize=(6,5.2))
    fig3, axs3 = plt.subplots(1,1,figsize=(6,5.2))
    fig4, axs4 = plt.subplots(1,1,figsize=(6,5.2))
    fig5, axs5 = plt.subplots(1,1,figsize=(6,5.2)) 
    radheatfig, radheataxs = plt.subplots(1,1,figsize=(6,5.2)) 
    upstempfig, upstempaxs = plt.subplots(1,1,figsize=(6,5.2)) 
    fig6, axs6 = plt.subplots(1,1,figsize=(6,5.2))  # axis for conductive/convective heat flux
    DLSfig, DLSaxs = plt.subplots(1,1,figsize=(6,5.2))  # axis for losses along killer flux tube
    for folder in folderList:

        Files = os.listdir(str(folder))
        Files = natsorted(Files)

        frontpos = []
        ionFlux = []
        peakheatLoad = []
        C = []
        Tu = []
        Qu = []
        QF = []
        fmom = []
        Tt = []
        Lpar = 0
        falpha = []
        radfraction = []
        colornum = 0
        Nu= []
        Cdls = []
        if "Tight" in folder:
            label = "Tight"
        else:
            label = "Open"
        detached = 0
        for File in Files:
        
            if File == "input.dat" or ".33" in File or ".34" in File \
                or ".46" in File or ".44" in File:
                continue
            # if "Tight" in folder and "12MW" in folder:
            #     if "ne9" in File or "ne10" in File:
            #         continue
            fileName = str(folder)+"\\"+str(File)+"\\balance.nc"
            rootgrp = Dataset(str(fileName), "r", format="NETCDF4")


            SOLring1 = 0
            RING = rootgrp["jsep"][0]+4
            SEPARATRIX= rootgrp["jsep"][0]+2



            # DETERMINE LOCATION OF X-POINTS
            XPTs = []
            for i in range(len(rootgrp['rightix'][0])):
                if rootgrp['rightix'][0][i] != i:
                    XPTs.append(i)
            XPTs = np.array(XPTs)+1
            midplaneix = int((XPTs[-1]+XPTs[-2])/2)

            quantities2d,SOLring1 = unpackSOLPS(fileName, -1,RING,Xpoint=len(rootgrp['rightix'][0])-midplaneix)
            quantities2d,rootg = return2d(fileName)
            Lpar = np.max(SOLring1.Spar)
            
            if colornum==2:
                cummDloss = -1*np.cumsum((SOLring1.ionisLoss+SOLring1.recombLoss)*SOLring1.V)
                cumRadloss = np.cumsum(SOLring1.radTrans*SOLring1.V)
                cumImpLoss = -1*np.cumsum(SOLring1.qf*SOLring1.V)

            pos = SOLring1.calcFrontTemp(5)
            
            falpha.append(np.sqrt(SOLring1.determinefi()))
            partoPol = interpolate.interp1d(SOLring1.Spar,SOLring1.Spol,kind='cubic',fill_value=0,bounds_error=False)
            frontpos.append(partoPol(pos))
            # print(rootgrp)
            ionFlux.append(rootg["fna_tot"][1][0][RING][-2])
            peak,total = readWallflx(str(folder)+"\\"+str(File),"o")
            peakheatLoad.append(peak)
            # plt.plot(SOLring1.Spar,SOLring1.cond)
            imprad = np.array(rootgrp["b2stel_she_bal"][1])
            imprad = np.cumsum(-1*imprad[RING][XPTs[-1]:])
            eael = np.sum(np.array(rootgrp["eirene_mc_eael_she_bal"]),axis=0)

            # plt.plot(SOLring1.Spar,np.array(quantities2d["qpar"]*quantities2d["Area"])[RING][-49:],color="C"+str(colornum))
            # plt.plot(SOLring1.Spar,imprad[RING][-49:],color="C"+str(colornum))
            # plt.plot(SOLring1.Spar,np.cumsum(-1*eael[RING][-49:]),color="C"+str(colornum))
            Tu.append(SOLring1.te[-1])
            if (SOLring1.te[0]<30):
                Tt.append(SOLring1.te[0])
                totalpress = (SOLring1.te+SOLring1.ti+SOLring1.FlowVelocity**2)*SOLring1.ne
                fmom.append((SOLring1.te[0]+SOLring1.ti[0])*SOLring1.ne[0]/((SOLring1.te[-1]+SOLring1.ti[-1])*SOLring1.ne[-1]))
            Qu.append(quantities2d["qpar"][RING][XPTs[-1]]*quantities2d["Area"][RING][XPTs[-1]])
            QF.append(imprad[-1])
            Nu.append(SOLring1.ne[-1])
            # Ccalcs.append(SOLring1.returnCalculationsC())
            radfraction.append(imprad[-1]/((quantities2d["qpar"]*quantities2d["Area"])[RING][-50]))
            colornum=colornum+1
            Cdls.append(ChInt(SOLring1.Spar, SOLring1.B, SOLring1.Spar[-1],SOLring1.Spar[-1] ,pos))
            print("name is",fileName[-26:-11] )
            markerstyle = "o"
            print("input particle flux",np.sum(rootgrp['fne'][1][1,XPTs[0]:XPTs[1]])+np.sum(rootgrp['fne'][1][1,XPTs[-2]:XPTs[-1]]))
            if not detached:
                if SOLring1.te[0]<5:
                    detached = 1

                    templabel = "Open"
                    tempcolor = gridcolors[2]
                    heatcolor = "#FF9B42"
                    if "Tight" in folder:
                        templabel = "Tight"
                        tempcolor = gridcolors[0]
                        heatcolor = "#8F250C"
                        markerstyle = "^"
                    Spar= np.cumsum(quantities2d["sdiff"][RING,57:-1])
                    Sparinner = np.cumsum(quantities2d["sdiff"][RING,1:56])
                    print("Tu is",np.max(quantities2d["te"][RING,57:-1]))
                    # print(np.array(rootgrp["na"][1]))
                    print("nthresh is",SOLring1.ne[-1])
                    rad = np.sum(np.array(rootgrp['b2stel_she_bal']),axis=0)

                    radWeighted_field = np.sum(quantities2d["TotalField"]*rad)
                    radWeighted_field = radWeighted_field/np.sum(rad)
                    L = np.zeros(quantities2d["te"].shape)
                    for ind0 in range(len(quantities2d["te"])):
                        for ind1 in range(len(quantities2d["te"][ind0])):
                            L[ind0][ind1] = LfuncN(quantities2d["te"][ind0][ind1])
                    radWeighted_press = np.sum(rad/quantities2d["V"])
                    radWeighted_press = np.sum(0.03*quantities2d["ne"]**2*L*quantities2d["V"])
                    radWeighted_press = radWeighted_press/np.sum(quantities2d["ne"]**2*quantities2d["V"])

                    radWeighted_func = np.sum(quantities2d["ne"]**2*L*rad)
                    radWeighted_func = radWeighted_func/np.sum(rad)
                    # plt.show()
                    # plt.hist(L.flatten(),weights=(quantities2d["ne"]*quantities2d["V"]).flatten())
                    # plt.savefig("test"+power+".png")
                    print("average field is ", radWeighted_field)
                    print("average pressure is is ", radWeighted_press)
                    print("average func is is ", radWeighted_func)
                    print("radiation is",np.sum(rad))
                    axs2.plot(Spar,quantities2d["te"][RING,57:-1],
                            color=tempcolor,label=templabel,linewidth=2)
                    # axs2.plot(SOLring1.Spar,SOLring1.te,
                    #         color=tempcolor,label=templabel)                    

                    fluxlim = np.gradient(SOLring1.te)/np.gradient(SOLring1.Spar)
                    fluxlim = fluxlim*SOLring1.te**(5/2)
                    fluxlim = SOLring1.cond/fluxlim


                    # plt.plot(SOLring1.Spar,SOLring1.cond)
                    # plt.ylim([0,5E7])
                    # plt.xlim([0,70])
                    # plt.show()
                    axs3.plot(Sparinner,quantities2d["te"][RING,1:56],
                            color=tempcolor,label=templabel)

                    Rrsep = 1000*(quantities2d["r"][:,midplaneix]-quantities2d["r"][SEPARATRIX,midplaneix])
                    
                    popt, pcov = curve_fit(expFunc, Rrsep[SEPARATRIX+1:], quantities2d["qpar"][SEPARATRIX+1:,XPTs[-1]],p0=[np.amax(quantities2d["qpar"][:,XPTs[-1]]),4])
                    print("width is",str(np.round(popt[1],1)))
                    heatwidthlabel = templabel+r", $\lambda_{q}$ = "+str(np.round(popt[1],1))+"mm"
                    ionisloss = np.sum(np.array(rootgrp["eirene_mc_eael_she_bal"]),axis=0)
                    ionisloss = np.sum(ionisloss[SEPARATRIX:-1,midplaneix:XPTs[-1]],axis=1)
                    ionisloss = np.abs(ionisloss)/quantities2d["Area"][SEPARATRIX:-1,XPTs[-1]]
                    radTrans = (quantities2d["radHeate"]+quantities2d["radHeati"])[SEPARATRIX:-1,midplaneix:XPTs[-1]]-(quantities2d["radHeate"]+quantities2d["radHeati"])[SEPARATRIX+1:,midplaneix:XPTs[-1]]
                    inmidplane = int((XPTs[1]+XPTs[0])/2)
                    # radTrans = (quantities2d["radHeate"]+quantities2d["radHeati"])[SEPARATRIX:-1,XPTs[0]:inmidplane]-(quantities2d["radHeate"]+quantities2d["radHeati"])[SEPARATRIX+1:,XPTs[0]:inmidplane]
                    # print("total rad trans is",np.sum(radTrans)*10**(-6))
                    radTrans = np.sum(radTrans,axis=1)/quantities2d["Area"][SEPARATRIX:-1,XPTs[-1]]
                    # print("average heat is",(trapz(SOLring1.cond,SOLring1.Spar)/np.sum(SOLring1.Spar))**(2/7))
                    # plt.plot(SOLring1.Spar,SOLring1.cond)
                    # plt.show()
                    axs0.plot(Rrsep,10**(-6)*(quantities2d["qpar"])[:,XPTs[-1]],color=heatcolor,label = heatwidthlabel,marker=markerstyle)
                    axs4.plot(Rrsep,(quantities2d["ne"])[:,midplaneix],color=heatcolor,label = templabel,marker=markerstyle)
                    # axs5.plot(Rrsep,(quantities2d["te"])[:,midplaneix],color=heatcolor,label = heatwidthlabel,marker=markerstyle)
                    axs5.plot(Rrsep[:],(quantities2d["te"])[:,midplaneix],color=heatcolor,label = heatwidthlabel,marker=markerstyle)
                    halflen = int(len(quantities2d["te"][RING,57:-1])/2)
                    # axs6.plot(Spar[:65],(quantities2d["qpar"])[RING,57+halflen:-1][::-1],color=heatcolor,label = heatwidthlabel)
                    # axs6.plot(Spar[:65],(quantities2d["cond"])[RING,57+halflen:-1][::-1],color=heatcolor,
                    #           linestyle = "--",label = "conductive")
                    radheat = (quantities2d["radHeate"]+quantities2d["radHeati"])
                    print("heat flux entering outer",10**(-6)*np.sum(radheat[SEPARATRIX-1,XPTs[-2]:XPTs[-1]]))
                    print("heat flux entering inner",10**(-6)*np.sum(radheat[SEPARATRIX-1,XPTs[0]:XPTs[1]]))
                    print("i/o ratio",np.sum(radheat[SEPARATRIX,XPTs[-2]:XPTs[-1]])/np.sum(radheat[SEPARATRIX,XPTs[0]:XPTs[1]]))
                    ion_heat,elec_heat = CalculateSinks(rootgrp)
                    totalsink = np.abs(np.sum(ion_heat["sinks"],axis=0)+np.sum(elec_heat["sinks"],axis=0))
                    print("inner loss",10**(-6)*(np.sum(totalsink[SEPARATRIX-1:,:XPTs[2]])+np.sum(totalsink[:SEPARATRIX-1,:XPTs[0]])+np.sum(totalsink[:SEPARATRIX-1,XPTs[1]:XPTs[2]])))
                    print("outer loss",10**(-6)*(np.sum(totalsink[SEPARATRIX-1:,XPTs[2]+1:])+np.sum(totalsink[:SEPARATRIX-1,XPTs[4]:])+np.sum(totalsink[:SEPARATRIX-1,XPTs[2]:XPTs[3]])))
                    print("inner target flux is",10**(-6)*np.sum((quantities2d["iHeat"]+quantities2d["elHeat"])[:,1]))
                    print("outer target flux is",10**(-6)*np.sum((quantities2d["iHeat"]+quantities2d["elHeat"])[:,-2]))

                    
                    axs6.plot(Rrsep[1:-1],rootgrp['fne'][1][1:-1,midplaneix],color=heatcolor,marker=markerstyle,label=templabel+" radial ion flux")
                    totalionsink = np.array(rootgrp['eirene_mc_papl_sna_bal'])
                    totalionsink = np.sum(totalionsink,axis=0)[1]#/np.array(rootgrp['vol'])

                    axs6.plot(Rrsep[1:-1],np.cumsum(totalionsink[1:-1,midplaneix]),color=heatcolor,linestyle="--",label=templabel+" ionisation")

                    # plot the total and conducted radial heat flux

                    radheataxs.plot(Rrsep[1:-1],1E-6*radheat[2:,midplaneix],color=heatcolor,marker=markerstyle,label=templabel+" radial heat flux")
                    radheataxs.plot(Rrsep[1:-1],1E-6*(np.array(rootgrp["fhe_cond"])+np.array(rootgrp["fhi_cond"]))[1][2:,midplaneix],color=heatcolor,linestyle="--",label=templabel+" conducted radial heat flux")
                    
                    #plot radial temperature profile at the midplane
                    upstempaxs.plot(Rrsep[SEPARATRIX-1:-1],quantities2d["te"][SEPARATRIX-1:-1,midplaneix],color=heatcolor,marker=markerstyle,label=templabel+" radial heat flux")


                    radtransloss = -1*np.cumsum(radheat[RING,XPTs[4]-1:-2]-radheat[RING+1,XPTs[4]-1:-2])[::-1]
                    volumetricloss = -1*np.cumsum(totalsink[RING,XPTs[4]:-1])[::-1]

                    radtransloss = -1*np.cumsum(radheat[RING,midplaneix-1:-2]-radheat[RING+1,midplaneix-1:-2])[::-1]
                    radtransloss = radtransloss-radtransloss[-1]
                    volumetricloss = -1*np.cumsum(totalsink[RING,midplaneix:-1])[::-1]
                    
                    pe,pi,fmox_flua, eirene, radtrans = pressureBalance(rootgrp)
         
                    # plot losses along a SOL ring
                    # DLSaxs.plot(SOLring1.Spar,np.cumsum(radtrans[RING,midplaneix:-1][::-1]),
                    # color=heatcolor,label="dynamic",linestyle="--",)
                    DLSaxs.plot(SOLring1.Spar,(pe)[RING,midplaneix:-1][::-1],
                    color=heatcolor,label="electron static")
                    # DLSaxs.plot(SOLring1.Spar,-1*np.cumsum(quantities2d["imprad"][RING,midplaneix:-1])[::-1],
                    # color = heatcolor,linestyle="-.")
                    # DLSaxs.plot(SOLring1.Spar,-1*radtransloss,
                    # color=heatcolor,linestyle = "--")
                    # DLSaxs.plot(SOLring1.Spar[:len(quantities2d["elHeat"][RING])-XPTs[4]-1],-1*np.cumsum(np.sum(rootgrp["eirene_mc_eael_she_bal"],axis=0)[RING,XPTs[4]:-1])[::-1],
                    # color=heatcolor,linestyle = ":")
                    # DLSaxs.plot(SOLring1.Spar,-1*np.array(radtransloss)+np.array(volumetricloss),
                    # color=heatcolor,linestyle = ":")
                    # DLSaxs.plot(SOLring1.Spar,SOLring1.ne*SOLring1.te,
                    # color=heatcolor)                
                    peakheatflux = -1*np.min(radtransloss)
                    maximp = np.max(-1*np.cumsum(quantities2d["imprad"][RING,midplaneix:-1])[::-1])
                    maxradTrans = np.max(-1*np.cumsum(radheat[RING-1,midplaneix:-1]-radheat[RING,midplaneix:-1])[::-1])
                    maxionis = np.max(-1*np.cumsum(np.sum(rootgrp["eirene_mc_eael_she_bal"],axis=0)[RING,XPTs[4]:-1])[::-1])
                
                    print("")
                    print("rad trans is", np.min(radtransloss))
                    print("imp is", maximp/peakheatflux)
                    print("upstream press is",SOLring1.ne[-1]*SOLring1.te[-1])
                    print("imp efficiency", maximp/(SOLring1.ne[-1]*SOLring1.te[-1]))
                    print("ionisation is", maxionis/peakheatflux)
                    print("")
        fig0.show()
        label = 0
        Tu= np.array(Tu)
        Qu = np.array(Qu)
        Nu = np.array(Nu)
        peakheatLoad = np.array(peakheatLoad)
        print(Nu)
        C = Nu#/(Qu**(5/7))
        collis = 10**(-16)*Lpar*Nu/(Tu**2)
        Cdls = np.array(Cdls)
        index0 = determineC0(np.array(frontpos),C)



        if "Tight" in folder:
            axspress.plot(Tt,fmom,color=gridcolors[0],label="Tight")
            label = "Tight pos"
            if collisionality:
                axs1.plot(collis,frontpos,marker="^",label=label,color=gridcolors[0])
                ax12.plot(collis,peakheatLoad*10**(-6),label="Tight flux",color="#8F250C",linestyle="-.")
            else:
                axs1.plot(C,frontpos,marker="^",label=label,color=gridcolors[0])
                
                ax12.plot(C,ionFlux,label="Tight flux",color="#8F250C",linestyle="-.")

        else:
            axspress.plot(Tt,fmom,color=gridcolors[2],label="Open")
            label = "Open pos"

            if collisionality:
                axs1.plot(collis,frontpos,marker="o",label=label,color=gridcolors[2])
                ax12.plot(collis,peakheatLoad*10**(-6),label="Open flux",color="#FF9B42",linestyle="--")
            else:
                axs1.plot(C,frontpos,marker="o",label=label,color=gridcolors[2])
                ax12.plot(C,ionFlux,label="Open flux",color="#FF9B42",linestyle="--")
    
    ylabel = "s"+r'$_{f,pol}$'+" [m]"
    poslabel = "s"+r'$_{||}$'+" [m]"
    poslabel0 = "R-R"+r'$_{sep}$'+" [mm]"
    axs1.set_xlabel(r"$C/C_{t}$")
    axs1.set_xlabel("n"+r"$_{u}$" + " [m"+r"$^{-3}$" +"]")


    axs1.set_ylabel(ylabel,color=gridcolors[0])
    ax12.set_ylabel(r"$\Gamma_{t}$"+" [s"+r"$^{-1}$"+"]",color="#8F250C")
    ax12.tick_params(axis='y', labelcolor="#8F250C")
    axs1.tick_params(axis='y', labelcolor=gridcolors[0])
    axs1.legend()
    ax12.legend()
    axs1.set_title(power)
    if collisionality:
        axs1.set_xlabel(r"$\nu^{*}_{SOL,e}$" )
        ax12.set_ylabel("peak heat load"+" [MWm"+r"$^{-2}$"+"]",color="#8F250C")
        fig1.savefig("Figures/BaffleFluxEvolution"+power+".png",dpi=800,bbox_inches='tight')#
    else:
        fig1.savefig("Figures/BaffleEvolution"+power+".png",dpi=800,bbox_inches='tight')#

    axs0.set_xlabel(poslabel0)
    axs0.set_ylabel("q"+r'$_{||}$'+" [MWm"+r'$^{-2}$'+"]")

    # axs0.set_ylabel("ionisation loss")
    axs0.set_xlim([-5,27])
    axs0.set_title(power)
    fig0.tight_layout()
    axs0.legend()
    fig0.savefig("Figures/Baffle_Profiles/baffle_Heat_faloff"+power+".png",dpi=800,bbox_inches='tight')

    axs4.set_xlabel(poslabel0)
    axs4.set_ylabel("n"+r'$_{e}$'+" [m"+r'$^{-3}$'+"]")
    fig4.tight_layout()
    axs4.legend()
    axs4.set_title(power)
    fig4.savefig("Figures/Baffle_Profiles/baffle_densUpstream"+power+".png",dpi=800,bbox_inches='tight')
    axs5.set_xlabel(poslabel0)
    axs5.set_ylabel("T [eV]")
    axs5.set_title(power)
    fig5.tight_layout()
    axs5.legend()
    fig5.savefig("Figures/Baffle_Profiles/baffle_TempUpstream"+power+".png",dpi=800,bbox_inches='tight')
    
    axs6.set_xlabel(poslabel0)
    axs6.set_ylabel("particle flux [s"+r"$^{-1}$"+"]")
    axs6.set_title(power)
    axs6.legend()
    fig6.tight_layout()
    fig6.savefig("Figures/Baffle_Profiles/upstream_particle_flux"+power+".png",dpi=800,bbox_inches='tight')





    axs2.plot([0,0],[0,200],color="black",linestyle="--")
    axs2.annotate(text="upper target",xy=(1.5,95))
    axs2.plot([48.3,48.3],[0,200],color="black",linestyle="--")
    axs2.annotate(text="lower target",xy=(36,95))
    axs2.set_ylim([0,105])

    axs2.set_xlabel("s"+r'$_{||}$'+" [m]")
    axs2.set_ylabel("T [eV]")
    axs2.set_title(power)
    axs2.legend()
    fig2.savefig("Figures/outerTemp_Baffle"+power+".png",dpi=800,bbox_inches='tight')
    # fig0.savefig("Figures/heatlossBaffle.png",dpi=400,bbox_inches='tight')
    axs3.plot([0,0],[0,200],color="black",linestyle="--")
    axs3.annotate(text="lower target",xy=(1.5,40))
    axs3.plot([52.21,52.21],[0,200],color="black",linestyle="--")
    axs3.annotate(text="upper target",xy=(40,40))
    axs3.set_ylim([0,105])
    axs3.set_xlabel("s"+r'$_{||}$'+" [m]")
    axs3.set_ylabel("T [eV]")
    axs3.legend()
    fig3.savefig("Figures/innerTemp_Baffle"+power+".png",dpi=800,bbox_inches='tight')
    

    radheataxs.set_xlabel(poslabel0)
    radheataxs.set_ylabel("heat flux [MW]")
    radheataxs.set_title(power)
    radheataxs.legend()
    radheatfig.tight_layout()
    radheatfig.savefig("Figures/Baffle_Profiles/radHeatflux"+power+".png",dpi=800,bbox_inches='tight')
    
    upstempaxs.set_xlabel(poslabel0)
    upstempaxs.set_ylabel("T [eV]")
    upstempaxs.set_title(power)
    upstempaxs.legend()
    radheatfig.tight_layout()
    upstempfig.savefig("Figures/Baffle_Profiles/radTempUpstream"+power+".png",dpi=800,bbox_inches='tight')
    
    plt.show()


    DLSaxs.set_xlabel("s [m]")
    DLSaxs.set_ylabel("pressure")
    DLSaxs.set_title(power)
    DLSaxs.legend()
    DLSfig.tight_layout()
    DLSfig.savefig("Figures/Baffle_Profiles/pressureProfile"+power+".png",dpi=800,bbox_inches='tight')
    
    plt.show()



    axspress.set_xlabel("Tt")
    axspress.set_ylabel("fmom")
    figpress.savefig("Figures/Baffle_Profiles/fmom"+power+".png",dpi=800,bbox_inches='tight')
    
    plt.show()


def heatBalance_Multiple_Sims(Simlist,heatmode):
    
    plt.rcParams.update({'font.size': 12})
    barfig, baraxs = plt.subplots()
    threshfig, threshaxs = plt.subplots()
    heat_counts = 0
    if heatmode == "source_type":
        heat_counts =  {
        "nitrogen": [],
        "hydrogen": [],
        "non-radiative outer": [],
        "non-radiative inner": [],
        "non-radiative wall": [],

        }
    else:
        heat_counts =  {
        "divertor radiation": [],
        "main chamber SOL radiation": [],
        "core radiation": [],
        "non-radiative targets": [],
        "non-radiative wall": [],

        }   

    Powers = np.array([3,6,12])
    Ctight = []
    Copen = []
    for fileName in Simlist:

        rootgrp = Dataset(str(fileName)+"\\balance.nc", "r", format="NETCDF4")


        SOLring1 = 0
        RING = rootgrp["jsep"][0]+5
        SEPARATRIX = rootgrp["jsep"][0]+2

        quantities2d,SOLring1 = unpackSOLPS(str(fileName)+"\\balance.nc", -1,RING,Xpoint=52)
        quantities2d,rootg = return2d(str(fileName)+"\\balance.nc")

        # DETERMINE LOCATION OF X-POINTS
        XPTs = []
        for i in range(len(rootgrp['rightix'][0])):
            if rootgrp['rightix'][0][i] != i:
                XPTs.append(i)
        XPTs = np.array(XPTs)+1
        midplaneix = int((XPTs[-1]+XPTs[-2])/2)
        print(fileName)
        if "Tight" in fileName:
            Ctight.append(quantities2d["ne"][RING][midplaneix])
        else:
            Copen.append(quantities2d["ne"][RING][midplaneix])      

        heat_counts = heatBalance(fileName,rootgrp,quantities2d,XPTs,heatmode=heatmode,heat_counts=heat_counts)
        
    width = 0.5

    threshaxs.plot(Powers,Ctight,marker="o",linewidth=2,label="Tight")
    threshaxs.plot(Powers,Copen,marker="^",linewidth=2,label="Open")
    threshaxs.set_xlabel("Input Power [MW]")
    threshaxs.set_ylabel("threshold density [m"+r"$^{-3}$"+"]")
    threshaxs.legend()
    threshfig.savefig("Figures\\threshbaffle.png",dpi=800,bbox_inches="tight")

    bottom = np.zeros(len(threshSims))
    print(heat_counts)
    heatcolors = ["#442288","#6CA2EA","#B5D33D","#FED23F","#EB7D5B"]
    case = (
        "3MW \n open",
        "3MW \n tight",
        "6MW \n open",
        "6MW \n tight",
        "12MW \n open",
        "12MW \n tight"

    )

    for c in range(len(threshSims)):
        heatnum = 0
        for boolean, heat_count in heat_counts.items():
            if c==0:
                p = baraxs.bar(case[c], heat_count[c], width, label=boolean, bottom=bottom[c],color=heatcolors[heatnum])
            else:
                p = baraxs.bar(case[c], heat_count[c], width,  bottom=bottom[c],color=heatcolors[heatnum])
            bottom[c] += heat_count[c]
            heatnum = heatnum+1
    baraxs.legend(fontsize=12)
    baraxs.set_ylabel("portion of input power",fontsize=16)
    barfig.savefig("Figures//heat_"+heatmode+"_baffle.png",dpi=400,bbox_inches="tight")
    barfig.show()   
    plt.rcParams.update({'font.size': 16})
         
def plotGifs():
    folderList = []

    folderList = [
    # "D:\\my stuff\\PhD\\IsolatedAnalysis\\balfiles\MAST_Tight",
    "D:\\my stuff\\PhD\\IsolatedAnalysis\\balfiles\MAST_Open",
    ]

        

    for folder in folderList:
        Files = os.listdir(str(folder))
        Files = natsorted(Files)


        if "Tight" in folder:
            label = "Tight"
        else:
            label = "Open"
        for File in Files:

            if File == "input.dat":
                continue
            fileName = str(folder)+"/"+str(File)
            rootgrp = Dataset(str(fileName), "r", format="NETCDF4")

            # print(rootgrp)
            Xpoint = -1
            SOLring1 = 0
            print(rootgrp["jsep"][0])
            RING = rootgrp["jsep"][0]+4

            quantities2d,SOLring1 = unpackSOLPS(fileName+"\\balance.nc", -1,RING,Xpoint=52)
            quantities2d,rootg = return2d(fileName+"\\balance.nc")
            plot2d(rootg,quantities2d["te"])
        io.mimsave('Figures\\Te.gif', image_list, duration=0.5)
import matplotlib.path as mpltPath



def readParam(fname,paramname):
    # paramnames include: pdena for atom dens, pdenm for mol dens, volume for triangle volume 
    fortFile = "balFiles//"+fname+".46"
    dataFort= open(fortFile)
    tline = dataFort.readlines(1)
    linenum = 0
    while True:
        tline = dataFort.readlines(1)
        if  paramname in tline[0]:
            break

    data = []
    while True:
        tline = dataFort.readlines(1)
        co = re.findall("\D+\d+\.\d+\D+\d+",tline[0])

        for i in range(len(co)):
            data.append(float(co[i]))

        if  '*eirene' in tline[0]:
            break
    return data

def readParam44(fname,paramname):
    # paramnames include: pdena for atom dens, pdenm for mol dens, volume for triangle volume 
    fortFile = fname+"//fort.44"
    dataFort= open(fortFile)
    tline = dataFort.readlines(1)
    linenum = 0
    while True:
        tline = dataFort.readlines(1)
        if  paramname in tline[0]:
            break

    data = []
    while True:
        tline = dataFort.readlines(1)
        co = re.findall("\D+\d+\.\d+\D+\d+",tline[0])

        for i in range(len(co)):
            data.append(float(co[i]))

        if  '*eirene' in tline[0]:
            break
    return data







def plotNeutrals():
    fig = plt.figure(figsize=(4,5))
    
    gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
    axs = gs.subplots(sharey='row')
    axscounter = 0
    for fname in ["MAST_Open\\3MW\\ne2.5","MAST_Tight\\3MW\\ne2.0"]:# simulations at threshold of detachment
    # for fname in ["MAST_Open\\12MWpump\\ne7.0","MAST_Tight\\12MW\\ne8.0"]:
        # get triangle x/y data
        fd = open('balfiles\\'+fname+'.33','r')   
        dataTriangImport =  fd.readlines()
        dataTriang = []
        for i in range(len(dataTriangImport)):
            co = re.findall("\D+\d+\.\d+\D+\d+",dataTriangImport[i])
            for j in range(len(co)):
                dataTriang.append(float(co[j]))
        dataTriang = np.array(dataTriang)/100

        datax = dataTriang[0:int(len(dataTriang)/2)]
        datay = dataTriang[int(len(dataTriang)/2):]

        # get triangle indices
        fd = open('balFiles\\'+fname+ '.34','r')   
        indices = np.loadtxt(fd,skiprows=1,usecols=(1,2,3))
        axs[axscounter].set_aspect('equal')
        # axs[counter].set_xlim([0.4,0.9])
        # axs[counter].set_ylim([-1.7,-1])
        if axscounter!=0:
            # axs[axscounter].xaxis.set_visible(False)
            axs[axscounter].yaxis.set_visible(False)
            
        # else:
        axs[axscounter].set_xlabel("R [m]")
        axs[axscounter].set_ylabel("Z [m]")
        plotquantity = 0
        norm = 0

        # get relevant EIRENE quantities
        Atomdensities = np.array(readParam(fname,"pdena"))
        Moldensities = np.array(readParam(fname,"pdenm"))
        Vtria = np.array(readParam(fname,"volume"))
        Eatom = np.array(readParam(fname,"edena"))
        Emol = np.array(readParam(fname,"edenm"))

        Dmol = np.array(Moldensities)
        Datom = Atomdensities
        Dtot = Dmol+Datom
        Etot = Eatom+Emol



        plotquantity = np.log10(Dtot*1E6+1)
        clabel = "log(n" +r"$_{0}$"+")"
        norm = mpl.colors.Normalize(vmin=15, vmax=19)
        cmap = cm.plasma
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        TotalDivV = 0
        TotalMainV = 0
        TotalDivD = 0
        TotalMainD = 0
        TotalDivE = 0

        P1 = np.array([[1.5,1.6,1.6,1.5,1.5],[-1.65,-1.65,-1.6,-1.6,-1.65]])
        P1 = np.transpose(P1)
        P2 = np.array([[1.45,1.55,1.55,1.45,1.45],[-0.1,-0.1,0.1,0.1,-0.1]])
        P2 = np.transpose(P2)

        divertorPolygon = mpltPath.Path(P1)
        MainChamberPolygon = mpltPath.Path(P2)

        for i in range(len(indices)):
            ind0 = int(indices[i][0])-1
            ind1 = int(indices[i][1])-1
            ind2 = int(indices[i][2])-1
            x = [datax[ind0],datax[ind1],datax[ind2]]
            y =[datay[ind0],datay[ind1],datay[ind2]]
            Point = (np.mean(x),np.mean(y))

            color1 = m.to_rgba(0)
            
            containedDiv = divertorPolygon.contains_point(Point) 
            containedMain = MainChamberPolygon.contains_point(Point) 
            if containedDiv:
                TotalDivD = TotalDivD+Dtot[i]*Vtria[i]
                TotalDivV = TotalDivV+Vtria[i]
                TotalDivE = TotalDivE+Etot[i]*Vtria[i]
            elif containedMain:
                TotalMainD = TotalMainD+Dtot[i]*Vtria[i]
                TotalMainV = TotalMainV+Vtria[i]      
            color1 = m.to_rgba(plotquantity[i])
            axs[axscounter].fill(x,y,color=color1,linewidth=0.01)

        avDensDiv = TotalDivD/TotalDivV
        avDensMain = TotalMainD/TotalMainV

        print("average temperature is",TotalDivE/TotalDivD)
        print("average divertor is",avDensDiv)
        print("average main is",avDensMain)
        
        if "Tight" in fname:
            plotWALL("balFiles//MAST_Tight//input.dat",axs[axscounter])
        else:
            plotWALL("balFiles//MAST_Open//input.dat",axs[axscounter])
        # counter = counter+1

        axscounter = axscounter+1
    axs[1].set_title("3MW",x=-0.05)
    cbarax = plt.axes([1.1, 0.22, 0.01, 0.6], facecolor='none')
    cb1 = mpl.colorbar.ColorbarBase(cbarax, cmap=mpl.cm.plasma, orientation='vertical',norm=norm,label=clabel)  
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(pad=0)

    plt.savefig("Figures//Ddensity_Baffle.png",dpi=1000,bbox_inches='tight')
    plt.show()

def plotEnergyCost(simFiles,power):
    fig = plt.figure(figsize=(4,5))
    gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
    axs = gs.subplots(sharey='row')
    axscounter = 0
    files = 0
    vmax = 0.5E23
    if power == "3MW":
        files = [simFiles[0],simFiles[1]]
    elif power =="6MW":
        files = [simFiles[2],simFiles[3]]
    else:
        vmax = 2E23
        files = [simFiles[4],simFiles[5]]
    for fileName in files:

        rootgrp = Dataset(str(fileName), "r", format="NETCDF4")
        eirene_mc_papl_sna_bal = np.array(rootgrp['eirene_mc_papl_sna_bal'])
        eirene_mc_eael_she_bal = np.array(rootgrp['eirene_mc_eael_she_bal'])
        eirene_mc_eael_she_bal = np.sum(eirene_mc_eael_she_bal,axis=0)
        eirene_mc_papl_sna_bal = np.sum(eirene_mc_papl_sna_bal,axis=0)[1]
        average_cost = (np.sum(eirene_mc_eael_she_bal)/np.sum(eirene_mc_papl_sna_bal))/1.60E-19
        print("average cost is",average_cost)
        print("ionisation is",np.sum(eirene_mc_papl_sna_bal))
        plotvar = eirene_mc_eael_she_bal/eirene_mc_papl_sna_bal
        plotvar=plotvar/1.602E-19
        grid = "Tight"
        if "Open" in fileName:
            grid = "Open"
            rootgrp = Dataset(str(fileName), "r", format="NETCDF4")
            eirene_mc_papl_sna_bal = np.array(rootgrp['eirene_mc_papl_sna_bal'])
            eirene_mc_eael_she_bal = np.array(rootgrp['eirene_mc_eael_she_bal'])
        if axscounter !=0:
            axs[axscounter].yaxis.set_visible(False)
        plotvar = np.array(rootgrp['eirene_mc_papl_sna_bal'])
        plotvar = np.sum(plotvar,axis=0)[1]
        plotvar = plotvar/np.array(rootgrp["vol"])
        # plotvar = rootgrp["ne"]
        axs[axscounter].set_aspect('equal')
        norm = mpl.colors.Normalize(vmin=1E18, vmax=vmax)
        cmap = cm.plasma
        plot2d(rootgrp,plotvar,grid,ilim=0,norm=norm,
               cmap=cmap,axs=axs[axscounter])
        axs[axscounter].set_ylim([-2.2,0])
        axs[axscounter].set_xlabel("R [m]")
        axs[axscounter].set_ylabel("Z [m]")
        axscounter = axscounter+1
    axs[1].set_title(power,x=-0.05)

    cbarax = plt.axes([1.1, 0.37, 0.01, 0.28], facecolor='none')
    cb1 = mpl.colorbar.ColorbarBase(cbarax, cmap=mpl.cm.plasma, 
                orientation='vertical',norm=norm,label="ionisation source [m"+r"$^{-3}$"+"s"+r"$^{-1}$"+"]")  

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(pad=0)

    plt.savefig("Figures/Baffle_Profiles/"+power+"_ionis.png", dpi=1000,bbox_inches='tight')

    plt.show()
    plt.close()

    
def plotClosure(power):

    fig = plt.figure(figsize=(4,5))
    
    gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
    axs = gs.subplots(sharey='row')
    axscounter = 0
    files = 0
    
    if power == "3MW":
            files = [simFiles[0],simFiles[1]]
    elif power == "6MW":

            files = [simFiles[2],simFiles[3]]
    else:
            files = [simFiles[4],simFiles[5]]

    for fileName in files:
        rootgrp = Dataset(str(fileName), "r", format="NETCDF4")

        # DETERMINE LOCATION OF X-POINTS
        XPTs = []
        for i in range(len(rootgrp['rightix'][0])):
            if rootgrp['rightix'][0][i] != i:
                XPTs.append(i)
        XPTs = np.array(XPTs)+1


        quantities2d,rootg = return2d(fileName)
        eirene_mc_papl_sna_bal = np.sum(np.array(rootgrp['eirene_mc_papl_sna_bal']),axis=0)[1]
        plotvar = eirene_mc_papl_sna_bal.copy()*0

        currentsum = 0
        ionTarg_Current = np.array(quantities2d['parFluxi'])[1]
        ionTarg_Current = np.sum(ionTarg_Current[:,-1])
        for i in range(len(eirene_mc_papl_sna_bal[0])-1,-1,-1):
            currentsum = currentsum+np.sum(eirene_mc_papl_sna_bal[:,i])
            plotvar[:,i] = currentsum
        plotvar = np.abs(plotvar/ionTarg_Current)



        grid = "Tight"
        if "Open" in fileName:
            grid = "Open"

        axs[axscounter].set_aspect('equal')
        if axscounter!=0:
            if power =="3MW":
                axs[axscounter].annotate(text="b)",xy=(1.45,-1.35))
            elif power =="6MW":
                axs[axscounter].annotate(text="d)",xy=(1.45,-1.35))
            else:           
                axs[axscounter].annotate(text="d)",xy=(1.45,-1.35))     
            axs[axscounter].yaxis.set_visible(False)
        else:
            if power =="3MW":
                axs[axscounter].annotate(text="a)",xy=(1.45,-1.35))
            elif power =="6MW":
                axs[axscounter].annotate(text="c)",xy=(1.45,-1.35))
            else:           
                axs[axscounter].annotate(text="c)",xy=(1.45,-1.35))   

        axs[axscounter].set_xlabel("R [m]")
        axs[axscounter].set_ylabel("Z [m]")
        axs[axscounter].set_ylim([-2.2,-1.2])
        axs[axscounter].set_xlim([0.6,1.9])
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        cmap = cm.plasma
        plot2d(rootgrp,plotvar,grid,ilim=0,norm=norm,
               cmap=cmap,axs=axs[axscounter])
        axscounter = axscounter+1
    axs[1].set_title(power,x=-0.05)
    cbarax = plt.axes([1.02, 0.37, 0.01, 0.25], facecolor='none')
    cb1 = mpl.colorbar.ColorbarBase(cbarax, cmap=mpl.cm.plasma, orientation='vertical',norm=norm,label="neutral trapping")  

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(pad=0)

    plt.savefig("Figures/Baffle_Profiles/"+power+"closure.png", dpi=1000,bbox_inches='tight')

    plt.show()
    plt.close()

def plotLosses(power,degDetachment):
    """" plot volumetric plasma losses in 2d grid for open and tight grids"""
    fig1,axs1 = plt.subplots(1,1)
    files = 0
    vmax = 2
    if power == "3MW":
        vmax = 1
        if degDetachment == "Deep":
            files = [deepsims[0],deepsims[1]]    
        else:
            files = [threshSims[0],threshSims[1]]
    elif power == "6MW":
        vmax = 3
        if degDetachment == "Deep":
            files = [deepsims[2],deepsims[3]]    
        else:
            files = [threshSims[2],deepsims[2]]
    else:
        vmax = 6
        if degDetachment == "Deep":
            files = [deepsims[4],deepsims[5]]    
        else:
            files = [threshSims[4],threshSims[5]]


    fig = plt.figure(figsize=(4,5))
    
    gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
    axs = gs.subplots(sharey='row')

    vmin = 0

    # radialfig,radialaxs = plt.subplots()
    # poloidalfig,poloidalaxs = plt.subplots()
    axscounter = 0
    
    for fileName in files:
        rootgrp = Dataset(str(fileName), "r", format="NETCDF4")

        # DETERMINE LOCATION OF X-POINTS
        XPTs = []
        for i in range(len(rootgrp['rightix'][0])):
            if rootgrp['rightix'][0][i] != i:
                XPTs.append(i)
        XPTs = np.array(XPTs)+1
        midplaneix = int((XPTs[-1]+XPTs[-2])/2)
        SEPARATRIX= rootgrp["jsep"][0]+2
        quantities2d,rootg = return2d(fileName)
       
        RING = rootgrp["jsep"][0]+4

        grid = "Tight"
        if "Open" in fileName:
            grid = "Open"


        if axscounter!=0:
            axs[axscounter].yaxis.set_visible(False)


        axs[axscounter].set_xlabel("R [m]")
        axs[axscounter].set_ylabel("Z [m]")

        cmap = cm.plasma
        ion_heat,elec_heat = CalculateSinks(rootgrp)
        print("total v is",np.sum(quantities2d["V"]))
        totalsink = np.abs(np.sum(ion_heat["sinks"],axis=0)+np.sum(elec_heat["sinks"],axis=0))
        Rrsep =1000*(quantities2d["r"][:,midplaneix]-quantities2d["r"][SEPARATRIX,midplaneix])
        # radialaxs.plot(Rrsep[RING-2:],np.sum(totalsink[RING-2:,XPTs[2]:],axis=1))
        # radialaxs.plot(Rrsep[RING-2:],quantities2d["te"][RING-2:,midplaneix])
        totalsink = 10**(-6)*totalsink/quantities2d["V"]
        totalsink = 10**(-6)*np.abs(np.array(rootgrp["b2stel_she_bal"][1]))/quantities2d["V"]
        totalsink = 10**(-3)*np.abs(np.array(rootgrp["ua"][1]))
        totalsink = np.abs(np.array(rootgrp["ua"][1]))/np.sqrt((quantities2d["ti"]+quantities2d["te"])*1.6E-19/(2*1.67E-27))
        # totalsink =np.abs(np.array(rootgrp["fhe_cond"][0]+rootgrp["fhi_cond"][0]))/np.abs(quantities2d["iHeat"]+quantities2d["elHeat"])
                
        
        norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        plot2d(rootgrp,totalsink,grid,norm=norm,
               cmap=cmap,axs=axs[axscounter],ilim=0,plotmode="real")

        axs[axscounter].set_title(grid +" Baffle")
        axscounter = axscounter+1
    cbarax = plt.axes([1.1, 0.22, 0.01, 0.6], facecolor='none')
    cb1 = mpl.colorbar.ColorbarBase(cbarax, cmap=mpl.cm.plasma, orientation='vertical',norm=norm,label="Mach number")
    # cb1 = mpl.colorbar.ColorbarBase(cbarax, cmap=mpl.cm.plasma, orientation='vertical',norm=norm,label="nitrogen radiation density [MWm"+r"$^{-3}$"+"]")
    # cb1 = mpl.colorbar.ColorbarBase(cbarax, cmap=mpl.cm.plasma, orientation='vertical',norm=norm,label="n [m"+r"$^{-3}$"+"]") 

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(pad=0)

    plt.savefig("Figures/Baffle_Profiles/"+power+"_"+"powerSink"+"_"+degDetachment+"_"+".png", dpi=1000,bbox_inches='tight')

    plt.show()
    plt.close()


def plotEvolution(power):
    plt.rcParams.update({'font.size': 10})
    """" plot volumetric plasma losses in 2d grid for open and tight grids"""
    fig,axslist = plt.subplots(1,2,figsize=(6,3))
    contfig,contaxs = plt.subplots(1,2)
    contaxs[0].set_aspect("equal")
    contaxs[1].set_aspect("equal")
    files = 0
    vmax = 2
    geometry = "open"
    if power == "3MW":
        files = [threshSims[0],threshSims[1],deepsims[0],deepsims[1]]   
    elif power == "6MW":
        files = [threshSims[2],threshSims[3],deepsims[2],deepsims[3]]   
    else:
        files = [threshSims[4],threshSims[5],deepsims[4],deepsims[5]]   
    for folder in ["C:\\Users\cyd cowley\Desktop\PhD\IsolatedAnalysis\\balFiles\Mast_Open\\"+power+"\\",
                   "C:\\Users\cyd cowley\Desktop\PhD\IsolatedAnalysis\\balFiles\Mast_Tight\\"+power+"\\"]:
        axsnum = 0
        cmap = 0
        files = os.listdir(folder)
        # files = os.listdir("C:\\Users\cyd cowley\Desktop\PhD\IsolatedAnalysis\\balFiles\Mast_Open\\6MW")
        files = natsorted(files)
        if "Tight" in folder:
            print("yes tight")
            axsnum = 1
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",["#F51A1A","#691B09"])
            plotWALL("balFiles\MAST_Tight//input.dat", contaxs[axsnum])
            files = files[1:]
        else:
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",["#FF9B42","#974F10"])
            plotWALL("balFiles\MAST_Open//input.dat", contaxs[axsnum])
            files = files[1:]

 
    
        m = matplotlib.cm.ScalarMappable(norm = mpl.colors.Normalize(vmin=0, vmax=len(files)), cmap=cmap)
        counter = 0
        
        for file in files:

            print(folder,file)
            colorplot = m.to_rgba(counter)  
            fileName = folder+file
            # fileName = "C:\\Users\cyd cowley\Desktop\PhD\IsolatedAnalysis\\balFiles\Mast_Open\\6MW\\"+file
            rootgrp = Dataset(str(fileName)+"\\balance.nc", "r", format="NETCDF4")

            # DETERMINE LOCATION OF X-POINTS
            XPTs = []
            for i in range(len(rootgrp['rightix'][0])):
                if rootgrp['rightix'][0][i] != i:
                    XPTs.append(i)
            XPTs = np.array(XPTs)+1
            midplaneix = int((XPTs[-1]+XPTs[-2])/2)
            SEPARATRIX= rootgrp["jsep"][0]+2
            quantities2d,rootg = return2d(fileName+"\\balance.nc")
        
            RING = rootgrp["jsep"][0]+4
            CUMSUM = -1*np.cumsum(np.sum(np.array(rootgrp["b2stel_she_bal"])[1][SEPARATRIX:-1,midplaneix:],axis=0))
            halfind = find_nearest(CUMSUM,np.max(CUMSUM)*0.5)
            # if counter == 0:
            #     contaxs[axsnum].plot(quantities2d["r"][SEPARATRIX+4,midplaneix+halfind],
            #                         quantities2d["z"][SEPARATRIX+4,midplaneix+halfind],
            #                         linestyle = "",marker = "x",color="k",label="50% N radiation")
            # else:
            #     contaxs[axsnum].plot(quantities2d["r"][SEPARATRIX+4,midplaneix+halfind],
            #                 quantities2d["z"][SEPARATRIX+4,midplaneix+halfind],
            #                 marker = "x",color="k")
                   
            grid = "Tight"
            detachlabel = "detached"
            heatcolor = "#8F250C"
            if "Open" in fileName:
                grid = "Open"
                heatcolor = "#FF9B42"
            linestyle = "--"
            if counter <2:
                linestyle = "-"
                detachlabel = "attached"

            cmap = cm.plasma
            ion_heat,elec_heat = CalculateSinks(rootgrp)
            totalsink = np.abs(np.sum(ion_heat["sinks"],axis=0)+np.sum(elec_heat["sinks"],axis=0))
            Rrsep =1000*(quantities2d["r"][:,midplaneix]-quantities2d["r"][SEPARATRIX,midplaneix])
            # axs.plot(Rrsep[RING-2:-1],np.sum(totalsink[RING-2:-1,XPTs[3]:XPTs[4]],axis=1),
            # color=heatcolor,linestyle = linestyle,label=detachlabel)


            # axs.plot(Rrsep[1:-1],rootgrp['fne'][1][1:-1,midplaneix],
            #     color=heatcolor,linestyle = linestyle)
            Pu = (quantities2d["ne"]*quantities2d["te"])[RING-2:,midplaneix]
            cond = np.array(rootgrp["fhe_cond"])[1]+np.array(rootgrp["fhi_cond"])[1]
            totrad = quantities2d["radHeate"]+quantities2d["radHeati"]
            label="n"+r"$_{u}$"+"="+str(int(quantities2d["ne"][RING][midplaneix]*10**(-18))/10)+r"$\times 10^{19}$"+" m"+r"$^{-3}$"
            conv = totrad-cond
            contaxs[axsnum].contour(quantities2d["r"],quantities2d["z"],quantities2d["te"],[10],colors=[colorplot],label=label)
            if counter ==0 or counter ==len(files)-1:
                
                contaxs[axsnum].plot([-1],[0],color=colorplot,label=label)
            #     axs.plot(Rrsep[1:-1],10**(-6)*(conv)[2:,midplaneix],
            #         color=colorplot,linestyle = "-.",
            #         label=label+" convected")
                
            #     axs.plot(Rrsep[1:-1],10**(-6)*(totrad)[2:,midplaneix],
            #         color=colorplot,linestyle = "-",
            #         label=label+" total")

                axslist[axsnum].plot(Rrsep[RING-2:],1.60E-19*(quantities2d["ne"]*quantities2d["te"])[RING-2:,midplaneix+10],
                    color=colorplot, label=label)
            #     # axs.plot(Rrsep[RING-2:],Pu/Pu[0],
            #     #     color=colorplot, label=label)
            #     # axs.plot(Rrsep[1:-1],-1E-6*np.sum(np.array(rootgrp["b2stel_she_bal"])[1][1:-1,XPTs[3]:XPTs[4]],axis=1),
            #     # color=colorplot, label=label) 
            else:
                axslist[axsnum].plot(Rrsep[RING-2:],1.60E-19*(quantities2d["ne"]*quantities2d["te"])[RING-2:,midplaneix+10],
                    color=colorplot)
            #     axs.plot(Rrsep[1:-1],1E-6*conv[2:,midplaneix],
            #         color=colorplot,linestyle = "-.",)
            #     axs.plot(Rrsep[1:-1],1E-6*totrad[2:,midplaneix],
            #         color=colorplot,linestyle = "-",)
                # axs.plot(Rrsep[RING-2:],(quantities2d["ne"]*quantities2d["te"])[RING-2:,midplaneix],
                # color=colorplot)
                # axs.plot(Rrsep[RING-2:],Pu/Pu[0],
                # color=colorplot)
                # axs.plot(Rrsep[1:-1],-1E-6*np.sum(np.array(rootgrp["b2stel_she_bal"])[1][1:-1,XPTs[3]:XPTs[4]],axis=1),
                # color=colorplot) 
            totalsink = 10**(-6)*totalsink/quantities2d["V"]
            if axsnum ==1:
                axslist[axsnum].yaxis.set_visible(False)
                contaxs[axsnum].yaxis.set_visible(False)
            counter = counter +1
    contaxs[0].set_xlim([0.6,1.7])
    contaxs[0].set_ylim([-2.2,0])
    contaxs[1].set_xlim([0.6,1.7])
    contaxs[1].set_ylim([-2.2,0])
    axslist[0].set_ylim([0,3E2])
    axslist[1].set_ylim([0,3E2])
    axslist[0].set_xlabel("R-R"+r'$_{sep}$'+" [mm]")
    axslist[1].set_xlabel("R-R"+r'$_{sep}$'+" [mm]")
    contaxs[0].set_xlabel("R [m]")
    contaxs[1].set_xlabel("R [m]")
    contaxs[0].set_ylabel("Z [m]")
    contaxs[1].set_ylabel("Z [m]")
    # plt.ylabel("heat flux [MW]")
    axslist[0].set_ylabel("P [pa]")
    # plt.ylabel("T [eV]")
    # plt.ylabel("summed nitrogen radiation [MW]")
    # plt.ylabel("radial heat flux [MW]")
    axslist[1].set_title("Tight")
    axslist[0].set_title("Open")
    contaxs[1].set_title("Tight")
    contaxs[0].set_title("Open")
    axslist[0].legend()
    axslist[1].legend()
    contaxs[0].legend()
    contaxs[1].legend()
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout(pad=0)
    contfig.subplots_adjust(wspace=0, hspace=0)
    contfig.tight_layout(pad=0)
    fig.savefig("Figures/Baffle_Profiles/pressEvolution.png", dpi=1000,bbox_inches='tight')
    contfig.savefig("Figures/Baffle_Profiles/contour.png", dpi=1000,bbox_inches='tight')
    plt.show()
    plt.close()


deepsims = [
    "C:\\Users\cyd cowley\Desktop\PhD\IsolatedAnalysis\\balFiles\Mast_Open\\3MW\\ne3.1\\balance.nc",
    "C:\\Users\cyd cowley\Desktop\PhD\IsolatedAnalysis\\balFiles\Mast_Tight\\3MW\\ne6.0\\balance.nc",
    "C:\\Users\cyd cowley\Desktop\PhD\IsolatedAnalysis\\balFiles\Mast_Open\\6MW\\ne4.5\\balance.nc",
    "C:\\Users\cyd cowley\Desktop\PhD\IsolatedAnalysis\\balFiles\Mast_Tight\\6MW/ne10.0",
    "C:\\Users\cyd cowley\Desktop\PhD\IsolatedAnalysis\\balfiles\MAST_Open\\12MW/\\ne7.0\\balance.nc",
    "C:\\Users\cyd cowley\Desktop\PhD\IsolatedAnalysis\\balFiles\Mast_Tight\\12MW/ne16.0",
]

threshSims = [
    "C:\\Users\cyd cowley\Desktop\PhD\IsolatedAnalysis\\balFiles\Mast_Open\\3MW\\ne2.8\\balance.nc",
    "C:\\Users\cyd cowley\Desktop\PhD\IsolatedAnalysis\\balFiles\Mast_Tight\\3MW\\ne2.0\\balance.nc",
    "C:\\Users\cyd cowley\Desktop\PhD\IsolatedAnalysis\\balFiles\Mast_Open\\6MW/ne3.2",
    "C:\\Users\cyd cowley\Desktop\PhD\IsolatedAnalysis\\balFiles\Mast_Tight\\6MW/ne4.0",
    "C:\\Users\cyd cowley\Desktop\PhD\IsolatedAnalysis\\balfiles\MAST_Open\\12MW\\ne4.5\\balance.nc",
    "C:\\Users\cyd cowley\Desktop\PhD\IsolatedAnalysis\\balFiles\Mast_Tight\\12MW\\ne9.0\\balance.nc",
]


# threshSims = [
#     "C:\\Users\cyd cowley\Desktop\PhD\IsolatedAnalysis\\balFiles\Mast_Open\\3MW/ne2.8.nc",
#     "C:\\Users\cyd cowley\Desktop\PhD\IsolatedAnalysis\\balFiles\Mast_Tight\\3MW/ne2.0.nc",
#     "C:\\Users\cyd cowley\Desktop\PhD\IsolatedAnalysis\\balFiles\Mast_Open\\6MW/ne3.2.nc",
#     "C:\\Users\cyd cowley\Desktop\PhD\IsolatedAnalysis\\balFiles\Mast_Tight\\6MW/ne4.0.nc",
#     "C:\\Users\cyd cowley\Desktop\PhD\IsolatedAnalysis\\balfiles\MAST_Open\\12MWpump/ne7.0.nc",
#     "C:\\Users\cyd cowley\Desktop\PhD\IsolatedAnalysis\\balFiles\Mast_Tight\\12MW/ne9.0.nc",
# ]

heatmode = "source_type"
# heatmode= "location"
# heatBalance_Multiple_Sims(threshSims,heatmode=heatmode)
power = "3MW"
# power = "6MW"
power = "12MW"
degDetachment = "Deep"
degDetachment = "Thresh"
perform_Analysis(power,threshSims,1)
# plotEvolution(power)
# plotLosses(power,degDetachment)
# plotNeutrals()
# plotEnergyCost(threshSims,power)
# plotClosure(power)