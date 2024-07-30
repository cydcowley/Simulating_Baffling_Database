from netCDF4 import Dataset
import numpy as np
import re

def return2d(balfile):

    rootgrp =Dataset(balfile, "r", format="NETCDF4")

    bb = rootgrp['bb']
    dv = np.array(rootgrp['vol'])
    quantities2d = {}
    #grid coordinates:
    quantities2d["r"] = (rootgrp['crx'][0]+rootgrp['crx'][1]+rootgrp['crx'][2]+rootgrp['crx'][3])/4
    quantities2d["z"] = (rootgrp['cry'][0]+rootgrp['cry'][1]+rootgrp['cry'][2]+rootgrp['cry'][3])/4
    hx = rootgrp['hx']
    #total magnetic field:
    quantities2d["TotalField"] = np.array(bb[3])
    quantities2d["Bpol"] = np.array(bb[0])
    #length of grid
    s = np.array(hx)*np.abs(np.array(bb[3])/np.array(bb[0]))
    quantities2d["sdiff"] = s
    quantities2d["sdiffpol"] = np.array(hx)
    # Parallel area:
    quantities2d["Area"] = np.array(dv)/s
    #Grid volume
    quantities2d["V"] = dv
    #specific flux ring to focus on
    # print(rootgrp['jsep'][0])
    sep = rootgrp['jsep'][0]
    ring = sep+5
    quantities2d["ring"] = ring
    #electron density (m^{-3})
    quantities2d["ne"] = np.array(rootgrp["ne"])
    #ion density (m^{-3})
    quantities2d["ni"] = np.array(rootgrp["na"][1])
    #Conductive electron heat flux (Wm^{-2}):
    fhe_cond = rootgrp['fhe_cond'][0]/np.abs(quantities2d["Area"])
    quantities2d["cond"] = np.abs(fhe_cond)
    #electron temperature (eV):
    te = np.array(rootgrp["te"])
    quantities2d["te"] = te/(1.60*10**(-19))
    #Z effective
    quantities2d["Zeff"]= quantities2d["ne"]/(np.sum(np.array(rootgrp["na"]),axis=0))
    #ion temperature (eV)
    ti = np.array(rootgrp["ti"])
    quantities2d["ti"] = ti/(1.60*10**(-19))
    #artificial impurity radiation (W):
    #imprad = np.sum(rootgrp["b2stel_she_bal"][2:],axis=0)
    imprad = np.sum(rootgrp["b2stel_she_bal"],axis=0)
    quantities2d["imprad"] = imprad
    #flow velocity
    vfluid = rootgrp["ua"][1]
    quantities2d["vfluid"] = vfluid
    # dab2 = rootgrp['dab2']
    # quantities2d["n0"] = dab2[0]
    quantities2d["qpar"] = rootgrp['fhe_cond'][0]+rootgrp['fhe_32'][0]+rootgrp['fhe_52'][0]+rootgrp['fhe_thermj'][0]+rootgrp['fhe_dia'][0]+rootgrp['fhe_ecrb'][0]
    quantities2d["qpar"] =quantities2d["qpar"] +rootgrp['fhe_strange'][0]+rootgrp['fhe_pschused'][0]
    quantities2d["qpar"]= quantities2d["qpar"]+rootgrp['fhi_cond'][0]+rootgrp['fhi_32'][0]+rootgrp['fhi_52'][0]+rootgrp['fhi_dia'][0]+rootgrp['fhi_ecrb'][0]
    quantities2d["qpar"]= quantities2d["qpar"]+rootgrp['fhi_strange'][0]+rootgrp['fhi_pschused'][0]  
    quantities2d["qpar"] = np.abs(quantities2d["qpar"])/(quantities2d["Area"])
    quantities2d["elHeat"] = rootgrp['fhe_cond'][0]+rootgrp['fhe_32'][0]+rootgrp['fhe_52'][0]+rootgrp['fhe_thermj'][0]+rootgrp['fhe_dia'][0]+rootgrp['fhe_ecrb'][0]
    quantities2d["elHeat"] =quantities2d["elHeat"] +rootgrp['fhe_strange'][0]+rootgrp['fhe_pschused'][0]
    quantities2d["elHeat"] = np.abs(quantities2d["elHeat"])
    quantities2d["iHeat"] = rootgrp['fhi_cond'][0]+rootgrp['fhi_32'][0]+rootgrp['fhi_52'][0]+rootgrp['fhi_dia'][0]+rootgrp['fhi_ecrb'][0]
    quantities2d["iHeat"] =quantities2d["iHeat"] +rootgrp['fhi_strange'][0]+rootgrp['fhi_pschused'][0]+rootgrp['fhi_inert'][0]+rootgrp['fhi_vispar'][0]+rootgrp['fhi_anml'][0]+rootgrp['fhi_kevis'][0]
    quantities2d["iHeat"] = np.abs(quantities2d["iHeat"])
    quantities2d["radHeate"] = rootgrp['fhe_cond'][1]+rootgrp['fhe_32'][1]+rootgrp['fhe_52'][1]+rootgrp['fhe_thermj'][1]+rootgrp['fhe_dia'][1]+rootgrp['fhe_ecrb'][1]
    quantities2d["radHeate"] =quantities2d["radHeate"] +rootgrp['fhe_strange'][1]+rootgrp['fhe_pschused'][1]    
    quantities2d["radHeati"] = rootgrp['fhi_cond'][1]+rootgrp['fhi_32'][1]+rootgrp['fhi_52'][1]+rootgrp['fhi_dia'][1]+rootgrp['fhi_ecrb'][1]
    quantities2d["radHeati"] =quantities2d["radHeati"] +rootgrp['fhi_strange'][1]+rootgrp['fhi_pschused'][1]    
    #parallel particle flux
    quantities2d["parFluxi"] = rootgrp['fna_pinch'][:,0]+rootgrp['fna_pll'][:,0]+rootgrp['fna_drift'][:,0]+rootgrp['fna_ch'][:,0]+rootgrp['fna_nanom'][:,0]
    quantities2d["parFluxi"] =quantities2d["parFluxi"]+rootgrp["fna_panom"][:,0] +rootgrp['fna_pschused'][:,0]

    #radial particle flux
    quantities2d["radFluxi"] = rootgrp['fna_pinch'][:,1]+rootgrp['fna_pll'][:,1]+rootgrp['fna_drift'][:,1]+rootgrp['fna_ch'][:,1]+rootgrp['fna_nanom'][:,1]
    quantities2d["radFluxi"] =quantities2d["radFluxi"]+rootgrp["fna_panom"][:,1] +rootgrp['fna_pschused'][:,1]

    R = rootgrp['crx'][1]
    # print(R[-1])
    quantities2d["fna"] = rootgrp["fna_tot"][0][1]#/quantities2d["Area"]
    return quantities2d,rootgrp

def plotWALL(filewalls,axis):
    datawall = open(filewalls)
    tline = datawall.readlines(1)
    linenum = 0
    while True:
        tline = datawall.readlines(1)
        if  '*** 3b. Data for additional surfaces' in tline[0]:
            break
    ns = int(datawall.readlines(1)[0])
    coords = []
    ic = 1
    for i in range(ns):
        displayname = datawall.readlines(1)
        rlbnd = int(datawall.readlines(1)[0][1])
        tmp = datawall.readlines(1)
        iliin = int(tmp[0][1:6])

        if iliin <0:
            tmp = datawall.readlines(1)
            continue
        if rlbnd == 2:
            check = datawall.readlines(1)[0]
            co = re.findall("\D+\d+\.\d+\D+\d+",check)
            for i in range(len(co)):
                co[i] = float(co[i])/100
            coords.append(co)
            stype = datawall.readlines(1)
            axis.plot([co[0],co[3]],[co[1],co[4]],color="black",linewidth=0.5)

def ImportGridue(fname: str = 'gridue') -> dict:
        """
        Import UEDGE grid file as dictionary.

        Parameters
        ----------
        fname : str, optional
            Path/file name to gridue formatted file.

        Returns
        -------
            A dict containing header and body information from the gridue file.

        """
        try:
            f = open(fname, mode='r')
            Values = []
            for i in range(5):
                Values.append([int(x) for x in next(f).split()])
            gridtype = "dn"
            HeaderItems = 0
            
            if gridtype == "dn":
                HeaderItems = ['nxm', 'nym','iyseparatrix1', 'iyseparatrix2',
                'ix_plate1', 'ix_cut1', '_FILLER_', 'ix_cut2', 'ix_plate2',
                'iyseparatrix3', 'iyseparatrix4',
                'ix_plate3', 'ix_cut3', '_FILLER_', 'ix_cut4', 'ix_plate4']
            
            else:
                HeaderItems = ['nxm', 'nym', 'ixpt1', 'ixpt2', 'iyseptrx1']
            flat_list = [item for sublist in Values for item in sublist]
            print(flat_list)
            Values = flat_list
            gridue_settings = dict(zip(HeaderItems, Values))
            print("settings are",gridue_settings)
            next(f)
            BodyItems = ['rm', 'zm', 'psi', 'br', 'bz', 'bpol', 'bphi', 'b']
            Str = {i: [] for i in BodyItems}
            k = iter(Str.keys())
            Key = next(k)
            for line in f:
                if line == 'iogridue\n':
                    continue
                if line == '\n':
                    try:
                        Key = next(k)
                    except:
                        continue

                else:
                    Str[Key].append(line)
            f.close()
            nx = gridue_settings['nxm'] + 2
            ny = gridue_settings['nym'] + 2
            for k, v in Str.items():
                L = (''.join(v).replace('\n', '').replace('D', 'e')).split()
                _l = iter(L)
                vv = next(_l)

                data_ = np.zeros((nx, ny, 5))
                for n in range(5):
                    for j in range(ny):
                        for i in range(nx):

                            data_[i][j][n] = float(vv)

                            try:
                                vv = next(_l)
                            except:
                                continue
                gridue_settings[k] = data_
            return gridue_settings
        except Exception as e:
            print(repr(e))

def NeonIonisationAMJUEL(T):
    lnreactivity= -4.164979646286E+01+2.217184105146E+01*(np.log(T)) \
        -1.042613793789E+01*(np.log(T)**2) +3.175650981066E+00*(np.log(T)**3) \
        -6.293446783142E-01*(np.log(T)**4)+7.941711930007E-02*(np.log(T)**5) \
        -6.140370720421E-03*(np.log(T)**6)+ 2.651559926489E-04*(np.log(T)**7) \
        -4.900429196295E-06*(np.log(T)**8)
    return 2.7**(lnreactivity)
