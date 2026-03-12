from operator import index
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os

from natsort import natsorted
import matplotlib.image as m

from scipy import interpolate
from PIL import Image

import matplotlib as mpl
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys


from UnpackSOLPS import unpackSOLPS, unpack2d

from scipy.optimize import curve_fit
from SharedFunctions import return2d, ImportGridue, plotWALL
import re
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid as cumtrapz

plt.rcParams["font.family"] = "serif"
params = {
    "legend.fontsize": "small",
    "font.size": "16",
    #  'figure.figsize': (4,3.2),
}

fig3, axs3 = plt.subplots(1, 1, figsize=(6, 5.2))
fig4, axs4 = plt.subplots(1, 1, figsize=(6, 5.2))
fig5, axs5 = plt.subplots(1, 1, figsize=(6, 5.2))

plt.rcParams.update(params)

customOrder = 0

colors = ["#FF9B42", "#8F250C"]
gridcolors = ["#53ba83", "#059b9a", "#095169", "#0c0636", "000000"]
heatcolors = ["#442288", "#6CA2EA", "#B5D33D", "#FED23F", "#EB7D5B"]


def determineC0(Spar, C):
    for k in range(len(C)):
        if Spar[k] > Spar[-1] / 100:

            return k - 1


image_list = []


def dls_Analysis(power):
    colornum = 0
    if power == "3MW":
        Files = [
            "balFiles/Mast_Tight/3MW/ne2.0/balance.nc",
            "balFiles/Mast_Open/3MW/ne2.6/balance.nc",
        ]
    elif power == "6MW":
        colornum = 1
        Files = [
            "balFiles/Mast_Tight/6MW/ne4.0/balance.nc",
            "balFiles/Mast_Open/6MW/ne3.2/balance.nc",
        ]
    elif power == "12MW":
        colornum = 2
        Files = [
            "balFiles/Mast_Tight/12MW/ne9.0/balance.nc",
            "balFiles/MAST_Open/12MW/ne4.5/balance.nc",
        ]
    plt.rcParams["axes.labelsize"] = "Large"

    Ccalcs = []
    maxrad = []
    fig1, axs1 = plt.subplots(1, 1, figsize=(6, 5.2))
    fig2, axs2 = plt.subplots(1, 1, figsize=(6, 5.2))
    axs22 = axs2.twinx()
    heatcolors2 = 0
    for fileName in Files:

        # if "Tight" in folder and "12MW" in folder:
        #     if "ne9" in File or "ne10" in File:
        #         continue
        rootgrp = Dataset(str(fileName), "r", format="NETCDF4")
        print("rootgroup is", rootgrp)
        SOLring1 = 0
        RING = rootgrp["jsep"][0] + 5
        SEPARATRIX = rootgrp["jsep"][0] + 2

        # DETERMINE LOCATION OF X-POINTS
        XPTs = []
        for i in range(len(rootgrp["rightix"][0])):
            if rootgrp["rightix"][0][i] != i:
                XPTs.append(i)
        XPTs = np.array(XPTs) + 1
        midplaneix = int((XPTs[-1] + XPTs[-2]) / 2)

        quantities2d, SOLring1 = unpackSOLPS(
            fileName, -1, RING, Xpoint=len(rootgrp["rightix"][0]) - midplaneix
        )
        quantities2d = unpack2d(rootgrp)
        xindex = len(rootgrp["rightix"][0]) - XPTs[-1] - 1

        calcs = SOLring1.returnCalculationsC(
            0,
            xindex=xindex,
        )
        calcs.insert(0, 1)
        calcs.append(SOLring1.ne[-1])
        Ccalcs.append(calcs)
        Rrsep = (
            quantities2d["r"][:, midplaneix] - quantities2d["r"][SEPARATRIX, midplaneix]
        ) * 1000
        # CALCULATE THE RADIAL-AVERAGED COLLISIONALITY IN THE SOL
        avdensity = np.trapezoid(
            y=quantities2d["ne"][SEPARATRIX:, XPTs[-1] - 1],
            x=Rrsep[SEPARATRIX:],
        ) / (Rrsep[-1] - Rrsep[SEPARATRIX])
        print("edge temp is", avdensity)

        # axs1.plot(
        #     SOLring1.Spar[:xindex],
        #     -1 * np.cumsum(SOLring1.qf * SOLring1.V)[:xindex],
        #     label="cumulative impurity losses",
        #     color=heatcolors[colornum],
        #     linewidth=2,
        # )
        # axs1.plot(
        #     SOLring1.Spar[:xindex],
        #     -1
        #     * np.cumsum(SOLring1.V * (SOLring1.ionisLoss + SOLring1.recombLoss))[
        #         :xindex
        #     ],
        #     label="cumulative deuterium losses",
        #     color=heatcolors[colornum],
        #     linestyle="--",
        #     linewidth=2,
        # )
        totalheat = (
            np.array(rootgrp["fhe_32"])
            + np.array(rootgrp["fhe_52"])
            + np.array(rootgrp["fhe_thermj"])
            + np.array(rootgrp["fhe_cond"])
            + np.array(rootgrp["fhe_dia"])
            + np.array(rootgrp["fhe_ecrb"])
            + np.array(rootgrp["fhe_strange"])
            + np.array(rootgrp["fhe_pschused"])
            + np.array(rootgrp["fhi_32"])
            + np.array(rootgrp["fhi_52"])
            + np.array(rootgrp["fhi_cond"])
            + np.array(rootgrp["fhi_dia"])
            + np.array(rootgrp["fhi_ecrb"])
            + np.array(rootgrp["fhi_strange"])
            + np.array(rootgrp["fhi_pschused"])
            + np.array(rootgrp["fhi_inert"])
            + np.array(rootgrp["fhi_vispar"])
            + np.array(rootgrp["fhi_visper"])
            + np.array(rootgrp["fhi_visq"])
            + np.array(rootgrp["fhi_anml"])
            + np.array(rootgrp["fhi_kevis"])
        )

        # totalmom = np.array(rootgrp["fmo_b2nxfv"])[1]
        print("heat entering=", np.sum(totalheat[1][-1][midplaneix : XPTs[-1] + 17]))

        print(
            "heat leaving=", np.sum(totalheat[1][SEPARATRIX][midplaneix : XPTs[-1] + 1])
        )

        radtrans = totalheat[1]
        radtrans = radtrans[RING + 1] - radtrans[RING]
        radtrans = radtrans[::-1]
        radtrans = radtrans
        partrans = quantities2d["elHeat"]
        partrans = partrans[RING]
        partrans = partrans[:xindex] - partrans[1 : xindex + 1]

        maxrad.append(np.max(radtrans))
        # print(rootgrp)
        neutralLosses = np.sum(
            np.array(rootgrp["eirene_mc_eael_she_bal"])
            + np.array(rootgrp["eirene_mc_emel_she_bal"])
            + np.array(rootgrp["eirene_mc_eiel_she_bal"])
            + np.array(rootgrp["eirene_mc_epel_she_bal"])
            + np.array(rootgrp["eirene_mc_eapl_shi_bal"])
            + np.array(rootgrp["eirene_mc_empl_shi_bal"])
            + np.array(rootgrp["eirene_mc_eipl_shi_bal"])
            + np.array(rootgrp["eirene_mc_eppl_shi_bal"]),
            axis=0,
        )
        # neutralLosses = (
        #     neutralLosses
        #     + np.array(rootgrp["b2stbr_phys_she_bal"])
        #     + np.array(rootgrp["b2stbr_bas_she_bal"])
        #     + np.array(rootgrp["b2stbr_first_flight_she_bal"])
        #     + np.array(rootgrp["b2stbc_she_bal"])
        #     + np.array(rootgrp["b2stbm_she_bal"])
        #     + np.array(rootgrp["ext_she_bal"])
        #     + np.array(rootgrp["b2srsm_she_bal"])
        #     + np.array(rootgrp["b2srdt_she_bal"])
        #     + np.array(rootgrp["b2srst_she_bal"])
        #     + np.array(rootgrp["reshe"])
        #     + np.array(rootgrp["b2stbr_phys_shi_bal"])
        #     + np.array(rootgrp["b2stbr_bas_shi_bal"])
        #     + np.array(rootgrp["b2stbr_first_flight_shi_bal"])
        #     + np.array(rootgrp["b2stbc_shi_bal"])
        #     + np.array(rootgrp["b2stbm_shi_bal"])
        #     + np.array(rootgrp["ext_shi_bal"])
        # )
        neutralLosses = neutralLosses
        neutralLosses = neutralLosses[RING][::-1]

        eirene_mc_papl_sna_bal = np.sum(
            np.array(rootgrp["eirene_mc_papl_sna_bal"]), axis=0
        )[1]

        ionTarg_Current = np.array(rootgrp["fna_tot"])[1][0]
        ionTarg_Current = np.sum(ionTarg_Current[:, -1])

        trapping = np.abs(
            np.sum(eirene_mc_papl_sna_bal[:, XPTs[-1] + 20 : -1]) / ionTarg_Current
        )
        print("trapping is", trapping)
        translosses = np.cumsum(radtrans)
        translosses = translosses[: xindex - 1]
        translosses = np.insert(translosses, 0, 0)

        neutralLosses = np.cumsum(neutralLosses)
        neutralLosses = neutralLosses[: xindex - 1]
        neutralLosses = np.insert(neutralLosses, 0, 0)
        Nlosses = -1 * np.cumsum(np.array(rootgrp["b2stel_she_bal"])[1][RING][::-1])
        Nlosses = Nlosses[: xindex - 1]
        Nlosses = np.insert(Nlosses, 0, 0)
        heatplot = (totalheat[0])[RING][::-1][:xindex]
        heatplot = heatplot - heatplot[0]

        momentumsource = np.sum(
            np.array(rootgrp["eirene_mc_mipl_smo_bal"])
            + np.array(rootgrp["eirene_mc_mmpl_smo_bal"])
            + np.array(rootgrp["eirene_mc_mapl_smo_bal"]),
            axis=0,
        )
        momentumsource = momentumsource[1]
        momentumsource = momentumsource[RING][midplaneix:-1]
        momentumsource = -1 * np.cumsum(momentumsource)[::-1]
        totparticles = np.array(rootgrp["fna_tot"])[1] + np.array(rootgrp["fne"])
        radtransparticle = totparticles[1]
        radtransparticle = radtransparticle[RING + 1] - radtransparticle[RING]
        radtransparticle = radtransparticle[midplaneix:-1]
        radtransparticle = np.cumsum(radtransparticle)[::-1]

        smo_vars = [var for var in rootgrp.variables if "smo" in var.lower()]
        print(smo_vars)
        momentumsourceB2 = 0
        for var_name in smo_vars[:]:
            if "eirene" not in var_name and "tot" not in var_name:
                data = rootgrp.variables[var_name][:]
                momentumsourceB2 += data
                #
        momentumsourceB2 = rootgrp["b2siav_smovv_bal"]  # source to due viscosity
        # momentumsourceB2 = rootgrp["b2sicf_smo_bal"]  # source due to diffusion
        momentumsourceB2 = rootgrp["b2sigp_smogp_bal"]  # pressure gradient
        momentumsourceB2 = momentumsourceB2[1][RING][midplaneix:-1]
        momentumsourceB2 = -1 * np.cumsum(momentumsourceB2)[::-1]

        momflu = -1 * (np.array(rootgrp["fmo_flua"])[1][0])[RING][midplaneix:-1][::-1]

        momvisc = (
            -1 * (np.array(rootgrp["fmo_cvsa"])[1][0])[RING][midplaneix:-1][::-1]
        )  # viscous flux
        if "Open" in fileName:
            axs1.plot(
                SOLring1.Spar[:xindex][0],
                Nlosses[0],
                color=gridcolors[heatcolors2 * 2],
                linestyle="-",
                label="Open",
            )
            axs1.plot(
                SOLring1.Spar[:xindex],
                translosses,
                color=gridcolors[heatcolors2 * 2],
                linestyle=":",
            )
            axs1.plot(
                SOLring1.Spar[:xindex],
                -1 * neutralLosses,
                color=gridcolors[heatcolors2 * 2],
                linestyle="--",
            )

            axs1.plot(
                SOLring1.Spar[:xindex],
                Nlosses,
                linestyle="-",
            )
            axs2.plot(
                SOLring1.Spar[0],
                momflu[0],
                color=gridcolors[heatcolors2 * 2],
                label="Open",
            )
            axs2.plot(
                SOLring1.Spar,
                momflu,
                color=gridcolors[heatcolors2 * 2],
            )
            axs2.plot(
                SOLring1.Spar,
                momvisc,
                color=gridcolors[heatcolors2 * 2],
                linestyle="--",
            )
            axs2.plot(
                SOLring1.Spar,
                momentumsourceB2,
                color=gridcolors[heatcolors2 * 2],
                linestyle="dashdot",
            )
            axs2.plot(
                SOLring1.Spar,
                momentumsource,
                color=gridcolors[heatcolors2 * 2],
                linestyle=":",
            )

        else:
            axs1.plot(
                SOLring1.Spar[:xindex],
                Nlosses,
                color=gridcolors[heatcolors2 * 2],
                linestyle="-",
                label="nitrogen losses",
            )
            axs1.plot(
                SOLring1.Spar[:xindex],
                translosses,
                color=gridcolors[heatcolors2 * 2],
                linestyle=":",
                label="radial transport",
            )
            axs1.plot(
                SOLring1.Spar[:xindex],
                -1 * neutralLosses,
                color=gridcolors[heatcolors2 * 2],
                linestyle="--",
                label="neutral losses",
            )

            axs1.plot(
                SOLring1.Spar[:xindex][0],
                Nlosses[0],
                color=gridcolors[heatcolors2 * 2],
                linestyle="-",
                label="Closed",
            )
            axs2.plot(
                SOLring1.Spar[0],
                momflu[0],
                color=gridcolors[heatcolors2 * 2],
                label="Closed",
            )
            axs2.plot(
                SOLring1.Spar,
                momflu,
                color=gridcolors[heatcolors2 * 2],
                label="fluid",
            )
            axs2.plot(
                SOLring1.Spar,
                momvisc,
                color=gridcolors[heatcolors2 * 2],
                label="viscous",
                linestyle="--",
            )
            axs2.plot(
                SOLring1.Spar,
                momentumsourceB2,
                color=gridcolors[heatcolors2 * 2],
                linestyle="dashdot",
                label="pressure grad",
            )
            axs2.plot(
                SOLring1.Spar,
                momentumsource,
                color=gridcolors[heatcolors2 * 2],
                linestyle=":",
                label="neutral source",
            )
        particlesource = np.sum(np.array(rootgrp["eirene_mc_papl_sna_bal"]), axis=0)
        particlesource = particlesource[1]
        particlesourceint = np.sum(particlesource[SEPARATRIX:, XPTs[-1] : -1], axis=1)
        if "Open" in fileName:
            linestylet = "--"
            colorpower = "#08495E"
        else:
            linestylet = "-"
            colorpower = "#23583B"
        if power == "3MW":
            if "Open" in fileName:
                colorpower = "#43C4EF"
            else:
                colorpower = "#8AD0AB"
        if power == "6MW":
            if "Open" in fileName:
                colorpower = "#129FCE"
            else:
                colorpower = "#46AF77"

        # axs3.plot(
        #     Rrsep[SEPARATRIX:],
        #     particlesourceint / ionTarg_Current,
        #     color=gridcolors[heatcolors2 * 2],
        #     linestyle="-",
        #     linewidth=3,
        #     alpha=alpha,
        #     # label="pressure grad",
        # )
        if "Open" in fileName:
            axs3.plot(
                Rrsep[SEPARATRIX + 5 :],
                rootgrp["te"][SEPARATRIX + 5 :, XPTs[-1] + 15] / (1.60e-19),
                color=colorpower,
                linestyle=linestylet,
                linewidth=3,
            )
            axs4.plot(
                Rrsep[SEPARATRIX:],
                particlesourceint / ionTarg_Current,
                color=colorpower,
                linestyle=linestylet,
                linewidth=3,
            )
            axs5.plot(
                Rrsep[SEPARATRIX + 1 :],
                totalheat[1, SEPARATRIX + 1 :, XPTs[-1] + 20]
                / (10**6 * float(power[:-2])),
                color=colorpower,
                linestyle=linestylet,
                linewidth=3,
            )
        else:
            axs3.plot(
                Rrsep[SEPARATRIX + 5 :],
                rootgrp["te"][SEPARATRIX + 5 :, XPTs[-1] + 15] / (1.60e-19),
                color=colorpower,
                linestyle=linestylet,
                linewidth=3,
                label=power,
            )
            axs4.plot(
                Rrsep[SEPARATRIX:],
                particlesourceint / ionTarg_Current,
                color=colorpower,
                linestyle=linestylet,
                linewidth=3,
                label=power,
                marker="o",
            )
            axs5.plot(
                Rrsep[SEPARATRIX + 1 :],
                totalheat[1, SEPARATRIX + 1 :, XPTs[-1] + 20]
                / (10**6 * float(power[:-2])),
                color=colorpower,
                linestyle=linestylet,
                linewidth=3,
                label=power,
                marker="o",
            )

        heatcolors2 += 1

    axs1.set_xlabel("s [m]")
    axs1.set_ylabel("power [W]")
    axs1.legend()
    axs1.set_title(power)
    fig1.tight_layout()

    axs2.set_xlabel("s [m]")
    axs2.set_ylabel("parallel momentum flux [N]")
    axs2.legend()
    axs2.set_title(power)
    fig2.tight_layout()
    fig2.savefig(
        "Figures//mom_diff_dls" + power + ".png",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    # axs2.set_xscale("log")

    fig1.savefig(
        "Figures//heat_diff_dls" + power + ".png",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    # fig1.show()

    # print(maxrad[1] / maxrad[0])
    Ccalcs = np.array(Ccalcs)
    plotCalcs = Ccalcs[1] / Ccalcs[0]
    if colornum == 0:
        axs0.plot(
            np.zeros(6) + plotCalcs[-1],
            color=heatcolors[colornum],
            label="SOLPS",
            linestyle="--",
        )
    else:
        axs0.plot(
            np.zeros(6) + plotCalcs[-1], color=heatcolors[colornum], linestyle="--"
        )

    axs0.plot(
        [
            "DLS Model",
            "True T" + r"$_{u}$",
            "Broad radiation",
            "True conduction",
            "True pressure variation",
            "True heat sinks",
        ],
        plotCalcs,
        alpha=1,
        marker="o",
        color=heatcolors[colornum],
        label=power,
        linewidth=2,
    )


fig0, axs0 = plt.subplots(1, 1, figsize=(8, 5.2))

threshSims = [
    "balFiles/Mast_Open/3MW/ne2.6/balance.nc",
    "balFiles/Mast_Tight/3MW/ne2.0/balance.nc",
    "balFiles/Mast_Open/6MW/ne3.2/balance.nc",
    "balFiles/Mast_Tight/6MW/ne4.0/balance.nc",
    "balFiles/MAST_Open/12MW/ne4.5/balance.nc",
    "balFiles/Mast_Tight/12MW/ne9.0/balance.nc",
]
dls_Analysis("3MW")
dls_Analysis("6MW")
dls_Analysis("12MW")

axs3.plot(
    [4],
    [0],
    color="#46AF77",
    linestyle="-",
    linewidth=3,
    label="Closed",
)

axs3.plot(
    [4],
    [0],
    color="#129FCE",
    linestyle="--",
    linewidth=3,
    label="Open",
)

axs4.plot(
    [4],
    [0],
    color="#46AF77",
    linestyle="-",
    linewidth=3,
    label="Closed",
)

axs4.plot(
    [4],
    [0],
    color="#129FCE",
    linestyle="--",
    linewidth=3,
    label="Open",
)
axs3.legend()
axs4.legend()

axs3.set_xlabel("R-R" + r"$_{sep}$" + " [mm]")
axs3.set_ylabel("T [eV]")

fig3.tight_layout()
fig3.savefig("Figures//temp_xpoint.png", dpi=300, transparent=True, bbox_inches="tight")
fig3.show()

axs4.set_xlabel("R-R" + r"$_{sep}$" + " [mm]")
axs4.set_ylabel(r"$\frac{S_{ioniz}}{\Gamma_{t}}$")

fig4.tight_layout()
fig4.savefig(
    "Figures//ioniz_xpoint.png", dpi=300, transparent=True, bbox_inches="tight"
)
fig4.show()
axs0.text(2.1, 1.1, "pressure variation", fontsize=10)
axs0.text(3.2, 1.7, "other heat sinks", fontsize=10)
axs0.arrow(3, 0.9, 0.6, 0.4, color="black", width=0.015, zorder=100)
axs0.arrow(4, 1.5, 0.7, 0.4, color="black", width=0.015, zorder=100)
axs0.set_ylim([0.5, 2])
axs0.legend()
axs0.tick_params(axis="x", labelrotation=90)
axs0.set_ylabel(r"$\frac{n_{u,thresh,open}}{n_{u,thresh,closed}}$")
fig0.savefig(
    "Figures//threshold_diff_dls.png", dpi=500, transparent=True, bbox_inches="tight"
)
