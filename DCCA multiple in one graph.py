# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:58:41 2019

@author: duqueh
"""

"""DCCA function modified from https://gist.github.com/jaimeide/a9cba18192ee904307298bd110c28b14"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.matlib import repmat
import pandas as pd

TMoYData = pd.read_csv("S:/Personal Work Areas/Lucy/MarkitCips/DiffusionIndex/Henryredone3MoYdata161018/Final data/DatesLineUp/3MoY.csv")

#PMIServ = TMoYData["CIPSservices"]
#PMIManu = TMoYData["CIPSmanu"]
#PMICons = TMoYData["CIPSconstruction"]

TMoYServ = TMoYData["3MoYServ"]
TMoYManu = TMoYData["3MoYManu"]
TMoYCons = TMoYData["3MoYCons"]

TMoYGrowthServ = TMoYData["IoS 3MoY growths"]
TMoYGrowthManu = TMoYData["IoM 3MoY growths"]
TMoYGrowthCons = TMoYData["IoC 3MoY growths"]



MoMData = pd.read_csv("S:\Personal Work Areas\Lucy\MarkitCips\DiffusionIndex\Henryredone3MoYdata161018\Final data\DatesLineUp/MoM.csv")

#PMIServ = MoMData["CIPSservices"]
#PMIManu = MoMData["CIPSmanu"]
#PMICons = MoMData["CIPSconstruction"]

MoMServ = MoMData["SA_MoMServ"]
MoMManu = MoMData["SA_MoMManu"]
MoMCons = MoMData["SA_MoMCons"]

MoMGrowthServ = MoMData["IoS MoM growths"]
MoMGrowthManu = MoMData["IoM MoM growths"]
MoMGrowthCons = MoMData["IoC MoM growths"]



TMoMData = pd.read_csv("S:\Personal Work Areas\Lucy\MarkitCips\DiffusionIndex\Henryredone3MoYdata161018\Final data\DatesLineUp/3MoM.csv")

PMIServ = TMoMData["CIPSservices"]
PMIManu = TMoMData["CIPSmanu"]
PMICons = TMoMData["CIPSconstruction"]

TMoMServ = TMoMData["SA_3MoMServ"]
TMoMManu = TMoMData["SA_3MoMManu"]
TMoMCons = TMoMData["SA_3MoMCons"]

TMoMGrowthServ = TMoMData["IoS 3MoM growths"]
TMoMGrowthManu = TMoMData["IoM 3MoM growths"]
TMoMGrowthCons = TMoMData["IoC 3MoM growths"]

TestSigLevels = pd.read_csv("S:\Personal Work Areas\Lucy\MarkitCips\DCCA work\Correct limits/DCCA confidence intevals for series length 95 trunc norm correct limits.csv")
kk = TestSigLevels["k"]
NinetyFive = TestSigLevels["NinetyFivePercentConfidenceIntervalUpper"]
LowerNinetyFive = TestSigLevels["NinetyFivePercentConfidenceIntervalLower"]

#Full IoS definition 3MoY diffusion index
FullIoSData = pd.read_csv("S:\Personal Work Areas\Lucy\MarkitCips\DiffusionIndex\Henryredone3MoYdata161018/DiffusionOutputFinalData_thresh5_FullIoS050819.csv")
FullIoSData = FullIoSData[14:96]
IoS3MoY = FullIoSData["3MoYServ"]


def sliding_window(xx,k):
    # Function to generate boxes given dataset(xx) and box size (k)
    import numpy as np
    # generate indexes. O(1) way of doing it :)
    idx = np.arange(k)[None, :]+np.arange(len(xx)-k+1)[:, None]
    return xx[idx],idx


def GetDCCAToList (x1, x2, W1, W2):

    # Plot
    cdata = np.array([x1,x2]).T
#    plt.plot(cdata)
#    plt.title('Sample time series')
#    plt.legend(['$x_1$','$x_2$'])
#    plt.show()
#    plt.clf()


    # Define
    nsamples,nvars = cdata.shape

    # Cummulative sum after removing mean
    cdata = cdata-cdata.mean(axis=0)
    xx = np.cumsum(cdata,axis=0)
    plt.plot(xx)
    plt.title('Cummulative sum')
    plt.legend(['$x_1$','$x_2$'])
    xx.shape
    plt.show()
    plt.clf()



    kList = []
    DCCAList = []

    for k in range(W1, W2):
        F2_dfa_x = np.zeros(nvars)
        allxdif = []
        for ivar in range(nvars): # do for all vars
            xx_swin , idx = sliding_window(xx[:,ivar],k)
            nwin = xx_swin.shape[0]
            b1, b0 = np.polyfit(np.arange(k),xx_swin.T,deg=1) # linear fit
            #x_hat = [[b1[i]*j+b0[i] for j in range(k)] for i in range(nwin)] # slow version
            x_hatx = repmat(b1,k,1).T*repmat(range(k),nwin,1) + repmat(b0,k,1).T
            # Store differences to the linear fit
            xdif = xx_swin-x_hatx
            allxdif.append(xdif)
            # Eq.4
            F2_dfa_x[ivar] = (xdif**2).mean()


        # Get the DCCA matrix
        dcca = np.zeros([nvars,nvars])
        for i in range(nvars): # do for all vars
            for j in range(nvars): # do for all vars
                # Eq.5 and 6
                F2_dcca = (allxdif[i]*allxdif[j]).mean()
                # Eq.1: DCCA
                dcca[i,j] = F2_dcca / np.sqrt(F2_dfa_x[i] * F2_dfa_x[j])

        #print(dcca)
        #print(dcca[0,1])
        #print(k)
    
        kList.append(k)
        print(kList)
        DCCAList.append(dcca[0,1])
    
    print(dict(zip(kList, DCCAList)))
    return [kList, DCCAList]





kListAndDCCAServPMI = GetDCCAToList(PMIServ, TMoMGrowthServ, 4, 19)

kListAndDCCAManuPMI = GetDCCAToList(PMIManu, TMoMGrowthManu, 4, 19)

kListAndDCCAConsPMI = GetDCCAToList(PMICons, TMoMGrowthCons, 4, 19)


kListAndDCCAServMBSDI = GetDCCAToList(TMoMServ, TMoMGrowthServ, 4, 19)

kListAndDCCAManuMBSDI = GetDCCAToList(TMoMManu, TMoMGrowthManu, 4, 19)

kListAndDCCAConsMBSDI = GetDCCAToList(TMoMCons, TMoMGrowthCons, 4, 19)


#kListAndDCCAServMBSDIFULLIOS = GetDCCAToList(IoS3MoY, TMoYGrowthServ, 4, 17)

fig, axes = plt.subplots()
axes.plot(kListAndDCCAServPMI[0], kListAndDCCAServPMI[1], color = 'r', linestyle = "--")
axes.plot(kListAndDCCAServMBSDI[0], kListAndDCCAServMBSDI[1], color = 'r')
#axes.plot(kListAndDCCAServMBSDIFULLIOS[0], kListAndDCCAServMBSDIFULLIOS[1], color = 'y')



axes.plot(kListAndDCCAManuPMI[0], kListAndDCCAManuPMI[1], color = 'g', linestyle = "--")
axes.plot(kListAndDCCAManuMBSDI[0], kListAndDCCAManuMBSDI[1], color = 'g')

axes.plot(kListAndDCCAConsPMI[0], kListAndDCCAConsPMI[1], color = 'b', linestyle = "--")
axes.plot(kListAndDCCAConsMBSDI[0], kListAndDCCAConsMBSDI[1], color = 'b')


axes.plot(kk, NinetyFive, color = 'k', linestyle = "-")
axes.plot(kk, LowerNinetyFive, color = 'k', linestyle = "-")
axes.legend(["Services PMI coefficient","Services MBS DI coefficient",
             "Manufacturing PMI coefficient", "Manufacturing MBS DI coefficient",
             "Construction PMI coefficient", "Construction MBS DI coefficient", 
             "95% confidence level", "95% confidence level"])

fig.set_size_inches(12, 8)
axes.set_ylabel('DCCA coefficient')
axes.set_xlabel('Window size (months)')
axes.set_title("DCCA coefficient between MBS 3MoY growths and diffusion indices")
#plt.savefig("S:\Personal Work Areas\Lucy\MarkitCips\Graphs\DCCA graphs\Correct limits for draft/3MoY DI vs DCCA with full IoS DI from 4-16", dpi=100, bbox_inches='tight')

x = pd.DataFrame(data = {"Services PMI and official estimates" : kListAndDCCAServPMI[1],
                         "Manufacturing PMI and official estimates" : kListAndDCCAManuPMI[1],
                         "Construction PMI and official estimates" : kListAndDCCAConsPMI[1],
                         "Services MBS DI and official estimates" : kListAndDCCAServMBSDI[1],
                         "Manufacturing MBS DI and official estimates" : kListAndDCCAManuMBSDI[1],
                         "Construction MBS DI and official estimates" : kListAndDCCAConsMBSDI[1],
                         "window size" : kListAndDCCAConsMBSDI[0]})
x.to_csv("S:\Personal Work Areas\Lucy\MarkitCips\Data for draft article 040919\Three-month on three-month graph data\DCCA/3MoMDCCAbetweenDiffusionIndicesandOfficialEstimates.csv")

####################################################
#
#kListAndDCCAServ = GetDCCAToList(PMIServ, TMoMServ, 4, 19)
#
#kListAndDCCAManu = GetDCCAToList(PMIManu, TMoMManu, 4, 19)
#
#kListAndDCCACons = GetDCCAToList(PMICons, TMoMCons, 4, 19)
#
#fig, axes = plt.subplots()
#
#axes.plot(kListAndDCCAServ[0], kListAndDCCAServ[1], color = 'r' )
#axes.plot(kListAndDCCAManu[0], kListAndDCCAManu[1], color = 'g')
#axes.plot(kListAndDCCACons[0], kListAndDCCACons[1], color = 'b')
#
#print(len(kk))
#print(len(kListAndDCCAServ[1]))
#
#axes.plot(kk, NinetyFive, color = 'k', linestyle = "--")
##axes.plot(kk[1:23], LowerNinetyFive[1:23], color = 'k', linestyle = "-")
#axes.legend(["Services PMI coefficient",
#             "Manufacturing PMI coefficient",
#             "Construction PMI coefficient",
#             "95% confidence level"])#, "95% confidence level"])
#
#fig.set_size_inches(12, 8)
#axes.set_ylabel('DCCA coefficient')
#axes.set_xlabel('Window size (months)')
#axes.set_title("DCCA coefficient between MBS MoM DI and PMI")
##plt.savefig("S:\Personal Work Areas\Lucy\MarkitCips\Graphs\DCCA graphs\Correct limits for draft/3MoY DI vs PMI DCCA from 4-16", dpi=100, bbox_inches='tight')
#
#x = pd.DataFrame(data = {"Window Size" : kListAndDCCAServ[0],
#                         "Services PMI and 3MoM DI" : kListAndDCCAServ[1],
#                         "Manufacturing PMI and 3MoM DI" : kListAndDCCAManu[1],
#                         "Construction PMI and 3MoM DI" : kListAndDCCACons[1],
#                         })
#    
#print(x)
#
#x.to_csv("S:\Personal Work Areas\Lucy\MarkitCips\Data for draft article 040919\Three-month on three-month graph data\DCCA/3MoMDCCABetweenPMIsAndDiffusionIndices.csv")