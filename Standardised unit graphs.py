# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:53:36 2019

@author: duqueh
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import os
import datetime as DT
import matplotlib
import statistics as stat
import numpy as np

DiffusionData = pd.read_csv('S:\Personal Work Areas\Lucy\MarkitCips\DiffusionIndex\Henryredone3MoYdata161018\Final data\FinalDIandPMIinStandardisedUnits.csv')

GrowthData = pd.read_csv("S:\Personal Work Areas\Lucy\MarkitCips\DiffusionIndex\Henryredone3MoYdata161018\Final data\Index growths in standardised units.csv")

IoSFullData = pd.read_csv('S:\Personal Work Areas\Lucy\MarkitCips\DiffusionIndex\Henryredone3MoYdata161018/DiffusionOutputFinalData_thresh5_WithFullIoS.csv')

IoS3MoYFull = IoSFullData['3MoMFullIoSServSTD']

CIPSServices = DiffusionData['CIPSservices']
CIPSManu = DiffusionData['CIPSmanu']
CIPSCons = DiffusionData['CIPSconstruction']

MoMManuSA = DiffusionData['SA_MoMManu']
MoMServSA = DiffusionData['SA_MoMServ']
MoMConsSA = DiffusionData['SA_MoMCons']

TMoYServ = DiffusionData['3MoYServ']
TMoYManu = DiffusionData['3MoYManu']
TMoYCons = DiffusionData['3MoYCons']

TMoMServ = DiffusionData['SA_3MoMServ']
TMoMManu = DiffusionData['SA_3MoMManu']
TMoMCons = DiffusionData['SA_3MoMCons']



IoSMoMGrowths = GrowthData['IoS MoM growths']
IoMMoMGrowths = GrowthData['IoM MoM growths']
IoCMoMGrowths = GrowthData['IoC MoM growths']

IoS3MoMGrowths = GrowthData['IoS 3MoM growths']
IoM3MoMGrowths = GrowthData['IoM 3MoM growths']
IoC3MoMGrowths = GrowthData['IoC 3MoM growths']

IoS3MoYGrowths = GrowthData['IoS 3MoY growths']
IoM3MoYGrowths = GrowthData['IoM 3MoY growths']
IoC3MoYGrowths = GrowthData['IoC 3MoY growths']


RawDate = DiffusionData['Period']
Date = []

for t in RawDate:
    Date.append(DT.datetime.strptime(t, "%b-%y"))

years = matplotlib.dates.YearLocator()   # Gives intervals every year that can be plotted
yearsFmt = matplotlib.dates.DateFormatter('%Y')
months = matplotlib.dates.MonthLocator(interval=3)  # changing interval will alter the frequency that the months are plotted
monthsFmt = matplotlib.dates.DateFormatter('%Y/%b')



fig, axes = plt.subplots()
#axes.plot(Date, CIPSServices, 'r')
axes.plot(Date, TMoYServ, 'r')
axes.bar(Date, IoS3MoYGrowths, width=31)
axes.plot(Date, IoS3MoYFull, 'y')
axes.grid(True)
#axes.legend(loc=2)
axes.legend(['PMI Construction', 'MBS 3MoY Construction DI', '3MoY IoC growths'])

axes.set_ylabel('Standard deviations from the mean')
axes.set_xlabel('Date')
axes.set_title('Construction PMI vs 3MoY MBS DI vs 3MoY IoC growths') #MBS MoM 
axes.xaxis.set_major_locator(months)
axes.xaxis.set_major_formatter(monthsFmt)
for tick in axes.get_xticklabels():
    tick.set_rotation(45)
#axes.set_xlim([DT.date(2012, 1, 1), DT.date(2016, 7, 1)])
#plt.axvline(x=DT.date(2012, 1, 1), color='r')
#plt.axvline(x=DT.date(2016, 7, 1), color='r')
fig.set_size_inches(12, 8)
#plt.savefig('S:\Personal Work Areas\Lucy\MarkitCips\Graphs\Graphs for draft\Construction 3MoY MBS, growths vs PMI', dpi=100, bbox_inches='tight')


