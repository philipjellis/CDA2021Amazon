import datetime
import numpy as np
import pandas as pd
from collections import namedtuple

class Const(object):
    OUTPUTNAME = "TEST1"
    OUTPUTNAME = "2-NexusPF"
    NPEEP = 1000
    RUNS = 1000
    FUNDVAL = 100000000.0
    FUNDFILE = "NEXUSPFGROWTH.csv"
    ANNCUM = "ANN"
    INCOME = 0.05
    WMFEE = 0.01
    PREMIUM = 0.0055000000000000005
    RUNINLAPSERATE = 0.07
    INCOMELAPSERATE = 0.03
    LAPSERATETHRESHOLD = 0.30000000000000004
    LAPSERATESUBTHRESHOLD = 0.01
    DISCOUNT = 0.035
    MORTALITYTABLEFEMALE = "iam_2012female_qx"
    MORTALITYTABLEMALE = "iam_2012male_qx"
    MORTALITYMULTIPLIER = 1
    TAPER = 0.75
    TAPERYEARS = 20
    RUNINEXTRA = 0
    ASSUMP_LAP = "OLD"
    ASSUMP_EXP = "NEW"
    ASSUMP_MOR = "NEW"
    LAPSECAP = 1.5
    STOCHASTIC = True
    DEBUG = False
    MORTSPREADSHEET = "IAM20122581_2582.xlsx"
    LAPSEUTILIZATION = "LapseUtilization.xlsx"
    YEARONEINCOME = 0.5
    HALFYEARNPV = True
    INFLATIONRATE = 1.02
    MAXAGE = 120 # saves looking it up in a mortality table.  Need to change if we change tables.
    # empty line
    TODAY = np.datetime64('today', 'D')
    THISYEAR = datetime.datetime.today().year
    LASTYEAR = THISYEAR - 1 # values such as NPV are end of year, so starting NPV is last year's
    FUNDDIRECTORY = '/home/test/funds/'
    HOMEDIRECTORY = '/home/test/working/'
    OUTPUTDIR = '/home/test/working/results/'
    AFTERFEES = (1 - PREMIUM) * (1 - WMFEE)
    PROCESSORS = 47
    MORTSPREADSHEET = HOMEDIRECTORY + MORTSPREADSHEET
    LAPSEFILE = HOMEDIRECTORY + LAPSEUTILIZATION 
    MARKET = FUNDDIRECTORY + FUNDFILE
    CENSUS = HOMEDIRECTORY + 'census.json'
    LAPSETABLE = pd.read_excel(LAPSEFILE,skiprows=2,index_col=0)  
    QUALUTILIZED = pd.read_excel(LAPSEFILE,2,skiprows=1,index_col=0)
    UNQUALUTILIZED = pd.read_excel(LAPSEFILE,1,skiprows=1,index_col=0)
    MORT_Fq = pd.read_excel(MORTSPREADSHEET,0,index_col=0)
    MORT_Mq = pd.read_excel(MORTSPREADSHEET,1,index_col=0)
    MORT_FG2 = pd.read_excel(MORTSPREADSHEET,2,index_col=0)
    MORT_MG2 = pd.read_excel(MORTSPREADSHEET,3,index_col=0)
    MORT_F = pd.read_excel(MORTSPREADSHEET,4,index_col=0)
    pmort = {}
    # empty line
    DISCO = 1 + DISCOUNT
    HALF_YR = DISCO ** .5 if HALFYEARNPV else 1.0 # this adds half a year to the NPV figures, arguable
    V = 1 / DISCO
    PERSONFUND = FUNDVAL * 1.0 / NPEEP
    MONTHS = 'Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split()
    prm_tuple = namedtuple('params',['group','npeep','sex','startage','fundsize','premiumpct','incomepct','wmpct','runinextra'])
    PRINTRUNS = 1 # prints every n run numbers, to check progress
