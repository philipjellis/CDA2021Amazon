from multiprocessing import Pool, Queue, Manager
from random import random,seed
from stox11 import StockMarket, Fund, loadcsvqueue, mort, outxl
import numpy as np
import pandas as pd
import numpy_financial as npf
import sys
import logging
from collections import namedtuple, OrderedDict
import csv
import time
from datetime import datetime
import os
import pickle
from qs import mortdick
#from iam2012 import morty
#from iam2012 import morty2

#
# FROM LINE 30 to 60
# DO NOT MESS WITH THESE LINES THEY ARE SET UP BY SED IN BATCH MODE!!!!!
#Following lines are just an example
#





OUTPUTNAME = "TEST1"
NPEEP = 1000
RUNS = 7 # leave this as 7 - for when mp7master is run in ipython
FUNDVAL = 100000000
FUNDFILE = "122018AAA6040.csv"
ANNCUM = "CUM"
INCOME = 0.05
WMFEE = 0.01
RUNINLAPSERATE = 0.04
INCOMELAPSERATE = 0.03
LAPSERATETHRESHOLD = 0.3
LAPSERATESUBTHRESHOLD = 0.01
DISCOUNT = 0.03
MORTALITYTABLEFEMALE = "iam_2012female_qx"
MORTALITYTABLEMALE = "iam_2012male_qx"
MORTALITYMULTIPLIER = 1
TAPER = 0.75
TAPERYEARS = 20
RUNINEXTRA = 0
STOCHASTIC = False 
PREMIUM = 0.0055000000000000005
ASSUMP_LAP = "NEW"
ASSUMP_EXP = "NEW"
ASSUMP_MOR = "NEW"
LAPSECAP = 1.5
DEBUG = False
YEARONEINCOME = 0.5
HALFYEARNPV = True






TODAY = np.datetime64('today', 'D')
THISYEAR = datetime.today().year
LASTYEAR = THISYEAR - 1 # used because values such as NPV are end of year, so starting NPV is last year's
FUNDDIRECTORY = '/home/test/funds/'
HOMEDIRECTORY = '/home/test/working/'
MAXAGE = 120 # saves looking it up in a mortality table.  Need to change if we change tables.
INFLATIONRATE = 1.02
PREMIUM = .0055
AFTERFEES = (1 - PREMIUM) * (1 - WMFEE)
SIMPLE_FILES = False # enables use of files with q=0 or lapse = 0 or whatever
PROCESSORS = 47
os.environ['NUMEXPR_MAX_THREADS'] = str(PROCESSORS) #'48'
#if SIMPLE_FILES:
#    MORTSPREADSHEET = 'debugmort.xlsx'
#    LAPSEFILE = HOMEDIRECTOR + 'debuglapse.xlsx'
#    market = FUNDDIRECTORY + 'fundsESG1.csv'
#    census = 'PolicyData.xlsx'
#else:
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
HALF_YR = DISCO ** .5 if HALFYEARNPV else 1.0 # this adds half a year to the NPV figures, arguable - cash flow timing
V = 1 / DISCO
PERSONFUND = FUNDVAL * 1.0 / NPEEP
THISYEAR = datetime.now().year
LASTYEAR = THISYEAR - 1
FUNDFILENAME = FUNDDIRECTORY + FUNDFILE
OUTPUTDIR = '/home/test/working/results/'
MONTHS = 'Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split()
NOWSTR = datetime.today().strftime("%Y_%m_%d")
prm_tuple = namedtuple('params',['group','npeep','sex','startage','fundsize','premiumpct','incomepct','wmpct','runinextra'])
BASELAPSES = pd.read_excel(LAPSEFILE,skiprows=2,index_col=0)
PRINTRUNS = 1 # prints every n run numbers, to check progress
intro_tuple = namedtuple('p',['group','npeep','sex','startage','fundsize',])

mtable= {'M':{},'F':{}} # mortality table

def censusaudit(q,probq): # first q is the Peep partial stats, the second q is the probabilistic q
    def month_str(y):
        #yr = int(y)
        #mon = int(round((y-yr)/(1/12)))
        #return str(yr) + ' ' + MONTHS[mon]
        return int(y)

    def pct_str(pct):
        return "{:.2f}%".format(pct)

    def money_str(x):
        return "${:,.0f}".format(x)

    first_time = True
    while not q.empty():
        peepdata = q.get()
        if first_time:
            first_time = False
            calcs = {}
            for col in peepdata.keys():
                calcs[col] = {'DthAgSm':[],'DthCt':[],'BstAgSm':[],'MaxDth':[],'BstCt':[],'LpsAgSm':[],
                        'LpsCt':[],'BstVlSm':[],'BstDthAgSm':[],'Prem':[],'NPVPrem':[],'NPeep':[]}
        for key,data in peepdata.items():
            for key2,val in data.items():
                calcs[key][key2].append(val)
    outcolnames = ['Total people in trial','Percent deaths','Percent bust','Percent lapsed',
        'Mean age at death','Max age at death','Mean age when bust','Mean age at lapse',
        'Mean year at death','Mean year when bust','Mean year when lapsed','Mean payout',
        'Mean years in benefit','Mean Premiums paid','NPV Mean Premiums paid']
    output = OrderedDict()
    if STOCHASTIC:
        for key,data in calcs.items():
            npeep = sum(data['NPeep'])
            startage = int(key[1:])
            totagedeath = sum(data['DthAgSm'])
            totdied = sum(data['DthCt'])
            totagebust = sum(data['BstAgSm'])
            totbust = sum(data['BstCt'])
            valbust = sum(data['BstVlSm'])
            premiums = sum(data['Prem']) / npeep
            npvpremiums = sum(data['NPVPrem']) / npeep
            yrsbust = sum(data['BstDthAgSm']) - totagebust + totbust # totust effectively adds 1 to each person, 
            # as they all get paid in the year they die.  Odd I know.  
            totagelapse = sum(data['LpsAgSm'])
            totlapsed = sum(data['LpsCt'])
            maxdeath = max(data['MaxDth'])
            meanagedeath = totagedeath / totdied if totdied else 0
            avgeyrdeath = THISYEAR + meanagedeath - startage
            meanagebust = totagebust / totbust if totbust else 0
            avgeyrbust = THISYEAR + meanagebust - startage
            avgebenpaid = valbust / totbust if totbust else 0
            yrsinpay = yrsbust / totbust if totbust else 0
            meanagelapse = totagelapse / totlapsed if totlapsed else 0
            avgeyrlapse = THISYEAR + meanagelapse - startage
            pctdied = totdied * 100 / npeep
            pctlapsed = totlapsed * 100 / npeep
            pctbust = totbust * 100 / npeep
            outdata = [npeep, pctdied, pctbust, pctlapsed, meanagedeath, maxdeath, meanagebust, meanagelapse,
                       avgeyrdeath, avgeyrbust, avgeyrlapse, avgebenpaid, yrsinpay, premiums, npvpremiums]
            output[key] = OrderedDict(zip(outcolnames,outdata))
    else: # we get the data from the prob q
        ix=['Total people in trial','Percent deaths','Percent bust','Percent lapsed',
        'Mean age at death','Max age at death','Mean age when bust','Mean age at lapse',
        'Mean year at death','Mean year when bust','Mean year when lapsed','Mean payout','Mean years in benefit']
        outdfs = {} 
        num=0
        while not probq.empty():
            peepdata = probq.get()
            for key,dick in peepdata.items():
                if key not in outdfs:
                    outdfs[key] = pd.DataFrame(index=ix)
                outdfs[key][num]=outdfs[key].index.map(dick)
            num += 1
        result = pd.DataFrame(index=ix)
        for k,v in outdfs.items():
            sm = v.sum(axis=1)
            mn = v.mean(axis=1)
            vt = v.T # transpose for selecting out busts > 0
            vtbust = vt[vt['Mean age when bust'] > 0]
            meanagewhenbust = vtbust.mean()['Mean age when bust']
            meanyearwhenbust = vtbust.mean()['Mean year when bust']
            meanpayout = vtbust.mean()['Mean payout']
            meanyrspayout = vtbust.mean()['Mean years in benefit']
            outdata = [sm['Total people in trial'],
                         mn['Percent deaths'],
                         mn['Percent bust'],
                         mn['Percent lapsed'],
                         mn['Mean age at death'],
                         mn['Max age at death'],
                         meanagewhenbust,
                         mn['Mean age at lapse'],
                         mn['Mean year at death'],
                         meanyearwhenbust,
                         mn['Mean year when lapsed'],
                         meanpayout,
                         meanyrspayout]
            output[key] = OrderedDict(zip(outcolnames,outdata))
    df = pd.DataFrame(output,index=outcolnames)
    #now the total column
    tot = df.sum(axis=1)
    npeep = tot['Total people in trial']
    pcts = df.loc['Total people in trial'] / npeep
    totcol = [npeep]
    for k in df.index[1:]:
        totcol.append((df.loc[k] * pcts).sum())
    df['Total'] = totcol
    return df

def intro(params):
    ddick = OrderedDict([
        ('01 Output name', OUTPUTNAME),
        ('02 People', NPEEP),
        ('03 Runs', RUNS),
        ('04 Fundvalue', "${:,}".format(FUNDVAL)),
        ('05 FundFile', FUNDFILE),
        ('06 Annual or Cumulative fund data', 'Cumulative' if ANNCUM.upper() == "CUM" else 'Annual'),
        ('07 Income', "{:.2f}%".format(INCOME*100)),
        ('08 Wealth Management Fee', "{:.2f}%".format(WMFEE * 100)),
        ('09 Run in lapse rate', "{:.2f}%".format(100 * RUNINLAPSERATE)),
        ('10 Income payment lapse rate', "{:.2f}%".format(100 * INCOMELAPSERATE) ),
        ('11 Threshold covered assets', "{:.2f}%".format(100 * LAPSERATETHRESHOLD)),
        ('12 Lapse rate below threshold', "{:.2f}%".format(100*LAPSERATESUBTHRESHOLD)),
        ('13 Discount rate', "{:.2f}%".format(100*DISCOUNT)),
        ('14 Female mortality table', MORTALITYTABLEFEMALE ),
        ('15 Male mortality table', MORTALITYTABLEMALE),
        ('16 Mortality multiplier', MORTALITYMULTIPLIER ),
        ('17 Linear mortality taper for new participants', "{:.2f}%".format(100 * TAPER)),
        ('18 Years in taper', TAPERYEARS ),
        ('19 Insurance premium as % of covered assets', "{:.2f}%".format(100 * PREMIUM)),
        ('20 Expenses assumption - Old or New', ASSUMP_EXP ),
        ('21 Mortality assumption - Old or New', ASSUMP_MOR ),
        ('22 Lapse assumption - Old or New', ASSUMP_LAP ),
        ('23 Mortality Spreadsheet', MORTSPREADSHEET ),
        ('24 Lapse Utilization Spreadsheet', LAPSEUTILIZATION )
        ])
    parameters = OrderedDict([('Parameter Values',ddick)])
    pframe = pd.DataFrame(parameters)
    paramlist = [intro_tuple(*i) for i in params]
    censusdicks = OrderedDict()
    for p in paramlist:
        censusdick = OrderedDict([
            ('Sex', p.sex),
            ('Start age', p.startage),
            ('Number of people', p.npeep),
            ('Fund size', p.fundsize)
            ])
        censusdicks[p.group] = censusdick
    censusframe = pd.DataFrame(censusdicks)
    return pframe,censusframe


def claims(po,inco):
    po['t'] = po.sum(axis=1) # sum each person not each year (this is done back in funk)
    pogt0 = po[po['t']>0] # select statement
    nbust = len(pogt0) # total number of bust
    totpo = po['t'].sum()
    result = pd.Series([totpo,nbust],index = ['totpo','nbust'],name = inco)
    return result

def benefs(wdd,df,inco):
    df2 = df[df['agedeath']>0] # just the ones with a date of death
    totdied = len(df2)
    wdd2 = wdd.sum(axis=1)
    wdd3 = wdd2[wdd2>0] # number of beneficiaries
    nbeneficiaries = len(wdd3)
    totbenefit = wdd3.sum()
    df['yeardied'] = df['agedeath'] + THISYEAR - df['startage']
    totyeardied = df['yeardied'].sum()
    return pd.Series([nbeneficiaries,totbenefit,totdied,totyeardied],
            index = ['nbeneficiaries','totbenefit','totdied','totyeardied'], 
            name = inco)

def getq(q):
    result = []
    while not q.empty():
        result.append(q.get())
    return result

def tvar(col, pctile): # calculates tail var on a column of data
    leng = int(len(col) * pctile / 100.0)
    return  col.sort_values(ascending = False)[leng:].mean()

def peep(inconame, params):
    # this is a general setup.  It setups the people dataframe and then adds the dataframes for 
    # premiums, expenses etc, all of which are indexed the same way - ie by person.
    names = [inconame + params.group + 'p' + str(i) for i in range(params.npeep)]
    p = pd.DataFrame(index=names)
    premiums = pd.DataFrame(index=names)
    expenses = pd.DataFrame(index=names)
    payouts = pd.DataFrame(index=names)
    cashflow = pd.DataFrame(index=names)
    wmfees = pd.DataFrame(index=names)
    income = pd.DataFrame(index=names)
    value = pd.DataFrame(index=names)
    wd_d = pd.DataFrame(index=names)
    wd_l = pd.DataFrame(index=names)
    p['startage'] = params.startage
    p['run'] = inconame
    p['group'] = params.group
    p['age'] = params.startage
    p['sex'] = params.sex
    p['fundsize'] = params.fundsize
    for k in ['agedeath','agebust','agelapsed']:
        p[k] = None
    for k in ['finished']:
        p[k] = False
    return p, premiums, expenses, payouts, cashflow, wmfees, income, value, wd_d, wd_l

def diedty(startage,age,sex,finished):
    if not finished:
        if random() < mtable[sex][startage][age]:
            return True
    return False

def bustty(funds,group,agebust,finished,year):
    if not finished: # can't go bust if dead or lapsed
        if agebust is None: # and not already bust
            if funds[group].bustyear[year]: # but they are now
                return True # this will set the bust year
    return False

def lapsedty(year, funds, group, startage, age, finished):
    if not finished:
        fd = funds[group]
        if ASSUMP_LAP == 'OLD':
                if (age - startage > 1) and (age > 64): # then not in runin
                    if fd.fundhist[year] / fd.startvalue > LAPSERATETHRESHOLD:
                        lapserate = INCOMELAPSERATE
                    elif fd.fundhist[year] == 0:
                        lapserate = 0.0
                    else:
                        lapserate = LAPSERATESUBTHRESHOLD
                else:
                    lapserate = RUNINLAPSERATE 
        else: # ASSUMP_LAP = NEW
            multiplier = 1.0
            if (age - startage > 1) and (age > 64): # then not in runin
                multiplier = .5
            nyears = min([age - startage + 1]) # starts in year 1
            baselapse = BASELAPSES[nyears][startage] * multiplier
            dynlapse = min([(fd.fundhist[year] / fd.startvalue) ** 3,LAPSECAP])
            lapserate = baselapse * dynlapse
        if random() < lapserate:
            return True
        else:
            return False
    return False

def premiumty(year, funds, group, finished):
    fd = funds[group]
    if (not finished):
        return float(fd.premiums[year])
    return 0.0

def expty(year, funds, group, finished):
    fd = funds[group]
    if not finished:
        return fd.expenses[year]
    return 0.0

def payoutty(year, funds, group, finished):
    fd = funds[group]
    if not finished:
        return float(fd.payout[year])
    return 0.0

def wmfeety(year, funds, group, finished):
    fd = funds[group]
    if not finished:
        return float(fd.wmfee[year])
    return 0.0

def valuety(year,funds,group,finished):
    #returns the fund value at end of year used in showing value to beneficiary
    # NB the y+1 is because the fundhist stores the fund value at the start of a year. If they die during the year
    # they will just have the end of year amount left, ie after charges and after market movements.
    fd = funds[group]
    if not finished:
        try:
            return fd.fundhist[year + 1]
        except:
            return fd.fundhist[year]
            logging.warning ('Serious index error valuety in mp6.py:',str(group),':',str(year))
            1/0
    return 0.0

def wdty(year,funds,group,done): # done is either True for died or True for lapsed, depending how called
    if done:
        fd = funds[group]
        return fd.eoyhist[year]
    return 0.0

def incomety(year, funds, group, finished):
    fd = funds[group]
    if not finished:
        return float(fd.incomes[year])
    return 0.0

vvaluety = np.vectorize(valuety)
vdiedty = np.vectorize(diedty)
vlapsedty = np.vectorize(lapsedty)
vpremiumty = np.vectorize(premiumty)
vexpty = np.vectorize(expty)
vpayoutty = np.vectorize(payoutty)
vwmfeety = np.vectorize(wmfeety)
vincomety = np.vectorize(incomety)
vbustty = np.vectorize(bustty)
vwdty = np.vectorize(wdty)

def runno(name):
    num = int(name[3:])
    if num % PRINTRUNS == 0: 
        print(name, ' started')

def set_params(mortable):
    cs = pd.read_json(CENSUS) # always same name - file sent from desktop 
    cs.index = [i.lower() for i in cs.index]
    youngest = min(cs.loc['startage'])
    oldest = len(mortdick[mortable]) # or 50... can't remember why I sometimes put 50 here..
    years = int(oldest - youngest)
    prms = [[csn,int(csd.n),csd.sex,int(csd.startage),float(csd.fundsize)] for (csn,csd) in cs.items()]
    return prms, years

""" START"""

def _lookupmort(key, age, pmort):
    if age > 120: age = 120
    return pmort[key].loc[age] 

lookupmort_p = np.vectorize(_lookupmort, otypes=[np.float])

def morty(key):
    sex = key[0]
    age = int(key[1:])
    if sex == 'F': 
        table = MORT_Fq
        improve = 1 - MORT_FG2
    else:
        table = MORT_Mq
        improve = 1 - MORT_MG2
    table['n'] = [max([0,i - age + 8]) for i in table.index] # power to raise the G2 to
    return table['q'] * np.power(improve['G2'],table['n']) * MORT_F['F']

def _lapsedty(startage, age, fundsize, value):
    try:
        baselapse = LAPSETABLE[1 + age - startage][startage] # 1 is because first year is year 1!
    except:
        return 0.0
    dynlapse = min([np.power(value / fundsize,3),1.5])
    return baselapse * dynlapse

lapsedty_p = np.vectorize(_lapsedty, otypes=[np.float])

def _utilized(startage, age, qualified):
    if qualified:
        table = QUALUTILIZED
    else:
        table = UNQUALUTILIZED
    try:
        rate = table[age + 1][startage] # plus one because gets called the year before in prep for next y
    except:
        return 0.0
    if not pd.isnull(rate): # then we have a number
        return rate 
    return 0.0 

utilized_p = np.vectorize(_utilized, otypes=[np.float])

def fillq(q,nlines=1000000):
    if 'AAA' in MARKET: # then we have a AAA file which is cumulative
        df_ = pd.read_csv(MARKET,index_col = False, names = range(2020,2091))
        ad1names = range(2020,2090)
        ad2names = range(2021,2091)
        ad1 = df_[ad1names]
        ad2 = df_[ad2names]
        ad2.columns = ad1names
        df = ad2 / ad1
    else:
        df = pd.read_csv(MARKET,names=range(2020,2090))
        df += 1
    stop = nlines
    line = 0
    for i in range(len(df)):
        q.put(df.iloc[i])
        line += 1
        if line > stop: break

def summer(df,name):
    try:
        df = df.to_frame()
    except:
        pass
    summ = df.sum()
    summ.name = name
    return summ

def punk(inconame, queues, peepframe, years, pmort):
    runno(inconame)
    people = peepframe
    gr = queues['Funds'].get() # read the next line of the price file
    gr.name = inconame
    # work out how many years to iterate...
    #'n','fundsize', 'PolicyId','sex','DOB','qualified' are column names
    premiums,wmfees,expenses,income,payout,cashflows,npvs = [pd.DataFrame(index=people.index) for i in range(7)]
    stats, cashflow, NPV, peepleft, diedthisy, bustthisy = [pd.DataFrame(index=people.index) for i in range(6)]
    lapsedthisy, value, wd_death, wd_lapse = [pd.DataFrame(index=people.index) for i in range(4)]
    # now get starting data types correct
    for col in ['n','fundsize']: people[col] = people[col].astype(float)
    for col in ['age']: people[col] = people[col].astype(int)
    people['runindone'] = False
    people['agebust'] = 0
    people['bust'] = False 
    people['nNoIncome'] = people['n'] 
    people['nIncome'] = 0.0
    people['nIncome'] = people['nIncome']
    people['nStart'] = people['n']
    people['nIncomeLY'] = 0.0
    people['nIncomeNew'] = 0.0
    people['value'] = people['fundsize'] * people['n']
    people['shadowstart'] = people['fundsize'] #* people['n']
    people['shadow'] = people['fundsize'] #* people['n']
    people['expincome'] = 0.0
    people['expincome_base'] = 0.0
    people['expincome_a'] = 0.0
    for col in ['nIncome','nIncomeLY','nIncomeNew','expincome','expincome_base','expincome_a']: 
        people[col] = people[col].astype(float)
    for col in ['agebust']: people[col] = people[col].astype(int)
    n = 0 # number of years executed
    if DEBUG: 
        writer = pd.ExcelWriter(OUTPUTDIR+inconame+'punkdebug.xlsx',engine = "openpyxl", mode = 'w')
        people.to_excel(writer,'Peoplestart')
    #years = range(2020,2024)
    for y in years:
        inflation = INFLATIONRATE ** n
        try:
            growth = gr[y]
        except:
            growth = 1
            #print("Warning, growth = 1 in year ",y)
        people['PersValSoY'] = people['value'] / people['n']
        #
        # Step 1 Valuation for the year - take out premiums, fees, income, payouts at start of year
        #
        premiums[y] = people['value'] * PREMIUM
        people['value'] = people['value'] - premiums[y]
        wmfees[y] = people['value'] * WMFEE
        people['value'] = people['value'] - wmfees[y]
        income[y] = np.where(people['runindone'] == True, np.minimum(people['value'],people['expincome']),0)
        payout[y] = np.where(people['runindone'] == True, (people['expincome'] - income[y]), 0.0)
        people.loc[(people.runindone == True) & (people.bust == False) & (payout[y] > 0), "agebust"] = y
        bustthisy[y] = np.where((people.runindone == True) & (people.bust == False) & (payout[y] > 0),people['n'],0)
        people.loc[(people.runindone == True) & (people.bust == False) & (payout[y] > 0), "bust"] = True 
        expenses[y] = 1.1 * (people['value'] * .0005 + 25 * inflation * people['n'] + payout[y] * .01)
        cashflow[y] = premiums[y] - expenses[y]
        people['value'] = people['value'] - income[y]
        #
        # Step 2 Census calculations - see who dies, lapses, decides to utilize during the year
        #
        # Step 2a get the qs and ps for dying and lapseing
        people['q'] = lookupmort_p(people['mortkey'], people['age'], pmort)
        people['p'] = 1 - people['q']
        diedthisy[y] = people['n'] * people['q']
        people['util'] = utilized_p(people['startage'], people['age'], people['qualified'])
        people['qLapseNoInc'] = lapsedty_p(people['startage'], people['age'], people['fundsize'], people['PersValSoY']) 
        people['qLapseInc'] = people['qLapseNoInc'] / 2.0
        people['pLapseNoInc'] = 1 - people['qLapseNoInc']
        people['pLapseInc'] = 1 - people['qLapseInc']
        #
        # Step 2b numbers of lapses
        #NB calculate lapsedthisy BEFORE you calculate nIncome and nNoIncome - same calculation
        # lapsedthisy - survive AND succumb to lapse Income or NoIncome
        lapsedthisy[y] = people['p'] * (people['nNoIncome'] * people['qLapseNoInc'] + people['nIncome'] * people['qLapseInc'])
        people['lapsed'] = lapsedthisy[y]
        people['nIncome'] = people['nIncome'] * people['p'] * people['pLapseInc'] # survive mortality and survive lapse
        people['nNoIncome'] = people['nNoIncome'] * people['p'] * people['pLapseNoInc'] # same but for NoIncome
        people['nEoY'] = people['nIncome'] + people['nNoIncome']
        # wd_death just the % of people who died * value after fees and premiums and income
        wd_death[y] = diedthisy[y] * people['value'] / people['n']
        people['died'] = diedthisy[y]
        people['WD_death'] = wd_death[y]
        # wd_lapse the % of people who lapsed out of the survivors
        wd_lapse[y] = lapsedthisy[y] * people['value'] / people['n'] #lapsedthisy[y] * people['value'] / (people['n'] - diedthisy[y])
        people['WD_Lapse'] = wd_lapse[y]
        # Step 3 Shadow calculations
        people['shadow'] = people['shadow'] * AFTERFEES * growth
        people['ShadowIncomeBase'] = np.maximum(people['shadowstart'],people['shadow'])
        # Ste 4 end of year processing
        n += 1
        people['age'] += 1
        inflation *= INFLATIONRATE
        people['value'] = np.where(people['n'] > 0,(people['value'] * growth * people['nEoY'] / people['n']),0.0)
        value[y] = people['value']
        people['n'] = people['nEoY']
        people['runindone'] = np.where((people['age'] > 64) & (n > 1),True,False)
        # Step 5 now we are at the beginning of a year.  Calculate the income folk will get
        #
        #      use the new census info to calculate the new expected income for next year
        #      a) break the people in income down to InIncomeLY and the new utilizes nIncomeNew
        people['nIncomeLY'] = people['nIncome'] # store last year for pro rata below
        people['nIncomeNew'] = people['nNoIncome'] * people['util']
        people['nNoIncome'] = people['nNoIncome'] - people['nIncomeNew']
        people['nIncome'] = people['nIncomeLY'] + people['nIncomeNew']
        #      b) get the income base for any new joiners - the shadow income
        people['incomebase'] = people['ShadowIncomeBase'] * INCOME
        #      c) calculate new expected income
        #      expincome_a is the expected income level for a single person pro rata'd amount between New and LY
        people['inc_base_LY'] = people['expincome_base']
        people['expincome_base'] = np.where(people['nIncome'] > 0,((people['incomebase'] * people['nIncomeNew'] / people['nIncome']) + (people['inc_base_LY'] * people['nIncomeLY'] / people['nIncome'])),0.0) # this is the amount of income by one averagely utilized person of average utilized income
        people['expincome_TY'] = np.where(people['nIncome'] > 0, # see YEARONEINCOME, may or may not be 50%
            ((people['incomebase'] * YEARONEINCOME * people['nIncomeNew'] / people['nIncome']) + (people['inc_base_LY'] * people['nIncomeLY'] / people['nIncome'])),
            0.0) # this is the amount of income by one averagely utilized person of average utilized income
        people['expincome'] = people['expincome_TY'] * people['nIncome'] # this is the expected income in total - payable
        peepleft[y] = people['n']
        if DEBUG: people.to_excel(writer,'People'+str(y))
    cashflow = premiums - expenses - payout
    
    if DEBUG:
        people.to_excel(writer,'People')
        income.to_excel(writer,'Income')
        wmfees.to_excel(writer,'WMFees')
        premiums.to_excel(writer,'Premiums')
        payout.to_excel(writer,'Payout')
        expenses.to_excel(writer,'Expenses')
        diedthisy.to_excel(writer,'DiedTY')
        lapsedthisy.to_excel(writer,'LapsedTY')
        bustthisy.to_excel(writer,'BustTY')
        writer.save()
    peepdata = {}
    """ This next piece pulls out some summary statistics.  Much more compact than passing off everyone's details.
    """
    outdata = {}
    for key in set(people.mortkey):
        sex = key[0]
        age = int(key[1:])
        sexcond = (people.sex == sex)
        outp = {}
        cond = (people.sex == sex) & (people.startage == age)
        n = people.loc[cond].nStart.sum()
        deaths = diedthisy.loc[cond].sum()
        busts= bustthisy.loc[cond].sum()
        lapses = lapsedthisy.loc[cond].sum()
        payouts = payout.loc[cond].sum()
        totdied = deaths.sum() # sum both dimensions
        totlapsed =  lapses.sum() 
        totbust =  busts.sum() 
        pctdied = 100.0 * totdied / n
        pctlapsed = 100.0 * totlapsed / n
        pctbust = 100.0 * totbust / n
        pctdeaths = deaths / totdied
        pctlapses= lapses / totlapsed
        pctbusts = busts / totbust
        avge_y_d = sum([yr * pct for yr, pct in zip(pctdeaths.index,pctdeaths)])
        avge_y_l = sum([yr * pct for yr, pct in zip(pctlapses.index,pctlapses)])
        avge_y_b = sum([yr * pct for yr, pct in zip(pctbusts.index,pctbusts)])
        age_d = int(avge_y_d) - THISYEAR + age
        age_l= int(avge_y_l) - THISYEAR + age
        age_b= int(avge_y_b) - THISYEAR + age if not np.isnan(avge_y_b) else 0
        maxage_d = 120
        avge_po = payouts.sum() / n
        yrsinpo = len([i for i in payouts if i > 0] )
        outp['Total people in trial'] = n
        outp['Percent deaths'] = pctdied
        outp['Percent bust'] = pctbust
        outp['Percent lapsed'] = pctlapsed
        outp['Mean age at death'] = age_d
        outp['Max age at death'] = 120
        outp['Mean age when bust'] = age_b 
        outp['Mean age at lapse'] = age_l
        outp['Mean year at death'] = avge_y_d
        outp['Mean year when bust'] = avge_y_b
        outp['Mean year when lapsed'] = avge_y_l
        outp['Mean payout'] = avge_po
        outp['Mean years in benefit']= yrsinpo
        outdata[key] = outp
    queues['PeepProb'].put(outdata)
    gr.name = inconame
    cashflowplus = cashflow.sum() #* HALF_YR # take out the half year stuff - pje
    npv = pd.Series([(npf.npv(DISCOUNT,cashflowplus[i:])) for i in range(len(cashflowplus))],index=years) / DISCO #
    npv.name = inconame #
    queues['Premiums'].put(summer(premiums,inconame)) 
    queues['Expenses'].put(summer(expenses,inconame)) 
    queues['Payouts'].put(summer(payout,inconame))
    #Calculate statistics
    #  Part 1 claims
    totpo = payout.sum().sum()
    nbust = bustthisy.sum().sum()
    claimstats = pd.Series([totpo,nbust],index=['totpo','nbust'],name=inconame)
    # Part 2 beneficiaries
    #  rows with payouts
    nbusts = bustthisy.sum().sum()
    totbenefit = wd_death.sum().sum()
    totdied = diedthisy.sum().sum()
    nbeneficiaries = totdied - nbusts 
    died1 = diedthisy.copy()
    died2 = pd.DataFrame(index=died1.index)
    for col in died1:
        died2[col] = col * diedthisy[col]
    totyeardied = died2.sum().sum()
    benefstats = pd.Series([nbeneficiaries,totbenefit,totdied,totyeardied],
            index = ['nbeneficiaries','totbenefit','totdied','totyeardied'], 
            name = inconame)
    # Part 3 - join them up
    stats = pd.concat([claimstats,benefstats])
    stats.name = inconame
    # Write out queue data
    queues['Stats'].put(stats)
    queues['WMFees'].put(summer(wmfees,inconame)) 
    queues['Income'].put(summer(income,inconame)) 
    queues['CashFlow'].put(summer(cashflow,inconame)) 
    queues['NPV'].put(npv) 
    queues['Peepleft'].put(summer(peepleft,inconame)) #
    queues['Diedthisy'].put(summer(diedthisy,inconame)) #
    queues['Bustthisy'].put(summer(bustthisy,inconame)) #
    queues['Lapsedthisy'].put(summer(lapsedthisy,inconame)) #
    queues['Value'].put(summer(value,inconame)) #
    queues['WD_Death'].put(summer(wd_death,inconame)) #
    queues['WD_Lapse'].put(summer(wd_lapse,inconame)) #
    queues['Anngrowth'].put(gr)
    #print(inconame,' finished')
 
def outxl(writer, tabname, df, extracol):
    ix,name,svals = extracol # index, name and startingvals for extra column
    df.insert(ix,name,svals)
    df.to_excel(writer,tabname)

def funk(inconame, queues, params, nyears):
    from stox11 import StockMarket, Fund
    years = nyears
    runno(inconame)
    people, premiums, expenses, value = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    payouts, cashflow, income, wmfees = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    wd_death, wd_lapse = pd.DataFrame(), pd.DataFrame()
    anngrowth = queues['Funds'].get() # read the next line of the generated price file
    anngrowth.name = inconame
    queues['Anngrowth'].put(anngrowth)
    mkt = StockMarket(inconame,anngrowth,FUNDFILE[:-4]) 
    fd = OrderedDict()
    paramlist = [prm_tuple(*i + [PREMIUM, INCOME, WMFEE, RUNINEXTRA]) for i in params]
    peepleft = pd.Series([0 for i in years],index=years,name=inconame)
    diedthisy = pd.Series([0 for i in years],index=years,name=inconame)
    lapsedthisy = pd.Series([0 for i in years],index=years,name=inconame)
    bustthisy = pd.Series([0 for i in years],index=years,name=inconame)

    for grp in paramlist:
        fd[grp.group] = Fund(inconame, mkt,years, grp, ASSUMP_EXP, YEARONEINCOME) 
        gpe, gprem, gexp, gpay, gcash, gfees, gincome, gval, gwd_d, gwd_l = peep(inconame, grp)
        if ASSUMP_MOR == 'OLD':
            mtable[grp.sex][grp.startage] = mort(MORTALITYTABLEMALE, MORTALITYTABLEFEMALE,grp.sex, grp.startage, TAPER, TAPERYEARS,MORTALITYMULTIPLIER)
        else:
            mtable[grp.sex][grp.startage] = morty(grp.sex + str(grp.startage))
        people = pd.concat([people, gpe])
        premiums =  pd.concat([premiums, gprem])
        expenses = pd.concat([expenses, gexp])
        payouts = pd.concat([payouts, gpay])
        cashflow = pd.concat([cashflow, gcash])
        wmfees = pd.concat([wmfees, gfees])
        value = pd.concat([value, gval])
        wd_death = pd.concat([wd_death, gwd_d])
        wd_lapse = pd.concat([wd_lapse, gwd_l])
        income = pd.concat([income, gcash])
    for y in years:
        premiums[y] = vpremiumty(y, fd, people['group'], people['finished'])
        expenses[y] = vexpty(y, fd, people['group'], people['finished'])
        payouts[y] = vpayoutty(y, fd, people['group'], people['finished'])
        wmfees[y] = vwmfeety(y, fd, people['group'], people['finished'])
        income[y] = vincomety(y, fd, people['group'], people['finished'])
        cashflow[y] = premiums[y] - expenses[y] - payouts[y]
        people['bustty'] = vbustty(fd, people['group'], people['agebust'], people['finished'],y)
        people.loc[people['bustty'] == True, 'agebust'] = people['age']
        people['diedty'] = vdiedty(people['startage'], people['age'], people['sex'], people['finished'])
        people.loc[people['diedty'] == True, 'agedeath'] = people['age'] 
        people.loc[people['diedty'] == True, 'finished'] = True
        people['lapsedty'] = vlapsedty(y, fd, people['group'], people['startage'], 
                                        people['age'], people['finished'])
        people.loc[people['lapsedty'] == True, 'agelapsed'] = people['age']
        people.loc[people['lapsedty'] == True, 'finished'] = True
        value[y] = vvaluety(y,fd, people['group'],people['finished']) # used to go after income[y]
        wd_death[y] = vwdty(y,fd, people['group'],people['diedty'])
        wd_lapse[y] = vwdty(y,fd, people['group'],people['lapsedty'])
        people.loc[people['diedty'] == True, 'benvalue'] = value[y] # used to go after died bit
        people['age'] += 1
        peepleft[y] = (~people['finished']).sum()
        diedthisy[y] = len(people.loc[people['diedty'] == True])
        lapsedthisy[y] = len(people.loc[people['lapsedty'] == True])
        bustthisy[y] = len(people.loc[people['bustty'] == True])
    peepdata = {}
    """ This next piece pulls out some summary statistics.  Much more compact than passing off everyone's details.
        But not so useful if we actually want the distribution of payouts and sd.
    """
    for age in sorted(set(people.startage)):
        sumpo = payouts.sum(axis=1) # this just totals the payout for each person
        sumpo.name = 'payout'
        sample = pd.concat([people,sumpo],axis=1) # and this adds total payout column to people df
        for sex in ['M','F']:
            group = (sample['startage'] == age) & (sample['sex'] == sex)
            selection = sample.loc[group]
            premiumsel = premiums.loc[group] # these the premiums
            premiumtot = premiumsel.sum().sum()
            premiumbyyr = premiumsel.sum()
            premiumnpv = npf.npv(DISCOUNT,premiumbyyr) / DISCO
            if not selection.empty:
                sm = selection.sum()
                ct = selection.count()
                busted = selection.agebust.notnull() # this enables age at death of the ones who bust
                try:
                    maxdeath = max(selection[selection.agedeath.notnull()]['agedeath'])
                except:
                    maxdeath = 0
                busteds = selection[busted]
                bustedsum = busteds.sum()
                peepdata[sex+str(age)] = {'DthAgSm': sm.agedeath, 'DthCt': ct.agedeath , 
                        'BstAgSm': sm.agebust, 'BstCt': ct.agebust , 'MaxDth': maxdeath,
                                 'LpsAgSm': sm.agelapsed, 'LpsCt': ct.agelapsed , 
                                 'BstVlSm': sm.payout, 'BstDthAgSm': bustedsum.agedeath,
                                 'Prem': premiumtot, 'NPVPrem': premiumnpv,
                                 'NPeep': int(len(selection))
                                 }
    """ This next piece gets the market returns applicable to each person so we can do a distribution of returns"""
    #people['returnage'] = min([people['agedeath'],people['agebust'],people['agelapsed']]) # last year they were in market

    queues['Peep'].put(peepdata)
    premiumssum = premiums.sum()
    premiumssum.name = inconame
    expensessum = expenses.sum()
    expensessum.name = inconame
    payoutssum = payouts.sum()
    payoutssum.name = inconame
    wmfeessum = wmfees.sum()
    wmfeessum.name = inconame
    cashflowsum = cashflow.sum()
    cashflowsum.name = inconame
    incomesum = income.sum()
    incomesum.name = inconame
    valuesum = value.sum()
    valuesum.name = inconame
    wd_dsum = wd_death.sum()
    wd_dsum.name = inconame
    wd_lsum = wd_lapse.sum()
    wd_lsum.name = inconame
    cashflowplus = cashflowsum * HALF_YR
    npv = pd.Series([(npf.npv(DISCOUNT,cashflowplus[i:])) for i in range(len(cashflowplus))],index=years) / DISCO
    npv.name = inconame
    queues['Premiums'].put(premiumssum)
    queues['Expenses'].put(expensessum)
    queues['Payouts'].put(payoutssum)
    claimstats = claims(payouts,inconame)
    benefstats = benefs(wd_death,people,inconame)
    stats = pd.concat([claimstats,benefstats])
    stats.name = inconame
    queues['Stats'].put(stats)
    queues['WMFees'].put(wmfeessum)
    queues['Income'].put(incomesum)
    queues['CashFlow'].put(cashflowsum)
    queues['NPV'].put(npv)
    queues['Peepleft'].put(peepleft)
    queues['Diedthisy'].put(diedthisy)
    queues['Bustthisy'].put(bustthisy)
    queues['Lapsedthisy'].put(lapsedthisy)
    queues['Value'].put(valuesum)
    queues['WD_Death'].put(wd_dsum)
    queues['WD_Lapse'].put(wd_lsum)
    if DEBUG:
        writer = pd.ExcelWriter(OUTPUTDIR + OUTPUTNAME + inconame + 'funkdebug' + '.xlsx')
        people.to_excel(writer,'People')
        premiums.to_excel(writer,'Premiums')
        wmfees.to_excel(writer,'WMFees')
        expenses.to_excel(writer,'Expenses')
        income.to_excel(writer,'Income')
        payouts.to_excel(writer,'Payouts')
        value.to_excel(writer,'Value')
        wd_death.to_excel(writer,'WithdrawalsDeath')
        wd_lapse.to_excel(writer,'WithdrawalsLapse')
        cashflow.to_excel(writer,'Cashflow')
        pd.DataFrame(peepdata).to_excel(writer,'Peepdata')
        stats.to_excel(writer,'Stats')
        writer.save()
    runnum = int(inconame[3:])
    if runnum % 50 == 0:
        outfil(OUTPUTNAME + inconame + ' finished')

def outfil(txt):
    try: # this might fail occasionally, but surely not often!  We don't care if it is not often.
        outf = open(OUTPUTDIR + 'output.txt','a')
        outf.write(txt+'\n')
        outf.close()
    except:
        pass

if __name__ == '__main__':
    start_time = int(time.time())
    # STEP 1 
    #     set up the parameters and other admin
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    pool = Pool(processes=PROCESSORS) #47)
    seed(1)
    # these are the special queues that don't mind being written to by parallel processes
    m = Manager()
    queues = {}
    queues['Premiums'] = m.Queue() 
    queues['Expenses'] = m.Queue() 
    queues['Payouts'] = m.Queue() 
    queues['WMFees'] = m.Queue() 
    queues['Income'] = m.Queue() 
    queues['CashFlow'] = m.Queue() 
    queues['Peep'] = m.Queue() # queue for stats calculations
    queues['PeepProb'] = m.Queue() # probabilistic version, different calcs
    queues['Funds'] = m.Queue() 
    queues['Value'] = m.Queue() 
    queues['WD_Death'] = m.Queue() 
    queues['WD_Lapse'] = m.Queue() 
    queues['100Prices'] = m.Queue() 
    queues['Stats'] = m.Queue() 
    queues['NPV'] = m.Queue() 
    queues['Peepleft'] = m.Queue()
    queues['Anngrowth'] = m.Queue()
    queues['Diedthisy'] = m.Queue()
    queues['Lapsedthisy'] = m.Queue()
    queues['Bustthisy'] = m.Queue()
    loadcsvqueue(queues['Funds'],queues['100Prices'],FUNDFILENAME,ANNCUM)
    inputs = []
    params, years = set_params(MORTALITYTABLEFEMALE)
    nyears = range(THISYEAR, THISYEAR + years)
    if STOCHASTIC:
        for n in range(RUNS):
            insco = 'Run' + str(n) # this is just the name - the actual instance is created in funk above
            inputs.append([insco, queues, params, nyears])
        if DEBUG:
            for inp in inputs:
                funk(*inp) # use this for debugging - you cannot debug inside pool.starmap
        else:
            results = pool.starmap(funk, inputs)
    else:
        peep = pd.read_json(CENSUS).T
        peep.columns = [i.lower() for i in peep.columns]
        peep['age'] = peep['startage']#  [((TODAY - peep['DOB']) / np.timedelta64(1,'Y')).astype(int)
        #peep['startage'] = ((TODAY - peep['DOB']) / np.timedelta64(1,'Y')).astype(int)
        peep['mortkey'] = peep['sex'] + peep['age'].astype(int).astype(str)
        for key in set(peep['mortkey']): pmort[key] = morty(key)
        #years = range(THISYEAR, THISYEAR + MAXAGE - min(peep['age']) +1) # + 1 to get age 120
        start_time = int(time.time())
        for n in range(RUNS):
            insco = 'Run' + str(n) # this is just the name - the actual instance is created in funk above
            inputs.append([insco, queues, peep.copy(), nyears, pmort])
        # STEP 2 
        #   This is where the work happens ...
        if DEBUG:
            for inp in inputs:
                punk(*inp) 
        else:
            results = pool.starmap(punk, inputs)

    # STEP 3 
    #   Now let's process the output into the census data and some census spreadsheets
    # write a big audit spreadsheet, for those that want it
    writer = pd.ExcelWriter(OUTPUTDIR + OUTPUTNAME + '.xlsx',engine = "openpyxl", mode = 'a')
    peepdf = censusaudit(queues['Peep'],queues['PeepProb'])
    introdf, censusdf = intro(params)
    introdf.to_excel(writer,'Parameters')
    censusdf.to_excel(writer,'Census')
    peepdf.to_excel(writer,'CensusStatistics')
    premdf = pd.concat(getq(queues['Premiums']),axis=1).T.sort_index()
    zeeros = np.zeros(premdf.shape[0])
    npp = np.full(premdf.shape[0],NPEEP)
    fval = np.full(premdf.shape[0],FUNDVAL)
    outxl(writer,'Premiums',premdf,[0,2019,zeeros])
    expdf = pd.concat(getq(queues['Expenses']),axis=1).T.sort_index()
    outxl(writer,'Expenses',expdf,[0,2019,zeeros])
    payoutdf = pd.concat(getq(queues['Payouts']),axis=1).T.sort_index()
    outxl(writer,'Payouts',payoutdf,[0,2019,zeeros])
    statsdf = pd.concat(getq(queues['Stats']),axis=1).T.sort_index() # joins them together with each statistic a column
    cfdf = pd.concat(getq(queues['CashFlow']),axis=1).T.sort_index()
    outxl(writer,'CashFlows',cfdf,[0,2019,zeeros])
    npvdf = pd.concat(getq(queues['NPV']),axis=1).T.sort_index()
    npvdf.columns = [i-1 for i in npvdf.columns]
    npvdf.to_excel(writer,'NPV')
    feesdf = pd.concat(getq(queues['WMFees']),axis=1).T.sort_index()
    outxl(writer,'Fees',feesdf,[0,2019,zeeros])
    incomedf = pd.concat(getq(queues['Income']),axis=1).T.sort_index()
    outxl(writer,'Income',incomedf,[0,2019,zeeros])
    peepleftdf = pd.concat(getq(queues['Peepleft']),axis=1).T.sort_index()
    outxl(writer,'PeopleLeft',peepleftdf,[0,2019,npp])
    diedthisydf = pd.concat(getq(queues['Diedthisy']),axis=1).T.sort_index()
    outxl(writer,'DiedTY',diedthisydf,[0,2019,zeeros])
    bustthisydf = pd.concat(getq(queues['Bustthisy']),axis=1).T.sort_index()
    outxl(writer,'BustTY',bustthisydf,[0,2019,zeeros])
    lapsedthisydf = pd.concat(getq(queues['Lapsedthisy']),axis=1).T.sort_index()
    outxl(writer,'LapsedTY',lapsedthisydf,[0,2019,zeeros])
    valueremainingdf = pd.concat(getq(queues['Value']),axis=1).T.sort_index()
    outxl(writer,'ValueRemaining',valueremainingdf,[0,2019,fval])
    wd_deathdf = pd.concat(getq(queues['WD_Death']),axis=1).T.sort_index()
    outxl(writer,'WD_Death',wd_deathdf,[0,2019,zeeros])
    wd_lapsedf = pd.concat(getq(queues['WD_Lapse']),axis=1).T.sort_index()
    outxl(writer,'WD_Lapse',wd_lapsedf,[0,2019,zeeros])
    anngrowthdf = pd.concat(getq(queues['Anngrowth']),axis=1).T.sort_index()
    outxl(writer,'AnnGrowth',anngrowthdf,[0,2019,zeeros])
    writer.save()
    # end of audit spreadsheet
    # STEP 4 
    #  Prep the 100 price file
    price100 = pd.concat(getq(queues['100Prices']),axis=0)
    #
    # STEP 5
    #   These next few lines are commented out - they were used for some kind of Bayesian analysis
    #   Good idea in principle, but not working now.
    #busted = peepdf.loc[~peepdf.agebust.isnull()]
    #busted['yearbust'] = THISYEAR + busted['agebust'] - busted['startage']
    #busted['yearsbust'] = busted['agedeath'] - busted['agebust']
    #bustmean = busted.mean()
    #losses = set(returnlt0(cfsumt)) # these are the column names that had sum(cf) < 0
    #downyr5 = set(returnlt(fundhdf.T,[0,5])) # these are the column names that were down in year 5
    #ploss = len(losses) * 100.0 / totalpop
    #plossgdyr = 'N/A' if not len(downyr5) else len(losses.intersection(downyr5)) / len(downyr5) # loss given down in year 5
    # STEP 6
    #  Calculate the Tail Vars from the NPV Cashflow figures
    npvavg = npvdf.mean()
    npvsthisy = npvdf[THISYEAR] # LAST YEAR for NPV purposes is 12/31 of last year - ie thisyear
    npvtv70 = tvar(npvsthisy,70)
    npvtv90 = tvar(npvsthisy,90)
    if RUNS > 49: # I Tail VAR 98 if at least 100 runs, else zero
        npvtv98 = tvar(npvsthisy,98)
        npvtv97 = tvar(npvsthisy,97) 
    else:
        npvtv98 = 0.0 # 0 as N/A or None causes calculation problems.  Will be > 49 runs in most cases.
        npvtv97 = 0.0 
    spectrumthisy = npvsthisy.sort_values()
    worstone = spectrumthisy.iloc[0] # The worst run
    tot_paid = statsdf['totpo'].sum()
    tot_payees = statsdf['nbust'].sum()
    pct_claim = tot_payees * 100.0 / (NPEEP * RUNS)
    avg_payout = tot_paid/tot_payees
    tot_died = statsdf['totdied'].sum()
    tot_ydied = statsdf['totyeardied'].sum()
    avg_ydeath = tot_ydied / tot_died
    nbenefs = statsdf['nbeneficiaries'].sum()
    totben = statsdf['totbenefit'].sum()
    avg_deathben = totben / nbenefs
    yearbust = 0
    yearsbust = 0
    # risk based capital calculation
    cte98 = npvtv98
    cte70_1 = npvsthisy.copy()
    cte70_1[(cte70_1 > 0)] = 0
    cte70 = tvar(cte70_1,70)
    taxcte70 = .9281 * cte70
    vm21 = .25 * ((cte98 - cte70) * (1 - 0.21) - (cte70 - taxcte70) * .21)
    report_data = OrderedDict([('CTE 0 NPV CF $', npvavg[THISYEAR]),
                              ('CTE 70 $', npvtv70),
                              ('CTE 70 LT0 $', cte70),
                              ('CTE 90 $', npvtv90),
                              ('CTE 97 $', npvtv97),
                              ('CTE 98 $', npvtv98),
                              ('C3P2 $',vm21),
                              ('800RBC $', 4 * vm21),
                              ('Worst NPV CF $', worstone),
                              ('% claim', pct_claim),
                              ('Number died',tot_died),
                              ('Avg Year Death',avg_ydeath),
                              ('Number payees',tot_payees),
                              ('Avg. claim size $', avg_payout),
                              ('Exp. bft. at death $', avg_deathben),
                              #('Avg. year bust',int(yearbust)),
                              #('Avg. years bust',yearsbust),
                              ])
    #
    # STEP 7 write out the pickle files for producing the python notebooks
    #
    name = OUTPUTNAME + '.P'
    pickle.dump(report_data, open(OUTPUTDIR + 'table_' + name,'wb')) # this produces summary report
    pickle.dump(npvavg, open(OUTPUTDIR + 'npvavg_' + name,'wb')) # this produces NPV year by year
    pickle.dump(price100, open(OUTPUTDIR + 'price100df_' + name,'wb')) # this produces spaghetti plots
    pickle.dump(spectrumthisy, open(OUTPUTDIR + 'spectrumthisy_' + name,'wb')) # this is range of NPV for histo
    pickle.dump(introdf, open(OUTPUTDIR + 'parameters_' + name,'wb')) # this is range of NPV for histo
    pickle.dump(censusdf, open(OUTPUTDIR + 'census.P','wb')) # this is range of NPV for histo
    pickle.dump(peepdf, open(OUTPUTDIR + 'censusstats_' + name,'wb')) # this is range of NPV for histo
    # write out the time taken
    tt = int(time.time()) - start_time
    fin_text = OUTPUTNAME + ' done, ' + str(tt) + ' seconds, ' + str(round(tt / RUNS,2)) + ' seconds per run.'
    outfil(fin_text)
