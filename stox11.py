import pandas as pd
import datetime
import numpy as np
import math
#from qs import mortdick
from random import random, gauss, seed
from collections import OrderedDict, deque, namedtuple
import copy
import logging
import csv

THISYEAR = datetime.datetime.now().year
DEBUG = False 

def gau():
    return gauss(0,1)

def taper(mt, age, taper, taperyears):
    mt = mt.copy()
    anndelt = float(taper) / taperyears # annual amount to creep back to 1 over taper period
    t = taper
    for y in range(age, int(age+taperyears)):
        yearoftaper = y - age
        mt[y] = round(mt[y] * ((1 - t) + anndelt * yearoftaper),6)
    return mt 

def mort(maletable, femaletable, sex, age, taperpct = 1, taperyears = 0, multiplier = 1):
    if sex.upper() == 'M':
        mt = mortdick[maletable]
    else:
        mt = mortdick[femaletable]
    if taperyears > 0:
        mt = taper(mt, age, taperpct, taperyears)
    mt *= multiplier
    return mt

def outxl(writer, tabname, df, extracol):
    ix,name,svals = extracol # index, name and startingvals for extra column
    df.insert(ix,name,svals)
    df.to_excel(writer,tabname)

class Fund(object):
    def __init__(self, inconame, market, years, params, assump, YEARONEINCOME):
        self.yearoneincome = YEARONEINCOME # inherited from set up spreadsheet, whether you get all income in y1 or 50%
        self.assump = assump
        self.years = years
        self.inco = inconame
        self.group = params.group 
        self.bust = False
        self.mkt = market
        self.startage = params.startage
        self.age = params.startage
        self.units = 0.0
        self.prices = self.mkt.prices[THISYEAR]
        self.startvalue = params.fundsize # starting balance 
        self.rebalance(self.startvalue)
        self.premiums ={} 
        self.bustyear ={} 
        self.wmfee = {}
        self.incomes = {}
        self.rawhist = OrderedDict([('group',self.group),(THISYEAR,self.startvalue)])
        self.fundhist = OrderedDict([('group',self.group),(THISYEAR,self.startvalue)])
        self.eoyhist = OrderedDict([('group',self.group),(THISYEAR,self.startvalue)])
        self.expenses = {}
        self.payout = {}
        self.ann_income = 0.0 
        self.runin = 0
        self.wmpct = params.wmpct
        self.premiumpct = params.premiumpct
        self.incomepct = params.incomepct
        self.infrate= 1.02
        self.inflation = 1
        self.runningin = True
        self.runinextra = params.runinextra #used in runin_done
        self.runincount = 0                 # "
        for y in self.years:
            #print ('year',y)
            if self.bust:
                self.premiums[y] = 0
                self.bustyear[y] = True
                self.wmfee[y] = 0
                self.incomes[y] = 0
                self.payout[y] = self.ann_income
                self.expenses[y] = 1.1 * (25 * self.inflation + .01 * self.payout[y])
            else:
                self.bustyear[y] = False # may get overwritten if we go bust this year
                self.payout[y] = 0 # may get overwritten below if we go bust this year
                self.premiums[y] = self.takeout(self.mtm() * self.premiumpct)
                self.wmfee[y] = self.takeout(self.mtm() * self.wmpct)
                if not self.runningin: # then we are good to pay income
                    inc_paid = self.takeout(self.ann_income)
                    self.incomes[y] = inc_paid
                    if inc_paid != self.ann_income: # there was a shortfall
                        self.payout[y] = self.ann_income - inc_paid # payout the shortfall
                        self.bustyear[y] = True # and declare we are bust
                else:
                    self.incomes[y] = 0
                    self.ann_income = self.mtm() * self.incomepct
                self.expenses[y] = 1.1 * (self.mtm() * .0005 + 25 * self.inflation + .01 * self.payout[y])
            if y != self.years[-1]: self.moveon1yr(y) # don't move on the last year because  no more prices
        self.eoyhist[y] = self.mtm()
        if DEBUG: self.generatehist()

    def generatehist(self):
        ix = 'Price FundHist Premium Expenses Payout Income Bust WMFee'.split()
        columns = []
        for y in self.years:
            col = pd.Series([self.mkt.prices[y],
                            self.fundhist[y],
                            self.premiums[y],
                            self.expenses[y],
                            self.payout[y],
                            self.incomes[y],
                            self.bustyear[y],
                            self.wmfee[y]])
            col.name = y
            col.index = ix
            columns.append(col)
        sheet = pd.concat([col for col in columns],axis=1)
        sheet.to_excel('/home/working/results/' + self.mkt.fund + self.inco + self.group + '.xlsx')

    def mtm(self):
        return self.units * self.prices

    def takeout(self, amt):
        if self.bust: 
            return 0
        mtm = self.mtm()
        pctout = amt / mtm
        if pctout > 1.0: # then BUST!
            self.bust = True
            self.units = 0.0
            return mtm # return total amount taken - insurance may cover rest
        else:
            self.units *= (1 - pctout)
            return float(amt)

    def summary(self):
        return pd.DataFrame(self.fundhist,index=[self.inco])

    def moveon1yr(self,y):
        if self.runningin:
            if (self.age < 65) or (self.runin < 2): # then we are still running in
                self.ann_income = self.mtm() * self.incomepct 
            else: # we have finished running in
                self.ann_income = max([self.ann_income,self.startvalue * self.incomepct])
                self.runningin = False
                yr_1_inc = self.ann_income * self.yearoneincome
                inc_paid = self.takeout(yr_1_inc) # half year's benefit in first year.
                self.incomes[y] = inc_paid
                self.expenses[y] = 1.1 * (self.mtm() * .0005 + 25 * self.inflation + .01 * self.payout[y])
                if inc_paid != yr_1_inc: # there was a shortfall
                    1/0 # SURELY this should never get executed.
        self.eoyhist[y] = self.mtm() 
        self.prices = self.mkt.prices[y + 1] # have bad feeling about this!
        value_eoy = self.mtm()
        self.rebalance(value_eoy)
        self.fundhist[y+1] = value_eoy
        self.inflation *= self.infrate
        self.age += 1
        self.runin += 1
        """
        print ('income')
        print(self.incomes)
        print('expenses')
        print (self.expenses)
        input('press a key')
        """

    def rebalance(self,mtmval): # this is either called with mtm() or the init value
        self.units = mtmval / self.prices

class StockMarket(object):
    """ This is really now just a placeholder for the prices queue to put
        a new set of prices.  Throwback to the old lognormal approach"""
    def __init__(self, inconame, pregen, fund):
        self.inco = inconame
        pregen1 = pregen
        pregen2 = [1.0] + list(pregen1)
        pregen3 = pd.Series(pregen2)
        pregen3.index = range(THISYEAR, THISYEAR + len(pregen3))
        self.prices = pd.Series([pregen3.loc[:i].product() for i in pregen3.index]) # (re)generate price series
        self.prices.index = pregen3.index
        #self.prices = pregen
        self.fund = fund
        
def loadcsvqueue(queue, queue100, fn, anncum):
    #
    # This works with the fund simulations.  One col for each year. 
    #  Set the annret Flag to True if you are receiving annual returns
    #  Set the annret Flag to False if you are receiving annual fund values
    # 
    # queue  contains everything
    # queue100 just contains first 100 rows, for printing spaghetti charts etc
    # queue is just the annual returns, for output to audit spreadsheet
    df = pd.read_csv(fn,header=None)
    if anncum == 'ANN': #then annuals
        anns = df + 1
    else: # then 'CUM', cumulative file
        num = df[df.columns[1:]]
        num.columns = range(len(num.columns))
        denom = df[df.columns[:-1]]
        anns = num / denom
    anns.columns = range(THISYEAR,THISYEAR + len(anns.columns))
    for i in range(len(anns)): queue.put(anns.iloc[i])
    for i in range(100): 
        if i < len(anns):
            row = anns.iloc[i]
        else:
            row = anns.iloc[len(anns)-1] # repeat last item if v short - ie debugging
        npr = np.array([np.prod(row[0:i]) for i in range(len(row))])
        ds = pd.Series(npr)
        ds.index = range(THISYEAR, THISYEAR + len(ds))
        queue100.put(ds)
    return # no need to return the queues, they are full now


