import pandas as pd
import numpy as np
from const import Const as K
import cProfile
import pstats

uttable = {}

def _utilizedty(qualified,startage,age):
    # first if age > 95 return 0, nothing happens post 95
    # this is V SLOW - would be worth speeding up...
    key = str(qualified) + str(startage) + str(age)
    if key in uttable: 
        return uttable[key]
    else:
        if age > 95: 
            uttable[key ] = 0.0
            return 0.0
        if qualified:
            table = K.QUALUTILIZED
        else:
            table = K.UNQUALUTILIZED
        ny = age - 65
        psb4 = table.loc[startage][:ny].fillna(0)
        pleft = 1 - sum(psb4)
        pthisy = table.loc[startage][age]
        papplic = pthisy / pleft
        if np.isnan(papplic):
            papplic = 0.0
        uttable[key] = papplic
        return papplic 

vutilize = np.vectorize(_utilizedty, otypes=[np.float])

def _lapsedty(year, runin, startage, age, fundval, startval):
    if K.ASSUMP_LAP == 'OLD':
        if runin:
            if fundval / startval > K.LAPSERATETHRESHOLD:
                lapserate = K.INCOMELAPSERATE
            elif fundval == 0:
                lapserate = 0.0
            else:
                lapserate = K.LAPSERATESUBTHRESHOLD
        else:
            lapserate = K.RUNINLAPSERATE 
    else:
        multiplier = 1.0
        if not runin: # then not in runin
            multiplier = .5
        nyears = min([age - startage + 1]) # starts in year 1
        baselapse = K.LAPSETABLE[nyears][startage] * multiplier
        dynlapse = min([(fd.fundhist[year] / fd.startvalue) ** 3,K.LAPSECAP])
        lapserate = baselapse * dynlapse
    return lapserate

vlapse = np.vectorize(_lapsedty, otypes=[np.float])

def _lookupmort(key, age):
    if age > 120: age = 120
    return K.pmort[key].loc[age] 

vmort = np.vectorize(_lookupmort, otypes=[np.float])

def morty(sex,age):
    if sex == 'F': 
        table = K.MORT_Fq
        improve = 1 - K.MORT_FG2
    else:
        table = K.MORT_Mq
        improve = 1 - K.MORT_MG2
    table['n'] = [max([0,i - age + 8]) for i in table.index] # power to raise the G2 to
    return table['q'] * np.power(improve['G2'],table['n']) * K.MORT_F['F']

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

def setuppeep(census,inconame):
    peep = pd.DataFrame()
    for group in census.columns:
        grp = census[group]
        sage = int(grp.StartAge)
        names = [inconame+group+grp.Sex+str(sage)+'p'+str(i) for i in range(grp.N)]
        df = pd.DataFrame(index=names)
        df['startage'] = sage
        df['age'] = sage
        df['sex'] = grp.Sex
        df['startval'] = float(grp.FundSize) 
        key = grp.Sex + str(sage)
        df['key'] = key
        if K.ASSUMP_MOR == 'OLD':
            K.pmort[key] = mort(K.MORTALITYTABLEMALE, 
                         K.MORTALITYTABLEFEMALE,
                         grp.Sex, 
                         sage, 
                         K.TAPER, 
                         K.TAPERYEARS,
                         K.MORTALITYMULTIPLIER)
        else:
            K.pmort[key] = morty(grp.Sex,sage)
        peep = pd.concat([peep,df])
    for k in ['agedeath','agebust','agelapsed']:
        peep[k] = None
    for k in ['bust','finished','lapsed','died','runin','utilized','qualified']:
        peep[k] = False
    for k in ['income','randtest','vwdlapsed','vwddied']:
        peep[k] = 0.0
    return peep 

class Funds(object):
    def __init__(self,peep,market,years,inconame):
        self.inconame = inconame
        self.year = K.THISYEAR
        self.years = years
        self.p = peep
        self.fd = pd.DataFrame(self.p.startval) # fund
        self.fd.columns = ['val']
        self.pm =  pd.DataFrame(index=self.p.index) # premiums
        self.income =  pd.DataFrame(index=self.p.index) # income 
        self.payout =  pd.DataFrame(index=self.p.index)  
        self.wmfee =  pd.DataFrame(index=self.p.index) 
        self.expenses = pd.DataFrame(index=self.p.index) 
        self.market = market
        self.inflation = 1.0
        self.chkall()
        self.peepleft = pd.Series(dtype='float',index=self.years)
        self.diedthisy = pd.Series(dtype='float',index=self.years)
        self.lapsedthisy = pd.Series(dtype='float',index=self.years)
        self.bustthisy = pd.Series(dtype='float',index=self.years)
        self.wd_death = pd.Series(dtype='float',index=self.years)
        self.wd_lapse = pd.Series(dtype='float',index=self.years)
        self.peepleft = pd.Series(dtype='float',index=self.years)
        self.bustty = 0.0 # this is just an annual counter
        self.bustthisy = pd.Series(dtype='float',index=self.years) #this is the annual record
        
    def chkrunin(self):
        self.runin = (~self.finished) & (self.p['age'] > 64) & (self.year - K.THISYEAR > 1) # 2? think about when this executes

    def chkutiliz(self):
        self.utilized = (self.p['utilized'] == True)

    def chklapsed(self):
        self.lapsed = (self.p['lapsed'] == True)

    def chkdied(self):
        self.died = (self.p['died'] == True)

    def chkfinished(self):
        self.finished = self.lapsed | self.died

    def chkbust(self):
        self.bust = (self.p['bust'] == True) 

    def chkall(self):
        self.chkbust()
        self.chkdied()
        self.chkutiliz()
        self.chklapsed()
        self.chkfinished()
        self.chkrunin()

    def takepct(self,pct):
        self.amt = self.fd.loc[~self.finished & ~self.bust] * pct # only applies to these peep
        self.fd.loc[~self.finished] -= self.amt
        return self.amt

    def takeamt(self,selection,amount):
        initialbal = self.fd[selection].copy()
        bal = self.fd[selection]['val'] - amount 
        bust = (bal < 0)
        newbust = self.p[selection][bust]
        self.bustty += len(newbust)
        if len(newbust) > 0:
            self.p.loc[newbust,'agebust'] = self.p[newbust].startage + self.year - K.THISYEAR
            self.payout[self.year] = -bal[newbust]
            self.p.loc[newbust,'bust'] = True
            self.fd[newbust] = 0
        amtpaid = initialbal - self.fd[selection]
        return amtpaid

    def addnewrunins(self):
        oldrunins = self.runin
        self.chkrunin() # reset self.runin
        newrunins = ~oldrunins & self.runin
        test = pd.concat([self.p[newrunins].startval,self.fd[newrunins]],axis=1) # compare fundval and val
        income = test.max(axis=1) * K.INCOME
        self.p.loc[newrunins,'income'] = income

    def addnewutilizeds(self):
        mightutilize = ~self.finished & ~self.utilized & self.runin
        might_ut = self.p.loc[mightutilize]
        nmight_ut = len(might_ut)
        if nmight_ut > 0:
            self.p.loc[mightutilize,'randtest'] = np.random.random(nmight_ut)
            self.p.loc[mightutilize,'testval'] = vutilize(might_ut.qualified,might_ut.startage,might_ut.age)
            self.p.loc[mightutilize,'utilized'] = np.where(self.p[mightutilize].randtest < self.p[mightutilize].testval,True,False)

    def takeincome(self):
        self.addnewrunins()
        self.addnewutilizeds()
        #pay payouts if not bust
        payees = self.runin & ~self.bust & ~self.finished
        oldbust = self.bust
        self.income[self.year] = self.takeamt(payees,self.p[payees]['income'])
        #make payout if old bust
        busteds = oldbust & ~self.finished # make sure we avoid any newly bust created in takeamt
        self.payout[self.year] = 0.0 # default
        self.payout.loc[busteds,self.year] = self.p[busteds].income # overwrite defaults with busteds

    def takeexpenses(self):
        if K.ASSUMP_EXP == 'NEW':
            exp = 1.1 * (self.fd.val[~self.finished] * .0005 + 25 * self.inflation + .01 * self.payout[~self.finished][self.year])
        else:
            exp = self.fd.val[~self.finished] * .0007 + 50 * self.inflation
        self.expenses[self.year] = exp
        self.inflation *= K.INFLATIONRATE

    def moveon1yr(self):
        self.year += 1
        self.gr = 1 + self.market[self.year] # / self.market[self.year - 1] decide which way to do this SOON
        self.fd[~self.finished & ~self.bust] *= self.gr
        self.p.loc[~self.finished,'age'] += 1

    def diedty(self):
        self.p.loc[~self.finished,'randtest'] = np.random.random(len(self.p[~self.finished]))
        self.p.loc[~self.finished,'testval'] = vmort(self.p.key,self.p.age)
        self.p.loc[~self.finished,'died'] = np.where(self.p[~self.finished].randtest < self.p[~self.finished].testval,True,False)
        self.wd_death[self.year] = self.fd.loc[~self.finished][self.p.died == True].val.sum()
        self.p.loc[~self.finished,'agedeath'] = np.where(self.p[~self.finished].died == True,self.p[~self.finished].age,0)
        self.p.loc[~self.finished,'yeardied'] = np.where(self.p[~self.finished].died == True,self.year,0)

    def lapsedty(self):
        self.p.loc[~self.finished,'randtest'] = np.random.random(len(self.p[~self.finished]))
        self.p.loc[~self.finished,'testval'] = vlapse(self.year,self.p.runin,self.p.startage,self.p.age, self.fd.val,self.p.startval)
        self.p.loc[~self.finished,'lapsed'] = np.where(self.p[~self.finished].randtest < self.p[~self.finished].testval,True,False)
        self.wd_lapse[self.year] = self.fd.loc[~self.finished][self.p.lapsed == True].val.sum()
        self.p.loc[~self.finished],'agelapsed'] = np.where(self.p[~self.finished].lapsed == True,self.p.age,0)

    def keepcount(self):
        self.peepleft[self.year] = len(self.p[~finished]) 
        self.bustthisy[self.year] = self.bustty
        self.bustty = 0.0

    def grouptotals(self):
        self.peepdata = {}
        sexes = set(self.p.sex)
        sumpo = self.payout.sum(axis=1) # this just totals the payout for each person
        sumpo.name = 'payout'
        sample = pd.concat([self.p,sumpo],axis=1) # and this adds total payout column to people df
        for key in sorted(set(self.p.key)):
            sex = key['0']
            startage = int(key[1:])
            group = (self.p['startage'] == startage) & (self.p['sex'] == sex)
            selection = self.p.loc[group]
            premiumsel = self.pm.loc[group] # these the premiums
            premiumtot = premiumsel.sum().sum()
            premiumbyyr = premiumsel.sum()
            premiumnpv = npf.npv(DISCOUNT,premiumbyyr) / DISCO
            sm = selection.sum()
            ct = selection.count()
            busted = (selection.bust == True) # this enables age at death of the ones who bust
            try:
                maxdeath = max(selection.agedeath)
            except:
                maxdeath = 0
            busteds = selection[busted]
            bustedsum = busteds.sum()
            self.peepdata[sex+str(age)] = {'DthAgSm': sm.agedeath, 'DthCt': ct.agedeath , 
                    'BstAgSm': sm.agebust, 'BstCt': ct.agebust , 'MaxDth': maxdeath,
                             'LpsAgSm': sm.agelapsed, 'LpsCt': ct.agelapsed , 
                             'BstVlSm': sm.payout, 'BstDthAgSm': bustedsum.agedeath,
                             'Prem': premiumtot, 'NPVPrem': premiumnpv,
                             'NPeep': int(len(selection))
                             }
    def makesums(self):
        self.premiumssum = self.pm.sum()
        self.premiumssum.name = self.inconame
        self.expensessum = self.expenses.sum()
        self.expensessum.name = self.inconame
        self.payoutssum = self.payout.sum()
        self.payoutssum.name = self.inconame
        self.wmfeessum = self.wmfee.sum()
        self.wmfeessum.name = self.inconame
        self.cashflowsum = self.cashflow.sum()
        self.cashflowsum.name = self.inconame
        self.incomesum = self.income.sum()
        self.incomesum.name = self.inconame
        self.valuesum = self.fd.sum()
        self.valuesum.name = self.inconame
        self.wd_dsum = self.wd_death.sum()
        self.wd_dsum.name = self.inconame
        self.wd_lsum = self.wd_lapse.sum()
        self.wd_lsum.name = self.inconame
        self.cashflowplus = self.cashflowsum * K.HALF_YR # !!! NOTE look at HALF_YR
        self.npv = pd.Series([(npf.npv(K.DISCOUNT,self.cashflowplus[i:])) for i in range(len(self.cashflowplus))],index = self.years) / K.DISCO
        self.npv.name = self.inconame

    def claims(self):
        self.payout['total'] = self.payout.sum(axis=1) # sum each person not each year (this is done back in funk)
        pogt0 = self.po[po['total']>0] # select statement
        nbust = len(pogt0) # total number of bust
        totpo = po['t'].sum()
        result = pd.Series([totpo,nbust],index = ['totpo','nbust'])
        return result

    def benefs():
        thedead = self.p[self.p['died' == True] # just the ones with a date of death
        totdied = len(thedead)
        wddtotal = self.wd_death.sum(axis=1)
        wddbenef = wddtotal[wddtotal>0] # number of beneficiaries
        nbeneficiaries = len(wddbenef)
        totbenefit = wddbenef.sum()
        totyeardied = self.p.yeardied.sum()
        return pd.Series([nbeneficiaries,totbenefit,totdied,totyeardied],
                index = ['nbeneficiaries','totbenefit','totdied','totyeardied']) 

    def statistics(self):
        claimstats = self.claims()
        benefstats = self.benefs()
        stats = pd.concat([claimstats,benefstats])
        stats.name = inconame

    def writeqs(self):
        queues['Peep'].put(self.peepdata)
        queues['Premiums'].put(self.premiumssum)
        queues['Expenses'].put(self.expensessum)
        queues['Payouts'].put(self.payoutssum)
        queues['Stats'].put(self.stats)
        queues['WMFees'].put(self.wmfeessum)
        queues['Income'].put(self.incomesum)
        queues['CashFlow'].put(self.cashflowsum)
        queues['NPV'].put(self.npv)
        queues['Peepleft'].put(self.peepleft)
        queues['Diedthisy'].put(self.diedthisy)
        queues['Bustthisy'].put(self.bustthisy)
        queues['Lapsedthisy'].put(self.lapsedthisy)
        queues['Value'].put(self.valuesum)
        queues['WD_Death'].put(self.wd_dsum)
        queues['WD_Lapse'].put(self.wd_lsum)

    def debug(self):
        writer = pd.ExcelWriter(K.OUTPUTDIR + K.OUTPUTNAME + self.inconame + 'funkdebug' + '.xlsx')
        self.p.to_excel(writer,'People')
        self.pm.to_excel(writer,'Premiums')
        self.wmfee.to_excel(writer,'WMFees')
        self.expenses.to_excel(writer,'Expenses')
        self.income.to_excel(writer,'Income')
        self.payout.to_excel(writer,'Payouts')
        self.fd.to_excel(writer,'Value')
        self.wd_death.to_excel(writer,'WithdrawalsDeath')
        self.wd_lapse.to_excel(writer,'WithdrawalsLapse')
        self.cashflow.to_excel(writer,'Cashflow')
        pd.DataFrame(self.peepdata).to_excel(writer,'Peepdata')
        stats.to_excel(writer,'Stats')
        writer.save()


                             

def funk(market,census,inconame):
    inconame = inconame
    peep = setuppeep(census,inconame)
    youngest = min(peep.startage)
    years = range(K.THISYEAR,K.THISYEAR + K.MAXAGE - youngest)
    fds = Funds(peep,market,years,inconame)
    for y in years:
        print(y)
        fds.year = y
        fds.pm[y] = fds.takepct(K.PREMIUM)
        fds.wmfee[y] = fds.takepct(K.WMFEE)
        fds.takeincome()
        fds.takeexpenses()
        fds.diedty()
        fds.lapsedty()
        fds.addnewutilizeds()
        fds.keepcount() # do died thisy etc etc
        if y < years[-1]: fds.moveon1yr() # don't move on on last go, no more funds to pull down
    fds.cashflow = fds.premiums - fds.expenses - fds.payout
    fds.grouptotals()
    fds.makesums()
    fsd.writeqs()
    if K.DEBUG: self.debug() # write detailed output, not just for debugging. There is a lot!


if __name__ == '__main__':
    run = 123
    inconame = 'Run'+str(run)
    if run % 50 == 0:
        outfil(OUTPUTNAME + inconame + 'finished')
    fd = pd.read_csv(K.MARKET, header=None, index_col=None)
    fd.columns = range(K.THISYEAR,K.THISYEAR + len(fd.columns))
    market = fd.loc[run] # arbitray number for testing funk
    census = pd.read_json(K.CENSUS)
    funk(market,census,inconame)
    #profile = cProfile.Profile()
    #profile.runcall(funk,market,census,run)
    #ps = pstats.Stats(profile)
    #ps.print_stats()
    """
    funk(market,census,run)
    t = {'Terri':{'age':51,'years':0,'runin':False},
         'Phil':{'age':51,'years':30,'runin':False},
         'ANO':{'age':65,'years':3,'runin':False}}
    fd = {'Terri':{'val':100},
         'Phil':{'val':1000},
         'ANO':{'val':50}}
    tt = pd.DataFrame(t).T
    fd = pd.DataFrame(fd).T
    """
    print('done')


