from queue import Queue
from collections import OrderedDict
import pickle
import glob
import pandas as pd
import seaborn as sns
from IPython.core.display import HTML
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
""" This reads through the results for the Multi Group scenario.
    it is similar to but shorter than the singleagesex version
"""
""" STEP ZERO here is where you might want to poke around and change stuff
    The rest you should be able to leave alone
"""
JUPYTER = True # set to true to get images displaying in Jupyter
SMALLFT = 18
MEDIUMFT = 24
BIGFT = 30
TESTFT = 18 # easy to change
FUNDSIZE = 100000000 # find a way to get this in from the MC run
sns.set(context="poster") # or talk,.. poster makes the font big and lines clear
sns.set_context("poster", font_scale=1, rc={"lines.linewidth":4,"axes.labelsize":TESTFT})
datadict =  {'table':{}, 'price100df':{}, 'npvavg':{}, 'censusstats':{}, 'parameters':{}, 'npvavg':{}, 'spectrumthisy':{}}

def findstr(l,s): # finds a substring in list and returns index or None if not found
    finds = [i for i in range(len(l)) if s in l[i]]
    if finds:
        return finds[0]
    return None

def fix_lines(ax): # a bit of a hack to sort out graph lines etc
    ax.axvline(5, color='grey', linewidth=2, linestyle='--')
    ax.axvline(10, color='grey', linewidth=2, linestyle='--')
    ax.axvline(15, color='grey', linewidth=2, linestyle='--')
    ax.axvline(20, color='grey', linewidth=2, linestyle='--')
    ax.axvline(25, color='grey', linewidth=2, linestyle='--')
    ax.axvline(30, color='grey', linewidth=2, linestyle='--')
    ax.axvline(35, color='grey', linewidth=2, linestyle='--')
    ax.axvline(40, color='grey', linewidth=2, linestyle='--')
    ax.axvline(45, color='grey', linewidth=2, linestyle='--')
    ax.axvline(50, color='grey', linewidth=2, linestyle='--')
    ax.text(5,10,'5%', fontsize=TESTFT) 
    ax.text(10,10,'10%', fontsize=TESTFT)
    ax.text(15,10,'15%', fontsize=TESTFT)
    ax.text(20,10,'20%', fontsize=TESTFT) 
    ax.text(25,10,'25%', fontsize=TESTFT)
    ax.text(30,10,'30%', fontsize=TESTFT) 
    ax.text(35,10,'35%', fontsize=TESTFT) 
    ax.text(40,10,'40%', fontsize=TESTFT)
    ax.text(45,10,'45%', fontsize=TESTFT)
    ax.text(50,10,'50%', fontsize=TESTFT) 
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    #ax.xaxis.set_minor_formatter(mtick.PercentFormatter())
    return ax

def formt(v,col,fstr):
    temp = pd.Series(float(i) for i in v[col])
    v[col] = pd.Series([fstr.format(i) for i in temp],index = v.index)
    return v

"""
STEP 1 read the pickle files
"""
census = pickle.load(open('census.P','rb'))
ct = census.T
ct['Fund size'] = pd.Series(["${0:,.0f}".format(val) for val in ct['Fund size']],index=ct.index)
census = census.T

display(HTML('<h1>Monte Carlo Results</h1>'))

display(HTML('<br><p>This output shows the results for the following</p>'))
try:
    df = pd.read_excel('standard.xlsx',index_col=0)
    display(HTML('<ol>'))
    for c in df.columns:
        display(HTML('<li>' + str(c) + ':  ' + df[c].Description + '</li>'))
    display(HTML('</ol>'))
except:
    pass
display(HTML('<h2>The cohorts comprising the census</h2>'))
if JUPYTER: display(census)

for typ in datadict.keys():
    for fil in glob.glob(typ + '*.p'):
        pieces = fil[:-2].split('_') # remove the .p the last two chars
        nm = pieces[1]
        if len(pieces) > 2:
            sex = pieces[2][0]
            age = pieces[2][1:3]
            lis = [nm,age,sex]
        else:
            lis = [nm]
        name = '_'.join(lis)
        datadict[typ][name] = pickle.load(open(fil,"rb"))
results = pd.DataFrame(datadict['table'])
resultst = results.T.iloc[:,:13]
format_dict = {}
for k in resultst.columns:
    if '%' in k:
        format_dict[k] = "{:.2f}%"
    elif '$' in k:
        format_dict[k] = "${:,.0f}"
    elif 'year' in k:
        format_dict[k] = "{:.0f}"
    else:
        format_dict[k] = "{:,.0f}"
resultst.to_excel('CohortRiskStatistics.xlsx')
"""
# STEP 3 - results table
"""
resultsdf = resultst.style.format(format_dict)
if JUPYTER: display(resultsdf)
"""
STEP 8 the parameters
"""
display(HTML('<h2>The parameter files</h2>'))
df = pd.concat(datadict['parameters'].values(),axis=1)
df.columns = datadict['parameters'].keys()
if JUPYTER: display(df)
"""
STEP 9 the census stats
"""
display(HTML('<h2>The census statistics from each run</h2>'))
for k,v in datadict['censusstats'].items():
    vt = v.T
    vt = formt(vt,'Total people in trial',"{:,.0f}")
    for col in vt.columns:
        if 'YEARS' in col.upper():
            vt = formt(vt,col,"{:.1f}")
        elif 'PAID' in col.upper():
            vt = formt(vt,col,"${:,.0f}")
        elif 'YEAR' in col.upper():
            vt = formt(vt,col,"{:.0f}")
        elif 'MEAN' in col.upper():
            vt = formt(vt,col,"{:,.1f}")
        elif 'PERCENT' in col.upper():
            vt = formt(vt,col,"{:.2f}%")
    display(HTML('<h3>' + k + '</h3>'))
    display(vt.T)

"""
# Step 4 NPV graph
"""
# try to get interest rate - this a bit flaky
params = datadict['parameters']
firstparam = params[list(params.keys())[0]]
cols = [i.upper() for i in firstparam.index]
discount_ix = findstr(cols,'DISCOUNT')
if discount_ix:
    intrate = firstparam['Parameter Values'].iloc[discount_ix] # PJE this is a bit fragile!
else:
    intrate = 0.0
df = pd.DataFrame(datadict['spectrumthisy'])
df = pd.concat([df[col].sort_values(ignore_index=True) for col in df],axis=1)
df = df * 100 / FUNDSIZE # gives as % of fundsize
df.index = [100 * i / len(df) for i in range(1,1+len(df))]
fig, ax = plt.subplots(figsize=(20,20))
fix_lines(ax)
sns.lineplot(data=df,dashes=False, palette="tab10", linewidth=3)
ax.grid(b=True, which='minor', color='w', linewidth=0.5)
ax.set_xticks([5,10,15,20,25,30,35,40,45,50,55,60],minor=True)
titstr = 'Ranked PV at ' + intrate +' of cash flows by Fund and Scenario'
plt.title(titstr)
plt.xlabel('Percent of runs')
plt.ylabel('% of initial protected assets')
plt.savefig(titstr.replace(' ','_') + '.png')
if JUPYTER: plt.show()
"""
# Step 5 NPV Cashflow
"""
npvks = datadict['npvavg'].keys()
npvdf = pd.concat([datadict['npvavg'][i] for i in npvks],axis=1) / 1000 #put results in $000
npvdf.columns = npvks
fig, ax = plt.subplots(figsize=(20,20))
sns.lineplot(data=npvdf,dashes=False, palette="tab10", linewidth=4.5)
titstr = 'Future NPV Cashflow of each fund'
plt.title(titstr)
plt.xlabel('Year')
plt.ylabel('NPV of future cashflows ($000)')
plt.savefig(titstr.replace(' ','_') + '.png')
if JUPYTER: plt.show()
"""
# Step 6 Range of cashflows
"""
for k,v in datadict['spectrumthisy'].items():
    fig, ax = plt.subplots(figsize=(20,20))
    sns.distplot(v/1000,kde=False,bins=40)
    titstr = 'Distribution of NPV Cashflows for '+k+' Fund ($000)'
    plt.title(titstr)
    plt.savefig(titstr.replace(' ','_') + '.png')
    if JUPYTER: plt.show()
"""
# STEP 7 the spaghetti graphs - we just use seaborn lineplot as you can show several sets at once with it
"""
df = pd.concat(datadict['price100df'].values(),axis=1)
df.columns = datadict['price100df'].keys()
fig, ax = plt.subplots(figsize=(20,20))
ax = sns.lineplot(data=df)
plt.title('Growth of simulated Portfolios')
plt.ylabel('Fund value ($), 2020 = 1')
plt.savefig(titstr.replace(' ','_') + '.png')
if JUPYTER: plt.show()
