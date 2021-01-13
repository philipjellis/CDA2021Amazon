import pandas as pd
import numpy as np
from collections import OrderedDict
import datetime

AMAZON = False
MAX = 121 # will give max age of 120 below
THISYEAR = datetime.datetime.now().year
TABLESTART = 2012
if AMAZON:
    MORTDIR = '/home/test/mort/'
else:
    MORTDIR = ''

def morty(sex,startage,fn='IAM20122581_2582.xlsx'):
    filename = MORTDIR + fn
    if sex.upper() == 'M':
        mtable = pd.read_excel(filename,1,index_col=0)
        g2 = pd.read_excel(filename,3,index_col=0)
    else:
        mtable = pd.read_excel(filename,0,index_col=0)
        g2 = pd.read_excel(filename,2,index_col=0)
    f = pd.read_excel(filename,4,index_col=0)
    df = pd.concat([mtable,g2,f],axis=1)
    df['n'] = THISYEAR - TABLESTART + df.index - startage # number of years for improvemvent scale
    qs = df['q'] * (1 - df['G2']) ** df['n'] * df['F'] # simple
    return qs[qs.index >= startage].round(6)
    
def morty2(key):
    sex = key[0]
    age = int(key[1:])
    return morty(sex,age)
