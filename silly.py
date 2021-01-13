import pandas as pd
import numpy as np
t = {'Terri':{'startage': 60,'age':61,'years':0,'runin':False},
        'Phil':{'startage':60,'age':62,'years':30,'runin':False},
        'ANO':{'startage':60,'age':75,'years':3,'runin':False}}
fd = {'Terri':{'val':100},
     'Phil':{'val':1000},
     'ANO':{'val':50}}
tt = pd.DataFrame(t).T
fd = pd.DataFrame(fd).T

