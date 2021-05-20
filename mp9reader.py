import pandas as pd
"""
This no longer reads a spreadsheet with input parameters and census / cohort data
It reads two JSON files sent from the desktop program simple.py, which reads the spreadsheet.
This was done so that we can check out the JSON files once they are here.  You cannot read the spreadsheets 
easily once they are up on the Linux box.
"""
def checknumber(num, limit, name): #num is input numeric string, limit is max value, name is name for error msg
    msg = ''
    try:
        if num > limit:
            msg += '{0} is {1}, I expect it to be less than {2).  Speak to Phil if this what you want?\n'.format(name,num,limit)
    except:
        msg += '{0} is not numeric. I got {1}\n'.format(name,num)
    return msg

def errr(column):
    msg = ""
    msg += checknumber(column.runs, 5000, 'Runs')
    msg += checknumber(column.wmfee, 0.1, 'Wealth management fee')
    msg += checknumber(column.income, 0.1, 'Income percent')
    msg += checknumber(column.discountrate, 0.1, 'Discount rate')
    return msg

#def appendchange(
scenarios = pd.read_json('/home/test/working/results/scenario.json')
#print (scenarios)
scenarios.index = [i.replace(' ','').lower() for i in scenarios.index]
census = pd.read_json('/home/test/working/results/census.json')
census.index = [i.replace(' ','').lower() for i in census.index]
cst = census.T
population = cst.n.sum()
fundsize = (cst.n * cst.fundsize).sum()
writer = pd.ExcelWriter('/home/test/working/inputs.xlsx')
scenarios.to_excel(writer,'Scenarios')
census.to_excel(writer,'Census')
writer.save()
outf = open('/home/test/working/mp9run.sh','w')
outf.write('#!/bin/bash\n')
outf.write('cp /home/test/working/inputs.xlsx  /home/test/working/results/inputs.xlsx\n')
outf.write('cp /home/test/working/read5.py  /home/test/working/results/read5.py\n')
outf.write('cp /home/test/working/mp9master.py  /home/test/working/results/mp9master.py\n')
outf.write('cp /home/test/working/stox11.py  /home/test/working/results/stox11.py\n')
totalerror = ''
for col,col_data in scenarios.items():
    print ('Processing ',col)
    outf.write('cp /home/test/working/mp9master.py /home/test/working/mp9.py\n')
    changes = []
    errors = errr(col_data)
    # PJE might want to switch this back to work both ways - everything forced to be cumulative now
    anncum = 'CUM'
    if errors == '': # then no errors, get going
        changes.append('OUTPUTNAME = "' + col + '"')
        changes.append('NPEEP = ' + str(population)) 
        changes.append('RUNS = ' + str(col_data.runs))
        changes.append('FUNDVAL = ' + str(fundsize))
        changes.append('FUNDFILE = "' + col_data.csvfile[:col_data.csvfile.find('.')].upper() + '.csv"')
        changes.append('ANNCUM = "' + anncum + '"')
        changes.append('INCOME = ' + str(col_data.income))
        changes.append('WMFEE = ' + str(col_data.wmfee))
        changes.append('PREMIUM = ' + str(col_data.premium))
        changes.append('DISCOUNT = ' + str(col_data.discountrate))
        census = col_data.censusfile
        if pd.isna(census):
            censusjson = 'census.json'
        else:
            censusjson = census.split('.')[0] +'.json' # remove .xlsx, which is the census ss. We use json file
        changes.append('CENSUSFILE = "' + censusjson + '"')
        changes.append('STOCHASTIC = ' + str(col_data.stochastic == 1))
        changes.append('DEBUG = ' + str(col_data.debug == 1))
        changes.append('PRUDENT = ' + str(col_data.prudent == 1))
        changes.append('MORTSPREADSHEET = "' + str(col_data.mortspreadsheet) + '"')
        changes.append('YEARONEINCOME = ' + str(col_data.yearoneincome))
        changes.append('LAPSEUTILIZATION = "' + str(col_data.lapseutilization) + '"')
        lineno = 25 # changes all happen to lines following line 30
        for change in changes: # this fills lines 30 to 60 up with these parameter settings
            outf.write("""sed -i '""" + str(lineno) + """s/^.*/""" + change + """/' /home/test/working/mp9.py\n""")
            lineno += 1
        while lineno < 50: # if we haven't filled them up then just fill up the remainder with comments
            outf.write("""sed -i '""" + str(lineno) + """s/^.*/# empty line/' /home/test/working/mp9.py\n""")
            lineno += 1
        outf.write('/home/ubuntu/anaconda3/bin/python /home/test/working/mp9.py\n')
    else:
        print('error!!',errors)
        totalerror += errors
if totalerror != '': # then errors.  Write them to a file and make the script read them
    errorfile = open('/home/test/working/output.txt','w')
    errorfile.write(totalerror)
    errorfile.close()
outf.write('echo "FINISHED" >> /home/test/working/results/output.txt')
outf.close()
print ('Done')
