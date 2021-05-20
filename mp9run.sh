#!/bin/bash
cp /home/test/working/inputs.xlsx  /home/test/working/results/inputs.xlsx
cp /home/test/working/read5.py  /home/test/working/results/read5.py
cp /home/test/working/mp9master.py  /home/test/working/results/mp9master.py
cp /home/test/working/stox11.py  /home/test/working/results/stox11.py
cp /home/test/working/mp9master.py /home/test/working/mp9.py
sed -i '25s/^.*/OUTPUTNAME = "Standard2020"/' /home/test/working/mp9.py
sed -i '26s/^.*/NPEEP = 1000/' /home/test/working/mp9.py
sed -i '27s/^.*/RUNS = 1000/' /home/test/working/mp9.py
sed -i '28s/^.*/FUNDVAL = 100000000.0/' /home/test/working/mp9.py
sed -i '29s/^.*/FUNDFILE = "122020AAA6040.csv"/' /home/test/working/mp9.py
sed -i '30s/^.*/ANNCUM = "CUM"/' /home/test/working/mp9.py
sed -i '31s/^.*/INCOME = 0.05/' /home/test/working/mp9.py
sed -i '32s/^.*/WMFEE = 0.01/' /home/test/working/mp9.py
sed -i '33s/^.*/PREMIUM = 0.0055000000000000005/' /home/test/working/mp9.py
sed -i '34s/^.*/DISCOUNT = 0.03/' /home/test/working/mp9.py
sed -i '35s/^.*/CENSUSFILE = "census.json"/' /home/test/working/mp9.py
sed -i '36s/^.*/STOCHASTIC = False/' /home/test/working/mp9.py
sed -i '37s/^.*/DEBUG = False/' /home/test/working/mp9.py
sed -i '38s/^.*/PRUDENT = True/' /home/test/working/mp9.py
sed -i '39s/^.*/MORTSPREADSHEET = "IAM20122581_2582.xlsx"/' /home/test/working/mp9.py
sed -i '40s/^.*/YEARONEINCOME = 1.0/' /home/test/working/mp9.py
sed -i '41s/^.*/LAPSEUTILIZATION = "Utilization2021_1_26.xlsx"/' /home/test/working/mp9.py
sed -i '42s/^.*/# empty line/' /home/test/working/mp9.py
sed -i '43s/^.*/# empty line/' /home/test/working/mp9.py
sed -i '44s/^.*/# empty line/' /home/test/working/mp9.py
sed -i '45s/^.*/# empty line/' /home/test/working/mp9.py
sed -i '46s/^.*/# empty line/' /home/test/working/mp9.py
sed -i '47s/^.*/# empty line/' /home/test/working/mp9.py
sed -i '48s/^.*/# empty line/' /home/test/working/mp9.py
sed -i '49s/^.*/# empty line/' /home/test/working/mp9.py
/home/ubuntu/anaconda3/bin/python /home/test/working/mp9.py
cp /home/test/working/mp9master.py /home/test/working/mp9.py
sed -i '25s/^.*/OUTPUTNAME = "Standard2019"/' /home/test/working/mp9.py
sed -i '26s/^.*/NPEEP = 1000/' /home/test/working/mp9.py
sed -i '27s/^.*/RUNS = 1000/' /home/test/working/mp9.py
sed -i '28s/^.*/FUNDVAL = 100000000.0/' /home/test/working/mp9.py
sed -i '29s/^.*/FUNDFILE = "122019AAA6040.csv"/' /home/test/working/mp9.py
sed -i '30s/^.*/ANNCUM = "CUM"/' /home/test/working/mp9.py
sed -i '31s/^.*/INCOME = 0.05/' /home/test/working/mp9.py
sed -i '32s/^.*/WMFEE = 0.01/' /home/test/working/mp9.py
sed -i '33s/^.*/PREMIUM = 0.0055000000000000005/' /home/test/working/mp9.py
sed -i '34s/^.*/DISCOUNT = 0.03/' /home/test/working/mp9.py
sed -i '35s/^.*/CENSUSFILE = "census.json"/' /home/test/working/mp9.py
sed -i '36s/^.*/STOCHASTIC = False/' /home/test/working/mp9.py
sed -i '37s/^.*/DEBUG = False/' /home/test/working/mp9.py
sed -i '38s/^.*/PRUDENT = True/' /home/test/working/mp9.py
sed -i '39s/^.*/MORTSPREADSHEET = "IAM20122581_2582.xlsx"/' /home/test/working/mp9.py
sed -i '40s/^.*/YEARONEINCOME = 1.0/' /home/test/working/mp9.py
sed -i '41s/^.*/LAPSEUTILIZATION = "Utilization2021_1_26.xlsx"/' /home/test/working/mp9.py
sed -i '42s/^.*/# empty line/' /home/test/working/mp9.py
sed -i '43s/^.*/# empty line/' /home/test/working/mp9.py
sed -i '44s/^.*/# empty line/' /home/test/working/mp9.py
sed -i '45s/^.*/# empty line/' /home/test/working/mp9.py
sed -i '46s/^.*/# empty line/' /home/test/working/mp9.py
sed -i '47s/^.*/# empty line/' /home/test/working/mp9.py
sed -i '48s/^.*/# empty line/' /home/test/working/mp9.py
sed -i '49s/^.*/# empty line/' /home/test/working/mp9.py
/home/ubuntu/anaconda3/bin/python /home/test/working/mp9.py
cp /home/test/working/mp9master.py /home/test/working/mp9.py
sed -i '25s/^.*/OUTPUTNAME = "Standard2019_a"/' /home/test/working/mp9.py
sed -i '26s/^.*/NPEEP = 1000/' /home/test/working/mp9.py
sed -i '27s/^.*/RUNS = 1000/' /home/test/working/mp9.py
sed -i '28s/^.*/FUNDVAL = 100000000.0/' /home/test/working/mp9.py
sed -i '29s/^.*/FUNDFILE = "122019AAA6040_A.csv"/' /home/test/working/mp9.py
sed -i '30s/^.*/ANNCUM = "CUM"/' /home/test/working/mp9.py
sed -i '31s/^.*/INCOME = 0.05/' /home/test/working/mp9.py
sed -i '32s/^.*/WMFEE = 0.01/' /home/test/working/mp9.py
sed -i '33s/^.*/PREMIUM = 0.0055000000000000005/' /home/test/working/mp9.py
sed -i '34s/^.*/DISCOUNT = 0.03/' /home/test/working/mp9.py
sed -i '35s/^.*/CENSUSFILE = "census.json"/' /home/test/working/mp9.py
sed -i '36s/^.*/STOCHASTIC = False/' /home/test/working/mp9.py
sed -i '37s/^.*/DEBUG = False/' /home/test/working/mp9.py
sed -i '38s/^.*/PRUDENT = True/' /home/test/working/mp9.py
sed -i '39s/^.*/MORTSPREADSHEET = "IAM20122581_2582.xlsx"/' /home/test/working/mp9.py
sed -i '40s/^.*/YEARONEINCOME = 1.0/' /home/test/working/mp9.py
sed -i '41s/^.*/LAPSEUTILIZATION = "Utilization2021_1_26.xlsx"/' /home/test/working/mp9.py
sed -i '42s/^.*/# empty line/' /home/test/working/mp9.py
sed -i '43s/^.*/# empty line/' /home/test/working/mp9.py
sed -i '44s/^.*/# empty line/' /home/test/working/mp9.py
sed -i '45s/^.*/# empty line/' /home/test/working/mp9.py
sed -i '46s/^.*/# empty line/' /home/test/working/mp9.py
sed -i '47s/^.*/# empty line/' /home/test/working/mp9.py
sed -i '48s/^.*/# empty line/' /home/test/working/mp9.py
sed -i '49s/^.*/# empty line/' /home/test/working/mp9.py
/home/ubuntu/anaconda3/bin/python /home/test/working/mp9.py
cp /home/test/working/mp9master.py /home/test/working/mp9.py
sed -i '25s/^.*/OUTPUTNAME = "Standard2021Mar"/' /home/test/working/mp9.py
sed -i '26s/^.*/NPEEP = 1000/' /home/test/working/mp9.py
sed -i '27s/^.*/RUNS = 1000/' /home/test/working/mp9.py
sed -i '28s/^.*/FUNDVAL = 100000000.0/' /home/test/working/mp9.py
sed -i '29s/^.*/FUNDFILE = "032021AAA6040.csv"/' /home/test/working/mp9.py
sed -i '30s/^.*/ANNCUM = "CUM"/' /home/test/working/mp9.py
sed -i '31s/^.*/INCOME = 0.05/' /home/test/working/mp9.py
sed -i '32s/^.*/WMFEE = 0.01/' /home/test/working/mp9.py
sed -i '33s/^.*/PREMIUM = 0.0055000000000000005/' /home/test/working/mp9.py
sed -i '34s/^.*/DISCOUNT = 0.03/' /home/test/working/mp9.py
sed -i '35s/^.*/CENSUSFILE = "census.json"/' /home/test/working/mp9.py
sed -i '36s/^.*/STOCHASTIC = False/' /home/test/working/mp9.py
sed -i '37s/^.*/DEBUG = False/' /home/test/working/mp9.py
sed -i '38s/^.*/PRUDENT = True/' /home/test/working/mp9.py
sed -i '39s/^.*/MORTSPREADSHEET = "IAM20122581_2582.xlsx"/' /home/test/working/mp9.py
sed -i '40s/^.*/YEARONEINCOME = 1.0/' /home/test/working/mp9.py
sed -i '41s/^.*/LAPSEUTILIZATION = "Utilization2021_1_26.xlsx"/' /home/test/working/mp9.py
sed -i '42s/^.*/# empty line/' /home/test/working/mp9.py
sed -i '43s/^.*/# empty line/' /home/test/working/mp9.py
sed -i '44s/^.*/# empty line/' /home/test/working/mp9.py
sed -i '45s/^.*/# empty line/' /home/test/working/mp9.py
sed -i '46s/^.*/# empty line/' /home/test/working/mp9.py
sed -i '47s/^.*/# empty line/' /home/test/working/mp9.py
sed -i '48s/^.*/# empty line/' /home/test/working/mp9.py
sed -i '49s/^.*/# empty line/' /home/test/working/mp9.py
/home/ubuntu/anaconda3/bin/python /home/test/working/mp9.py
echo "FINISHED" >> /home/test/working/results/output.txt