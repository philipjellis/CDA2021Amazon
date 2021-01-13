import wx
import os
import pandas as pd
import numpy as np
import boto3
import botocore
import paramiko
import threading
import time
import shutil
from openpyxl import load_workbook

wildcard = "Excel sheets (*.xlsx)|*.xlsx|"     \
           "All files (*.*)|*.*"
IPADDRESS = "ec2-3-22-226-74.us-east-2.compute.amazonaws.com"
HOMEDIR = '/home/test/working/'
RESULTSDIR = HOMEDIR + 'results/'

ROLLFORWARD = 'c:/LimeTreeDox/ARM/MeritInsurance/MC/working/DummyOutputSS.xlsx'
key = paramiko.RSAKey.from_private_key_file('c:/LimeTreeDox/ARM/PJEAmazonJupyter.pem')
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

AppBaseClass = wx.App


class Monty(wx.Frame):
    def __init__(self, parent, title):
        wx.Frame.__init__(self, parent, -1, title)

        # Create the menubar and menu
        menuBar = wx.MenuBar()
        menu = wx.Menu()
        
        #empty data
        self.fn, self.outdir = None, None
        self.status  = 'ok'

        # add Instruction to the menu
        menu.Append(101, "&Help\tAlt-H","This will show the instructions")
        menu.Append(wx.ID_EXIT, "E&xit\tAlt-X", "Exit")
        self.Bind(wx.EVT_MENU, self.Help, id=101)
        self.Bind(wx.EVT_MENU, self.Exit, id=wx.ID_EXIT)
        menuBar.Append(menu,"&Help")
        self.SetMenuBar(menuBar)

        self.CreateStatusBar()

        # Now create the Panel to put the other controls on.
        panel = wx.Panel(self)

        text = wx.StaticText(panel, -1, "Merit Insurance - Monte Carlo portfolio simulation")
        text.SetFont(wx.Font(14, wx.SWISS, wx.NORMAL, wx.BOLD))
        text.SetSize(text.GetBestSize())

       #Now the filename and output directory
        self.rftext = wx.StaticText(panel, -1, "RollForward Spreadsheet "+ROLLFORWARD)
        self.rftext.SetFont(wx.Font(8, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sstext = wx.StaticText(panel, -1, "Input Spreadsheet")
        self.sstext.SetFont(wx.Font(8, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.dirtext = wx.StaticText(panel, -1, "Output Directory")
        self.dirtext.SetFont(wx.Font(8, wx.SWISS, wx.NORMAL, wx.BOLD))
        # The buttons
        infil = wx.Button(panel, -1, "1. Choose the spreadsheet", (50,50))
        self.Bind(wx.EVT_BUTTON, self.choosefile, infil)
        outdir = wx.Button(panel, -1, "2. Choose the output directory", (50,50))
        self.Bind(wx.EVT_BUTTON, self.choosedirectory, outdir)
        process = wx.Button(panel, -1, "3. Process...")
        self.Bind(wx.EVT_BUTTON, self.Process, process)
        # Use a sizer to layout the controls, stacked vertically 10 pix border
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(text, 0, wx.ALL, 10)
        #sizer.Add(text2, 0, wx.ALL, 10)
        #sizer.Add(iptext, 0, wx.ALL, 10)
        sizer.Add(self.rftext, 0, wx.ALL, 10)
        sizer.Add(self.sstext, 0, wx.ALL, 10)
        sizer.Add(self.dirtext, 0, wx.ALL, 10)
        sizer.Add(infil, 0, wx.ALL, 10)
        sizer.Add(outdir, 0, wx.ALL, 10)
        sizer.Add(process, 0, wx.ALL, 10)
        panel.SetSizer(sizer)
        panel.Layout()

        # And also use a sizer to manage the size of the panel such
        # that it fills the frame
        self.SetStatusText('Remote process waiting')
        sizer = wx.BoxSizer()
        sizer.Add(panel, 1, wx.EXPAND)
        self.SetSizer(sizer)
        self.Fit()
        self.CenterOnScreen(wx.BOTH)


    def Exit(self, evt):
        """Event handler for the button click."""
        self.Close()

    def Help(self, evt):
        text = '''There are four steps...\n
First, set up the spreadsheet with the scenarios and census.  
\tYou can use the spreadsheet Standard.xlsx as a template
\tThen choose the file you have set up.\n
Second, choose an output directory.\n
Third, the Process button will check the spreadsheet,
\tsend it to Amazon for processing, and
\tsend any output back to this computer.\n
Fourth you can review the graphs and spreadsheet
and run the dashboard with the files in the Output directory.
        '''
        self.msg(text)

    def choosefile(self, evt):
        dlg = wx.FileDialog(
            self, message="Choose a file",
            defaultDir=os.getcwd(),
            defaultFile="",
            wildcard=wildcard,
            style=wx.FD_OPEN | 
                  wx.FD_CHANGE_DIR | wx.FD_FILE_MUST_EXIST |
                  wx.FD_PREVIEW
            )
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.fn = path
            self.sstext.SetLabel("Input Spreadsheet: " + self.fn)
        dlg.Destroy()
        self.SetStatusText('Input spreadsheet chosen.  Step 1 complete.')

    def sendfile(self):
        #try:
        client.connect(hostname=IPADDRESS, username="ubuntu", pkey=key)
        client.exec_command('rm /home/test/working/results/*')
        ftp_client=client.open_sftp()
        ftp_client.put(self.census, HOMEDIR + 'census.json')
        ftp_client.put(self.scenario, HOMEDIR + 'scenario.json')
        # get the output file names
        wb = pd.read_excel(self.fn,index_col=0)
        outputfns = [i + '.xlsx' for i in wb.columns] # col B and on contain the scenario names
        for fn in outputfns:
            ftp_client.put(ROLLFORWARD, RESULTSDIR + fn)
        ftp_client.close()
        client.close()
        shutil.copy(self.fn,'c:/LimeTreeDox/ARM/MeritInsurance/MC/working/results/Standard.xlsx')
        self.SetStatusText('Files sent')
        #except:
        #    self.msg('Failed at send file')

    def getfile(self):
        try:
            client.connect(hostname=IPADDRESS, username="ubuntu", pkey=key)
            ftp_client=client.open_sftp()
            fs = ftp_client.listdir(RESULTSDIR)
            for f in fs:
                ftp_client.get(RESULTSDIR + f, self.outdir + '/' + f)
            ftp_client.close()
            client.close()
            self.SetStatusText('Files retrieved')
        except:
            self.msg('Failed at get file')

    def do_spreadsheet(self):
        if self.fn and self.outdir:
            dfcs = pd.read_excel(self.fn,'Census',skiprows=1,index_col=0)
            dfcs2 = dfcs.loc[dfcs.index.drop('Total')] # assumes the total row is named 'Total'
            dfcs2.T.to_json(self.census)
            dfsc = pd.read_excel(self.fn,'Scenarios',index_col=0)
            dfsc2 = dfsc[:len(dfsc)-1] # skip the last row - just a comment in the ss
            dfsc2.to_json(self.scenario)
            self.SetStatusText('Spreadsheet processed')
        else:
            self.msg('You have not selected a spreadsheet and output directory yet')
            
    def msg(self, msg):
        dlg = wx.MessageDialog(self, msg,
                                     'Message',
                                      wx.OK | wx.ICON_INFORMATION
                                      #wx.YES_NO | wx.NO_DEFAULT | wx.CANCEL | wx.ICON_INFORMATION
                                      )
        dlg.ShowModal()
        dlg.Destroy()

    def choosedirectory(self,evt):
        dlg = wx.DirDialog(self, "Choose a directory:",
                          style=wx.DD_DEFAULT_STYLE
                           #| wx.DD_DIR_MUST_EXIST
                           #| wx.DD_CHANGE_DIR
                           )
        if dlg.ShowModal() == wx.ID_OK:
            self.outdir = dlg.GetPath()
            self.census = self.outdir + '/census.json'
            self.scenario = self.outdir + '/scenario.json'
            self.dirtext.SetLabel("Output Directory: " + self.outdir)
        dlg.Destroy()
        self.SetStatusText('Directory chosen,  Step 2 complete.')

    def worker(self):
        try:
            client.connect(hostname=IPADDRESS, username="ubuntu", pkey=key)
            stdin, stdout, stderr = client.exec_command('./mp9starter.sh')
            client.close()
            self.SetStatusText('Process started on Amazon')
        except :
            self.msg('Worker failed')

    def getupdate(self):
        client.connect(hostname=IPADDRESS, username="ubuntu", pkey=key)
        stdin, stdout, stderr = client.exec_command('tail ' + RESULTSDIR + 'output.txt')
        output = stdout.readlines()
        client.close()
        if len(output) > 0:
            lastline = output[-1]
        else:
            lastline = 'Running...'
        return lastline

    def checker(self):
        start = True
        while start:
            time.sleep(5)
            status = self.getupdate()
            self.SetStatusText(status)
            if status == 'FINISHED\n':
                start = False
        self.getfile()
        self.SetStatusText('Files retrieved and in output directory')

    def Process(self, evt):
        self.do_spreadsheet()
        self.sendfile()
        t = threading.Thread(target=self.worker)
        u = threading.Thread(target=self.checker)
        t.start()
        u.start()

class MyApp(AppBaseClass):
    def OnInit(self):
        frame = Monty(None, "Merit Insurance Monte Carlo Simulator")
        self.SetTopWindow(frame)
        frame.Show(True)
        return True

app = MyApp()
app.MainLoop()

