__author__ = 'garta'

import pylab # for plot
import sys
import os
from importlib import import_module
libnames = ['ppf','getdat']
relative_imports = []


for libname in libnames:
    try:
        lib = import_module(libname)
    except:
        exc_type, exc, tb = sys.exc_info()
        print(os.path.realpath(__file__))
        print(exc)
    else:
        globals()[libname] = lib

for libname in relative_imports:
    try:
        anchor = libname.split('.')
        libr = anchor[0]
        package = anchor[1]

        lib = import_module(libr,package=package)
    except:
        exc_type, exc, tb = sys.exc_info()
        print(os.path.realpath(__file__))
        print(exc)
    else:
        globals()[libname] = lib

import numpy
import matplotlib.pyplot as plt
import signalsTableJET_MAG as STJET
import csv
import scipy
import pandas as pd
import os




class MAGTool:

    def __init__(self,JPN):
        self.JPN = JPN # JET Pulse Number
        self.description = 'Tools to work with magnetic signals'
        self.author = 'artaserse'

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    def downloadCheck(self,JPN,nameSignalsTable,signalsTable):
        # check status
        ierM = []
        signalsName = signalsTable.keys()
        signalsPath = signalsTable.values()
        # _____download signals
        for signals in signalsName:
           signalsPath = signalsTable[signals]
           if signalsPath[0:3]=='PPF': # PPF
               dda   = signalsPath[4:8]
               dtype = signalsPath[9:]
               ihdat,iwdat,data,x,t,ier = ppf.ppfget(JPN,dda,dtype,fix0=0,reshape=1,no_x=0,no_t=0)
               #ihdat,iwdat,data,x,t,ier = ppf.ppfget(JPN,dda,dtype)
               dataExp = data
               tExp = t
           elif signalsPath[0:2] == 'DA' or signalsPath[0:2] == 'PF': # JPF
               data,tvec,nwds,title,units,ier= getdat.getdat(signalsTable[signals],JPN)
               dataExp = data
               tExp = tvec
           if ier!=0:
               ierM.append(1)
           else:
               ierM.append(0)
        return ierM

    @staticmethod
    def downloadSingleChannel(JPN,signalPath):
        ier = 0
        #  download single channel
        data,tvec,nwds,title,units,ier= getdat.getdat(signalPath,JPN)
        if ier!=0:
            print(signalPath, 'Error to retreive it')
        else:
            print(signalPath,' Downloaded !!')

        return tvec,data,ier




    def download(self,JPN,nameSignalsTable,signalsTable,plotSignal):
        # ppf.ppfgo(JPN,262)
        # ppf.ppfuid('gszepesi','r')

        # return a Dictionary of dictionary with value 'v' and time 't' of the exp signal
        # example:
        # dictDataExp = {
        #    'CX01': { 'v':'......','t':'......'},
        #    'CX02': { 'v':'......','t':'......'},......
        fileDir = os.path.dirname('__file__')
        fileDir = os.path.dirname(os.path.realpath('__file__'))
        filename = os.path.join(fileDir,'logFile/logFileDownload_' + nameSignalsTable[13:] + '_' + str(JPN) + '.csv')
        with open(filename,'w+') as out_file:
            out_str = '#'+ str(JPN) + '\n'
            out_file.write(out_str)
            out_str = 'signalName,Outcome from download\n'
            out_file.write(out_str)

            signalsName = signalsTable.keys()
            signalsPath = signalsTable.values()
            # _____download signals
            dictDataExp = {}
            for signals in signalsName:
               out_str = '' + signals + ','
               #ier = 0
               #dataExp = 0
               #tExp = 0
               dictDataExpTMP = {}
               signalsPath = signalsTable[signals]
               #print(signalsPath[0:4] + '@@@')
               #print(signalsPath[0:2] + '@' )
               if signalsPath[0:3]=='PPF': # PPF

                   dda   = signalsPath[4:8]
                   dtype = signalsPath[9:]

                   if dda == 'EFIT':
                       ihdat,iwdat,data,x,t,ier = ppf.ppfget(int(JPN),dda,dtype,fix0=0,reshape=0,no_x=0,no_t=0)
                       # print(' NO RESHAPE of ' + signalsPath   + ' WARNING if is BPCA!!!!!!!!!!!')
                   else:
                       ihdat,iwdat,data,x,t,ier = ppf.ppfget(int(JPN),dda,dtype,fix0=0,reshape=1,no_x=0,no_t=0)
                       # print('RESHAPE of ' + signalsPath)


                   #print(dda)
                   #print(dtype)
                   #if dda == 'EFIT':
                       # needs to be reshaped
                   # ihdat,iwdat,data,x,t,ier = ppf.ppfget(JPN,dda,dtype,fix0=0,reshape=1,no_x=0,no_t=0)
                       # 1st BPCA Pick Up coils signal EFIT rec
                       # data[:,0]
                       #plt.figure,plt.plot(t,data[:,0]),plt.show()
                   #else:
                   #    ihdat,iwdat,data,x,t,ier = ppf.ppfget(JPN,dda,dtype)
                   # if not reshape data 2D are downloaded as single vector constructed like this
                   # a(0),b(0),c(0),...,First Sample of last column vector,
                   # than
                   # a(1),b(1),c(1),...,Second Sample of last column vector,
                   # ....
                   # a(nT),b(nT),c(nT),...,Last Sample of last column vector,
                   # ....
                   dataExp = data
                   tExp = t
                   dictDataExpTMP['x'] = x        # nr of columns dor a 2D array
               elif signalsPath[0:2] == 'DA' or signalsPath[0:2] == 'PF': # JPF
                   data,tvec,nwds,title,units,ier= getdat.getdat(signalsTable[signals],JPN)
                   dataExp = data
                   tExp = tvec
               if ier!=0:
                   print(signals, 'Error to retreive it')
                   out_str += '@@%^&$£ ~~~~ Error to retreive it'
                   out_str +='\n' # new Line
                   out_file.write(out_str)
               else:
                   dictDataExpTMP['v'] = dataExp        # signal value
                   dictDataExpTMP['t'] = tExp# signal time
                   # print(signals,' Downloaded !!')
                   out_str += 'Downloaded :)))'
                   out_str +='\n' # new Line
                   out_file.write(out_str)
                   if plotSignal:
                       pylab.figure()
                       pylab.plot(tExp,dataExp,label=signalsTable[signals])
                       pylab.xlabel('time[s]')
                       pylab.title(signals)
                       pylab.legend(loc='lower right')
                       pylab.show()
                   dictDataExp[signals] = dictDataExpTMP

            return dictDataExp

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    def TFcompensation(self,JPN,expDataDictRaw,signalTableNameRaw,dictDataExpTF,flagPlot):
        # TF compensation of raw signals using JPF TF coefficients
        # STFcomp = Sraw - TFECoeff*ITFEVN - TFOCoeff*ITFODD
        # JPN: (int) Jet pulse number
        # expDataDictRaw: (dict) dictionary with exp. data Raw
        # signalTableNameRaw: (dict) dictionary with name and path of signal raw
        # dictDataExpTF: (dict) dictionary with exp data (ITFEVN,ITFODD) to toroidally
        #                compensate the raw signals
        # flagPlot: (boolean) if 1 plot each raw signal aginst the TF compensated
        # extra output: logFileTFcoeff ASCII file with JPF TF coeff

         #________________________
        def downloadTFCoeff(JPN,signalsTable):

            signalsNameSorted = sorted(signalsTable) # become a LIST!!!!!!!!
            signalsName = signalsTable.keys()
            signalsPath = signalsTable.values()

            fileDir = os.path.dirname('__file__')
            fileDir = os.path.dirname(os.path.realpath('__file__'))
            filename = os.path.join(fileDir,'logFile/logFileTFcoeff_' + str(JPN) + '.csv')

            # write log file with TF coefficients
            with open(filename,'w') as out_file:
                out_str = '#'+ str(JPN) + '\n'
                out_file.write(out_str)
                out_str = 'signalName,TFE,TFO\n'
                out_file.write(out_str)

                # _____download signals
                dictDataExp = {}
                for signals in signalsNameSorted:
                      signalsPath = signalsTable[signals]

                      # download TF coeff
                      # TFE
                      TFEpath = signalsTable[signals] + '<TFE'
                      TFEname = signals + '_TFE'
                      dataTFE,nwds,title,units,ier = getdat.getsca(TFEpath,JPN)
                      dictDataExp[TFEname] =  dataTFE
                      print(TFEpath,' Downloaded !!')
                      # TFO
                      TFOpath = signalsTable[signals] + '<TFO'
                      TFOname = signals + '_TFO'
                      dataTFO,nwds,title,units,ier = getdat.getsca(TFOpath,JPN)
                      dictDataExp[TFOname] = dataTFO
                      print(TFOpath,' Downloaded !!')


                      out_str = '' + signals + ','
                      out_str += str(dataTFE)
                      out_str += ',' + str(dataTFO)
                      out_str +='\n' # new Line
                      out_file.write(out_str)
                      # ~~~~~~~~~~~~~~~~~~~ end write file
            out_file.close()
            return   dictDataExp

    #_____________________


        def TFCompensationSC(sRaw,TFECoeff,TFOCoeff,TFEVN,TFODD):
            # Toroidal field compensation of a raw signal ''Sraw''
            # using coefficients TFECoeff and TFOCoeff of the signal
            # and the even and odd TF current ITFEVN and ITFODD
            #
            # STFcomp = Sraw - TFECoeff*ITFEVN - TFOCoeff*ITFODD

            ITFEVN = TFEVN
            ITFODD = TFODD
            sTFcomp = sRaw - TFECoeff*ITFEVN - TFOCoeff*ITFODD
            return   sTFcomp

        dictTFcoeff = downloadTFCoeff(JPN,signalTableNameRaw)


        TFEVN = dictDataExpTF['TFEVN']['v']
        TFODD = dictDataExpTF['TFODD']['v']

        dictDataExpTFcomp = {}
        dictTMP ={}
        dictSC_KS_TMP = {}
        dictSC_KE_TMP = {}
        dictTFcompTMP = {}

        signalsNameRawSorted = sorted(signalTableNameRaw) # become a LIST!!!!!!!!
        for zz in signalsNameRawSorted:

            sRaw = expDataDictRaw[zz]['v']
            tRaw = expDataDictRaw[zz]['t']
            nameTFE = zz + '_TFE'
            TFECoeff =  dictTFcoeff[nameTFE]
            nameTFO = zz + '_TFO'
            TFOCoeff =  dictTFcoeff[nameTFO]

            print(nameTFE)
            print(nameTFO)
            v_STFcomp= TFCompensationSC(sRaw,TFECoeff,TFOCoeff,TFEVN,TFODD)


            # Shape Controller
            nameSC = zz[:1] + 'Z' + zz[2:]
            nameSC_KS = zz[:1] + 'Z' + zz[2:] + '_KS'
            nameSC_KE = zz[:1] + 'Z' + zz[2:] + '_KE'
            pathSC_KS = 'PF/SC-' + nameSC + '<KS' # KC1D
            print(pathSC_KS)
            pathSC_KE = 'PF/SC-' + nameSC + '<KE' # KC1E
            print(pathSC_KE)

            # SC from KC1D
            data,tvec,nwds,title,units,ier= getdat.getdat(pathSC_KS,JPN)
            s_SC_KS = data
            t_SC_KS = tvec
            # SC from KC1E
            data,tvec,nwds,title,units,ier= getdat.getdat(pathSC_KE,JPN)
            s_SC_KE = data
            t_SC_KE = tvec


            dictTMP['raw']   = expDataDictRaw[zz]
            dictTFcompTMP['v'] = v_STFcomp
            dictTFcompTMP['t'] = tRaw
            dictTMP['TFcomp']= dictTFcompTMP
            dictTMP['TFECoeff'] =  TFECoeff
            dictTMP['TFOCoeff'] =  TFOCoeff
            dictSC_KS_TMP['v'] = s_SC_KS
            dictSC_KS_TMP['t'] = t_SC_KS
            dictSC_KE_TMP['v'] = s_SC_KE
            dictSC_KE_TMP['t'] = t_SC_KE
            dictTMP['SC_KS']= dictSC_KS_TMP
            dictTMP['SC_KE']= dictSC_KE_TMP
            dictDataExpTFcomp[zz] = dictTMP
            dictTMP = {}
            dictSC_KS_TMP = {}
            dictSC_KE_TMP = {}
            dictTFcompTMP = {}


            if flagPlot:
               pylab.figure()
               pylab.plot(tRaw,sRaw,'b',       label=zz + ' RAW')
               pylab.plot(tRaw,v_STFcomp,'r',   label=zz + ' TF comp')
               pylab.plot(t_SC_KS,s_SC_KS,'m',label=nameSC_KS)
               pylab.plot(t_SC_KE,s_SC_KE,'k',label=nameSC_KE)
               pylab.xlabel('time[s]')
               pylab.legend(loc='lower right')
               pylab.show()


        return dictDataExpTFcomp



    @staticmethod
    def TFcompSingleChannel(JPN,signalPath,DAQ):
        # TF compensation of raw signals using JPF TF coefficients
        # STFcomp = Sraw - TFECoeff*ITFEVN - TFOCoeff*ITFODD

        tRaw,sRaw,ier = MAGTool.downloadSingleChannel(JPN,signalPath)
        strTmp = signalPath.split('-')
        signal = strTmp[1]


        dictTFcoeff = downloadTFCoeff(JPN,signalPath)
        nameTFE = signal + '_TFE'
        TFECoeff =  dictTFcoeff[nameTFE]
        nameTFO = signal + '_TFO'
        TFOCoeff =  dictTFcoeff[nameTFO]

        if 'KC1D' in DAQ:
            signalsTable = 'signalsTable_TF_KC1D'
        elif 'KC1Z' in DAQ:
            signalsTable = 'signalsTable_TF_KC1Z'
        elif 'KC1E' in DAQ:
            signalsTable = 'signalsTable_TF_KC1E'


        nameSignals_TF = STJET.signalsTableJET(signalsTable)


        dictTFcurrent = {}
        dictTFtmp = {}
        for jj in nameSignals_TF:
            signalPathTF = nameSignals_TF[jj]
            tData,vData,err = MAGTool.downloadSingleChannel(JPN,signalPathTF)
            if err==0:
                dictTFtmp['v'] = vData
                dictTFtmp['t'] = tData
            dictTFcurrent[jj]=dictTFtmp
            dictTFtmp = {}


        TFEVN = dictTFcurrent['TFEVN']['v']
        TFODD = dictTFcurrent['TFODD']['v']


        v_STFcomp= TFCompensationSingleChannel(sRaw,TFECoeff,TFOCoeff,TFEVN,TFODD)
        #
        return tRaw,v_STFcomp




  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



    def compareOctantsSameDAQSystem(self,DAQSystem,dictExp,idStr,sig1,sig2,iRow,iColumn):
        # @@@@@@@@@@@@@@@@@ COMPARE OCTANTS but SAME DAQSystem @@@@@@@@@@@@@@@@@@@
         # es. for KC1D or KC1Z compare CX vs CY and SX vs SY
        # es. for KC1E compare C1 vs C5 and S1 vs S5
        #  compareOctantsSameDAQSystem(self,DAQSystem,dictExp,idStr,sig1,sig2,iRow,iColumn)
        # DAQSystem:[string] subsystem choose one of the following 'KC1D','KC1E','KC1Z'
        # dictExp:[dictionary] dictionary with the experimental data, es. 'CX01': { 'v':'......','t':'......'}
        # idStr:[string] must be the first 2 char of signal name to choose which signals familiy to compare,
        #  es. pick up coils 'CX'
        # sig1,sig2: [string] signals name to compare
        # iRow, iColumn: [int] number f row and column for the subplot of sensor family to be plot in hte same figure


        if (sig1 == '' or sig2 == ''):
            print('@@@@@@@ No single signal comparison @@@@@@@@@')
            # Plot all of them comparing different octant
            # if KC1D or KC1Z compare X (oct.3) vs Y (oct.7)
            # if KC1E compare compare 1 (oct.1) vs 5 (oct.5)
            if DAQSystem == 'KC1D' or DAQSystem == 'KC1Z':
                print('KC1D or KC1Z')
                if idStr == 'CX':
                    idStr1 = idStr # 'CX'
                    idstr2 = 'CY'
                elif idStr== 'SX':
                    idStr1 = idStr # 'SX'
                    idstr2 = 'SY'

            elif DAQSystem == 'KC1E':
                print('KC1E')
                if idStr == 'C1':
                    idStr1 = idStr # 'C1'
                    idstr2 = 'C5'
                elif idStr == 'S1':
                    idStr1 = idStr # 'S1'
                    idstr2 = 'S5'

            dictListKeys = list()
            posS1 = []
            posS2 = []
            for ii in dictExp.keys():
                dictListKeys.append(ii)
            print('List Not Sorted: ')
            print(dictListKeys)
            dictListKeys.sort() # sorted
            # now I can search for CX and CY and they will be sorted
            print('List sorted')
            print(dictListKeys)
            for jj in dictListKeys:
                if idStr1 in jj:
                    posS1.append(dictListKeys.index(jj))
                    #print(idStr1 + ' is in pos ' + str(dictListKeys.index(jj)))
                if idstr2 in jj:
                    posS2.append(dictListKeys.index(jj))
                    #print(idstr2 + ' is in pos ' + str(dictListKeys.index(jj)))
            print(posS1)
            print(posS2)


            indexSubPlot = 0
            plt.figure()
            for zz in posS1:
                signal1 = dictExp[dictListKeys[posS1[zz]]]
                v_signal1 = signal1['v']
                t_signal1 = signal1['t']
                signal2 = dictExp[dictListKeys[posS2[zz]]]
                v_signal2 = signal2['v']
                t_signal2 = signal2['t']

                # PLOT
                indexSubPlot = indexSubPlot + 1
                plt.subplot(iRow,iColumn,indexSubPlot)
                plt.plot(t_signal1,v_signal1,label=dictListKeys[posS1[zz]])
                plt.plot(t_signal2,v_signal2,label=dictListKeys[posS2[zz]])
                plt.legend(loc='best',prop={'size':6})
                plt.xlabel('time[s]')
            plt.suptitle('compare signals from different octant of ' + DAQSystem)
            plt.show()

            # Plot Error
            indexSubPlot = 0
            plt.figure()
            for zz in posS1:
                signal1 = dictExp[dictListKeys[posS1[zz]]]
                v_signal1 = signal1['v']
                t_signal1 = signal1['t']
                signal2 = dictExp[dictListKeys[posS2[zz]]]
                v_signal2 = signal2['v']
                t_signal2 = signal2['t']

                # PLOT
                indexSubPlot = indexSubPlot + 1
                plt.subplot(iRow,iColumn,indexSubPlot)
                plt.plot(t_signal1,v_signal2-v_signal1,label=dictListKeys[posS2[zz]] + '-' + dictListKeys[posS1[zz]])
                plt.legend(loc='best',prop={'size':6})
                plt.xlabel('time[s]')
            plt.suptitle('error between signals of different octants ' + DAQSystem)
            plt.show()


        else:
            print('@@@@@@@@ SNGLE SIGNAL COMPARISON @@@@@@@@@@')
            signal1 = dictExp[sig1]
            v_signal1 = signal1['v']
            t_signal1 = signal1['t']
            signal2 = dictExp[sig2]
            v_signal2 = signal2['v']
            t_signal2 = signal2['t']


            pylab.figure()
            pylab.plot(t_signal1,v_signal1,label=sig1)
            pylab.plot(t_signal2,v_signal2,label=sig2)
            pylab.legend(loc='best')
            pylab.xlabel('time[s]')
            pylab.show()

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    def compareDAQSystem(self,DAQSystem1,dictExp1,idStr1,DAQSystem2,dictExp2,idStr2,iRow,iColumn):
        # Compare different system
        # ex. KC1D vs KC1Z compare CX_KC1D vs CX_KC1Z, same for CY,SX and SY, choose against who in case of KC1E
        # ex. KC1D vs KC1E compare CX_KC1D vs C1_KC1E, same for CY,SX and SY
        # DAQSystem1,DAQSystem2[string]: string to identify DAQ system, ex. 'KC1D','KC1Z','KC1E'
        # dictExp1,dictExp2[dictionary]: dictionary with exp. data from DAQ system 1 and 2
        # idStr1,idStr2[string]:string which identify signals. For ex. 'CX' means compare with CY

        def sortDict(dictExp,DAQSystem,idStr):
            # sort the keys of a dictionary using the string in idStr
            # returns:
            # the sorted keys of a dictionary in Input, the keys are sorted according the idStr which identify 2 subset of keys,
            # ex. CX and CY, or SX and SY [KC1D or KC1Z], or C1 and C5, or S1 and S1 [KC1E]
            # also return 2 posS1,posS2 vector containig the position of the sorted dictionary keys organized in a list
            dictListKeys = list()
            posS1 = []
            posS2 = []
            for ii in dictExp.keys():
                dictListKeys.append(ii)
            print('List Not Sorted: ')
            print(dictListKeys)
            dictListKeys.sort() # sorted
            # now I can search for CX and CY and they will be sorted
            print('List sorted')
            print(dictListKeys)
            if DAQSystem == 'KC1D' or DAQSystem == 'KC1Z':
                if idStr == 'CX':
                    idStr1 = 'CX'
                    idStr2 = 'CY'
                elif idStr == 'SX':
                    idStr1 = 'SX'
                    idStr2 = 'SY'
            elif DAQSystem == 'KC1E':
                if idStr == 'C1':
                    idStr1 = 'C1'
                    idStr2 = 'C5'
                elif idStr == 'S1':
                    idStr1 = 'S1'
                    idStr2 = 'S5'
            for jj in dictListKeys:
                if idStr1 in jj:
                    posS1.append(dictListKeys.index(jj))
                    #print(idStr1 + ' is in pos ' + str(dictListKeys.index(jj)))
                if idStr2 in jj:
                    posS2.append(dictListKeys.index(jj))
                    #print(idstr2 + ' is in pos ' + str(dictListKeys.index(jj)))
            print(posS1)
            print(posS2)
            return dictListKeys,posS1,posS2


        dictListKeysSorted1,pos1_1,pos1_2 = sortDict(dictExp1,DAQSystem1,idStr1)
        dictListKeysSorted2,pos2_1,pos2_2 = sortDict(dictExp2,DAQSystem2,idStr2)

        # octant CX,SX or C1,S1
        indexSubPlot = 0
        plt.figure()
        for zz in pos1_1:
            signal1 = dictExp1[dictListKeysSorted1[pos1_1[zz]]]
            v_signal1 = signal1['v']
            t_signal1 = signal1['t']
            signal2 = dictExp2[dictListKeysSorted2[pos2_1[zz]]]
            v_signal2 = signal2['v']
            t_signal2 = signal2['t']

            # PLOT
            indexSubPlot = indexSubPlot + 1
            plt.subplot(iRow,iColumn,indexSubPlot)
            plt.plot(t_signal1,v_signal1,label=dictListKeysSorted1[pos1_1[zz]] + ' ' + DAQSystem1)
            plt.plot(t_signal2,v_signal2,label=dictListKeysSorted2[pos2_1[zz]] + ' ' + DAQSystem2)
            plt.legend(loc='best',prop={'size':6})
            plt.xlabel('time[s]')
        plt.suptitle('Compare signals from : ' + DAQSystem1 + ' vs ' + DAQSystem2 )
        plt.show()

        if 0: #Plot error CX,SX,C1 from different DAQSystems
            indexSubPlot = 0
            plt.figure()
            for zz in pos1_1:
                signal1 = dictExp1[dictListKeysSorted1[pos1_1[zz]]]
                v_signal1 = signal1['v']
                t_signal1 = signal1['t']
                signal2 = dictExp2[dictListKeysSorted2[pos2_1[zz]]]
                v_signal2 = signal2['v']
                t_signal2 = signal2['t']

                # PLOT
                indexSubPlot = indexSubPlot + 1
                plt.subplot(iRow,iColumn,indexSubPlot)
                v1_interp = numpy.interp(t_signal2,t_signal1,v_signal1)
                plt.plot(t_signal2,v_signal2-v1_interp,\
                         label=[dictListKeysSorted1[pos1_1[zz]] + ' ' + DAQSystem1 + ' - ' + \
                               dictListKeysSorted2[pos2_1[zz]] + ' ' + DAQSystem2])
                plt.legend(loc='best',prop={'size':6})
                #plt.xlim(40,70)
                plt.xlabel('time[s]')
            plt.suptitle('Error between Octants (S_DAQ2-S_DAQ1) : ' + DAQSystem1 + ' vs ' + DAQSystem2 )
            plt.show()



        # Other octant CY,CY or C5,S5
        indexSubPlot = 0
        plt.figure()
        for zz in pos1_1:
            print('pos1_1:' + str(zz))
            print('pos1_2:' + str(pos1_2[zz]))
            print('pos2_2:' + str(pos2_2[zz]))
            signal1 = dictExp1[dictListKeysSorted1[pos1_2[zz]]]
            v_signal1 = signal1['v']
            t_signal1 = signal1['t']
            signal2 = dictExp2[dictListKeysSorted2[pos2_2[zz]]]
            v_signal2 = signal2['v']
            t_signal2 = signal2['t']

            # PLOT
            indexSubPlot = indexSubPlot + 1
            plt.subplot(iRow,iColumn,indexSubPlot)
            plt.plot(t_signal1,v_signal1,label=dictListKeysSorted1[pos1_2[zz]] + ' ' + DAQSystem1)
            plt.plot(t_signal2,v_signal2,label=dictListKeysSorted2[pos2_2[zz]] + ' ' + DAQSystem2)
            plt.legend(loc='best',prop={'size':6})
            plt.xlabel('time[s]')
        plt.suptitle('Compare signals from : ' + DAQSystem1 + ' vs ' + DAQSystem2 )
        plt.show()

        if 0: #Plot error CY,SY,S5 from different DAQSystems
            indexSubPlot = 0
            plt.figure()
            for zz in pos1_1:
                signal1 = dictExp1[dictListKeysSorted1[pos1_2[zz]]]
                v_signal1 = signal1['v']
                t_signal1 = signal1['t']
                signal2 = dictExp2[dictListKeysSorted2[pos2_2[zz]]]
                v_signal2 = signal2['v']
                t_signal2 = signal2['t']

                # PLOT
                indexSubPlot = indexSubPlot + 1
                plt.subplot(iRow,iColumn,indexSubPlot)
                v1_interp = numpy.interp(t_signal2,t_signal1,v_signal1)
                plt.plot(t_signal2,v_signal2-v1_interp,\
                         label=[dictListKeysSorted1[pos1_2[zz]] + ' ' + DAQSystem1 + ' - ' + \
                               dictListKeysSorted2[pos2_2[zz]] + ' ' + DAQSystem2])
                plt.legend(loc='best',prop={'size':6})
                #plt.xlim(40,70)
                plt.xlabel('time[s]')
            plt.suptitle('Error between Octants (S_DAQ2-S_DAQ1) : ' + DAQSystem1 + ' vs ' + DAQSystem2 )
            plt.show()



#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    def computeIPLA(self,prefixPU,dictDataExp_PU,dictExpData_ID_VLrr,fileNameCoeffIPLA,flagPlotIPLA):
    # _______ Compute Ip _______________
    # Ip = (1/mu0)*sum(Bp[i]*l[i]) - sum(ND[i]*ID[i]) - a*IMII - (VLRRU+VLRRL)/Rrr  )
    # where a=0, neglected IMII contribution
    # dictCoeffIp = csvR.CSV2DictGA(nameFileCoeffIp)
    # new Ver: compute IPLA only for a set of pick up coils, CX,CY,CZ,C1,C5

        # retreive IPLA coefficients ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        vesselArc  = [] # numbered from 1:18 as the pick up coils
        mappNameCX = []
        mappNameCY = []
        mappNameID  = []
        weightID   = []
        Rrr = []
        with open(fileNameCoeffIPLA) as csvfile:
            wholeFileContent = csv.reader(csvfile)
            for row in wholeFileContent:
                #print(row)
                if row[0][0]!='#':
                    if 'CX' in row[0]:
                        # build matrix of vessel arcs
                        vesselArc.append(float(row[1]))
                        mappNameCX.append(row[0])
                    elif 'CY' in row[0]:
                        mappNameCY.append(row[0])
                    elif 'ID' in row[0]:
                        mappNameID.append(row[0])
                        weightID.append(float(row[1]))
                    else :
                        Rrr =  float(row[1])
        print(mappNameCX)
        print('vessel arc:')
        print(vesselArc)
        print(mappNameCY)
        print('vessel arc:')
        print(vesselArc)
        print(mappNameID)
        print('weight ID:')
        print(weightID)
        print(dictDataExp_PU.keys())
        print(dictExpData_ID_VLrr.keys())
        # END retreive IPLA coefficients ~~~~~~~~~~~~~~~~~~~~~~~~~


        # COMPUTE IPLA ~~~~~~~~~~~~~~~~~~~~~~~~~~
        mu0 = 4*numpy.pi*1e-7
        #1/mu0 = 7.958*1e5 #(KSI sergei)
        KSI = 7.958*1e5 #(KSI sergei)

        # build contribution pick_up coils
        # contribution to IPLA from CX(oct. 3) or CY(oct.7) or C1(oct.1) or C5(oct.5) or CZ(SC)
        IPLA_PU = 0
        IPLA_PU1 = 0 # using KSI
        for ii,vv in enumerate(vesselArc):
            # CX and CY have the same coefficients
            namePU = mappNameCX[ii][0] + prefixPU[1] + mappNameCX[ii][2:]
            s_PU = dictDataExp_PU[namePU]
            v_PU = s_PU['v']
            t_PU = s_PU['t']
            print([mappNameCX[ii] + '------> ' + namePU])

            IPLA_PU = IPLA_PU + numpy.asarray(v_PU)*vesselArc[ii]/mu0
            IPLA_PU1 = IPLA_PU1 + numpy.asarray(v_PU)*vesselArc[ii]*KSI


            if 0:
                pylab.figure()
                pylab.plot(t_PU,v_PU,label=namePU)
                pylab.legend(loc='best')
                pylab.xlabel('time[s]')
                pylab.show()


        # contribution to IPLA from divertor currents ID
        IPLA_ID = 0
        for ii,vv in enumerate(weightID):
            s_ID = dictExpData_ID_VLrr[mappNameID[ii]]
            v_ID = s_ID['v']
            t_ID = s_ID['t']
            IPLA_ID = IPLA_ID + numpy.asarray(v_ID)*weightID[ii]

            if 0: # abs vs no abs
                pylab.figure()
                pylab.plot(t_ID,v_ID,label='no abs')
                pylab.plot(t_ID,abs(v_ID),label='abs')
                pylab.legend(loc='best')
                pylab.xlabel('time[s]')
                pylab.title(mappNameID[ii])
                pylab.show()

        # contribution to IPLA from restaint rings
        s_VLrrU = dictExpData_ID_VLrr['VLRRU']
        v_VLrrU = s_VLrrU['v']
        t_VLrrU = s_VLrrU['t']
        s_VLrrL = dictExpData_ID_VLrr['VLRRL']
        v_VLrrL = s_VLrrL['v']
        t_VLrrL = s_VLrrL['t']
        IPLA_VLRR = (numpy.asanyarray(v_VLrrU) + numpy.asanyarray(v_VLrrL) )/Rrr

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # IPLA from KC1D
        s_IPLA_KC1D = dictExpData_ID_VLrr['IPLA_SC_KC1D']
        v_IPLA_KC1D = s_IPLA_KC1D['v']
        t_IPLA_KC1D = s_IPLA_KC1D['t']

        # IPLA from KC1Z
        s_IPLA_KC1Z = dictExpData_ID_VLrr['IPLA_C2Z']
        v_IPLA_KC1Z = s_IPLA_KC1Z['v']
        t_IPLA_KC1Z = s_IPLA_KC1Z['t']

        # IPLA from KC1E
        s_IPLA_KC1E = dictExpData_ID_VLrr['IPLA_SC_KC1E']
        v_IPLA_KC1E = s_IPLA_KC1E['v']
        t_IPLA_KC1E = s_IPLA_KC1E['t']

        # IPLA from C2 'DA/C2-IPLA'
        s_IPLA_C2 = dictExpData_ID_VLrr['IPLA_C2']
        v_IPLA_C2 = s_IPLA_C2['v']
        t_IPLA_C2 = s_IPLA_C2['t']

        # IPLA from CF 'DA/CF-IPLA'
        s_IPLA_CF = dictExpData_ID_VLrr['IPLA_CF']
        v_IPLA_CF = s_IPLA_CF['v']
        t_IPLA_CF = s_IPLA_CF['t']

        # IPLA from CDE 'DA/CDE-IPLA'
        s_IPLA_CDE = dictExpData_ID_VLrr['IPLA_CDE']
        v_IPLA_CDE = s_IPLA_CDE['v']
        t_IPLA_CDE = s_IPLA_CDE['t']

        # # IPLA from MAGJ
        # s_IPLA_MAGJ = dictExpData_ID_VLrr['IPLA_MAGJ']
        # v_IPLA_MAGJ = s_IPLA_MAGJ['v']
        # t_IPLA_MAGJ = s_IPLA_MAGJ['t']
        #
        # # IPLA from MAGN
        # s_IPLA_MAGN = dictExpData_ID_VLrr['IPLA_MAGN']
        # v_IPLA_MAGN = s_IPLA_MAGN['v']
        # t_IPLA_MAGN = s_IPLA_MAGN['t']
        #
        # # IPLA from MAGF
        # s_IPLA_MAGF = dictExpData_ID_VLrr['IPLA_MAGF']
        # v_IPLA_MAGF = s_IPLA_MAGF['v']
        # t_IPLA_MAGF = s_IPLA_MAGF['t']


        IPLA_PU_interp = numpy.interp(t_ID,t_PU,IPLA_PU)
        compIPLA = IPLA_PU_interp-IPLA_ID-IPLA_VLRR



        if 0:
            pylab.figure()
            pylab.plot(t_PU,IPLA_PU/1e6,'k',label= 'IPLA PU')
            pylab.plot(t_ID,IPLA_ID/1e6,'r',label= 'IPLA ID')
            pylab.plot(t_ID,(IPLA_PU-IPLA_ID)/1e6,'.y',label= 'IPLA (PU-ID)')
            #pylab.plot(t_ID,(IPLA_PU+IPLA_ID)/1e6,'m',label= 'IPLA (PU+ID)')
            pylab.plot(t_VLrrL,IPLA_VLRR/1e6,'g',label= 'IPLA VLRR')
            pylab.plot(t_ID,compIPLA/1e6,'c',label= 'IPLA (PU-ID-VLrr)')
            pylab.plot(t_ID,(KSI*IPLA_PU1-IPLA_ID-IPLA_VLRR)/1e6,'oc',label= 'IPLA KSI*(PU-ID-VLrr)')
            #pylab.plot(t_IPLA_MAGN,v_IPLA_MAGN/1e6,':c',label= 'MAGN')
            # if 1:
            #    pylab.plot(t_MAGF,MAGF/1e6,':k',label= 'MAGF')
            #    pylab.plot(t_MGZF,MGZF/1e6,':r',label= 'MGZF')
            #    pylab.plot(t_MGZJ,MGZJ/1e6,':b',label= 'MGZJ')
            pylab.ylabel('[MA]')
            pylab.legend(loc='lower right',prop={'size':8})
            pylab.xlabel('time[s]')
            pylab.title('IPLA')
            pylab.show()



        IPLA_KC1D_interp = numpy.interp(t_ID,t_IPLA_C2,v_IPLA_C2)
        IPLA_KC1Z_interp = numpy.interp(t_ID,t_IPLA_KC1Z,v_IPLA_KC1Z)
        IPLA_KC1E_interp = numpy.interp(t_ID,t_IPLA_KC1E,v_IPLA_KC1E)

        if flagPlotIPLA :
            # Compare IPLA computed from CX,CY and KC1D Shape controller (SC)
            pylab.figure()
            pylab.plot(t_ID,compIPLA/1e6,':k',label= prefixPU + ' - ID - VLRR/Rrr')
            pylab.plot(t_IPLA_KC1D,v_IPLA_KC1D/1e6,'k',label= 'IPLA SC(KC1D)')
            pylab.plot(t_IPLA_KC1Z,v_IPLA_KC1Z/1e6,'r',label= 'IPLA (KC1Z)')
            pylab.plot(t_IPLA_KC1E,v_IPLA_KC1E/1e6,'g',label= 'IPLA SC(KC1E)')
            if 1:
                pylab.plot(t_IPLA_C2,v_IPLA_C2/1e6,'m',  label= 'IPLA DA C2')
                pylab.plot(t_IPLA_CDE,v_IPLA_CDE/1e6,'--m',label= 'IPLA DA CDE')
                pylab.plot(t_IPLA_CF,v_IPLA_CF/1e6,'om',  label= 'IPLA DA CF')
            # if 1: # PPF MAG
            #     pylab.plot(t_IPLA_MAGJ,v_IPLA_MAGJ/1e6,'b',  label= 'IPLA (MAGJ)')
            #     pylab.plot(t_IPLA_MAGN,v_IPLA_MAGN/1e6,':b', label= 'IPLA (MAGN)')
            #     pylab.plot(t_IPLA_MAGF,v_IPLA_MAGF/1e6,'--b',label= 'IPLA (MAGF)')


            pylab.ylabel('[MA]')
            pylab.legend(loc='lower right',prop={'size':8})
            pylab.xlabel('time[s]')
            pylab.title('IPLA')
            pylab.show()

            #  Plot error in IPLA computation between idSystem and IPLA from KC1D or KC1Z
            pylab.figure()
            pylab.plot(t_ID,(compIPLA-IPLA_KC1D_interp)/1e3,'k',label= '(IPLA[' + prefixPU + '] - IPLA[KC1D]')
            pylab.plot(t_ID,(compIPLA-IPLA_KC1Z_interp)/1e3,'r',label= '(IPLA[' + prefixPU + '] - IPLA[KC1Z]')
            pylab.plot(t_ID,(compIPLA-IPLA_KC1E_interp)/1e3,'g',label= '(IPLA[' + prefixPU + '] - IPLA[KC1E]')
            pylab.ylabel('[kA]')
            pylab.legend(loc='lower right',prop={'size':8})
            pylab.xlabel('time[s]')
            pylab.title('IPLA disagreement with DAQSystems')
            pylab.show()

            #  Plot error in IPLA computation between idSystem and IPLA from KC1D or KC1Z
            smoothData_err_PU_KC1D = pd.rolling_mean(compIPLA-IPLA_KC1D_interp,5)
            smoothData_err_PU_KC1Z = pd.rolling_mean(compIPLA-IPLA_KC1Z_interp,5)
            smoothData_err_PU_KC1E = pd.rolling_mean(compIPLA-IPLA_KC1E_interp,5)
            pylab.figure()
            pylab.plot(t_ID,(compIPLA-IPLA_KC1D_interp)/1e3,'k',label= '(IPLA[' + prefixPU + '] - IPLA[KC1D]')
            pylab.plot(t_ID,(compIPLA-IPLA_KC1Z_interp)/1e3,'r',label= '(IPLA[' + prefixPU + '] - IPLA[KC1Z]')
            pylab.plot(t_ID,(compIPLA-IPLA_KC1E_interp)/1e3,'g',label= '(IPLA[' + prefixPU + '] - IPLA[KC1E]')
            pylab.plot(t_ID,smoothData_err_PU_KC1D/1e3,'c',label= 'smoothed: (IPLA[' + prefixPU + '] - IPLA[KC1D]')
            pylab.plot(t_ID,smoothData_err_PU_KC1Z/1e3,'c',label= 'smoothed: (IPLA[' + prefixPU + '] - IPLA[KC1Z]')
            pylab.plot(t_ID,smoothData_err_PU_KC1E/1e3,'c',label= 'smoothed: (IPLA[' + prefixPU + '] - IPLA[KC1E]')
            pylab.ylabel('[kA]')
            pylab.legend(loc='lower right',prop={'size':8})
            pylab.xlabel('time[s]')
            pylab.title('AVERAGE IPLA disagreement with DAQSystems')
            pylab.show()

        # return compIPLA, t_ID, IPLA_SC_KC1D_interp,IPLA_SC_KC1Z_interp,IPLA_SC_KC1E_interp, \
        #        t_IPLA_MAGJ,v_IPLA_MAGJ,t_IPLA_MAGF, v_IPLA_MAGF,t_IPLA_MAGN, v_IPLA_MAGN, \
        #        t_PU, IPLA_PU,t_ID,IPLA_ID,t_VLrrL,IPLA_VLRR
        return compIPLA, t_ID, IPLA_KC1D_interp,IPLA_KC1Z_interp,IPLA_KC1E_interp


    def computePVPM(self,prefixPU,dictDataExp_PU,prefixSL,dictDataExp_SL,dictExpData_ID_VLrr,\
        fileNameCoeffPVPM,flagPlotPVPM):
    # _______ Compute PVPM ___PLASMA VERTICAL MOMENT_____MOMENT order 1____
    # PVPM = (1/mu0)*sum(Bp[i]*l[i] + Sad[i]*s[i])  - sum(ND[i]*ID[i]) - a*IMII - (VLRRU+VLRRL)/Rrr  )
    # where a=0, neglected IMII contribution
    # dictCoeffPVPM = csvR.CSV2DictGA(nameFileCoeffPVPM)


        # retreive PVPM coefficients ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        coeffPU  = [] # numbered from 1:18 as the pick up coils
        coeffSL  = [] # numbered from 1:14 as the saddle loops
        mappNamePU = []
        mappNameSL = []
        mappNameID  = []
        weightID   = []
        Rrr = []
        with open(fileNameCoeffPVPM) as csvfile:
            wholeFileContent = csv.reader(csvfile)
            for row in wholeFileContent:
                print(row)
                if row[0][0]!='#':
                    if prefixPU in row[0]:
                        coeffPU.append(float(row[1]))
                        mappNamePU.append(row[0])
                    elif prefixSL in row[0]:
                        coeffSL.append(float(row[1]))
                        mappNameSL.append(row[0])
                    elif 'ID' in row[0]:
                        mappNameID.append(row[0])
                        weightID.append(float(row[1]))
                    else :
                        Rrr =  float(row[1])
        print(mappNamePU)
        print('coeff PU:')
        print(coeffPU)
        print(mappNameSL)
        print('vessel arc:')
        print(coeffSL)
        print(mappNameID)
        print('weight ID:')
        print(weightID)
        print(dictDataExp_PU.keys())
        print(dictDataExp_SL.keys())
        print(dictExpData_ID_VLrr.keys())
        # END retreive PVPM coefficients ~~~~~~~~~~~~~~~~~~~~~~~~~


        # COMPUTE PVPM ~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 1/mu0=KSI !!!!!!!!!!!!!!!!!!!!!!
        mu0 = 4*numpy.pi*1e-7
        #1/mu0 = 7.958*1e5 #(KSI sergei)
        KSI = 7.958*1e5 #(KSI sergei)

        # build contribution pick_up coils
        # contribution to PVPM from CX(oct. 3) or CY(oct.7) or C1(oct.1) or C5(oct.5) or CZ(SC)
        PVPM_PU = 0
        PVPM_PU_SG = 0 # using KSI coeff Sergei Gerasimov
        for ii,vv in enumerate(coeffPU):
            # CX and CY have the same coefficients
            namePU = mappNamePU[ii][0] + prefixPU[1] + mappNamePU[ii][2:]
            s_PU = dictDataExp_PU[namePU]
            v_PU = s_PU['v']
            t_PU = s_PU['t']
            print([mappNamePU[ii] + '------> ' + namePU])

            PVPM_PU    = PVPM_PU + numpy.asarray(v_PU)*coeffPU[ii]/mu0
            PVPM_PU_SG = PVPM_PU_SG + numpy.asarray(v_PU)*coeffPU[ii]*KSI

        # build contribution pick_up coils
        # contribution to PVPM from SX(oct. 3) or SY(oct.7) or S1(oct.1) or S5(oct.5) or SZ(SC)
        PVPM_SL = 0
        PVPM_SL_SG = 0 # using KSI
        for ii,vv in enumerate(coeffSL):
            # SX and SY have the same coefficients
            nameSL = mappNameSL[ii][0] + prefixSL[1] + mappNameSL[ii][2:]
            s_SL = dictDataExp_SL[nameSL]
            v_SL = s_SL['v']
            t_SL = s_SL['t']
            print([mappNameSL[ii] + '------> ' + nameSL])

            PVPM_SL    = PVPM_SL + numpy.asarray(v_SL)*coeffSL[ii]/mu0
            PVPM_SL_SG = PVPM_SL_SG + numpy.asarray(v_SL)*coeffSL[ii]*KSI

        if 0:
            pylab.figure()
            pylab.plot(t_SL,PVPM_PU,label='PVPM from PU')
            pylab.plot(t_SL,PVPM_PU_SG,'.',label='PVPM from PU [KSI]')
            pylab.plot(t_SL,PVPM_SL,label='PVPM from SL')
            pylab.plot(t_SL,PVPM_SL_SG,'.',label='PVPM from SL [KSI]')
            pylab.legend(loc='best')
            pylab.xlabel('time[s]')
            pylab.show()


        # contribution to PVPM from divertor currents ID
        PVPM_ID = 0
        for ii,vv in enumerate(weightID):
            s_ID = dictExpData_ID_VLrr[mappNameID[ii]]
            v_ID = s_ID['v']
            t_ID = s_ID['t']
            PVPM_ID = PVPM_ID + numpy.asarray(v_ID)*weightID[ii]



        # contribution to PVPM from restaint rings
        s_VLrrU = dictExpData_ID_VLrr['VLRRU']
        v_VLrrU = s_VLrrU['v']
        t_VLrrU = s_VLrrU['t']
        s_VLrrL = dictExpData_ID_VLrr['VLRRL']
        v_VLrrL = s_VLrrL['v']
        t_VLrrL = s_VLrrL['t']
        PVPM_VLRR = (numpy.asanyarray(v_VLrrU) + numpy.asanyarray(v_VLrrL) )/Rrr


        PVPM_PU_interp = numpy.interp(t_ID,t_PU,PVPM_PU) # to deal with using KC1Z,KC1E
        PVPM_SL_interp = numpy.interp(t_ID,t_PU,PVPM_SL)
        compPVPM = PVPM_PU_interp + PVPM_SL_interp - PVPM_ID - PVPM_VLRR
    # ALWAYS PLASMA CURRENT MOMENTS ARE on Divertor Currents TIME SCALE


        return compPVPM, t_ID


    def computePRPM(self,prefixPU,dictDataExp_PU,prefixSL,dictDataExp_SL,dictExpData_ID_VLrr,\
        fileNameCoeffPRPM,flagPlotPRPM):
    # _______ Compute PRPM ____PLASMA RADIAL MOMENT_MOMENT ORDER 2_____________
    # PVPM = (1/mu0)*sum(Bp[i]*l[i] + Sad[i]*s[i])  - sum(ND[i]*ID[i]) - a*IMII - (VLRRU+VLRRL)/Rrr  )
    # where a=0, neglected IMII contribution
    # dictCoeffPRPM = csvR.CSV2DictGA(nameFileCoeffPRPM)


        # retreive PRPM coefficients ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        coeffPU  = [] # numbered from 1:18 as the pick up coils
        coeffSL  = [] # numbered from 1:14 as the saddle loops
        mappNamePU = []
        mappNameSL = []
        mappNameID  = []
        weightID   = []
        Rrr = []
        with open(fileNameCoeffPRPM) as csvfile:
            wholeFileContent = csv.reader(csvfile)
            for row in wholeFileContent:
                print(row)
                if row[0][0]!='#':
                    if prefixPU in row[0]:
                        coeffPU.append(float(row[1]))
                        mappNamePU.append(row[0])
                    elif prefixSL in row[0]:
                        coeffSL.append(float(row[1]))
                        mappNameSL.append(row[0])
                    elif 'ID' in row[0]:
                        mappNameID.append(row[0])
                        weightID.append(float(row[1]))
                    else :
                        Rrr =  float(row[1])
        print(mappNamePU)
        print('coeff PU:')
        print(coeffPU)
        print(mappNameSL)
        print('vessel arc:')
        print(coeffSL)
        print(mappNameID)
        print('weight ID:')
        print(weightID)
        print(dictDataExp_PU.keys())
        print(dictDataExp_SL.keys())
        print(dictExpData_ID_VLrr.keys())
        # END retreive PRPM coefficients ~~~~~~~~~~~~~~~~~~~~~~~~~


        # COMPUTE PRPM ~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 1/mu0=KSI !!!!!!!!!!!!!!!!!!!!!!
        mu0 = 4*numpy.pi*1e-7
        #1/mu0 = 7.958*1e5 #(KSI sergei)
        KSI = 7.958*1e5 #(KSI sergei)

        # build contribution pick_up coils
        # contribution to PRPM from CX(oct. 3) or CY(oct.7) or C1(oct.1) or C5(oct.5) or CZ(SC)
        PRPM_PU = 0
        PRPM_PU_SG = 0 # using KSI coeff Sergei Gerasimov
        for ii,vv in enumerate(coeffPU):
            # CX and CY have the same coefficients
            namePU = mappNamePU[ii][0] + prefixPU[1] + mappNamePU[ii][2:]
            s_PU = dictDataExp_PU[namePU]
            v_PU = s_PU['v']
            t_PU = s_PU['t']
            print([mappNamePU[ii] + '------> ' + namePU])

            PRPM_PU    = PRPM_PU + numpy.asarray(v_PU)*coeffPU[ii]/mu0
            PRPM_PU_SG = PRPM_PU_SG + numpy.asarray(v_PU)*coeffPU[ii]*KSI

        # build contribution pick_up coils
        # contribution to PRPM from SX(oct. 3) or SY(oct.7) or S1(oct.1) or S5(oct.5) or SZ(SC)
        PRPM_SL = 0
        PRPM_SL_SG = 0 # using KSI
        for ii,vv in enumerate(coeffSL):
            # SX and SY have the same coefficients
            nameSL = mappNameSL[ii][0] + prefixSL[1] + mappNameSL[ii][2:]
            s_SL = dictDataExp_SL[nameSL]
            v_SL = s_SL['v']
            t_SL = s_SL['t']
            print([mappNameSL[ii] + '------> ' + nameSL])

            PRPM_SL    = PRPM_SL + numpy.asarray(v_SL)*coeffSL[ii]/mu0
            PRPM_SL_SG = PRPM_SL_SG + numpy.asarray(v_SL)*coeffSL[ii]*KSI

        if 0:
            pylab.figure()
            pylab.plot(t_SL,PRPM_PU,label='PRPM from PU')
            pylab.plot(t_SL,PRPM_PU_SG,'.',label='PRPM from PU [KSI]')
            pylab.plot(t_SL,PRPM_SL,label='PRPM from SL')
            pylab.plot(t_SL,PRPM_SL_SG,'.',label='PRPM from SL [KSI]')
            pylab.legend(loc='best')
            pylab.xlabel('time[s]')
            pylab.show()


        # contribution to PRPM from divertor currents ID
        PRPM_ID = 0
        for ii,vv in enumerate(weightID):
            s_ID = dictExpData_ID_VLrr[mappNameID[ii]]
            v_ID = s_ID['v']
            t_ID = s_ID['t']
            PRPM_ID = PRPM_ID + numpy.asarray(v_ID)*weightID[ii]



        # contribution to PRPM from restaint rings
        s_VLrrU = dictExpData_ID_VLrr['VLRRU']
        v_VLrrU = s_VLrrU['v']
        t_VLrrU = s_VLrrU['t']
        s_VLrrL = dictExpData_ID_VLrr['VLRRL']
        v_VLrrL = s_VLrrL['v']
        t_VLrrL = s_VLrrL['t']
        PRPM_VLRR = (numpy.asanyarray(v_VLrrU) + numpy.asanyarray(v_VLrrL) )/Rrr


        PRPM_PU_interp = numpy.interp(t_ID,t_PU,PRPM_PU) # to deal with using KC1Z,KC1E
        PRPM_SL_interp = numpy.interp(t_ID,t_PU,PRPM_SL)
        compPRPM = PRPM_PU_interp + PRPM_SL_interp - PRPM_ID - PRPM_VLRR
    # ALWAYS PLASMA CURRENT MOMENTS ARE on Divertor Currents TIME SCALE


        return compPRPM, t_ID



#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


    def compareSignals(self,JPN,listOfSignals):
        # compare a list of signals organized in a dictionary with signalsName and signalsPath
        signalsName = listOfSignals.keys()
        signalsPath = listOfSignals.values()

        dictDataExp = {}
        for jj in  signalsName:
               dictDataExpTMP = {}
               data,tvec,nwds,title,units,ier= getdat.getdat(listOfSignals[jj],JPN)
               if ier:
                   print(jj, 'Error to retreive it')
               else:
                   dictDataExpTMP['v'] = data        # signal value
                   dictDataExpTMP['t'] = tvec # signal time
                   print(jj,' Downloaded !!')

                   dictDataExp[jj] = dictDataExpTMP


        pylab.figure()
        for zz in signalsName:
            s = dictDataExp[zz]
            v = s['v']
            t = s['t']
            pylab.plot(t,v,label=[zz + ': ' + listOfSignals[zz]])
        pylab.xlabel('time[s]')
        pylab.title('Compare a list of signals')
        pylab.legend(loc='lower right',prop={'size':8})
        pylab.show()

        return dictDataExp


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    def strikePointsXLOC(self,nameListSP2Extract,expDataXLOC,geomInfoGapXLOC,timeEquil):
        """

        :param nameListSP2Extract:
        :param expDataXLOC:
        :param geomInfoGapXLOC:
        :param timeEquil:
        :return:
        """
        #reurn gap at current Time(timeEquil) and XLOC boundary

        nameListGap = expDataXLOC.keys() # from dict experimental
        nameListGap = nameListSP2Extract
        timeXLOC = expDataXLOC['ROG']['t']
        iTimeXLOC = numpy.where(numpy.abs(timeEquil-timeXLOC)<2*min(numpy.diff(timeXLOC)))# twice of the min of XLOC delta time
        if  iTimeXLOC : # empty
            iTimeXLOC = numpy.where(numpy.abs(timeEquil-timeXLOC)<50*min(numpy.diff(timeXLOC)))# twice of the min of XLOC delta time

        iTXLOC = iTimeXLOC[0][0]
        # print('Time XLOC:' + str(timeXLOC[iTXLOC]))
        SP0 = {}
        rSP0 = []
        zSP0 = []

        for jj in nameListSP2Extract:
            vSP = expDataXLOC[jj]['v']
            tSP = expDataXLOC[jj]['t']
            if 0:
                pylab.figure()
                pylab.plot(tSP,vSP)
                pylab.title(str(jj))
                pylab.show()
            SP0[jj]=vSP[iTXLOC] #as dictionary
            # print(jj + ': ' + str(vSP[iTXLOC]))

            R1 = geomInfoGapXLOC[jj]['R1']
            R2 = geomInfoGapXLOC[jj]['R2']
            R3 = geomInfoGapXLOC[jj]['R3']
            R4 = geomInfoGapXLOC[jj]['R4']
            Z1 = geomInfoGapXLOC[jj]['Z1']
            Z2 = geomInfoGapXLOC[jj]['Z2']
            Z3 = geomInfoGapXLOC[jj]['Z3']
            Z4 = geomInfoGapXLOC[jj]['Z4']
            M1 = geomInfoGapXLOC[jj]['M1']
            M2 = geomInfoGapXLOC[jj]['M2']
            M3 = geomInfoGapXLOC[jj]['M3']
            M4 = geomInfoGapXLOC[jj]['M4']

            # M1comp = 0
            # M2comp = numpy.sqrt((R2-R1)**2+(Z2-Z1)**2)
            # M3comp = numpy.sqrt((R3-R2)**2+(Z3-Z2)**2)
            # M4comp = numpy.sqrt((R4-R3)**2+(Z4-Z3)**2)

            #mapList = [M1comp,M2comp,M3comp,M4comp]
            mapList = [M1,M2,M3,M4]
            rList = [R1,R2,R3,R4]
            zList = [Z1,Z2,Z3,Z4]
            rSP = numpy.interp(SP0[jj],mapList,rList)
            zSP = numpy.interp(SP0[jj],mapList,zList)
            rSP0.append(rSP)
            zSP0.append(zSP)


        return SP0,rSP0,zSP0,iTXLOC


    def gapXLOC(self,nameListGap2Extract,expDataXLOC,geomInfoGapXLOC,timeEquil):
        #reurn gap at current Time(timeEquil) and XLOC boundary

        nameListGap = expDataXLOC.keys() # from dict experimental
        nameListGap = nameListGap2Extract
        timeXLOC = expDataXLOC['ROG']['t']

        iTimeXLOC = numpy.where(numpy.abs(timeEquil-timeXLOC)<2*min(numpy.diff(timeXLOC)))# twice of the min of XLOC delta time
        if  iTimeXLOC  : # empty
             iTimeXLOC = numpy.where(numpy.abs(timeEquil-timeXLOC)<50*min(numpy.diff(timeXLOC)))# twice of the min of XLOC delta time



        iTXLOC = iTimeXLOC[0][0]
        # print('Time XLOC:' + str(timeXLOC[iTXLOC]))
        gap0 = {}
        rG0 = []
        zG0 = []
        count = 0
        for jj in nameListGap:
            vGAP = expDataXLOC[jj]['v']
            gap0[jj]=vGAP[iTXLOC] # as dictionary
            # print(jj + ': ' + str(vGAP[iTXLOC]))

            R1 = geomInfoGapXLOC[jj]['R1']
            R2 = geomInfoGapXLOC[jj]['R2']
            Z1 = geomInfoGapXLOC[jj]['Z1']
            Z2 = geomInfoGapXLOC[jj]['Z2']

            rG = R1+gap0[jj]*(R2-R1)/numpy.sqrt((R2-R1)**2+(Z2-Z1)**2)
            zG = Z1+gap0[jj]*(Z2-Z1)/numpy.sqrt((R2-R1)**2+(Z2-Z1)**2)
            rG0.append(rG)
            zG0.append(zG)


        return gap0,rG0,zG0,iTXLOC



    def gapWALLS(self,nameListGapWALLS,expDataWALLS,timeEquil):
        #reurn r and z of gap point at current Time(timeEquil),WALLS boundary

        nameListGap = expDataWALLS.keys() # from dict experimental
        timeWALLS = expDataWALLS['IWLGR01']['t']

        iTimeWALLS = numpy.where(numpy.abs(timeEquil-timeWALLS)<10e-3)# WALLS delta time

        iTWALLS = iTimeWALLS[0][0]
        # print(timeWALLS[iTWALLS])
        rWALLS = []
        zWALLS = []
        for jj,vv in enumerate(nameListGapWALLS):
            nameR = vv[0:4] + 'R' + vv[4:]
            nameZ = vv[0:4] + 'Z' + vv[4:]

            vR = expDataWALLS[nameR]['v']
            vZ = expDataWALLS[nameZ]['v']

            rWALLS.append(vR[iTWALLS])
            zWALLS.append(vZ[iTWALLS])

            #print(vv + ' -> ' + nameR + ' -> ' + nameZ)

        return rWALLS,zWALLS,iTWALLS


    def readEFITFlux(self,expDataDictJPNobj,timeEquil):
        # given EFIT exp data it return:
        #  rC,zC: coordinates of the boundary @ t0=timeEquil ,
        # psi: flux map @ t0
        # rGrid,zGrid: coordinates of the EFIT grid where Grad Shafranov Equation is solved

        EFIT = expDataDictJPNobj

        RBND_v = EFIT['RBND']['v']
        RBND_x = EFIT['RBND']['x']
        RBND_t = EFIT['RBND']['t']
        rBND = RBND_v
        rC=numpy.reshape(rBND,(len(RBND_t),len(RBND_x)))

        ZBND_v = EFIT['ZBND']['v']
        ZBND_x = EFIT['ZBND']['x']
        ZBND_t = EFIT['ZBND']['t']
        zBND = ZBND_v
        zC=numpy.reshape(zBND,(len(ZBND_t),len(ZBND_x)))

        timeEFIT = RBND_t # one of the _t variables
        iCurrentTime = numpy.where(numpy.abs(timeEquil-timeEFIT)<2*min(numpy.diff(timeEFIT)))# twice of the min of EFIT delta time
        # print(timeEFIT[iCurrentTime])

        iTEFIT = iCurrentTime[0][0]

        rC0 = rC[iTEFIT,:]
        zC0 = zC[iTEFIT,:]



        return rC0,zC0

    @staticmethod
    def readGapFile(fileNameGap):
        fileDir = os.path.dirname('__file__')
        fileDir = os.path.dirname(os.path.realpath('__file__'))
        fileName = os.path.join(fileDir,  fileNameGap)
        # read gaps and strike points
        gapSpDict = {}
        with open(fileName,'r') as f:
            wholeFileContent = csv.reader(f)
            dictTMP = {}
            for line in wholeFileContent:
#                print(line)
                if line[0][0]!='#':

                    if len(line) == 5:
                        # nameGap R1  Z1  R2  Z2 # GAPS
                        dictTMP['R1'] = float(line[1])
                        dictTMP['Z1'] = float(line[2])
                        dictTMP['R2'] = float(line[3])
                        dictTMP['Z2'] = float(line[4])

                    elif len(line)>5 :
                        #nameStrikePoint R1 R2 R3 R4 Z1 Z2 Z3 Z4 #STRIKE POINTS
                        dictTMP['R1'] = float(line[1])
                        dictTMP['R2'] = float(line[2])
                        dictTMP['R3'] = float(line[3])
                        dictTMP['R4'] = float(line[4])
                        dictTMP['Z1'] = float(line[5])
                        dictTMP['Z2'] = float(line[6])
                        dictTMP['Z3'] = float(line[7])
                        dictTMP['Z4'] = float(line[8])
                        # mapping (abscissa curvilinea
                        dictTMP['M1'] = float(line[9])
                        dictTMP['M2'] = float(line[10])
                        dictTMP['M3'] = float(line[11])
                        dictTMP['M4'] = float(line[12])
                    gapSpDict[line[0].strip()]  = dictTMP
                    dictTMP = {}


       # print(gapSpDict)
        return gapSpDict

    @staticmethod
    def readFWFile(fileNameFW):
        fileDir = os.path.dirname('__file__')
        fileDir = os.path.dirname(os.path.realpath('__file__'))
        fileName = os.path.join(fileDir,'Database/' + fileNameFW)
        # read First Wall ILW points
        with open(fileName,'r') as f:
            wholeFileContent = csv.reader(f)
            xFW = []
            yFW = []
            for line in wholeFileContent:
 #               print(line)
                if line[0][0]!='#':
                    xFW.append(float(line[0]))
                    yFW.append(float(line[1]))
            # to return the FW as a closed line
            xFW.append(xFW[0])
            yFW.append(yFW[0])
        return xFW, yFW



    def searchDryRun(self,JPNstandardDryRun,JPNstart,JPNend):
        # TO BE IMPLEMENTED...not tested YET!!!!!!!!!!!!!!!

        pulses = numpy.linspace(JPNstart,JPNend,num=1,retstep=True)
        listOfSignals = {
            'IPLA'  :'PF/SC-IPLA<KS',    # IPLA SC KC1D
            'DSRP'  :'PF/PTN-DSRTP<PFP', # occoured disruption
            'NSB'   :'PF/PTN-NSB<PFP',   # NSB (not sosteinable breakdown)
            }
        signalsName = listOfSignals.keys()
        signalsPath = listOfSignals.values()

        dictDataExp = {}
        for ss in pulses:
            print(pulses[0])
            for jj in  signalsName:
                   dictDataExpTMP = {}
                   try:
                       data,tvec,nwds,title,units,ier= getdat.getdat(listOfSignals[jj],pulses[0])
                   except ier==0:
                       data,nwds,title,units,ier = getdat.getsca(listOfSignals[jj],pulses[0])
                   if ier==0:
                       print(jj, 'Error to retreive it')
                   else:
                       dictDataExpTMP['v'] = data        # signal value
                       dictDataExpTMP['t'] = tvec # signal time
                       print(jj,' Downloaded !!')

                       dictDataExp[jj] = dictDataExpTMP




    # @@@@@@@@@@@@  BONUS routine @@@@@@@@@@@@@@@@

    def extractSignalsFromTFcompDict(self,dictTF,labelExtract,idDAQ):
        # estract data from TF dictionary such as 'raw,'TFcomp','TFO',....
        nameList = dictTF.keys()
        dataToExtract = {}
        dictTMP = {}
        for nameSignal in nameList:
            sD = dictTF[nameSignal]
            vS = sD[labelExtract]['v']
            tS = sD[labelExtract]['t']
            dictTMP['v'] = vS
            dictTMP['t'] = tS
            dictTMP['comment'] = idDAQ + ' ' + labelExtract + ' ' + nameSignal # to add a comment in private ppf
            dataToExtract[nameSignal]= dictTMP
            dictTMP = {}
        return dataToExtract


    def writePrivatePPF(self,recDataDict,ddaPrivate,uidPrivate):
        # write private ppf from a dictionary content
        # use uidPrivate = 'garta'
        pulse = self.JPN
        dda   = ddaPrivate
        ier = ppf.ppfgo(pulse)
        ppf.ppfuid(uidPrivate, "w")
        time, date, ier = ppf.pdstd(pulse)


        nameList = recDataDict.keys()
        for dtyp in nameList:
            sD   = recDataDict[dtyp]
            vS   = sD['v']
            tS   = sD['t']
            desc = sD['comment']

            ier = ppf.ppfopn(pulse, date, time, desc)
            irdat     = ppf.ppfwri_irdat(1,len(tS),refx=-1,reft=-1)
            ihdat     = ppf.ppfwri_ihdat("", "", "", "F", "F", "F",desc)
            iwdat,ier = ppf.ppfwri(pulse,dda,dtyp,irdat,ihdat,vS,[0],tS)

        seq, ier  = ppf.ppfclo(pulse, dda, 0)


    def computeErrorPerc(self,idStr,expDataDict1,expDataDict2,iRow,iColumn,flagFreq,tStart,tEnd):
        # is already done inside the MAGTool  compareDAQSystem activated by a flag in the if,
        # here the difference is choose a time window [tStart,tEnd] to visualize the results
        # usuallly in the main plasma phase than with a compute the % err of a signals
        # from same or different DAQ. Err% is defined as  err% = 100*(sKC1Z-sKC1D)/sKC1D
        # and with a flag decide if take into account that different DAQ system have different sampling
        # rate moreover KC1D has a fast sampling time window, which means within same DAQ there is different
        # sampling freq, so in comparing (doing the difference of 2 signals from different DAQ) the criteria
        # is to downsample the signal with higher sampling once detected the various sampling windows
        #
        # idStr : signal prefix string to order the keys in the dictionary, ex. 'CX' or 'CY' or 'SX' or 'S1'
        # expDataDict1,expDataDict2: dict with exp data with 'v'(value) and 't'(time) field for each signal,
        # iRow,iColumn: integer for nr of row and column in the subplot
        # flagFreq: boolean to decide if find sampling window and compare the DAQ in each sampling window,
        #         downsampling that one with higher sampling freq
        # tStart,tEnd: float wich indicates time window to visualiza comparison
        #
        # return dictErr,dictErrSmoothed in a format that can be written as ppf

        list2compare = list()
        keysList = expDataDict1.keys()
        for jj in expDataDict1.keys():
            if idStr in jj:
                list2compare.append(jj)
        list2compare.sort()

        # err% = 100*(KC1Z-KC1D)/KC1D
        if not flagFreq: #subplot
            dictErr ={}
            dictErrTmp ={}
            dictErrSmooth ={}
            dictErrSmoothTmp ={}
            dictDataKC1ZInterp = {}
            dictDataKC1ZInterpTmp = {}
            dictDataKC1DSliced = {}
            dictDataKC1DSlicedTmp = {}
            indexSubPlot = 0
            plt.figure()
            for ww in list2compare:
                s_KC1D   = expDataDict1[ww]
                v_S1 = s_KC1D['v']
                t_S1 = s_KC1D['t']
                s_KC1Z   = expDataDict2[ww]
                v_S2 = s_KC1Z['v']
                t_S2 = s_KC1Z['t']

                # m = sum(v_S/len(v_S)) # mean
                # print(m)
                # varS = sum([(xi - m)**2 for xi in v_S])/len(v_S) # variance
                # print(varS)
                # DOWNSAMPLING KC1Z
                vS2_Interp = numpy.interp(t_S1,t_S2,v_S2) # KC1Z interp on KC1D time scale
                errPercentage = ((vS2_Interp-v_S1)/v_S1)*100
                errPercentageSmooted = pd.rolling_mean(errPercentage,5)
                iPlot = numpy.where((t_S1>= tStart) & (t_S1<= tEnd))

                dictErrTmp['v']= errPercentage[iPlot]
                dictErrTmp['t']= t_S1[iPlot]
                dictErrTmp['comment'] = 'err % sKC1Z-sKC1D'
                dictErr[ww] = dictErrTmp
                dictErrSmoothTmp['v']= errPercentageSmooted[iPlot]
                dictErrSmoothTmp['t']= t_S1[iPlot]
                dictErrSmoothTmp['comment'] = 'err % sKC1Z-sKC1D smoothed'
                dictErrSmooth[ww] = dictErrSmoothTmp
                dictDataKC1ZInterpTmp['v']= vS2_Interp[iPlot]
                dictDataKC1ZInterpTmp['t']= t_S1[iPlot]
                dictDataKC1ZInterpTmp['comment'] = 'sKC1Z interp. on tKC1D'
                dictDataKC1ZInterp[ww] = dictDataKC1ZInterpTmp
                dictDataKC1DSlicedTmp['v']= v_S1[iPlot]
                dictDataKC1DSlicedTmp['t']= t_S1[iPlot]
                dictDataKC1DSlicedTmp['comment'] = 'sKC1D sliced'
                dictDataKC1DSliced[ww] = dictDataKC1DSlicedTmp
                dictErrTmp = {}
                dictErrSmoothTmp = {}
                dictDataKC1ZInterpTmp = {}
                dictDataKC1DSlicedTmp ={}

                # plot
                indexSubPlot = indexSubPlot + 1
                plt.subplot(iRow,iColumn,indexSubPlot)
                pylab.plot(t_S1[iPlot],errPercentage[iPlot],'r',label= ww + ' err %')
                meanErrPerc = str(round(numpy.mean(errPercentageSmooted[iPlot]),2))
                pylab.plot(t_S1[iPlot],errPercentageSmooted[iPlot],'k',label= ww + ' <err %> = ' + str(meanErrPerc))
                plt.xlabel('time[s]')
                plt.legend(loc='best',prop={'size':8})

            plt.suptitle( '#' + str(self.JPN) + '  err% = 100*(KC1Z-KC1D)/KC1D')
            plt.show()

        elif flagFreq:
            # compute sampling frequency frpm 1 representative signal lets say CX02
            ww = 'CX02'
            s_KC1D   = expDataDict1[ww]
            v_S1 = s_KC1D['v']
            t_S1 = s_KC1D['t']
            s_KC1Z   = expDataDict2[ww]
            v_S2 = s_KC1Z['v']
            t_S2 = s_KC1Z['t']
            # iPlot_KC1D = numpy.where((t_S1_all>= tStart) & (t_S1_all<= tEnd))
            # t_S1 = t_S1_all[iPlot_KC1D]
            # v_S1 = v_S1_all[iPlot_KC1D]
            # iPlot_KC1Z = numpy.where((t_S2_all>= tStart) & (t_S2_all<= tEnd))
            # t_S2 = t_S2_all[iPlot_KC1Z]
            # v_S2 = v_S2_all[iPlot_KC1Z]

            # KC1D usually has 70Hz sampling time and a fast windows of 5kHz
            freqS_KC1D = 1/numpy.diff(t_S1)
            # KC1Z usually has 500Hz sampling time
            freqS_KC1Z = 1/numpy.diff(t_S2)

            # derivative in time of freq
            dfreqS_KC1D = numpy.diff(freqS_KC1D)/freqS_KC1D[:-1]
            # greater than 10Hz (the mnax is 20Hz)per samples (derivative of freqSig is always
            #  zero a part where sampling frequency change, has a jump there)
            vCutOffPlus  = 0.8*numpy.max(dfreqS_KC1D)
            vCutOffMinus = 0.8*numpy.min(dfreqS_KC1D)
            iCutOffPlus  = numpy.where(dfreqS_KC1D>vCutOffPlus)
            iCutOffMinus = numpy.where(dfreqS_KC1D<vCutOffMinus)
            tCutOff = numpy.asarray([t_S1[iCutOffPlus[0][-1]] , t_S1[iCutOffMinus[0][0]]])
            tJumpKC1D = 30e-3

            # find time index of KC1D signals witouht transition from downsampling to higher sampling
            i_KC1D = [numpy.where(t_S1<=(tCutOff[0]-tJumpKC1D))[0][-1], \
                      numpy.where(t_S1>=(tCutOff[0]+tJumpKC1D))[0][0],  \
                      numpy.where(t_S1<=(tCutOff[1]-tJumpKC1D))[0][-1], \
                      numpy.where(t_S1>=(tCutOff[1]+tJumpKC1D))[0][0]]
            i_KC1Z = [numpy.where(t_S2<=(tCutOff[0]-tJumpKC1D))[0][-1], \
                      numpy.where(t_S2>=(tCutOff[0]+tJumpKC1D))[0][0],  \
                      numpy.where(t_S2<=(tCutOff[1]-tJumpKC1D))[0][-1], \
                      numpy.where(t_S2>=(tCutOff[1]+tJumpKC1D))[0][0]]

            #print(str(tCutOff-tJumpKC1D))
            #print(str(tCutOff+tJumpKC1D))
            #print(str(t_S1[i_KC1D]))
            f, (ax1, ax2, ax3) = pylab.subplots(3, sharex=True)
            ax1.plot(t_S1,v_S1,'r',label= ww + ' KC1D')
            ax1.plot(t_S2,v_S2,'k',label= ww + ' KC1Z')
            #ax1.plot(t_S1_artif,v_S1_artif,'g',label='KC1D reduced')
            ax1.plot(t_S1[i_KC1D],[0,0,0,0],'ok')
            ax1.plot(t_S2[i_KC1Z],[0,0,0,0],'xc')
            ax1.legend(loc='lower right',prop={'size':12})
            ax2.plot(t_S1[:-1],freqS_KC1D,'r',label='sampl.freq. KC1D')
            ax2.plot(t_S2[:-1],freqS_KC1Z,'k',label='sampl.freq. KC1Z')
            ax2.legend(loc='lower right',prop={'size':12})
            ax3.plot(t_S1[:-2],dfreqS_KC1D,'g',label='d/dsamples of sampl.freq. KC1D')
            ax3.plot(tCutOff,[0,0],'*k')
            ax3.plot(t_S1[i_KC1D],[0,0,0,0],'xk')
            ax3.plot(tCutOff+tJumpKC1D,[0,0],'or')
            ax3.plot(tCutOff-tJumpKC1D,[0,0],'oc')
            ax3.legend(loc='lower right',prop={'size':12})
            plt.show()


            indexSubPlot = 0
            plt.figure()
            for ww in list2compare:
                s_KC1D   = expDataDict1[ww]
                v_S1 = s_KC1D['v']
                t_S1 = s_KC1D['t']
                s_KC1Z   = expDataDict2[ww]
                v_S2 = s_KC1Z['v']
                t_S2 = s_KC1Z['t']

                # splitto il vettore finale in 3 fasi
                # 1st phase sampl freq KC1Z>sampl freq KC1D -> downsampling KC1Z
                t_phase1_KC1D = t_S1[0:i_KC1D[0]]
                v_phase1_KC1D = v_S1[0:i_KC1D[0]]
                t_phase1_KC1Z = t_S2[0:i_KC1Z[0]]
                v_phase1_KC1Z = v_S2[0:i_KC1Z[0]]
                v_artif_phase1 = numpy.interp(t_phase1_KC1D,t_phase1_KC1Z,v_phase1_KC1Z) # KC1Z
                # 2nd phase sampl freq KC1D>sampl freq KC1Z -> downsampling KC1D
                t_phase2_KC1D = t_S1[i_KC1D[1]:i_KC1D[2]]
                v_phase2_KC1D = v_S1[i_KC1D[1]:i_KC1D[2]]
                t_phase2_KC1Z = t_S2[i_KC1Z[1]:i_KC1Z[2]]
                v_phase2_KC1Z = v_S2[i_KC1Z[1]:i_KC1Z[2]]
                v_artif_phase2 = numpy.interp(t_phase2_KC1Z,t_phase2_KC1D,v_phase2_KC1D) # KC1D
                # 3rd phase sampl freq KC1Z>sampl freq KC1D -> downsampling KC1Z
                t_phase3_KC1D = t_S1[i_KC1D[3]:-1]
                v_phase3_KC1D = v_S1[i_KC1D[3]:-1]
                t_phase3_KC1Z = t_S2[i_KC1Z[3]:-1]
                v_phase3_KC1Z = v_S2[i_KC1Z[3]:-1]
                v_artif_phase3 = numpy.interp(t_phase3_KC1D,t_phase3_KC1Z,v_phase3_KC1Z) # KC1Z


                errPerc_phase_1 = 100*(v_artif_phase1-v_phase1_KC1D)/v_phase1_KC1D
                # errPerc_phase_1[errPerc_phase_1==-numpy.inf]=0
                # errPerc_phase_1[errPerc_phase_1==numpy.inf]=0
                # errPerc_phase_1[numpy.isnan(errPerc_phase_1)]=0
                errPerc_phase_2 = 100*(v_phase2_KC1Z-v_artif_phase2)/v_artif_phase2
                # errPerc_phase_2[errPerc_phase_2==-numpy.inf]=0
                # errPerc_phase_2[errPerc_phase_2==numpy.inf]=0
                # errPerc_phase_2[numpy.isnan(errPerc_phase_2)]=0
                errPerc_phase_3 = 100*(v_artif_phase3-v_phase3_KC1D)/v_phase3_KC1D
                # errPerc_phase_3[errPerc_phase_3==-numpy.inf]=0
                # errPerc_phase_3[errPerc_phase_3==numpy.inf]=0
                # errPerc_phase_3[numpy.isnan(errPerc_phase_3)]=0
                errPerc_phase_1_smooth = pd.rolling_mean(errPerc_phase_1,5)
                errPerc_phase_1_smooth[numpy.isnan(errPerc_phase_1_smooth)]=0
                errPerc_phase_2_smooth = pd.rolling_mean(errPerc_phase_2,5)
                errPerc_phase_2_smooth[numpy.isnan(errPerc_phase_2_smooth)]=0
                errPerc_phase_3_smooth = pd.rolling_mean(errPerc_phase_3,5)
                errPerc_phase_3_smooth[numpy.isnan(errPerc_phase_3_smooth)]=0

                # t_S1_artif = numpy.append(t_S1[0:i_KC1D[0]],\
                #               t_S1[i_KC1D[1]:i_KC1D[2]])
                # t_S1_artif= numpy.append(t_S1_artif,t_S1[i_KC1D[3]:-1])
                #
                # v_S1_artif = numpy.append(v_S1[0:i_KC1D[0]],\
                #               v_S1[i_KC1D[1]:i_KC1D[2]])
                # v_S1_artif= numpy.append(v_S1_artif,v_S1[i_KC1D[3]:-1])


                meanErrPercSmooted = numpy.mean([numpy.mean(errPerc_phase_1_smooth),\
                                                 numpy.mean(errPerc_phase_2_smooth),\
                                                 numpy.mean(errPerc_phase_3_smooth)])
                #iPlot = numpy.where((t_S1>= tStart) & (t_S1<= tEnd))
                # plot
                indexSubPlot = indexSubPlot + 1
                plt.subplot(iRow,iColumn,indexSubPlot)
                pylab.plot(t_phase1_KC1D,errPerc_phase_1,'r',label= ww + ' err %')
                pylab.plot(t_phase2_KC1Z,errPerc_phase_2,'r')#,label= ww + ' err %')
                pylab.plot(t_phase3_KC1D,errPerc_phase_3,'r')#,label= ww + ' err %')
                pylab.plot(t_phase1_KC1D,errPerc_phase_1_smooth,'k',label= ww + ' <err %> ' + str(round(numpy.mean(meanErrPercSmooted),2)))
                pylab.plot(t_phase2_KC1Z,errPerc_phase_2_smooth,'k')#,label= ww + ' <err %> ' + str(round(numpy.mean(meanErrPercSmooted),2)))
                pylab.plot(t_phase3_KC1D,errPerc_phase_3_smooth,'k')#,label= ww + ' <err %> ' + str(round(numpy.mean(meanErrPercSmooted),2)))

                plt.xlabel('time[s]')
                #plt.xlim(tStart,tEnd)
                #plt.axis('equal')
                plt.legend(loc='best',prop={'size':8})

            plt.suptitle( '#' + str(self.JPN) + '  err% = 100*(KC1Z-KC1D)/KC1D')
            plt.show()

            # not implemented for freq anal yet
            dictErr = {}
            dictErrSmooth = {}
            dictDataKC1ZInterp = {}
            dictDataKC1DSliced = {}

        return dictErr,dictErrSmooth,dictDataKC1ZInterp,dictDataKC1DSliced


    @staticmethod
    def computeErrorPercOctant(listKeySorted,pos1ExpdData,pos2ExpData,expDataDict):
        # compute err % between octant for same DAQ and shot
        # dictErr,dictErrSmooth = computeErrorPercOctant(listKeySorted,pos1ExpdData,pos2ExpData,expDataDict)
        nrSamplesToAvg = 5
        # err% = 100*(s2-s1)/s1
        dictErr ={}
        dictErrTmp ={}
        dictErrSmooth ={}
        dictErrSmoothTmp ={}
        dictErrAbsTmp = {}
        dictErrAbs = {}
        dictErrAbsSmoothTmp = {}
        dictErrAbsSmooth = {}

        meanErrPercSmoothed = []
        meanErrPercAbsSmoothed = []

        nSens = len(pos1ExpdData)
        for ii in pos1ExpdData:
            ww = listKeySorted[ii]
            s_1   = expDataDict[ww]
            v_S1 = s_1['v']
            t_S1 = s_1['t']
            zz = listKeySorted[ii+nSens]
            s_2   = expDataDict[zz]
            v_S2 = s_2['v']
            t_S2 = s_2['t']


            # err %
            errPercentage = (v_S2-v_S1)/v_S1*100
            errPercentageSmoothed = pd.rolling_mean(errPercentage,nrSamplesToAvg)
            meanErrPercSmoothed.append(numpy.mean(errPercentageSmoothed))

            # err % abs
            errPercentageAbs = abs(v_S2-v_S1)/abs(v_S1)*100
            errPercentageAbsSmooted = pd.rolling_mean(errPercentageAbs,nrSamplesToAvg)
            meanErrPercAbsSmoothed.append(numpy.mean(errPercentageSmoothed))

            # err %
            dictErrTmp['v']= errPercentage
            dictErrTmp['t']= t_S1
            dictErrTmp['comment'] = 'err % (vDR-vSDR)/vSDR'
            dictErr[ww] = dictErrTmp

            # smooth err %
            dictErrSmoothTmp['v']= errPercentageSmoothed
            dictErrSmoothTmp['t']= t_S1
            dictErrSmoothTmp['comment'] = 'err % (vDR-vSDR)/vSDR smoothed'
            dictErrSmooth[ww] = dictErrSmoothTmp

            # abs err %
            dictErrAbsTmp['v']= errPercentageAbs
            dictErrAbsTmp['t']= t_S1
            dictErrAbsTmp['comment'] = 'err % abs(vDR-vSDR)/abs(vSDR)'
            dictErrAbs[ww] = dictErrAbsTmp

            # smooth abs err %
            dictErrAbsSmoothTmp['v']= errPercentageAbsSmooted
            dictErrAbsSmoothTmp['t']= t_S1
            dictErrAbsSmoothTmp['comment'] = 'err % abs(vDR-vSDR)/abs(vSDR) smoothed'
            dictErrAbsSmooth[ww] = dictErrAbsSmoothTmp



            dictErrTmp = {}
            dictErrSmoothTmp = {}
            dictErrTmp = {}
            dictErrSmoothTmp = {}
            dictErrAbsTmp = {}
            dictErrAbsSmoothTmp = {}

        return dictErr,dictErrSmooth,dictErrAbs,dictErrAbsSmooth,\
               meanErrPercSmoothed,meanErrPercAbsSmoothed


    @staticmethod
    def computeErrorPercSameDAQ(listKeySorted,pos1ExpdData,pos2ExpData,expDataDict1,expDataDict2):
               # dictErr1,dictErrSmooth1 = MAGTool.computeErrorPercSameDAQ(dictListKeysSortedJPNTDR,\
               #  pos1_JPNTDR,pos1_JPNDR,dictExpDataJPNTDR,dictExpDataJPNDR)

        # compute %err between shot for same DAQ
        #
        # idStr : signal prefix string to order the keys in the dictionary, ex. 'CX' or 'CY' or 'SX' or 'S1'
        # expDataDict1,expDataDict2: dict with exp data with 'v'(value) and 't'(time) field for each signal,
        # iRow,iColumn: integer for nr of row and column in the subplot
        #
        # return dictErr,dictErrSmoothed in a format that can be written as ppf


        # err% = 100*(s2-s1)/s1
        dictErr ={}
        dictErrTmp ={}
        dictErrSmooth ={}
        dictErrSmoothTmp ={}

        for ii in pos1ExpdData:
            ww = listKeySorted[ii]
            s_1   = expDataDict1[ww]
            v_S1 = s_1['v']
            t_S1 = s_1['t']
            s_2   = expDataDict2[ww]
            v_S2 = s_2['v']
            t_S2 = s_2['t']



            errPercentage = (v_S2-v_S1)/v_S1*100
            errPercentageSmooted = pd.rolling_mean(errPercentage,5)


            dictErrTmp['v']= errPercentage
            dictErrTmp['t']= t_S1
            dictErrTmp['comment'] = 'err % (vDR-vSDR)/vSDR'
            dictErr[ww] = dictErrTmp
            dictErrSmoothTmp['v']= errPercentageSmooted
            dictErrSmoothTmp['t']= t_S1
            dictErrSmoothTmp['comment'] = 'err % (vDR-vSDR)/vSDR smoothed'
            dictErrSmooth[ww] = dictErrSmoothTmp
            dictErrTmp = {}
            dictErrSmoothTmp = {}

        return dictErr,dictErrSmooth,


    def sortDict(self,dictExp,DAQSystem,kindSignals):
        # sort the keys of a dictionary using the string in idStr
        # returns:
        # the sorted keys of a dictionary in Input, the keys are sorted according the idStr which identify 2 subset of keys,
        # ex. CX and CY, or SX and SY [KC1D or KC1Z], or C1 and C5, or S1 and S1 [KC1E]
        # also return 2 posS1,posS2 vector containig the position of the sorted dictionary keys organized in a list
        dictListKeys = list()
        posS1 = []
        posS2 = []
        for ii in dictExp.keys():
            dictListKeys.append(ii)
        print('List Not Sorted: ')
        print(dictListKeys)
        dictListKeys.sort() # sorted
        # now I can search for CX and CY and they will be sorted
        print('List sorted')
        print(dictListKeys)
        if DAQSystem == 'KC1D' or DAQSystem == 'KC1Z':
            if kindSignals == 'PU':
                idStr1 = 'CX'
                idStr2 = 'CY'
            elif kindSignals == 'SL':
                idStr1 = 'SX'
                idStr2 = 'SY'
            elif kindSignals == 'TP':
                idStr1 = 'TP2'
                idStr2 = 'TP6'
            elif kindSignals == 'TN':
                idStr1 = 'TN2'
                idStr2 = 'TN6'
            elif kindSignals == 'P8AB':
                idStr1 = 'A'
                idStr2 = 'B'
            elif kindSignals == 'FL':
                idStr1 = 'FL'
                idStr2 = []
            elif kindSignals == 'UP':
                idStr1 = 'UP'
                idStr2 = []
            elif kindSignals == 'UN':
                idStr1 = 'UN'
                idStr2 = []
            elif kindSignals == 'IC':
                idStr1 = 'I8'
                idStr2 = []
            elif kindSignals == 'BSL':
                idStr1 = 'S3'
                idStr2 = 'S7'
            elif kindSignals == 'OPL':
                idStr1 = 'P4'
                idStr2 = 'P8'
            elif kindSignals == 'DSL':
                idStr1 = 'TS2'
                idStr2 = 'TS6'

        elif DAQSystem == 'KC1E':
            if kindSignals == 'PU':
                idStr1 = 'C1'
                idStr2 = 'C5'
            elif kindSignals == 'SL':
                idStr1 = 'S1'
                idStr2 = 'S5'
            elif kindSignals == 'FL':
                idStr1 = 'FL'
                idStr2 = []
            elif kindSignals == 'PP':
                idStr1 = 'PP4'
                idStr2 = 'PP8'
            elif kindSignals == 'PN':
                idStr1 = 'PN4'
                idStr2 = 'PN8'
            elif kindSignals == 'UPE':
                idStr1 = 'UP4E'
                idStr2 = 'UP8E'
            elif kindSignals == 'UNE':
                idStr1 = 'UN4E'
                idStr2 = 'UN8E'
            elif kindSignals == 'BSL':
                idStr1 = 'S7'
                idStr2 = []
            elif kindSignals == 'DTCP':
                idStr1 = 'TP1'
                idStr2 = 'TP2'
            elif kindSignals == 'DTCN':
                idStr1 = 'TN1'
                idStr2 = 'TN2'
            elif kindSignals == 'EVPU':
                idStr1 = 'EC3'
                idStr2 = 'EC8'
            elif kindSignals == 'EVH':
                idStr1 = 'EH3'
                idStr2 = 'EH8'
            elif kindSignals == 'EVFL':
                idStr1 = 'EFL3'
                idStr2 = 'EFL8'
            elif kindSignals == 'EVCH':
                idStr1 = 'EHC'
                idStr2 = 'ECC'

        if not idStr2:  # if  empty (only 1 set  per type, not replication in different octants)
            posS2 = []
            for jj in dictListKeys:
                if idStr1 in jj:
                    posS1.append(dictListKeys.index(jj))
                    #print(idStr1 + ' is in pos ' + str(dictListKeys.index(jj)))

        else:
            for jj in dictListKeys:
                if idStr1 in jj:
                    posS1.append(dictListKeys.index(jj))
                    #print(idStr1 + ' is in pos ' + str(dictListKeys.index(jj)))
                if idStr2 in jj:
                    posS2.append(dictListKeys.index(jj))
                    #print(idstr2 + ' is in pos ' + str(dictListKeys.index(jj)))

        print(posS1)
        print(posS2)
        return dictListKeys,posS1,posS2


    @staticmethod
    def plotSignalsShotsDAQ(dictMoreExpData,dictLessExpData,JPN,iRow,iColumn,\
               kindSignals,DAQmore,DAQless) :

        if DAQmore == 'KC1Z' or DAQless =='KC1Z':
            if kindSignals=='TP':
                idStr1 = 'TP2'
                idStr2 = 'TP6'
                nSens = 11
            if kindSignals=='TN':
                idStr1 = 'TN2'
                idStr2 = 'TN6'
                nSens = 11
            if kindSignals=='OPL':
                idStr1 = 'P4'
                idStr2 = 'P8'
                nSens = 7
            if kindSignals=='DSL':
                idStr1 = 'TS2'
                idStr2 = 'TS6'
                nSens = 12
        if DAQmore == 'KC1E' or DAQless =='KC1E':
            if kindSignals=='BSL':
                idStr1 = 'S3'
                idStr2 = 'S7'
                idPrefix = ['U','L','I','O']
                nSens = 4

        # one octant
        indexSubPlot = 0
        plt.figure()
        for jjj in range(0,nSens):
            if DAQmore == 'KC1E' or DAQless =='KC1E':
                nameSensOct1 = idStr1 + idPrefix[jjj]
            else:
                nameSensOct1 = idStr1 + str(jjj+1).zfill(2)

            #One DAQ
            signal1 = dictMoreExpData[nameSensOct1]
            v_signal1 = signal1['v']
            t_signal1 = signal1['t']

            if nameSensOct1 in dictLessExpData.keys():
                signal2 = dictLessExpData[nameSensOct1]
                v_signal2 = signal2['v']
                t_signal2 = signal2['t']
            else:
                v_signal2 = []
                t_signal2 = []


            # PLOT
            indexSubPlot = indexSubPlot + 1
            plt.subplot(iRow,iColumn,indexSubPlot)
            if not(len(v_signal1)==0):
                plt.plot(t_signal1,v_signal1,'r',label='#' + str(JPN) + ': (' + DAQmore + ') ' + nameSensOct1)
                plt.legend(loc='best',prop={'size':8})
                plt.xlabel('time[s]')
            if not(len(v_signal2)==0):
                plt.plot(t_signal2,v_signal2,'b',label='#' + str(JPN) + ': (' + DAQless + ') ' + nameSensOct1)
                plt.legend(loc='best',prop={'size':8})
                plt.xlabel('time[s]')
        plt.suptitle('compare DAQs: ' + DAQmore + ' vs ' + DAQless)
        plt.show()


        # other octant
        indexSubPlot = 0
        plt.figure()
        for jjj in range(0,nSens):
            if DAQmore == 'KC1E' or DAQless =='KC1E':
                nameSensOct2 = idStr2 + idPrefix[jjj]
            else:
                nameSensOct2 = idStr2 + str(jjj+1).zfill(2)

            #One DAQ
            signal1 = dictMoreExpData[nameSensOct2]
            v_signal1 = signal1['v']
            t_signal1 = signal1['t']

            if nameSensOct2 in dictLessExpData.keys():
                signal2 = dictLessExpData[nameSensOct2]
                v_signal2 = signal2['v']
                t_signal2 = signal2['t']
            else:
                v_signal2 = []
                t_signal2 = []


            # PLOT
            indexSubPlot = indexSubPlot + 1
            plt.subplot(iRow,iColumn,indexSubPlot)
            if not(len(v_signal1)==0):
                plt.plot(t_signal1,v_signal1,'r',label='#' + str(JPN) + ': (' + DAQmore + ') ' +  nameSensOct2)
                plt.legend(loc='best',prop={'size':8})
                plt.xlabel('time[s]')
            if not(len(v_signal2)==0):
                plt.plot(t_signal2,v_signal2,'b',label='#' + str(JPN) + ': (' + DAQless + ') ' + nameSensOct2)
                plt.legend(loc='best',prop={'size':8})
                plt.xlabel('time[s]')
        plt.suptitle('compare DAQs: ' + DAQmore + ' vs ' + DAQless)
        plt.show()


    @staticmethod
    def plotSignalsOctants(dictExpDataJPN,JPN,iRow,iColumn,kindSignals,DAQ):


        if DAQ == 'KC1Z':
            if kindSignals=='TP':
                idStr1 = 'TP2'
                idStr2 = 'TP6'
                nSens = 11
            if kindSignals=='TN':
                idStr1 = 'TN2'
                idStr2 = 'TN6'
                nSens = 11
            if kindSignals=='OPL':
                idStr1 = 'P4'
                idStr2 = 'P8'
                nSens = 7
            if kindSignals=='DSL':
                idStr1 = 'TS2'
                idStr2 = 'TS6'
                nSens = 12

        if  DAQ == 'KC1Z' or DAQ == 'KC1D':
            if kindSignals=='PU':
                idStr1 = 'CX'
                idStr2 = 'CY'
                nSens = 18
            if kindSignals=='SL':
                idStr1 = 'SX'
                idStr2 = 'SY'
                nSens = 18

        indexSubPlot = 0
        plt.figure()
        for jjj in range(0,nSens):
            nameSens1 = idStr1 + str(jjj+1).zfill(2)
            nameSens2 = idStr2 + str(jjj+1).zfill(2)

            if nameSens1 in dictExpDataJPN.keys():
                signal1 = dictExpDataJPN[nameSens1]
                v_signal1 = signal1['v']
                t_signal1 = signal1['t']
            else:
                v_signal1 = []
                t_signal1 = []

            if nameSens2 in dictExpDataJPN.keys():
                signal2 = dictExpDataJPN[nameSens2]
                v_signal2 = signal2['v']
                t_signal2 = signal2['t']
            else:
                v_signal2 = []
                t_signal2 = []


            # PLOT
            indexSubPlot = indexSubPlot + 1
            plt.subplot(iRow,iColumn,indexSubPlot)
            if not(len(v_signal1)==0):
                plt.plot(t_signal1,v_signal1,'r',label='#' + str(JPN) + ':' + nameSens1)
                plt.legend(loc='best',prop={'size':8})
                plt.xlabel('time[s]')
            if not(len(v_signal2)==0):
                plt.plot(t_signal2,v_signal2,'b',label='#' + str(JPN) + ':' + nameSens2)
                plt.legend(loc='best',prop={'size':8})
                plt.xlabel('time[s]')
        plt.suptitle('compare signals: ' + DAQ)
        plt.show()



    @staticmethod
    def plotSignalsDAQ(dictExpDataJPNTDR,dictListKeysSortedJPNTDR,pos_JPNTDR,\
                 dictExpDataJPNDR,dictListKeysSortedJPNDR,pos_JPNDR,\
                       iRow,iColumn,JPNtrustDryRun,JPNdryRun,DAQ):

        indexSubPlot = 0
        plt.figure()
        for iii, vvv in enumerate(pos_JPNTDR):
            signal1 = dictExpDataJPNTDR[dictListKeysSortedJPNTDR[pos_JPNTDR[iii]]]
            v_signal1 = signal1['v']
            t_signal1 = signal1['t']
            signal2 = dictExpDataJPNDR[dictListKeysSortedJPNDR[pos_JPNDR[iii]]]
            v_signal2 = signal2['v']
            t_signal2 = signal2['t']

            # PLOT
            indexSubPlot = indexSubPlot + 1
            plt.subplot(iRow,iColumn,indexSubPlot)
            plt.plot(t_signal1,v_signal1,'r',label='#' + str(JPNtrustDryRun) + ':' + dictListKeysSortedJPNTDR[pos_JPNTDR[iii]])
            plt.plot(t_signal2,v_signal2,'b',label='#' + str(JPNdryRun) + ':' + dictListKeysSortedJPNDR[pos_JPNDR[iii]])
            plt.legend(loc='best',prop={'size':8})
            plt.xlabel('time[s]')
        plt.suptitle('compare signals: ' + DAQ)
        plt.show()



    @staticmethod
    def setSubplotSignalsTable(DAQ,kindSignals):
                # set signals table and size subplot
        if DAQ == 'KC1D':
            nameSignalsTable_TF = 'signalsTable_TF_KC1D' # EVEN and ODD TF current

            if kindSignals == 'PU': # are 18 in total per octant
                iRow = 5
                iColumn = 4
                nameSignalsTable = 'signalsTable_KC1D_PU' # pick up field coils IDC OCT.3&7 from KC1D

            elif kindSignals == 'SL': # are 14 in total per octant
                iRow = 4
                iColumn = 4
                nameSignalsTable = 'signalsTable_KC1D_SL' # saddle loops OCT.3&7 from KC1D

            elif kindSignals == 'TP': # are 11 in total per octant per type normal or tangential
                iRow = 4
                iColumn = 3
                nameSignalsTable = 'signalsTable_KC1D_DCO_TP' # tangential divertor coils old (1996) OCT.3&7 from KC1D

            elif kindSignals == 'TN': # are 11 in total per octant per type normal or tangential
                iRow = 4
                iColumn = 3
                nameSignalsTable = 'signalsTable_KC1D_DCO_TN' # normal divertor coils old (1996) OCT.3&7 from KC1D

            elif kindSignals == 'P8AB': # are 7 in total per type A/B
                iRow = 4
                iColumn = 2
                nameSignalsTable = 'signalsTable_KC1D_P8AB' # poloidal limiter coils 1994-2001 OCT.8 from KC1D

            elif kindSignals == 'FL':
                iRow = 4
                iColumn = 2
                nameSignalsTable = 'signalsTable_KC1D_FL' # full flux loops from KC1D

            elif kindSignals == 'UP':
                iRow = 2
                iColumn = 2
                nameSignalsTable = 'signalsTable_KC1D_UP' # upper tangential coils from KC1D

            elif kindSignals == 'UN':
                iRow = 3
                iColumn = 2
                nameSignalsTable = 'signalsTable_KC1D_UN' # upper normal coils from KC1D

            elif kindSignals == 'IC':
                iRow = 3
                iColumn = 1
                nameSignalsTable = 'signalsTable_KC1D_IC' # inner coils from KC1D

            elif kindSignals == 'BSL':
                iRow = 2
                iColumn = 2
                nameSignalsTable = 'signalsTable_KC1D_BSL' # Big Saddle Loops from KC1D

            elif kindSignals == 'OPL':
                iRow = 4
                iColumn = 2
                nameSignalsTable = 'signalsTable_KC1D_OPL' # Outer Pol.Limiter from KC1D

            elif kindSignals == 'DSL':
                iRow = 4
                iColumn = 3
                nameSignalsTable = 'signalsTable_KC1D_DSL' # Divertor Saddle Loops from KC1D

            elif kindSignals == 'PP' or kindSignals == 'PN' or \
                 kindSignals == 'UPE' or  kindSignals == 'UNE' or \
                 kindSignals == 'DTCP' or  kindSignals == 'DTCN' or \
                 kindSignals == 'EVPU' or kindSignals == 'EVH' or \
                 kindSignals == 'EVFL' or kindSignals == 'EVCH':
                sys.exit(kindSignals + ' NOT in KC1D')


        elif DAQ == 'KC1Z' :
            nameSignalsTable_TF = 'signalsTable_TF_KC1Z' # EVEN and ODD TF current

            if kindSignals == 'PU': # are 18 in total per octant
                iRow = 5
                iColumn = 4
                nameSignalsTable = 'signalsTable_KC1Z_PU' # pick up field coils IDC OCT.3&7 from KC1Z

            elif kindSignals == 'SL': # are 14 in total per octant
                iRow = 4
                iColumn = 4
                nameSignalsTable = 'signalsTable_KC1Z_SL' # saddle loops OCT.3&7 from KC1Z

            elif kindSignals == 'TP': # are 11 in total per octant per type normal or tangential
                iRow = 4
                iColumn = 3
                nameSignalsTable = 'signalsTable_KC1Z_DCO_TP' # tangential divertor coils old (1996) OCT.3&7 from KC1Z

            elif kindSignals == 'TN': # are 11 in total per octant per type normal or tangential
                iRow = 4
                iColumn = 3
                nameSignalsTable = 'signalsTable_KC1Z_DCO_TN' # normal divertor coils old (1996) OCT.3&7 from KC1Z

            elif kindSignals == 'P8AB': # are 7 in total per type A/B
                iRow = 4
                iColumn = 2
                nameSignalsTable = 'signalsTable_KC1Z_P8AB' # poloidal limiter coils 1994-2001 OCT.8 from KC1Z

            elif kindSignals == 'FL':
                iRow = 4
                iColumn = 2
                nameSignalsTable = 'signalsTable_KC1Z_FL' # full flux loops from KC1Z

            elif kindSignals == 'UP':
                iRow = 2
                iColumn = 2
                nameSignalsTable = 'signalsTable_KC1Z_UP' # upper tangential coils from KC1Z

            elif kindSignals == 'UN':
                iRow = 3
                iColumn = 2
                nameSignalsTable = 'signalsTable_KC1Z_UN' # upper normal coils from KC1Z


            elif kindSignals == 'IC':
                iRow = 3
                iColumn = 1
                nameSignalsTable = 'signalsTable_KC1Z_IC' # inner coils from KC1Z

            elif kindSignals == 'BSL':
                iRow = 2
                iColumn = 2
                nameSignalsTable = 'signalsTable_KC1Z_BSL' # Big Saddle Loops from KC1Z

            elif kindSignals == 'OPL':
                iRow = 4
                iColumn = 2
                nameSignalsTable = 'signalsTable_KC1Z_OPL' # Outer Pol.Limiter from KC1Z

            elif kindSignals == 'DSL':
                iRow = 4
                iColumn = 3
                nameSignalsTable = 'signalsTable_KC1Z_DSL' # Divertor Saddle Loops from KC1Z

            elif kindSignals == 'PP' or kindSignals == 'PN' or \
                 kindSignals == 'UPE' or  kindSignals == 'UNE' or \
                 kindSignals == 'DTCP' or  kindSignals == 'DTCN' or \
                 kindSignals == 'EVPU' or kindSignals == 'EVH' or \
                 kindSignals == 'EVFL' or kindSignals == 'EVCH':
                sys.exit(kindSignals + ' NOT in KC1Z')


        elif DAQ == 'KC1E' :
            nameSignalsTable_TF = 'signalsTable_TF_KC1E' # EVEN and ODD TF current

            if kindSignals == 'PU': # are 18 in total per octant
                iRow = 5
                iColumn = 4
                nameSignalsTable = 'signalsTable_KC1E_PU' # pick up field coils IDC OCT.1&5 from KC1E

            elif kindSignals == 'SL': # are 14 in total per octant
                iRow = 4
                iColumn = 4
                nameSignalsTable = 'signalsTable_KC1E_SL' # saddle loops OCT.1&5 from KC1E

            elif kindSignals == 'FL': # are 6 in total per octant
                iRow = 4
                iColumn = 2
                nameSignalsTable = 'signalsTable_KC1E_FL' # full flux loops from KC1E, 6 in total

            elif kindSignals == 'PP': # are 8 in total per octant
                iRow = 4
                iColumn = 2
                nameSignalsTable = 'signalsTable_KC1E_PP' # full flux loops from KC1E

            elif kindSignals == 'PN': # are 8 in total per octant
                iRow = 4
                iColumn = 2
                nameSignalsTable = 'signalsTable_KC1E_PN' # full flux loops from KC1E

            elif kindSignals == 'UPE': # are 4 in total per octant
                iRow = 2
                iColumn = 2
                nameSignalsTable = 'signalsTable_KC1E_UPE' # enhanced upper tang.coils from KC1E

            elif kindSignals == 'UNE': # are 4 in total per octant
                iRow = 2
                iColumn = 2
                nameSignalsTable = 'signalsTable_KC1E_UNE' # enhanced upper tang.coils from KC1E

            elif kindSignals == 'BSL':
                iRow = 2
                iColumn = 2
                nameSignalsTable = 'signalsTable_KC1E_BSL' # Big Saddle Loops from KC1E

            elif kindSignals == 'DTCP':
                iRow = 4
                iColumn = 2
                nameSignalsTable = 'signalsTable_KC1E_DTCP' # Divertor Target Coils Tang. from KC1E

            elif kindSignals == 'DTCN':
                iRow = 4
                iColumn = 2
                nameSignalsTable = 'signalsTable_KC1E_DTCN' # Divertor Target Coils Normal from KC1E
            elif kindSignals == 'EVPU':
                iRow = 3
                iColumn = 1
                nameSignalsTable = 'signalsTable_KC1E_EVPU' # Ex Vessel Pick Up from KC1E
            elif kindSignals == 'EVH':
                iRow = 3
                iColumn = 1
                nameSignalsTable = 'signalsTable_KC1E_EVH' # Ex Vessel Hall probe from KC1E
            elif kindSignals == 'EVFL':
                iRow = 3
                iColumn = 1
                nameSignalsTable = 'signalsTable_KC1E_EVFL' # Ex Vessel Flux Loop from KC1E
            elif kindSignals == 'EVCH':
                iRow = 2
                iColumn = 1
                nameSignalsTable = 'signalsTable_KC1E_EVCH' # Ex Vessel PU and Hall from KC1E

            elif kindSignals == 'TP' or kindSignals == 'TN' or \
                 kindSignals == 'P8AB' or kindSignals == 'UP' or \
                 kindSignals == 'UN' or kindSignals == 'IC' or \
                 kindSignals == 'OPL' or  kindSignals == 'DSL' :
                sys.exit(kindSignals + ' NOT in KC1E')


        return iRow,iColumn,nameSignalsTable,nameSignalsTable_TF


# TF coefficients common functions

def downloadTFCoeff(JPN,signalPath):
    # _____download signals
    dictDataExp = {}
    strTmp = signalPath.split('-')
    signal = strTmp[1]
    # download TF coeff
    # TFE
    TFEpath = signalPath + '<TFE'
    TFEname = signal + '_TFE'
    dataTFE,nwds,title,units,ier = getdat.getsca(TFEpath,JPN)
    dictDataExp[TFEname] =  dataTFE
    print(TFEpath,' Downloaded !!')
    # TFO
    TFOpath = signalPath + '<TFO'
    TFOname = signal + '_TFO'
    dataTFO,nwds,title,units,ier = getdat.getsca(TFOpath,JPN)
    dictDataExp[TFOname] = dataTFO
    print(TFOpath,' Downloaded !!')

    return   dictDataExp


def TFCompensationSingleChannel(sRaw,TFECoeff,TFOCoeff,TFEVN,TFODD):
    # Toroidal field compensation of a raw signal ''Sraw''
    # using coefficients TFECoeff and TFOCoeff of the signal
    # and the even and odd TF current ITFEVN and ITFODD
    #
    # STFcomp = Sraw - TFECoeff*ITFEVN - TFOCoeff*ITFODD

    ITFEVN = TFEVN
    ITFODD = TFODD
    sTFcomp = sRaw - TFECoeff*ITFEVN - TFOCoeff*ITFODD
    return   sTFcomp



