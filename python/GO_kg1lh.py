#!/usr/bin/env python
"""
Class that runs CORMAT_py GUI
"""


# ----------------------------
__author__ = "Bruno Viola"
__Name__ = "KG1L_py"
__version__ = "0.1"
__release__ = "0"
__maintainer__ = "Bruno Viola"
__email__ = "bruno.viola@ukaea.uk"
__status__ = "Testing"
# __status__ = "Production"

from threading import Thread
from multiprocessing.pool import Pool
import matplotlib.pyplot as plt
# from scipy.signal import sosfiltfilt, butter
import pdb
# from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing.pool import ThreadPool
import threading
import argparse
import pickle
import logging
import math
from types import SimpleNamespace
from logging.handlers import RotatingFileHandler
from logging import handlers
import os
import pathlib
# from pickle import dump,load
import pickle
import platform
import datetime
import sys
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
from consts import Consts
# from find_disruption import find_disruption
from PyQt4 import QtCore, QtGui

from PyQt4.QtCore import *
from PyQt4.QtGui import *
from kg1_ppf_data import Kg1PPFData

from library import * #containing useful function
from efit_data import EFITData
from kg1l_data import KG1LData
# from mag_data import MagData
from matplotlib import gridspec
from matplotlib.backends.backend_qt4agg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.ticker import AutoMinorLocator
from ppf import *
from ppf_write import *
from signal_base import SignalBase
# from custom_formatters import MyFormatter,QPlainTextEditLogger,HTMLFormatter
# from support_classes import LineEdit,Key,KeyBoard,MyLocator
import inspect
import fileinput
import cProfile, pstats, io
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
import inspect
from  ppf import *
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from utility import *
import pandas as pd




sys.path.append('../../')
from eg_python_tools.my_flush import *
# qm = QtGui.QMessageBox
# qm_permanent = QtGui.QMessageBox
plt.rcParams["savefig.directory"] = os.chdir(os.getcwd())
myself = lambda: inspect.stack()[1][3]
logger = logging.getLogger(__name__)
# noinspection PyUnusedLocal

#--------
def map_kg1_efit_RM_pandas(arg):
    data = arg[0]
    chan = arg[1]

    if data.code.lower()=='kg1l':
        ntefit = len(data.EFIT_data.rmag.time)
        tefit = data.EFIT_data.rmag.time
        data_efit = data.EFIT_data.rmag.data
        data.EFIT_data.sampling_time = np.mean(np.diff(data.EFIT_data.rmag.time))

    else:
        ntefit = len(data.EFIT_data.rmag_fast.time)
        tefit = data.EFIT_data.rmag_fast.time
        data_efit = data.EFIT_data.rmag_fast.data
        data.EFIT_data.sampling_time = np.mean(np.diff(data.EFIT_data.rmag_fast.time))

    density = np.zeros(ntefit)
    ntkg1v = len(data.KG1_data.density[chan].time)
    tkg1v = data.KG1_data.density[chan].time
    tsmo = data.KG1LH_data.tsmo

    sampling_time_kg1v = np.mean(np.diff(tkg1v))

    rolling_mean=int(round(tsmo/sampling_time_kg1v))

    # density = pd.rolling_mean(data.KG1_data.density[chan].data,rolling_mean)
    density = pd.Series(data.KG1_data.density[chan].data).rolling(window=rolling_mean).mean()

    data.KG1LH_data.lid[chan] = SignalBase(data.constants)
    data.KG1LH_data.lid[chan].data = density
    data.KG1LH_data.lid[chan].time = data.KG1_data.density[chan].time

    return (data,chan)



#--------
def map_kg1_efit_RM(arg):
    data = arg[0]
    chan = arg[1]

    if data.code.lower()=='kg1l':
        ntefit = len(data.EFIT_data.rmag.time)
        tefit = data.EFIT_data.rmag.time
        data_efit = data.EFIT_data.rmag.data
        data.EFIT_data.sampling_time = np.mean(np.diff(data.EFIT_data.rmag.time))

    else:
        ntefit = len(data.EFIT_data.rmag_fast.time)
        tefit = data.EFIT_data.rmag_fast.time
        data_efit = data.EFIT_data.rmag_fast.data
        data.EFIT_data.sampling_time = np.mean(np.diff(data.EFIT_data.rmag_fast.time))

    density = np.zeros(ntefit)
    ntkg1v = len(data.KG1_data.density[chan].time)
    tkg1v = data.KG1_data.density[chan].time
    tsmo = data.KG1LH_data.tsmo

    sampling_time_kg1v = np.mean(np.diff(tkg1v))

    rolling_mean=int(round(tsmo/sampling_time_kg1v))

    # cumsum_vec = np.cumsum(np.insert(data.KG1_data.density[chan].data, 0, 0))
    # density = (cumsum_vec[rolling_mean:] - cumsum_vec[:-rolling_mean]) / rolling_mean
    density = movingaverage(data.KG1_data.density[chan].data, rolling_mean)



    data.KG1LH_data.lid[chan] = SignalBase(data.constants)
    data.KG1LH_data.lid[chan].data = density
    data.KG1LH_data.lid[chan].time = data.KG1_data.density[chan].time

    return (data,chan)


#--------
def map_kg1_efit(arg):
    data = arg[0]
    chan = arg[1]

    if data.code.lower()=='kg1l':
        ntefit = len(data.EFIT_data.rmag.time)
        tefit = data.EFIT_data.rmag.time
        data_efit = data.EFIT_data.rmag.data
        data.EFIT_data.sampling_time = np.mean(np.diff(data.EFIT_data.rmag.time))

    else:
        ntefit = len(data.EFIT_data.rmag_fast.time)
        tefit = data.EFIT_data.rmag_fast.time
        data_efit = data.EFIT_data.rmag_fast.data
        data.EFIT_data.sampling_time = np.mean(np.diff(data.EFIT_data.rmag_fast.time))

    density = np.zeros(ntefit)
    ntkg1v = len(data.KG1_data.density[chan].time)
    tkg1v = data.KG1_data.density[chan].time
    tsmo = data.KG1LH_data.tsmo



    for it in range(0, ntefit):
        # pdb.set_trace()
        sum = np.zeros(8)

        nsum = 0

        tmin = 1000.0

        jmin = 1

        # in principle they can be different (?!)
        ntkg1v = len(data.KG1_data.density[chan].time)
        tkg1v = data.KG1_data.density[chan].time

        for jj in range(0, ntkg1v):
            tdif = abs(tkg1v[jj] - tefit[it])

            if (tdif < tmin):
                tmin = tdif
                jmin = jj
            if (tkg1v[jj] >= tefit[it] + tsmo):
                break
            if (tkg1v[jj] > tefit[it] - tsmo):
                sum[chan - 1] = sum[chan - 1] + \
                                data.KG1_data.density[chan].data[jj]
                nsum = nsum + 1
        if nsum > 0:
            density[it] = sum[chan - 1] / float(nsum)
        else:
            density[it] = data.KG1_data.density[chan].data[jmin]

    data.KG1LH_data.lid[chan] = SignalBase(data.constants)
    data.KG1LH_data.lid[chan].data = density
    data.KG1LH_data.lid[chan].time = data.EFIT_data.rmag.time

    return (data,chan)



# ----------------------------
#--------
def compute_len_lad_xtan(arg):
    data = arg[0]
    chan = arg[1]

    length = np.empty(len(data.KG1LH_data.lid[chan].time))
    xtan = np.empty(len(data.KG1LH_data.lid[chan].time))
    lad = np.empty(len(data.KG1LH_data.lid[chan].time))


    if data.code.lower()=='kg1l':
        ntefit = len(data.EFIT_data.rmag.time)
        tefit = data.EFIT_data.rmag.time
        data_efit = data.EFIT_data.rmag.data
        time_efit = data.EFIT_data.rmag.time
        data.EFIT_data.sampling_time = np.mean(np.diff(data.EFIT_data.rmag.time))

    else:
        ntefit = len(data.EFIT_data.rmag_fast.time)
        tefit = data.EFIT_data.rmag_fast.time
        data_efit = data.EFIT_data.rmag_fast.data
        time_efit = data.EFIT_data.rmag_fast.time
        data.EFIT_data.sampling_time = np.mean(np.diff(data.EFIT_data.rmag_fast.time))

    xpt = data.r_coord[chan - 1]
    ypt = data.z_coord[chan - 1]
    delta = math.atan2(ypt,xpt)





    plt.figure()

    for IT in range(0, ntefit):
            TIMEM = tefit[IT]
            dtime = float(TIMEM)

            t, ier = flushinit(15, data.pulse, TIMEM, lunget=12, iseq=0,
                               uid='JETPPF', dda='EFIT', lunmsg=0)

            t,ier = flushquickinit(data.pulse,TIMEM)
            # print('\n')
            logger.log(5,'************* Time = {}s'.format(TIMEM))
            # logger.debug('************* Time = {}s'.format(t))
            #ier = Flush_getError(ier)
            #if ier !=0:
            #    logger.error('flush error {}'.format(ier))
            #    return

            # flusu2(nPsi, psi, npoint, npdim=0, work=None, jwork=None, lopt=2)

            r,z,ier  = Flush_getClosedFluxSurface(data.psim1, nPoints=360)
            r[:] = [x/100 for x in r]
            z[:] = [x/100 for x in z]
            # flux, ier = Flush_getlcfsFlux()
            # logger.debug('lcfsFlux is {}'.format(flux))
            plt.scatter(r,z,c='b')


            if ier != 0:
                logger.error('flush error {} in Flush_getClosedFluxSurface'.format(ier))
                return ier
            # look for xpoint
            iflsep, rx, zx, fx, ier = flush_getXpoint()
            # iflsep_all, rx_all, zx_all, fx_all, ier_all = flush_getAllXpoint()
            if ier != 0:
                logger.error('flush error {} in flush_getXpoint'.format(ier))
                return ier
            logger.log(5,'Time {}s; iflsep {}; rx {}; zx {}; fx {}; ier {} '.format(TIMEM,iflsep, rx, zx, fx, ier))



            
            if int(iflsep) == 0:
                logger.log(5,'iflsep is {}'.format(iflsep))
                logger.debug('Time {}s; NO X-point found'.format(TIMEM))
                psimax = data.psim1
                logger.log(5,'psimax is {}'.format(psimax))
                iskb = 1
                logger.log(5,'psimax is {}'.format(psimax))
            else:
                # logger.log(5,'iflsep is {}'.format(iflsep))
                logger.debug('Time {}s; X-point plasma'.format(TIMEM))
                # if int(iflsep) == 1:
                #
                #      logger.log(5,'found {} X-point '.format(iflsep))
                #      psimax = data.psim1
                #      logger.log(5,'psimax is {}'.format(psimax))
                #      if fx[0] >= data.psim1:
                #           logger.log(5,'fx is {}'.format(fx))
                #           iskb = 1
                #           logger.log(5,'iskb is {}'.format(iskb))
                #      else:
                #           iskb = 0
                #           logger.log(5,'iskb is {}'.format(iskb))
                #           growth = (data.psim1 / fx[0]) - 1
                #           logger.log(5,'growth is {}'.format(growth))
                # else:

                psimax = data.psim1 * fx[0]
                # psimax = 1

                iskb = 0

                # growth = data.psim1 - 1
                growth = fx[0] - 1

                logger.log(5, 'psimax is {}'.format(psimax))
                logger.log(5,'growth is {}'.format(growth))
                logger.log(5, 'iflsep is {}'.format(iflsep))
                logger.log(5, 'iskb is {}'.format(iskb))


                
                
                
            volume_m3, ier = Flush_getVolume(1)

            if ier != 0:
                logger.error('flush error {} in Flush_getVolume'.format(ier))
                return ier


            #
            if iskb != 1:
                logger.log(5,'iskb is {}'.format(iskb))
                ier = flush_blowUp(growth, volume_m3)
                logger.log(5,'blowup error {}'.format(ier))


            #
            # - ----------------------------------------------------------------------
            #             BEGIN            SCANNING             KG1             CHORDS
            # - ----------------------------------------------------------------------


            # -----------------------------------------------------------------------
            # FIND PSI AT TANGENT FLUX SURFACE (to make FLUL2 quicker)
            # -----------------------------------------------------------------------
            # angle, rTan, zTan, ier = Flush_getTangentsToSurfaces(xpt, ypt, fx[0], 1, data.EPSDD, 8)
            # logger.log(5,
            #     'get tangent to surfaces output is angle {} rTan {}, zTan {}'.format(
            #         angle, rTan, zTan))
            #
            # if ier != 0:
            #     logger.error('flush error {} in getTangentsToSurfaces'.format(ier))
            #     return
            # rTan1,zTan1, fTan1, ier  = Flush_GetTangentFlux(rTan,zTan,angle, data.EPSDD)
            rTan1,zTan1, fTan1, ier  = Flush_GetTangentFlux(xpt,ypt,delta, data.EPSDD)
            if ier != 0:
                logger.error('flush error {} in getTangentsToSurfaces'.format(ier))
                return ier
            #
            # if int(iflsep) != 0:
            #     print('\n')
            #     logger.debug('psimax is {}'.format(psimax))
            #     logger.debug('volume is {}'.format(volume_m3))
            #     logger.debug('growth is {}'.format(growth))
            #     logger.debug('iskb is {}'.format(iskb))
            #     logger.debug('fx[0] is {}'.format(fx[0]))
            #     # logger.debug('flux is {}'.format(flux))
            #     # plt.show()
            #     pdb.set_trace()


            # rTan, zTan, fTan, ier  = Flush_GetTangentFlux(xpt,ypt,delta, data.EPSDD)

            logger.log(5,'get tangent flux output is rTan {}, zTan {}, fTan {}'.format(rTan1, zTan1, fTan1))
            # logger.debug('get tangent flux output is rTan {}, zTan {}, fTan {}'.format(rTan, zTan, fTan))


            if ier != 0:
                logger.error('flush error {}  in FlushGetTangentFlux'.format(ier))
                return ier

            #
            # ----------------------------------------------------------------------
            # FIND INTERSECTION POINTS WITH PLASMA BOUNDARY
            # ----------------------------------------------------------------------

            NPSI = 1  # look for one surface
            nfound, r1, z1, r2, z2, r3, z3, r4, z4, ier  = Flush_getIntersections(xpt, ypt, delta, data.EPSF,NPSI, fx[0])
            if ier != 0:
                logger.error('flush error {}  in Flush_getIntersections'.format(ier))
                return ier
            cord=math.hypot(r2-r1,z2-z1)


            logger.log(5,'found {} intersection/s'.format(nfound))


    # -----------------------------------------------------------------------
    # final results
    # -----------------------------------------------------------------------
            if cord <0:
               cord =abs(cord)
            length[IT] = cord/100.0 # conversion from cm to m
            # logger.log(5, 'cord length for channel {} is {}'.format(chan, cord))
            logger.debug('cord length for channel {} is {}'.format(chan, length[IT]))
            # length[IT] = cord # conversion from cm to m
            if ( length[IT] >  0.0):
               lad[IT] = data.KG1LH_data.lid[chan].data[IT]/length[IT]
            else:
               lad[IT] = 0.0

            xtan[IT] = fTan1

    #
            data.KG1LH_data.lad[chan] = SignalBase(data.constants)
            data.KG1LH_data.lad[chan].data = lad
    #        data.KG1LH_data.lad[chan].time = data.EFIT_data.rmag.time
            data.KG1LH_data.lad[chan].time = data.KG1_data.density[chan].time
    #
            data.KG1LH_data.len[chan] = SignalBase(data.constants)
            data.KG1LH_data.len[chan].data = length
            # data.KG1LH_data.lad[chan].time = data.EFIT_data.rmag.time
            data.KG1LH_data.len[chan].time = time_efit
    #
            data.KG1LH_data.xta[chan] = SignalBase(data.constants)
            data.KG1LH_data.xta[chan].data = xtan
             # data.KG1LH_data.lad[chan].time = data.EFIT_data.rmag.time
            data.KG1LH_data.xta[chan].time = time_efit
    #
    return (data, chan)


# ----------------------------



def main(shot_no, code,read_uid, write_uid, test=False):
    '''
    Program to calculate the line averaged density for all channels of
    C kg1v. Other outputs are the tangent flux surface
    C and the distance to the edge divided by the chord length (curvature
    C of the edge for channel 4).
    :param shot_no:
    :param code:
    :param read_uid:
    :param write_uid:
    :param test:
    :return:
    '''

    data = SimpleNamespace()
    data.pulse = shot_no
    channels=np.arange(0, 8) + 1

    # C-----------------------------------------------------------------------
    # C constants
    # C-----------------------------------------------------------------------

    data.psim1 = 1.00
    data.EPSDD = 0.1                           # accuracy for flul1
    data.EPSF = 0.00001  # value recommended in FLUSH documentation

    # C-----------------------------------------------------------------------
    # C set smoothing time
    # C-----------------------------------------------------------------------

    if (code.lower() == 'kg1l'):

        tsmo = 0.025
    else:
        tsmo = 1.0e-4

    # ----------------------------
    #

# C-----------------------------------------------------------------------
# C init
# C-----------------------------------------------------------------------


    logger.info('\tStart KG1L/H \n')
    logger.info(
        '\t {} \n'.format(datetime.datetime.today().strftime('%Y-%m-%d')))
    cwd = os.getcwd()
    workfold = cwd
    home = cwd
    parent = Path(home)
    if "USR" in os.environ:
        logger.log(5, 'USR in env')
        # owner = os.getenv('USR')
        owner = os.getlogin()
    else:
        logger.log(5, 'using getuser to authenticate')
        import getpass
        owner = getpass.getuser()
    logger.log(5, 'this is your username {}'.format(owner))
    homefold = os.path.join(os.sep, 'u', owner)
    logger.log(5, 'this is your homefold {}'.format(homefold))
    home = str(Path.home())
    chain1 = '/common/chain1/kg1/'
    if code.lower()=='kg1l':
        extract_history(
            workfold + '/run_out.txt',
            chain1 + 'kg1l_out.txt')
    else:
        extract_history(
            workfold + '/run_out.txt',
            chain1 + 'kg1h_out.txt')


    logger.info(' copying to local user profile \n')
    logger.log(5, 'we are in %s', cwd)

    # -------------------------------
    # Read  config data.
    # -------------------------------
    logger.info(" Reading in constants. \n")
    # test_logger()
    # raise SystemExit

    try:
        data.constants = Consts("consts.ini", __version__)
    except KeyError:
        logger.error(" Could not read in configuration file consts.ini")
        sys.exit(65)

    read_uis = []
    for user in data.constants.readusers.keys():
        user_name = data.constants.readusers[user]
        read_uis.append(user_name)

    # -------------------------------
    # list of option to write ppf for current user
    # -------------------------------
    write_uis = []
    # -------------------------------
    # check if owner is in list of authorised users
    # -------------------------------
    if owner in read_uis:
        logger.info(
            'user {} authorised to write public PPF \n'.format(owner))
        write_uis.insert(0, 'JETPPF')  # jetppf first in the combobox list
        write_uis.append(owner)
        # users.append('chain1')
    else:
        logger.info(
            'user {} NOT authorised to write public PPF\n'.format(owner))
        write_uis.append(owner)


    data.code = code
    data.KG1_data = {}
    data.EFIT_data = {}

    data.temp, data.r_ref, data.z_ref, data.a_ref, data.r_coord, data.z_coord, data.data_coord, data.coord_err = [[],[],[],[],[],[],[],[]]



    data.KG1LH_data = KG1LData(data.constants)
    data.KG1LH_data1 = KG1LData(data.constants)
    data.KG1LH_data2 = KG1LData(data.constants)

    data.KG1LH_data.tsmo = tsmo


    logger.debug('checking pulse number')
    maxpulse = pdmsht()
    if (data.pulse > pdmsht()):
        logger.error('try a pulse lower than {} '.format(maxpulse))
        return 71


    logger.info('INIT DONE\n')
    
    

    # -------------------------------
    # 2. Read in KG1 data
    # -------------------------------
    data.KG1_data = Kg1PPFData(data.constants, data.pulse)
    
    

    success = data.KG1_data.read_data(data.pulse,
                                           read_uid=read_uid)
                                           
    if success is False:
        logger.error('error reading KG1 data')
        return 20										

    
    # at least one channel has to be flagged/validated
    status_flags=[]
    pulseok=True
    for chan in data.KG1_data.global_status.keys():
        status_flags.append(data.KG1_data.global_status[chan])
    for item in status_flags:
        if item in [1,2,3]:
            pulseok=False
            break
        else:
                pass
    
    if pulseok:
            logger.error('No validated LID channels in KG1V')
            return 9
            
    
    
    

    # -------------------------------
    # 2. Read in EFIT data
    # -------------------------------
    data.EFIT_data = EFITData(data.constants)
    ier = data.EFIT_data.read_data(data.pulse)
    
    if ier !=0:
        logger.error('error reading EFIT data')
        return 30



    # -------------------------------
    # 3. Read in line of sights
    # -------------------------------
    logging.info('reading line of sights')
    data.temp, data.r_ref, data.z_ref, data.a_ref, data.r_coord, data.z_coord, data.a_coord, data.coord_err =data.KG1_data.get_coord(data.pulse)
	
    if data.coord_err !=0:
        logger.error('error reading cords coordinates')
        return 22
    


    # -------------------------------
    # 4. map kg1v data onto efit time vector
    # -------------------------------


    


    if data.code.lower()=='kg1l':
        ntefit = len(data.EFIT_data.rmag.time)
        Tefit  = data.EFIT_data.rmag.time
    else:
        ntefit = len(data.EFIT_data.rmag_fast.time)
        Tefit = data.EFIT_data.rmag_fast.time


    logger.info("\n             loading data from pickle.\n")
    # with open('./test_data.pkl',
    with open('./test_data_kg1l.pkl',
              'rb') as f:
        [data.EFIT_data, data.KG1_data,
             data.KG1LH_data,data.KG1LH_data1,data.KG1LH_data2] = pickle.load(f)
    f.close()


    # test=True
    test=False
    if test:
    #
        logger.info('start mapping kg1v data onto efit time vector')
        start_time = time.time()
        with Pool(10) as pool:
            results = pool.map(map_kg1_efit, [(data, chan) for chan in channels])
        logger.info("--- {}s seconds ---".format((time.time() - start_time)))
        # pdb.set_trace()
        for i,r in enumerate(results):
            data.KG1LH_data.lid[i+1] = SignalBase(data.constants)
            data.KG1LH_data.lid[i+1].time = r[0].KG1LH_data.lid[r[1]].time
            data.KG1LH_data.lid[i+1].data = r[0].KG1LH_data.lid[r[1]].data
        #
        # #
        ## logger.info('start single thread')
        ## start_time = time.time()
        ## output=map_kg1_efit((data, 1))
        ## logger.info("--- {}s seconds ---".format((time.time() - start_time)))
        ## # plt.figure()
        ## plt.plot(output[0].KG1LH_data.lid[output[1]].time, output[0].KG1LH_data.lid[output[1]].data,label='original_ST')
        ##
        ## logger.info('start single thread RM')
        ## start_time = time.time()
        ## output = map_kg1_efit_RM((data, 1))
        ## logger.info("--- {}s seconds ---".format((time.time() - start_time)))
        ## plt.figure()
        ## plt.plot(output[0].KG1LH_data.lid[output[1]].time, output[0].KG1LH_data.lid[output[1]].data,label='rolling_mean_ST')
        # #

        #
        logger.info('start mapping kg1v data onto efit time vector')
        start_time = time.time()
        with Pool(10) as pool:
            results = pool.map(map_kg1_efit_RM, [(data, chan) for chan in channels])
        logger.info("--- {}s seconds ---".format((time.time() - start_time)))

        for i,r in enumerate(results):
            data.KG1LH_data1.lid[i+1] = SignalBase(data.constants)
            data.KG1LH_data1.lid[i+1].time = r[0].KG1LH_data.lid[r[1]].time
            data.KG1LH_data1.lid[i+1].data = r[0].KG1LH_data.lid[r[1]].data

        logger.info('start mapping kg1v data onto efit time vector')
        start_time = time.time()
        with Pool(10) as pool:
            results = pool.map(map_kg1_efit_RM_pandas,
                               [(data, chan) for chan in channels])
        logger.info("--- {}s seconds ---".format((time.time() - start_time)))

        for i, r in enumerate(results):
            data.KG1LH_data2.lid[i + 1] = SignalBase(data.constants)
            data.KG1LH_data2.lid[i + 1].time = r[0].KG1LH_data.lid[r[1]].time
            data.KG1LH_data2.lid[i + 1].data = r[0].KG1LH_data.lid[r[1]].data

    # logger.info("\n             dumping data to pickle.\n")
    # with open('./test_data.pkl', 'wb') as f:
    #     pickle.dump(
    #         [data.EFIT_data, data.KG1_data,
    #          data.KG1LH_data,data.KG1LH_data1,data.KG1LH_data2], f)
    # f.close()


    # logger.info('start time loop')
    # start_time = time.time()
    # with Pool(10) as pool:
    #     results = pool.map(compute_len_lad_xtan,
    #                        [(data, chan) for chan in channels])
    # logger.info("--- {}s seconds ---".format((time.time() - start_time)))
    # # pdb.set_trace()
    # for i, r in enumerate(results):
    #     data.KG1LH_data.lad[i + 1] = SignalBase(data.constants)
    #     data.KG1LH_data.lad[i + 1].time = r[0].KG1LH_data.lid[r[1]].time
    #     data.KG1LH_data.lad[i + 1].data = r[0].KG1LH_data.lid[r[1]].data
    #     data.KG1LH_data.len[i + 1] = SignalBase(data.constants)
    #     data.KG1LH_data.len[i + 1].time = r[0].KG1LH_data.len[r[1]].time
    #     data.KG1LH_data.len[i + 1].data = r[0].KG1LH_data.len[r[1]].data
    #     data.KG1LH_data.xta[i + 1] = SignalBase(data.constants)
    #     data.KG1LH_data.xta[i + 1].time = r[0].KG1LH_data.xta[r[1]].time
    #     data.KG1LH_data.xta[i + 1].data = r[0].KG1LH_data.xta[r[1]].data
    #
    #
    #
    # logger.info("\n             dumping data to pickle.\n")
    # with open('./test_data_kg1l.pkl', 'wb') as f:
    #     pickle.dump(
    #         [data.EFIT_data, data.KG1_data,
    #          data.KG1LH_data,data.KG1LH_data1,data.KG1LH_data2], f)
    # f.close()

    # raise SystemExit
    chan =5
    # compute_len_lad_xtan((data,chan))
    linewidth = 0.5
    markersize = 1


    xpt = data.r_coord[chan - 1]
    ypt = data.z_coord[chan - 1]
    delta = math.atan2(ypt,xpt)


    plot_point((xpt,ypt),delta,30,str(chan))



    plt.show()
    raise SystemExit







    plot = False
    # plot = True

    if plot:
        for chan in channels:

            plt.figure(chan)
            plt.plot(data.KG1_data.density[chan].time, data.KG1_data.density[chan].data,
                     label='kg1v_lid')

            plt.plot(data.KG1LH_data.lid[chan].time, data.KG1LH_data.lid[chan].data,label='kg1l_lid_original_MT', marker = 'o', linestyle='-.', linewidth=linewidth,
                                  markersize=markersize)

            plt.plot(data.KG1LH_data1.lid[chan].time, data.KG1LH_data1.lid[chan].data,label='kg1l_lid_rollingmean_MT', marker = 'x', linestyle=':', linewidth=linewidth,
                                 markersize=markersize)

            plt.plot(data.KG1LH_data2.lid[chan].time, data.KG1LH_data2.lid[chan].data,label='kg1l_lid_rollingmean_pandas_MT', marker = 'x', linestyle=':', linewidth=linewidth,
                                 markersize=markersize)

            plt.plot(data.KG1LH_data.lad[chan].time, data.KG1LH_data.lad[chan].data,label='kg1l_lad_original_MT', marker = 'x', linestyle=':', linewidth=linewidth,
                                 markersize=markersize)



            plt.legend(loc='best',prop={'size':12})

    kg1l_lid3, dummy = getdata(shot_no, 'KG1l', 'LID3')
    kg1l_lad3, dummy = getdata(shot_no, 'KG1l', 'LAD3')
    kg1l_len3, dummy = getdata(shot_no, 'KG1l', 'LEN3')
    kg1l_xtan3, dummy = getdata(shot_no, 'KG1l', 'xta3')


    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(kg1l_lid3['time'],kg1l_lid3['data'],label='xtan_jetppf')
    plt.plot(data.KG1LH_data.lid[chan].time, data.KG1LH_data.lid[chan].data,
             label='kg1l_lid_original_MT', marker='o', linestyle='-.',
             linewidth=linewidth,
             markersize=markersize)
    plt.legend(loc=0, prop={'size': 8})
    #
    #

    plt.subplot(4, 1, 2)
    plt.plot(kg1l_lad3['time'], kg1l_lad3['data'], label='lad_jetppf')
    plt.plot(data.KG1LH_data.lad[chan].time, data.KG1LH_data.lad[chan].data,
             label='kg1l_lad_original_MT', marker='x', linestyle='-.',
             linewidth=linewidth,
             markersize=markersize)
    plt.legend(loc=0, prop={'size': 8})



    plt.subplot(4, 1, 3)
    plt.plot(kg1l_xtan3['time'],kg1l_xtan3['data'],label='xtan_jetppf')
    plt.plot(data.KG1LH_data.xta[chan].time, data.KG1LH_data.xta[chan].data,
             label='kg1l_xtan_original_MT', marker='o', linestyle='-.',
             linewidth=linewidth,
             markersize=markersize)

    plt.legend(loc=0, prop={'size': 8})


    plt.subplot(4, 1, 4)
    plt.plot(kg1l_len3['time'], kg1l_len3['data'], label='len_jetppf')
    plt.plot(data.KG1LH_data.len[chan].time, data.KG1LH_data.len[chan].data,
             label='kg1l_len_original_MT', marker='x', linestyle='-.',
             linewidth=linewidth,
             markersize=markersize)
    plt.legend(loc=0, prop={'size': 8})



    plt.show(block=True)



    if plot:
        plt.show(block=True)






    # pdb.set_trace()
    logger.info("\n             Finished.\n")
#     return c
# 8 No validated data in KG1V
# 9 No validated LID channels in KG1V
# 20 No KG1V data
# 21 Could not read SF for KG1V
# 22 Error reading KG1V line-of-sight data
# 30 No EFIT data
# 31 No points in EFIT
# 70 Error in PDLPPF
# 71 Invalid shot number
# 72 No PPF exists for shot
# 73 NDDA too small on input
# 74 NSEQ too small on input

# KG1H
# 8 No validated data in KG1V
# 9 No validated LID channels in KG1V
# 20 No KG1V data
# 21 Could not read SF for KG1V
# 22 Error reading KG1V line-of-sight data
# 30 No EFIT data
# 31 No points in EFIT
# 70 Error in PDLPPF
# 71 Invalid shot number
# 72 No PPF exists for shot
# 73 NDDA too small on input
# 74 NSEQ too small on input
if __name__ == "__main__":

    # Ensure we are running python 3
    assert sys.version_info >= (
    3, 5), "Python version too old. Please use >= 3.5.X."

    # Parse arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pulse", type=int, help="Pulse number to run.",
                        required=True)
    parser.add_argument("-c", "--code", help="code to run.",
                        default="KG1L")
    parser.add_argument("-r", "--uid_read", help="UID to read PPFs from.",
                        default="JETPPF")
    parser.add_argument("-u", "--uid_write", help="UID to write PPFs to.",
                        default="")
    parser.add_argument("-d", "--debug", type=int,
                        help="Debug level. 0: Error, 1: Warning, 2: Info, 3: Debug, 4: Debug Plus",
                        default=2)
    parser.add_argument("-t", "--test",
                        help="""Run in test mode. In this mode code will run and if -uw=JETPPF then no PPF will be written, to avoid
                            over-writing validated data.""", default=False)


    args = parser.parse_args(sys.argv[1:])

    # Setup the logger
    debug_map = {0: logging.ERROR,
                 1: logging.WARNING,
                 2: logging.INFO,
                 3: logging.DEBUG,
                 4: 5}

    logging.basicConfig(level=debug_map[args.debug])

    logging.addLevelName(5, "DEBUG_PLUS")

    logger = logging.getLogger(__name__)
    # logging.basicConfig(
    #     # format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    #     format='%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    #     # datefmt='%d-%m-%Y:%H:%M:%S',
    #     level=logging.DEBUG)
    if args.uid_write == "":
        logger.warning("No write UID was given: no PPF will be written.")



    # Call the main code
    main(args.pulse, args.code,args.uid_read, args.uid_write, test=args.test)
