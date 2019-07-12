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
import pdb
# from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing.pool import ThreadPool
import threading
import argparse
import pickle
import logging
from my_flush import *
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

# C-----------------------------------------------------------------------
# C set smoothing time
# C-----------------------------------------------------------------------

    if (code.lower() == 'kg1l'):

            tsmo = 0.025
    else:
            tsmo = 1.0e-4



    # ----------------------------
    data.code = code
    data.KG1_data = {}
    data.EFIT_data = {}

    data.KG1LH_data = KG1LData(data.constants)
    data.KG1LH_data1 = KG1LData(data.constants)
    data.KG1LH_data2 = KG1LData(data.constants)

    data.KG1LH_data.tsmo = tsmo


    logger.info('INIT DONE\n')

    # -------------------------------
    # 2. Read in KG1 data
    # -------------------------------
    data.KG1_data = Kg1PPFData(data.constants, data.pulse)

    success = data.KG1_data.read_data(data.pulse,
                                           read_uid=read_uid)

    # ::todo:
    # at least one channel has to be flagged/validated

    logger.log(5, 'success reading KG1 data?'.format(success))
    # -------------------------------
    # 2. Read in EFIT data
    # -------------------------------
    data.EFIT_data = EFITData(data.constants)
    data.EFIT_data.read_data(data.pulse)



    # -------------------------------
    # 3. Read in line of sights
    # -------------------------------
    logging.info('reading line of sights')
    temp, r_ref, z_ref, a_ref, r_coord, z_coord, a_coord, coord_err =data.KG1_data.get_coord(data.pulse)




    # -------------------------------
    # 4. map kg1v data onto efit time vector
    # -------------------------------


    channels=np.arange(0, 8) + 1

    kg1v92121_lid3,dummy = getdata(92121, 'KG1V', 'LID3')
    kg1l92121_lad3,dummy = getdata(92121, 'KG1l', 'LAD3')


    logger.info('start time loop')
    if data.code.lower()=='kg1l':
        ntefit = len(data.EFIT_data.rmag.time)
        Tefit  = data.EFIT_data.rmag.time
    else:
        ntefit = len(data.EFIT_data.rmag_fast.time)
        Tefit = data.EFIT_data.rmag_fast.time


    test=True
    # test=False
    if test:
    #
        # logger.info('start mapping kg1v data onto efit time vector')
        # start_time = time.time()
        # with Pool(10) as pool:
        #     results = pool.map(map_kg1_efit, [(data, chan) for chan in np.arange(0, 8) + 1])
        # logger.info("--- {}s seconds ---".format((time.time() - start_time)))
        # # pdb.set_trace()
        # for i,r in enumerate(results):
        #     data.KG1LH_data.lid[i+1] = SignalBase(data.constants)
        #     data.KG1LH_data.lid[i+1].time = r[0].KG1LH_data.lid[r[1]].time
        #     data.KG1LH_data.lid[i+1].data = r[0].KG1LH_data.lid[r[1]].data
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
            results = pool.map(map_kg1_efit_RM, [(data, chan) for chan in np.arange(0, 8) + 1])
        logger.info("--- {}s seconds ---".format((time.time() - start_time)))

        for i,r in enumerate(results):
            data.KG1LH_data1.lid[i+1] = SignalBase(data.constants)
            data.KG1LH_data1.lid[i+1].time = r[0].KG1LH_data.lid[r[1]].time
            data.KG1LH_data1.lid[i+1].data = r[0].KG1LH_data.lid[r[1]].data



    linewidth = 0.5
    markersize = 1

    logger.info("\n             loading data from pickle.\n")
    with open('./test_data.pkl',
              'rb') as f:
        [data.EFIT_data, data.KG1_data,
             data.KG1LH_data,data.KG1LH_data1] = pickle.load(f)
    f.close()

    logger.info('start mapping kg1v data onto efit time vector')
    start_time = time.time()
    with Pool(10) as pool:
        results = pool.map(map_kg1_efit_RM_pandas,
                           [(data, chan) for chan in np.arange(0, 8) + 1])
    logger.info("--- {}s seconds ---".format((time.time() - start_time)))

    for i, r in enumerate(results):
        data.KG1LH_data2.lid[i + 1] = SignalBase(data.constants)
        data.KG1LH_data2.lid[i + 1].time = r[0].KG1LH_data.lid[r[1]].time
        data.KG1LH_data2.lid[i + 1].data = r[0].KG1LH_data.lid[r[1]].data


    for chan in channels:

        plt.figure(chan)
        plt.plot(data.KG1_data.density[chan].time, data.KG1_data.density[chan].data,
                 label='kg1v_lid')

        # plt.plot(data.KG1LH_data.lid[chan].time, data.KG1LH_data.lid[chan].data,label='kg1l_lid_original_MT', marker = 'o', linestyle='-.', linewidth=linewidth,
        #                      markersize=markersize)

        plt.plot(data.KG1LH_data1.lid[chan].time, data.KG1LH_data1.lid[chan].data,label='kg1l_lid_rollingmean_MT', marker = 'x', linestyle=':', linewidth=linewidth,
                             markersize=markersize)

        plt.plot(data.KG1LH_data2.lid[chan].time, data.KG1LH_data2.lid[chan].data,label='kg1l_lid_rollingmean_pandas_MT', marker = 'x', linestyle=':', linewidth=linewidth,
                             markersize=markersize)


        plt.legend(loc='best',prop={'size':12})
    plt.show(block=True)

    logger.info("\n             dumping data to pickle.\n")
    with open('./test_data.pkl', 'wb') as f:
        pickle.dump(
            [data.EFIT_data, data.KG1_data,
             data.KG1LH_data,data.KG1LH_data1], f)
    f.close()


    pdb.set_trace()



    # psim1 = 1.00
    #
    # for  IT in range(0,ntefit):
    #     TIMEM=Tefit[IT]
    #     dtime=float(TIMEM)
    #     t,ier = flushinit(15, data.pulse, TIMEM, lunget=12, iseq=0, uid='JETPPF', dda='EFIT', lunmsg=0)
    #
    #     ier = Flush_getError(ier)
    #     if ier !=0:
    #         logger.error('flush error {}'.format(ier))
    #         return
    #
    #     # flusu2(nPsi, psi, npoint, npdim=0, work=None, jwork=None, lopt=2)
    #
    #     aa=Flush_getClosedFluxSurface(psim1, nPoints=360)
    #
    #     if iflsep ==0:
    #         logger.debug('Time {}s; NO X-point found')
    #         psimax=psim1
    #         iskb=1
    #     else:
    #         logger.debug('Time {}s; NO X-point plasma')
    #         if iflsep ==1:
    #             psimax=psim1
    #             if psisep[0] >=psim1:
    #                 iskb=1
    #             else:
    #                 iskb=0
    #                 growth = (psim1/psisep[0])-1
    #         else:
    #             psimax=psim1*psisep[0]
    #             iskb=0
    #             growth = psim1-1
    #
    #
    #
    #     Flush_getTangentsToSurfaces(r, z, flux, iside, accuracy, nBeams)







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
