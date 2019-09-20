#!/usr/bin/env python
"""
Class that runs CORMAT_py GUI
"""


# ----------------------------
__author__ = "Bruno Viola"
__Name__ = "KG1L_py"
__version__ = "1.0"
__release__ = "1.0"
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
import inspect
from  ppf import *
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from utility import *
import pandas as pd
import gc





# sys.path.append('../../')
# from eg_python_tools.my_flush import *
from my_flush import *
# qm = QtGui.QMessageBox
# qm_permanent = QtGui.QMessageBox
plt.rcParams["savefig.directory"] = os.chdir(os.getcwd())
myself = lambda: inspect.stack()[1][3]
logger = logging.getLogger(__name__)
# noinspection PyUnusedLocal

#--------
import ast

def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v


#--------
def time_loop(arg):
    """
    computes time loop on efit time base.
    calls flush every step to initialise, get x-point flux, get intersections with Line of sight, get tangent flux to line of sight



    :param arg: 
    :return: 
    """
    data = arg[0] # struct containing all data
    chan = arg[1] # channel to analyse

    if data.KG1_data.global_status[chan] >3:
        logger.warning('channel data is not good - skipping ch. {}!'.format(chan))
        return (data,chan)

    if data.code.lower()=='kg1l':
        ntime_efit = len(data.EFIT_data.rmag.time)
        time_efit = data.KG1LH_data.lid[chan].time
        data_efit = data.EFIT_data.rmag.data
        ntefit = len(time_efit)
        data.EFIT_data.sampling_time = np.mean(np.diff(data.EFIT_data.rmag.time))
        ntkg1v = len(data.KG1_data.density[chan].time)
        tkg1v = data.KG1_data.density[chan].time
        sampling_time_kg1v = np.mean(np.diff(tkg1v))
        tsmo = data.KG1LH_data.tsmo
        rolling_mean = int(round(tsmo/sampling_time_kg1v))

    else:
        ntime_efit = len(data.EFIT_data.rmag_fast.time)
        time_efit = data.KG1LH_data.lid[chan].time
        data_efit = data.EFIT_data.rmag_fast.data
        ntefit=len(time_efit)
        data.EFIT_data.sampling_time = np.mean(np.diff(data.EFIT_data.rmag_fast.time))
        ntkg1v = len(data.KG1_data.density[chan].time)
        tkg1v = data.KG1_data.density[chan].time
        sampling_time_kg1v = np.mean(np.diff(tkg1v))
        tsmo = data.KG1LH_data.tsmo
        rolling_mean = int(round(sampling_time_kg1v / tsmo))
        data.EPSF = data.EPSF/10
        data.EPSDD = data.EPSDD/100




    density = data.KG1LH_data.lid[chan].data






    length = np.zeros(ntefit)
    xtan = np.zeros(ntefit)
    lad = np.zeros(ntefit)

    xpt = data.r_ref[chan - 1]
    ypt = data.z_ref[chan - 1]
    angle = data.a_ref[chan - 1]


    # convert to cm
    xpt = xpt * 100
    ypt = ypt * 100




    for IT in range(0, ntefit):

        TIMEM = time_efit[IT]



        logger.log(5,'computing lad/len/xtan \n')

        dtime = float(TIMEM)

        t, ier = flushinit(15, data.pulse, dtime, lunget=12, iseq=0,
                           uid='JETPPF', dda='EFIT', lunmsg=0)
        if ier != 0:
            logger.warning('flush error {} in flushinit'.format(ier))
            # return ier

        logger.log(5, '************* Time = {}s'.format(TIMEM))

        # look for xpoint
        iflsep, rx, zx, fx, ier = flush_getXpoint()

        if ier != 0:
            logger.warning('flush error {} in flush_getXpoint'.format(ier))
            # return ier
        logger.log(5,
                   'Time {}s; iflsep {}; rx {}; zx {}; fx {}; ier {} '.format(TIMEM,
                                                                              iflsep,
                                                                              rx,
                                                                              zx,
                                                                              fx,
                                                                              ier))

        if int(iflsep) == 0:
            logger.log(5, 'iflsep is {}'.format(iflsep))
            if IT==0:
                logger.debug('index {} - Time {}s; NO X-point found'.format(IT,TIMEM))
            elif IT %(1/tsmo)==0:
                logger.debug('index {} - Time {}s; NO X-point found'.format(IT,TIMEM))
            else:
                logger.log(5,
                'Time {}s; NO X-point found'.format(TIMEM))

            psimax = data.psim1
            logger.log(5, 'psimax is {}'.format(psimax))
            iskb = 1
            logger.log(5, 'psimax is {}'.format(psimax))
        else:
            if IT==0:
                logger.debug('index {} - Time {}s; X-point plasma'.format(IT,TIMEM))

            elif IT %(1/tsmo)==0:
                logger.debug('index {} - Time {}s; X-point plasma'.format(IT,TIMEM))
            else:
                logger.log(5,'Time {}s; X-point plasma'.format(TIMEM))
            # logger.log(5,'iflsep is {}'.format(iflsep))

        #     #probably all this is useless!
        #
        #     # if int(iflsep) == 1:
        #     #
        #     #      logger.log(5,'found {} X-point '.format(iflsep))
        #     #      psimax = data.psim1
        #     #      logger.log(5,'psimax is {}'.format(psimax))
        #     #      if fx[0] >= data.psim1:
        #     #           logger.log(5,'fx is {}'.format(fx))
        #     #           iskb = 1
        #     #           logger.log(5,'iskb is {}'.format(iskb))
        #     #      else:
        #     #           iskb = 0
        #     #           logger.log(5,'iskb is {}'.format(iskb))
        #     #           growth = (data.psim1 / fx[0]) - 1
        #     #           logger.log(5,'growth is {}'.format(growth))
        #     # else:
        #
        #     psimax = data.psim1 * fx[0]
        #     # psimax = 1
        #
        #     iskb = 0
        #
        #     # growth = data.psim1 - 1
        #     growth = fx[0] - 1
        #
        #     logger.log(5, 'psimax is {}'.format(psimax))
        #     logger.log(5, 'growth is {}'.format(growth))
        #     logger.log(5, 'iflsep is {}'.format(iflsep))
        #     logger.log(5, 'iskb is {}'.format(iskb))
        #
        # volume_m3, ier = Flush_getVolume(1)
        #
        # # if ier != 0:
        # #     logger.warning('flush error {} in Flush_getVolume'.format(ier))
        # #     return ier
        #
        # #
        # if iskb != 1:
        #     logger.log(5, 'iskb is {}'.format(iskb))
        #     ier = flush_blowUp(growth, volume_m3)
        #     logger.log(5, 'blowup error {}'.format(ier))
        # #end of useless stuff?

        # -----------------------------------------------------------------------
        # FIND PSI AT TANGENT FLUX SURFACE (to make FLUL2 quicker)
        # -----------------------------------------------------------------------
        rTan1,zTan1, fTan1, ier  = Flush_GetTangentFlux(xpt,ypt,angle, data.EPSDD)
        if ier != 0:
            logger.warning('flush error {} in Flush_GetTangentFlux'.format(ier))
            # return ier

        logger.log(5, 'get tangent flux output is rTan {}, zTan {}, fTan {}'.format(
            rTan1, zTan1, fTan1))


        #
        # ----------------------------------------------------------------------
        # FIND INTERSECTION POINTS WITH PLASMA BOUNDARY
        # ----------------------------------------------------------------------

        NPSI = 1  # look for one surface
        psimax = 1 # value of psi at the last closed surface
        nfound, r1, z1, r2, z2, r3, z3, r4, z4, ier = Flush_getIntersections(xpt,
                                                                             ypt,
                                                                             angle,
                                                                             data.EPSF,
                                                                             NPSI,
                                                                             psimax)
        if ier != 0:
            logger.warning('flush error {}  in Flush_getIntersections'.format(ier))
            # return ier
        cord = math.hypot(r2 - r1, z2 - z1)

        logger.log(5, 'found {} intersection/s'.format(nfound))

        # -----------------------------------------------------------------------
        # final results
        # -----------------------------------------------------------------------
        if cord < 0:
            cord = abs(cord)
        length[IT] = cord / 100.0  # conversion from cm to m
        logger.log(5,'cord length for channel {} is {}'.format(chan, length[IT]))
        # length[IT] = cord # conversion from cm to m
        if (length[IT] > 0.0):
            lad[IT] = density[IT] / length[IT]
        else:
            lad[IT] = 0.0

        xtan[IT] = fTan1









    data.KG1LH_data.lid[chan] = SignalBase(data.constants)
    data.KG1LH_data.lid[chan].data = density
    data.KG1LH_data.lid[chan].time = time_efit
    #
    data.KG1LH_data.lad[chan] = SignalBase(data.constants)
    data.KG1LH_data.lad[chan].data = lad

    data.KG1LH_data.lad[chan].time = time_efit
    #
    data.KG1LH_data.len[chan] = SignalBase(data.constants)
    data.KG1LH_data.len[chan].data = length

    data.KG1LH_data.len[chan].time = time_efit
    #
    data.KG1LH_data.xta[chan] = SignalBase(data.constants)
    data.KG1LH_data.xta[chan].data = xtan

    data.KG1LH_data.xta[chan].time = time_efit



    return (data, chan)

# ----------------------------


# ----------------------------



def main():
    '''
    Program to calculate the line averaged density for all channels of
    C kg1v. Other outputs are the tangent flux surface
    C and the distance to the edge divided by the chord length (curvature
    C of the edge for channel 4).
    :param shot_no:shot number to be processed
    :param code: KG1L or KG1H
    :param read_uid: UID for reading PPFs
    :param write_uid: UID for writing PPFs
    :param list_of_channels: list of channel to process (testing purposes only)
    :param algorithm: choose what algorithm to use for density filtering
    :param interp_method: choose what algorithm to use to resample signal.
    :param plot: True if user wants to plot data
    :param test: Set to True for testing mode, meaning the following:
                     - If write_uid is given and it is NOT JETPPF then a PPF will be written.

    :return:
    Return Codes::

    0 :  All OK
    1 :  Some channels were unavailable for processing.
    2 :  some channles were not validated
    5 :  init error
    9 :  No validated LID channels in KG1V
    11 : All available channels already have validated PPF data.
    20 : No KG1V data
    21 : Could not read SF for KG1V
    22 : Error reading KG1V line-of-sight data
    23 : could not filter data
    24 : could not perform time loop
    25 : could not filter data
    30 : No EFIT data
    31 : No points in EFIT
    65 : The initialisation file could not be read in.
    66 : Problem reading the geometry file.
    67 : Failed to write PPF.
    71 : Invalid shot number
    72 : No PPF exists for shot
    100: TEST MODE - NO PPF IS WRITTEN







    '''



    write_uid = os.getlogin()

    # pdb.set_trace()
    # with open('./data_3channels.pkl', 'rb') as f:
    with open('./data_8channels.pkl', 'rb') as f:
        [data,channels]=pickle.load(f)
    f.close()
    logger.info('loaded {} LID data for pulse {}, {} channels'.format(data.code,data.pulse,channels))
    plot= True
    processes = len(channels)
    shot_no = data.pulse




    # pdb.set_trace()
    data.code = 'kg1l'





    try:
        logger.info('Starting time loop')
        start_time = time.time()
        with Pool(processes) as pool:
            results = pool.map(time_loop,
                               [(data, chan) for chan in channels])
        logger.info("--- {}s seconds ---".format((time.time() - start_time)))

        for i, res in enumerate(results):
            if len(res[0].KG1LH_data.lad.keys()) != 0:
                # data.KG1LH_data.lid[i + 1] = SignalBase(data.constants)
                # data.KG1LH_data.lid[i + 1].time = res[0].KG1LH_data.lid[res[1]].time
                # data.KG1LH_data.lid[i + 1].data = res[0].KG1LH_data.lid[res[1]].data

                data.KG1LH_data.lad[i + 1] = SignalBase(data.constants)
                data.KG1LH_data.lad[i + 1].time = res[0].KG1LH_data.lad[res[1]].time
                data.KG1LH_data.lad[i + 1].data = res[0].KG1LH_data.lad[res[1]].data

                data.KG1LH_data.len[i + 1] = SignalBase(data.constants)
                data.KG1LH_data.len[i + 1].time = res[0].KG1LH_data.len[res[1]].time
                data.KG1LH_data.len[i + 1].data = res[0].KG1LH_data.len[res[1]].data

                data.KG1LH_data.xta[i + 1] = SignalBase(data.constants)
                data.KG1LH_data.xta[i + 1].time = res[0].KG1LH_data.xta[res[1]].time
                data.KG1LH_data.xta[i + 1].data = res[0].KG1LH_data.xta[res[1]].data
            else:
                continue
    except:
        logger.error('could not perform time loop')#
        return 24


    pdb.set_trace()


    # -------------------------------
    # 5. plot data
    # pdb.set_trace()
    # -------------------------------

    if plot:
        try:
            logging.info('plotting data and comparison with Fortran code')
            linewidth = 0.5
            markersize = 1

            # logging.info('plotting data')
            dda = data.code.upper()
            for chan in channels:
                if chan in data.KG1LH_data.lid.keys():

                #loading JETPPF data to use for comparison

                    kg1v_lid3, dummy = getdata(shot_no, 'KG1V', 'LID' + str(chan))
                    kg1l_lid3, dummy = getdata(shot_no, dda, 'LID'+str(chan))
                    kg1l_lad3, dummy = getdata(shot_no, dda, 'LAD'+str(chan))
                    kg1l_len3, dummy = getdata(shot_no, dda, 'LEN'+str(chan))
                    kg1l_xtan3, dummy = getdata(shot_no, dda, 'xta'+str(chan))


                    plt.figure()

                    ax_1= plt.subplot(4, 1, 1)
                    plt.plot(kg1l_lid3['time'],kg1l_lid3['data'],label='lid_jetppf_ch'+ str(chan))
                    plt.plot(kg1v_lid3['time'],kg1v_lid3['data'],label='KG1V_lid_jetppf_ch'+ str(chan))
                    plt.plot(data.KG1LH_data.lid[chan].time, data.KG1LH_data.lid[chan].data,
                             label=dda+'_lid_original_MT_ch'+ str(chan), marker='o', linestyle='-.',
                             linewidth=linewidth,
                             markersize=markersize)
                    # plt.plot(data.KG1LH_data1.lid[chan].time, data.KG1LH_data1.lid[chan].data,label=dda+'_lid_rollingmean_MT', marker = 'v', linestyle=':', linewidth=linewidth,
                    #                          markersize=markersize)
                    # plt.legend(loc='best',prop={'size':12})
                    #
                    #
                    # plt.plot(data.KG1LH_data2.lid[chan].time, data.KG1LH_data2.lid[chan].data,label=dda+'_lid_rollingmean_pandas_MT', marker = 'p', linestyle=':', linewidth=linewidth,
                    #                          markersize=markersize)
                    plt.legend(loc=0, prop={'size': 8})


                    plt.subplot(4, 1, 2,sharex=ax_1)
                    plt.plot(kg1l_lad3['time'], kg1l_lad3['data'], label='lad_jetppf_ch'+ str(chan))
                    plt.plot(data.KG1LH_data.lad[chan].time, data.KG1LH_data.lad[chan].data,
                             label=dda+'_lad_original_MT_ch'+ str(chan), marker='x', linestyle='-.',
                             linewidth=linewidth,
                             markersize=markersize)
                    plt.legend(loc=0, prop={'size': 8})

                    plt.subplot(4, 1, 3,sharex=ax_1)
                    plt.plot(kg1l_xtan3['time'],kg1l_xtan3['data'],label='xtan_jetppf_ch'+ str(chan))
                    plt.plot(data.KG1LH_data.xta[chan].time, data.KG1LH_data.xta[chan].data,
                             label=dda+'_xtan_original_MT_ch'+ str(chan), marker='o', linestyle='-.',
                             linewidth=linewidth,
                             markersize=markersize)

                    plt.legend(loc=0, prop={'size': 8})


                    plt.subplot(4, 1, 4,sharex=ax_1)
                    plt.plot(kg1l_len3['time'], kg1l_len3['data'], label='len_jetppf_ch'+ str(chan))
                    plt.plot(data.KG1LH_data.len[chan].time, data.KG1LH_data.len[chan].data,
                             label=dda+'_len_original_MT_ch'+ str(chan), marker='x', linestyle='-.',
                             linewidth=linewidth,
                             markersize=markersize)
                    plt.legend(loc=0, prop={'size': 8})



                    plt.savefig(cwd + os.sep + 'figures/' + data.code +'_'+ str(data.pulse) + 'ch_' + str(chan) + '_comparisons.png', dpi=300)
        except:
            logger.error('could not plot data')
            return 25
    plt.show()


    raise SystemExit
    # pdb.set_trace()



    # if plot:
    #     plt.show(block=True)


    # -------------------------------
    # 7. writing PPFs
    #pdb.set_trace()
    # -------------------------------

    logging.info('start writing PPFs')
    if (write_uid != "" and not test) or (test and write_uid.upper() != "JETPPF" and write_uid != ""):
        logger.info("\n             Writing PPF with UID {}".format(write_uid))

        err = open_ppf(data.pulse, write_uid)

        if err != 0:
            logger.error('failed to open ppf')
            return 67

        itref_kg1v = -1
        dda = data.code.upper()


        for chan in data.KG1LH_data.lid.keys():

            dtype_lid = "LID{}".format(chan)
            comment = "DATA FROM KG1 CHANNEL {}".format(chan)

            write_err, itref_written = write_ppf(data.pulse, dda,
                                                 dtype_lid,
                                                 data.KG1LH_data.lid[
                                                     chan].data,
                                                 time=
                                                 data.KG1LH_data.lid[
                                                     chan].time,
                                                 comment=comment,
                                                 unitd='M-2', unitt='SEC',
                                                 itref=itref_kg1v,
                                                 nt=len(
                                                     data.KG1LH_data.lid[
                                                         chan].time),
                                                 status=
                                                 data.KG1_data.status[
                                                     chan],
                                                 global_status=
                                                 data.KG1_data.global_status[
                                                     chan])
            if write_err != 0:
                logger.error(
                    "Failed to write {}/{}. Errorcode {}".format(dda, dtype_lid,
                                                                 write_err))
                return 67


        for chan in data.KG1LH_data.lad.keys():
            dtype_lid = "LAD{}".format(chan)
            comment = "Line Average Density CHANNEL {}".format(chan)


            write_err, itref_written = write_ppf(data.pulse, dda,
                                                 dtype_lid,
                                                 data.KG1LH_data.lad[
                                                     chan].data,
                                                 time=
                                                 data.KG1LH_data.lad[
                                                     chan].time,
                                                 comment=comment,
                                                 unitd='M-3', unitt='SEC',
                                                 itref=itref_kg1v,
                                                 nt=len(
                                                     data.KG1LH_data.lad[
                                                         chan].time),
                                                 status=
                                                 data.KG1_data.status[
                                                     chan],
                                                 global_status=
                                                 data.KG1_data.global_status[
                                                     chan])
            if write_err != 0:
                logger.error(
                    "Failed to write {}/{}. Errorcode {}".format(dda, dtype_lid,
                                                                 write_err))
                return 67

        for chan in data.KG1LH_data.len.keys():
            dtype_lid = "LEN{}".format(chan)
            comment = "CORD LENGTH KG1 CHANNEL {}".format(chan)

            write_err, itref_written = write_ppf(data.pulse, dda,
                                                 dtype_lid,
                                                 data.KG1LH_data.len[
                                                     chan].data,
                                                 time=
                                                 data.KG1LH_data.len[
                                                     chan].time,
                                                 comment=comment,
                                                 unitd='M', unitt='SEC',
                                                 itref=itref_kg1v,
                                                 nt=len(
                                                     data.KG1LH_data.len[
                                                         chan].time),
                                                 status=
                                                 data.KG1_data.status[
                                                     chan],
                                                 global_status=
                                                 data.KG1_data.global_status[
                                                     chan])
            if write_err != 0:
                logger.error(
                    "Failed to write {}/{}. Errorcode {}".format(dda, dtype_lid,
                                                                 write_err))
                return 67
        
        for chan in data.KG1LH_data.xta.keys():
            dtype_lid = "XTA{}".format(chan)
            
            comment = "Tangent flux lid{} ".format(chan)

            write_err, itref_written = write_ppf(data.pulse, dda,
                                                 dtype_lid,
                                                 data.KG1LH_data.xta[
                                                     chan].data,
                                                 time=
                                                 data.KG1LH_data.xta[
                                                     chan].time,
                                                 comment=comment,
                                                 unitd='  ', unitt='SEC',
                                                 itref=itref_kg1v,
                                                 nt=len(
                                                     data.KG1LH_data.xta[
                                                         chan].time),
                                                 status=
                                                 data.KG1_data.status[
                                                     chan],
                                                 global_status=
                                                 data.KG1_data.global_status[
                                                     chan])

            if write_err != 0:
                logger.error(
                    "Failed to write {}/{}. Errorcode {}".format(dda, dtype_lid,
                                                                 write_err))
                return 67

        mode = "smoothing time  kg1 {}".format(data.KG1LH_data.tsmo)
        dtype_mode = "TSMO"
        comment = mode
        write_err, itref_written = write_ppf(data.pulse, dda, dtype_mode,
                                            np.array([data.KG1LH_data.tsmo]),
                                             time=np.array([0]),
                                             comment=comment, unitd="SEC ",
                                             unitt=" ", itref=-1, nt=1,
                                             status=None)

        if write_err != 0:
            logger.error(
                "Failed to write {}/{}. Errorcode {}".format(dda, dtype_mode,
                                                             write_err))
            return 67

        mode = "Generated by {}".format(write_uid)
        dtype_mode = "MODE"
        comment = mode
        write_err, itref_written = write_ppf(data.pulse, dda, dtype_mode,
                                             np.array([1]),
                                             time=np.array([0]),
                                             comment=comment, unitd=" ",
                                             unitt=" ", itref=-1, nt=1,
                                             status=None)

        if write_err != 0:
            logger.error(
                "Failed to write {}/{}. Errorcode {}".format(dda, dtype_mode,
                                                             write_err))
            return 67


        err = close_ppf(data.pulse, write_uid,
                            data.constants.code_version,code)

        if err != 0:
            logger.error('failed to close ppf')
            return 67



    else:
        return_code = 100
        logger.info("No PPF was written. UID given was {}, test: {}".format(write_uid, test))
        return return_code



    logger.info("\n             Finished.\n")
    logger.info("--- {}s seconds --- \n \n \n \n ".format((time.time() - code_start_time)))

    gc.collect()
    return return_code

if __name__ == "__main__":

    # Ensure we are running python 3
    assert sys.version_info >= (
    3, 5), "Python version too old. Please use >= 3.5.X."


    # Setup the logger
    debug_map = {0: logging.ERROR,
                 1: logging.WARNING,
                 2: logging.INFO,
                 3: logging.DEBUG,
                 4: 5}

    logging.basicConfig(level=debug_map[2])

    logging.addLevelName(5, "DEBUG_PLUS")

    logger = logging.getLogger(__name__)



    # Call the main code
    main()
