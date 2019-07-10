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



import pdb
from multiprocessing.dummy import Pool as ThreadPool
import threading
import argparse
import logging
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



# qm = QtGui.QMessageBox
# qm_permanent = QtGui.QMessageBox
plt.rcParams["savefig.directory"] = os.chdir(os.getcwd())
myself = lambda: inspect.stack()[1][3]
logger = logging.getLogger(__name__)
# noinspection PyUnusedLocal

#--------
def map_kg1_efit(ntefit,tefit,data,tsmo,chan):
    for it in range(0, ntefit):
        sum = np.zeros(8)

        nsum = 0

        tmin = 1000.0

        jmin = 1
        density = np.zeros(ntefit)
        # for chan in data.KG1_data.density.keys():
        pdb.set_trace()
        ntkg1v = len(data.KG1_data.density[chan].time)
        tkg1v = data.KG1_data.density[chan].time
        for jj in range(0, ntkg1v):
            tdif = abs(tkg1v[jj] - tefit[it])
            if (tdif < tmin):
                tmin = tdif
                jmin = jj
            if (tkg1v[jj] >= tefit[it] + tsmo):
                if nsum > 0:
                    # pdb.set_trace()
                    # data.KG1LH_data.density[chan] = SignalBase(data.constants)
                    density[it] = sum[chan - 1] / float(nsum)
                else:
                    density[it] = data.KG1_data.density[chan].data[jmin]

            if (tkg1v[jj] < tefit[it] - tsmo):
                sum[chan - 1] = sum[chan - 1] + \
                                data.KG1_data.density[chan].data[jj]
                nsum = nsum + 1
                if nsum > 0:
                    # pdb.set_trace()
                    # data.KG1LH_data.density[chan] = SignalBase(data.constants)
                    density[it] = sum[chan - 1] / float(nsum)
                else:
                    density[it] = data.KG1_data.density[chan].data[jmin]
    return density

    # pdb.set_trace()

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

    # multi threading
    pool = ThreadPool(4)
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
        # constants = Kg1Consts("kg1_consts.ini", __version__)
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
    data.KG1_data = {}
    data.EFIT_data = {}
    data.KG1LH_data = {}

    data.KG1LH_data = {}
    logger.info('INIT DONE\n')

    # -------------------------------
    # 2. Read in KG1 data
    # -------------------------------
    data.KG1_data = Kg1PPFData(data.constants, data.pulse)

    success = data.KG1_data.read_data(data.pulse,
                                           read_uid=read_uid)



    logger.log(5, 'success reading KG1 data?'.format(success))
    # -------------------------------
    # 2. Read in EFIT data
    # -------------------------------
    data.EFIT_data = EFITData(data.constants)
    data.EFIT_data.read_data(data.pulse)


    # -------------------------------
    # 3. map kg1v data onto efit time vector
    # -------------------------------

    if code.lower()=='kg1l':
        ntefit = len(data.EFIT_data.rmag.time)
        tefit = data.EFIT_data.rmag.time
        data_efit = data.EFIT_data.rmag.data
        
    else:
        ntefit = len(data.EFIT_data.rmag_fast.time)
        tefit = data.EFIT_data.rmag_fast.time
        data_efit = data.EFIT_data.rmag_fast.data

    channels=range(1,8)
    # chan =1
    for chan in channels:
        logger.info('computing channel {}'.format(chan))
        start_time = time.time()
        threading.Thread(target=map_kg1_efit, args=(ntefit,tefit,data,tsmo,chan)).start()
        # results = pool.map(map_kg1_efit(ntefit,tefit),channels)
        logger.info("--- {}s seconds ---".format((time.time() - start_time)))
        pdb.set_trace()

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
