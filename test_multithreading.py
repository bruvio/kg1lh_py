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

import matplotlib.pyplot as plt
import pdb
# from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing.pool import ThreadPool
import threading
import argparse
import logging
from types import SimpleNamespace
import numpy as np
import time
import inspect
import logging

logger = logging.getLogger(__name__)

myself = lambda: inspect.stack()[1][3]
logger = logging.getLogger(__name__)
pool = ThreadPool(processes=8)

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return




#--------
def map_kg1_efit(data,chan):


    density = np.zeros(968)


    for it in range(0,data.ntefit):
        density[it] = it
        for jj in range(0,data.ntkg1v):
            density[it]=density[it]+jj

    data.KG1LH_data.lid[chan] = density

# ----------------------------

def main():
    data = SimpleNamespace()
    data.KG1LH_data = SimpleNamespace()
    data.ntkg1v = 30039
    data.ntefit = 968

    data.KG1LH_data.lid = [ [],[],[],[],[],[],[],[]]

    channels=range(1,8)



    # chan =1
    for chan in channels:
        logger.info('computing channel {}'.format(chan))
        start_time = time.time()
        twrv = ThreadWithReturnValue(target=map_kg1_efit, args=(data,chan))
        # pdb.set_trace()
        twrv.start()
        twrv.join()
        logger.info("--- {}s seconds ---".format((time.time() - start_time)))
        plt.figure()
        plt.plot(range(0,data.ntefit), data.KG1LH_data.lid[chan])
        plt.show()




        logger.info('computing channel {}'.format(chan))
        start_time = time.time()
        map_kg1_efit(data,chan)
        logger.info("--- {}s seconds ---".format((time.time() - start_time)))

        plt.figure()
        plt.plot(range(0,data.ntefit), data.KG1LH_data.lid[chan])
        plt.show()



    logger.info("\n             Finished.\n")

if __name__ == "__main__":
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
