"""
Class to read and store all efit data
"""

import logging

import numpy as np

from signal_base import SignalBase
from make_plots import make_plot
import pdb

logger = logging.getLogger(__name__)

# ----------------------------
__author__ = "B. VIOLA"


# ----------------------------


class EFITData:

    # ------------------------
    def __init__(self, constants):
        """
        Init function

        :param constants: instance of Kg1Consts class
        """
        self.constants = constants
        # density
        self.rmag = {}
        self.rmag_fast = {}
        self.sampling_time = None

        # ------------------------

    def read_data(self, shot_no, read_uid="JETPPF"):
        """
        Read in efit (RMAG)


        :param shot_no: shot number
        """
        # Read in efit

        # pdb.set_trace()
        node_name = self.constants.efit
        efit_signal = SignalBase(self.constants)
        dda = node_name[:node_name.find('/')]
        dtype = node_name[node_name.find('/') + 1:]

        status = efit_signal.read_data_ppf(dda, dtype, shot_no,
                                            read_bad=True,
                                            read_uid=read_uid)

        if efit_signal.data is not None:
            # Keep points where there is ip

            self.rmag = efit_signal
        else:
            logger.error('no EFIT/RMAG data!')
            return 30


        node_name = self.constants.efit_fast
        efit_signal = SignalBase(self.constants)
        dda = node_name[:node_name.find('/')]
        dtype = node_name[node_name.find('/') + 1:]

        status = efit_signal.read_data_ppf(dda, dtype, shot_no,
                                            read_bad=True,
                                            read_uid=read_uid)

        if efit_signal.data is not None:
            # Keep points where there is ip

            self.rmag_fast = efit_signal
        else:
            logger.error('no EHTR/RMAG data!')
            return 30
        return 0
