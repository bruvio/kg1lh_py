"""
Class to read and store all KG1L/H data
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


class KG1LData:

    # ------------------------
    def __init__(self, constants):
        """
        Init function

        :param constants: instance of Kg1Consts class
        """
        self.constants = constants
        # density
        self.lid = {}
        self.lad = {}
        self.len = {}
        self.xta = {}
        self.tsmo = None



        # ------------------------

