"""
Class to read and store KG1 PPF data for one channel.
Reads in LIDX, FCX, MIRX, JXBX, TYPX

"""

import logging
import getdat
import numpy as np
from ppf_write import write_ppf
from signal_kg1 import SignalKg1
from signal_base import SignalBase
from pdb import set_trace as bp
from library import * #containing useful function

logger = logging.getLogger(__name__)

# ----------------------------
__author__ = "B. Viola"
# ----------------------------


class Kg1PPFData(SignalBase):

    # ------------------------
    def __init__(self, constants,pulse):
        """
        Init function
        :param constants: instance of Kg1Consts class
        """
        # self.signal_type = ""  # kg1r, kg1c, kg1v
        # self.dcn_or_met = "" # dcn, met

        self.constants = constants
        self.pulse = pulse
        self.dda = "KG1V"
        self.density = {}

        # Time dependent status flags
        self.status = {}
        self.global_status = {}
        self.type = {}
        self.mode = ""

        # PPF data: If a KG1 PPF has already been written,
        # and the status flag of the data is 1,2 or 3
        # (ie. it has already been validated), then
        # this data is just copied over to the output file,
        # no further corrections are made.
        self.ppf = {}

        self.dfr = self.constants.DFR_DCN

    # ------------------------
    def read_data(self, shot_no, read_uid="JETPPF"):
        """
        Read in PPF data for KG1V for a given channel
        :param shot_no: shot number
        :param chan: channel
        :param read_uid: read UID
        :return: True if data was read in successfully, False otherwise
        """

        for chan in self.constants.kg1v.keys():
            nodename = self.constants.kg1v[chan]
            density = SignalKg1(self.constants,self.pulse)
            # corrections = SignalBase(self.constants)
            dda = nodename[:nodename.find('/')]
            dtype = nodename[nodename.find('/')+1:]
            status = density.read_data_ppf(dda, dtype, shot_no, read_bad=True, read_uid=read_uid)

            # We are only interested in keeping the data if it has already been validated
            # if density.data is not None and (0 < status < 4) and not all_status:
            #     logger.debug( "PPF data chan {}".format(status))
            if density.data is not None:
                self.density[chan] = density
                self.status[chan] = SignalBase(self.constants)
                self.status[chan].data = np.zeros(
                        len(self.density[chan].time))
                self.status[chan].time = self.density[chan].time
                dummy=check_SF(read_uid, shot_no)
                self.global_status[chan] = dummy[chan-1]

                # self.density[chan].corrections = SignalBase(self.constants)
                self.density[chan].signal_type = 'vert'
                if chan >4:
                    self.density[chan].signal_type = 'lat'
                    self.density[chan].correction_dcn = SignalBase(self.constants)
                    self.density[chan].correction_met = SignalBase(self.constants)

            # else:
            #     return False
            # else:
            #     return False

        for chan in self.constants.kg1r_ppf_type.keys():
            nodename = self.constants.kg1r_ppf_type[chan]
            sig_type = SignalBase(self.constants)
            dda = nodename[:nodename.find('/')]
            dtype = nodename[nodename.find('/')+1:]
            status = sig_type.read_data_ppf(dda, dtype, shot_no, read_bad=True, read_uid=read_uid)

            if sig_type.data is not None:
                ind_type = sig_type.ihdata.find("SIG TYPE:")+len("SIG TYPE:")+1
                ind_chan = sig_type.ihdata.find("CH.")-1
                self.type[chan] = sig_type.ihdata[ind_type:ind_chan]
        

        nodename_mode = self.constants.mode

        signal_mode = SignalBase(self.constants)
        
        signal_mode.read_data_ppf('KG1V', 'MODE', shot_no, read_bad=False, read_uid=read_uid)
        try:
            self.mode = signal_mode.ihdata[36:]
        except TypeError:
            logger.error('no KG1V/MODE data!')

        


        
        return True


    # ------------------------
     # ------------------------
    def get_coord(self, shot_no):
        """
        Get vacuum vessel temperature & extract spatial coordinates of KG4 chords from text file.
        Function copied from A. Boboc's kg4r_py code.

        :param shot_no: shot number
        """

        nodename_temp = self.constants.temp_node

        vv_temp, err = self.get_jpf_point(shot_no, nodename_temp)

        filename = self.constants.geometry_filename

        try:
            raw = np.loadtxt(filename, skiprows=3)

            temp = raw[:,0]
            index = np.where(temp==vv_temp)[0]
            Coord_Rref = raw[0,1:9]
            Coord_Zref = raw[0,9:17]
            Coord_Aref = raw[0,17:25]
            Coord_R = raw[index,1:9][0]
            Coord_Z = raw[index,9:17][0]
            Coord_A = raw[index,17:25][0]
            Coord_Temperature = vv_temp

            logger.debug('Vaccum Vessel temperature(deg.C) {}'.format(vv_temp))
            logger.debug('Rref {}'.format(Coord_Rref))
            logger.debug('Zref {}'.format(Coord_Zref))
            logger.debug('Aref {}'.format(Coord_Aref))
            logger.debug('R {}'.format(Coord_R))
            logger.debug('Z {}'.format(Coord_Z))
            logger.debug('A {}'.format(Coord_A))

            return (Coord_Temperature,Coord_R,Coord_Z,\
                    Coord_A,Coord_Rref,Coord_Zref,Coord_Aref,0)

        except IOError:
            pass
            ier = 1
            logger.debug("Reading KG4 R,Z,A coordinates data from {} failed".format(filename))
            return 1,1,1,1,1,1,1,66

    # ------------------------

