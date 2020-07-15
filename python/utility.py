import logging
import pdb
logger = logging.getLogger(__name__)
import sys
import os
from importlib import import_module
import shutil
from shapely.geometry import LineString,Polygon

libnames = ['ppf']
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

        lib = import_module(libr)
        # lib = import_module(libr,package=package)
    except:
        exc_type, exc, tb = sys.exc_info()
        print(os.path.realpath(__file__))
        print(exc)
    else:
        globals()[libr] = lib
from status_flag import GetSF
import numpy as np

from numpy import arange, asscalar

import shutil

import math
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool
from threading import Thread


pool = ThreadPool(processes=8)


class ThreadWithReturnValue(Thread):
    def __init__(
        self, group=None, target=None, name=None, args=(), kwargs={}, Verbose=None
    ):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


def thread_map(f, iterable, pool=None):
    """
    Just like [f(x) for x in iterable] but each f(x) in a separate thread.
    :param f: f
    :param iterable: iterable
    :param pool: thread pool, infinite by default
    :return: list if results
    """
    res = {}
    if pool is None:

        def target(arg, num):
            try:
                res[num] = f(arg)
            except:
                res[num] = sys.exc_info()

        threads = [
            Thread(target=target, args=[arg, i]) for i, arg in enumerate(iterable)
        ]
    else:

        class WorkerThread(Thread):
            def run(self):
                while True:
                    try:
                        num, arg = queue.get(block=False)
                        try:
                            res[num] = f(arg)
                        except:
                            res[num] = sys.exc_info()
                    except Empty:
                        break

        queue = Queue()
        for i, arg in enumerate(iterable):
            queue.put((i, arg))

        threads = [WorkerThread() for _ in range(pool)]

    [t.start() for t in threads]
    [t.join() for t in threads]
    return [res[i] for i in range(len(res))]


def getdata(shot, dda, dtype, uid=None, seq=None):
    if uid is None:
        ppf.ppfuid("jetppf", rw="R")
    else:
        ppf.ppfuid(uid, rw="R")
    if seq is None:
        ier = ppf.ppfgo(shot, seq=0)
    else:
        ier = ppf.ppfgo(shot, seq=seq)

    ihdata, iwdata, data, x, time, ier = ppf.ppfget(shot, dda, dtype)
    pulse, seq, iwdat, comment, numdda, ddalist, ier = ppf.ppfinf(comlen=50, numdda=50)

    name = dict()
    name["ihdata"] = ihdata
    name["iwdata"] = iwdata
    name["data"] = data
    name["x"] = x
    name["time"] = time
    name["ier"] = ier
    name["seq"] = seq
    name["pulse"] = pulse
    name["dda"] = dda
    name["dtype"] = dtype
    return name, seq


def plottimedata(*args):
    plt.figure()
    for arg in args:

        plt.plot(arg["time"], arg["data"], label=retrieve_name(arg))
        plt.legend(loc=0, prop={"size": 8})


def subplottimedata(*args):
    fig, ax = plt.subplots(nrows=len(args), sharex=True)
    for arg, i in enumerate(args):

        ax[arg].plot(i["time"], i["data"], label=retrieve_name(i))
        ax[arg].legend(loc=0, prop={"size": 8})

    # plt.plot(arg['time'], arg['data'], label=retrieve_name(arg))
    # plt.legend(loc=0, prop={'size': 8})


import inspect




def retrieve_name(var):
    """
        Gets the name of var. Does it from the out most frame inner-wards.
        :param var: variable to get name from.
        :return: string
        """
    for fi in reversed(inspect.stack()):
        names = [
            var_name
            for var_name, var_val in fi.frame.f_locals.items()
            if var_val is var
        ]
        if len(names) > 0:
            return names[0]


# cumsum_vec = np.cumsum(np.insert(data.KG1_data.density[1].data, 0, 0))
# density = (cumsum_vec[rolling_mean:] - cumsum_vec[:-rolling_mean]) / rolling_mean
#
#
def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, "same")


#
#
#
# x = data.KG1_data.density[1].time
# y = data.KG1_data.density[1].data
#
# plt.plot(x, y, "k.")
# y_av = movingaverage(y, rolling_mean)
# plt.plot(x, y_av, "r")
# # xlim(0, 1000)
# plt.xlabel("Months since Jan 1749.")
# plt.ylabel("No. of Sun spots")
# plt.grid(True)
# plt.show()
#
# pdb.set_trace()


def create_dict(*args):
    return dict({i: eval(i) for i in args})
def is_empty(structure):
    """

    :param structure:
    :return: check is structure is empty
    """
    if structure:
        return False
    else:
        return True

def plot_point(point, angle, length):
    '''
    point - Tuple (x, y)
    angle - Angle you want your end point at in degrees.
    length - Length of the line you want to plot.

    Will return start and end point of the segment
    '''

    # unpack the first point
    x, y = point
    # pdb.set_trace()
    # find the end points
    endy1 = y-length * math.sin((angle))
    endy2 = y+length * math.sin((angle))
    endx1 = x-length * math.cos((angle))
    endx2 = x+length * math.cos((angle))
    return endx1,endx2,endy1,endy2
    # plt.plot([endx1, endx2], [endy1, endy2])



def define_LOS(data):
    plt.figure(1, figsize=(10, 4), dpi=90)  # 1, figsize=(10, 4), dpi=180)

    endx1, endx2, endy1, endy2 = plot_point(
        [data.r_coord[0], data.z_coord[0]], data.a_coord[0],
        2)
    data.LOS1 = LineString([(endx1, endy1), (endx2, endy2)])
    plt.plot([endx1, endx2], [endy1, endy2], label='LOS1')

    endx1, endx2, endy1, endy2 = plot_point(
        [data.r_coord[1], data.z_coord[1]], data.a_coord[1],
        2)
    data.LOS2 = LineString([(endx1, endy1), (endx2, endy2)])
    plt.plot([endx1, endx2], [endy1, endy2], label='LOS2')

    endx1, endx2, endy1, endy2 = plot_point(
        [data.r_coord[2], data.z_coord[2]], data.a_coord[2],
        2)
    data.LOS3 = LineString([(endx1, endy1), (endx2, endy2)])
    plt.plot([endx1, endx2], [endy1, endy2], label='LOS3')

    endx1, endx2, endy1, endy2 = plot_point(
        [data.r_coord[3], data.z_coord[3]], data.a_coord[3],
        2)
    data.LOS4 = LineString([(endx1, endy1), (endx2, endy2)])
    plt.plot([endx1, endx2], [endy1, endy2], label='LOS4')

    endx1, endx2, endy1, endy2 = plot_point(
        [data.r_coord[4], data.z_coord[4]], data.a_coord[4],
        2)
    data.LOS5 = LineString([(endx1, endy1), (endx2, endy2)])
    plt.plot([endx1, endx2], [endy1, endy2], label='LOS5')

    endx1, endx2, endy1, endy2 = plot_point(
        [data.r_coord[5], data.z_coord[5]], data.a_coord[5],
        2)
    data.LOS6 = LineString([(endx1, endy1), (endx2, endy2)])
    plt.plot([endx1, endx2], [endy1, endy2], label='LOS6')

    endx1, endx2, endy1, endy2 = plot_point(
        [data.r_coord[6], data.z_coord[6]], data.a_coord[6],
        2)
    data.LOS7 = LineString([(endx1, endy1), (endx2, endy2)])
    plt.plot([endx1, endx2], [endy1, endy2], label='LOS7')

    endx1, endx2, endy1, endy2 = plot_point(
        [data.r_coord[7], data.z_coord[7]], data.a_coord[7],
        2)
    data.LOS8 = LineString([(endx1, endy1), (endx2, endy2)])
    plt.plot([endx1, endx2], [endy1, endy2], label='LOS8')
    plt.legend()

    return data.LOS1,data.LOS2,data.LOS3,data.LOS4,data.LOS5,data.LOS6,data.LOS7,data.LOS8




def test_logger():
    """
    function to test logger
    :return:
    """
    logger.info("info")
    logger.debug("debug")
    logger.warning("warn")
    logger.error("error")
    logger.log(5, "debug plus")


def reconnect(signal, newhandler=None, oldhandler=None):
    """
    function used to connect a signal to a different handler
    :param signal:
    :param newhandler:
    :param oldhandler:
    :return:
    """
    while True:
        try:
            if oldhandler is not None:
                signal.disconnect(oldhandler)
            else:
                signal.disconnect()
        except TypeError:
            break
    if newhandler is not None:
        signal.connect(newhandler)


def is_empty(any_structure):
    if any_structure:

        return False
    else:

        return True


def are_eq(a, b):
    """
    checks if two lists are equal
    :param a:
    :param b:
    :return:
    """
    return set(a) == set(b) and len(a) == len(b)


def autoscale_data(ax, data):
    """
    autoscale plot
    :param ax:
    :param data:
    :return:
    """
    ax.set_ylim(min(data), max(data))


# def find_nearest(array,value):
#     # logger.log(5, "looking for value {}".format(value))
#     idx = (np.abs(array-value)).argmin()
#
#     # logger.log(5," found at {} with index {}".format(array[idx].item(),idx))
#     return idx,array[idx].item()
def find_nearest(array, value):
    """

    :param array:
    :param value:
    :return: returns value and index of the closest element in the array to the value
    """
    import numpy as np
    import math

    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (
        idx == len(array)
        or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
    ):
        return idx - 1, array[idx - 1]
    else:
        return idx, array[idx]


def find_in_list_array(array, value):
    found = False
    array2list = np.array(array)  # numpy array
    try:
        index = list(array2list).index(value)
        found = True
        return found, index
    except ValueError:
        index = []
        return found, index


def find_listelements_in_otherlist2(list1, list2, tstep):
    """

    :param list1:
    :param list2:
    :param tstep: minimum distance between two data points
    :return:
    """
    #
    list1 = list(list1)
    list2 = list(list2)
    # [i for e in list1 for i in list2 if e in i]
    found_list = []
    index_list = []
    for i, value in enumerate(list2):
        found, index = find_in_list_array(list1, value)
        if found:
            index_list.append(index)

    # def find_listelements_in_otherlist(list1,list2):


#     list1=list(list1)
#     list2=list(list2)
#
#     [item for item in list1 if any(x in item for x in list2)]


def find_within_range(array, minvalue, maxvalue):

    # idxmin = (np.abs(array - min)).argmin()
    # idxmax = (np.abs(array - max)).argmax()

    l2 = []
    l3 = []
    if array is None:
        return l3, l2
    else:
        for i, value in enumerate(array):
            if value >= minvalue and value <= maxvalue:
                l2.append(value)
                l3.append(i)

        return l3, l2


def pyqt_set_trace():
    """
    Set a tracepoint in the Python debugger that works with Qt
    :return:
    """

    from PyQt4.QtCore import pyqtRemoveInputHook
    import pdb
    import sys

    pyqtRemoveInputHook()
    # set up the debugger
    debugger = pdb.Pdb()
    debugger.reset()
    # custom next to get outside of function scope
    debugger.do_next(None)  # run the next command
    users_frame = (
        sys._getframe().f_back
    )  # frame where the user invoked `pyqt_set_trace()`
    debugger.interaction(users_frame, None)


def norm(data):
    """normalise data
    """
    return (data) / (max(data) - min(data))


def normalise(signal, kg1_signal, dis_time):
    """

        :param signal:  second trace
        :param kg1_signal: KG1 signal
        :param dis_time: disruption time
        :return: Use ratio of maximum of signal - kg1 as the normalisation factor. Exclude region around the disruption.
        """
    if dis_time > 0:
        (ind_dis,) = np.where((kg1_signal.time < dis_time - 1))

        max_kg1 = max(kg1_signal.data[ind_dis])
    else:
        max_kg1 = max(kg1_signal.data)

    max_signal = max(signal.data)

    #    print("max kg1 {} max signal {}".format(max_kg1, max_signal))

    if max_signal == 0:
        logger.warning("divide by 0 ")
        max_signal = 1

    norm_factor = max_kg1 / max_signal
    dummy = np.multiply(signal.data, norm_factor)

    return dummy


def get_seq(shot_no, dda, read_uid="JETPPF"):
    """

    :param shot_no: pulse number
    :param dda:
    :param read_uid:
    :return: get sequence of a ppf
    """
    ier = ppf.ppfgo(shot_no, seq=0)
    if ier != 0:
        return None

    ppf.ppfuid(read_uid, rw="R")

    iseq, nseq, ier = ppf.ppfdda(shot_no, dda)

    if ier != 0:
        return None

    return iseq


def get_min_max_seq(shot_no, dda="KG1V", read_uid="JETPPF"):
    """

    :param shot_no:
    :param dda:
    :param read_uid:
    :return: return min and max sequence for given pulse, dda and readuid
    min is the unvalidated sequence
    max is the last validated sequence
    """
    kg1v_seq = get_seq(shot_no, dda, read_uid)
    unval_seq = -1
    val_seq = -1
    if kg1v_seq is not None:
        unval_seq = min(kg1v_seq)
        if len(kg1v_seq) > 1:
            val_seq = max(kg1v_seq)
            return unval_seq, val_seq
        else:
            val_seq = unval_seq
            return unval_seq, val_seq


# def check_SF(read_uid, pulse):
#     """
#
#     :param read_uid:
#     :param pulse:
#     :return: list of Status Flags
#     """
#     logging.info("\n")
#     logging.info("checking status FLAGS ")
#
#     ppfuid(read_uid, "r")
#
#     ppfssr([0, 1, 2, 3, 4])
#
#     channels = arange(0, 8) + 1
#     SF_list = []
#
#     pulse = int(pulse)
#
#     for channel in channels:
#         ch_text = "lid" + str(channel)
#
#         st_ch = GetSF(pulse, "kg1v", ch_text)
#         st_ch = asscalar(st_ch)
#         SF_list.append(st_ch)
#     logging.info("%s has the following SF %s", str(pulse), SF_list)
#
#     return SF_list


def check_SF(read_uid, pulse, seq,dda=None):
    """

    :param read_uid:
    :param pulse:
    :return: list of Status Flags
    """
    logging.info("\n")
    logging.info("checking status FLAGS ")
    if dda is None:
        dda = 'kg1v'
    else:
        dda = dda
    ier = ppf.ppfgo(pulse, seq=seq)
    ppf.ppfuid(read_uid, "r")

    ppf.ppfssr([0, 1, 2, 3, 4])

    channels = arange(0, 8) + 1
    SF_list = []

    pulse = int(pulse)

    for channel in channels:
        ch_text = "lid" + str(channel)

        st_ch = GetSF(pulse, dda, ch_text)
        st_ch = asscalar(st_ch)
        SF_list.append(st_ch)
    logging.info("%s has the following SF %s", str(pulse), SF_list)

    return SF_list

def extract_history(filename, outputfile):
    """
    running this script will create a csv file containing a list of all the
    ppf that have been created with Cormat_py code

    the script reads a log file (generally in /u/user/work/Python/Cormat_py)


    and writes an output file in the current working directory

    the file is formatted in this way
    shot: {} user: {} date: {} seq: {} by: {}
    user is the write user id
    by is the userid of the user of the code
    the output is appended and there is a check on duplicates

    if the user have never run KG1_py code the file will be empty

    :param filename: name of KG1L (or KG1H) diary to be read
    :param outputfile: name of the output file
    :return:

    """
    import os

    if os.path.exists(filename):

        with open(filename, "r") as f_in:
            lines = f_in.readlines()
            for index, line in enumerate(lines):
                if "shot" in str(line):
                    dummy = lines[index].split()
                    shot = int(dummy[1])
                    user = str(dummy[3])
                    date = str(dummy[5])

                    # #             dummy = lines[index + 1].split()
                    sequence = dummy[7]

                    writtenby = dummy[10]
                    # #
                    #             month =(dummy[6])
                    #             day =(dummy[7])
                    #             year =(dummy[9])
                    #             logging.info(month,day,year)
                    #             date = datetime.date(int(year),strptime(month,'%b').tm_mon , int(day))

                    # logging.info(shot, user, date, sequence,writtenby)
                    # return
                    string_to_write = "shot: {} user: {} date: {} seq: {} by: {}\n".format(
                        str(shot).strip(),
                        user.strip(),
                        str(date).strip(),
                        str(sequence).strip(),
                        writtenby.strip(),
                    )

                    if os.path.exists(outputfile):
                        if check_string_in_file(outputfile, string_to_write):
                            pass
                        else:
                            with open(outputfile, "a+") as f_out:
                                f_out.write(string_to_write)
                            f_out.close()
                    else:
                        with open(outputfile, "a+") as f_out:
                            f_out.write(string_to_write)
                        f_out.close()

        f_in.close()
    else:
        f_in = open(filename, "w")
        f_in.close()
        string_to_write = "shot: {} user: {} date: {} seq: {} by: {}\n".format(
            str(00000), "unknown", str("00-00-00"), str(000), "unknown"
        )
        if os.path.exists(outputfile):
            if check_string_in_file(outputfile, string_to_write):
                pass
            else:
                with open(outputfile, "a+") as f_out:
                    f_out.write(string_to_write)
                f_out.close()
        else:
            with open(outputfile, "a+") as f_out:
                f_out.write(string_to_write)
            f_out.close()


def check_string_in_file(filename, string):
    """

    :param filename:
    :param string:
    :return: checks if the string is in that file
    """
    with open(filename) as myfile:
        if string in myfile.read():
            return True
        else:
            return False


def equalsFile(firstFile, secondFile, blocksize=65536):
    """

    :param firstFile:
    :param secondFile:
    :param blocksize:
    :return: returns True if files are the same,i.e. secondFile has same checksum as first
    """

    if os.path.getsize(firstFile) != os.path.getsize(secondFile):
        return False
    else:
        firstFile = open(firstFile, "rb")
        secondFile = open(secondFile, "rb")
        buf1 = firstFile.read(blocksize)
        buf2 = secondFile.read(blocksize)
    while len(buf1) > 0:
        if buf1 != buf2:
            return False
        buf1, buf2 = firstFile.read(blocksize), secondFile.read(blocksize)
    return True


# =============================================================================
def pyqt_set_trace():
    """Set a tracepoint in the Python debugger that works with Qt"""
    from PyQt4.QtCore import pyqtRemoveInputHook
    import pdb
    import sys

    pyqtRemoveInputHook()
    # set up the debugger
    debugger = pdb.Pdb()
    debugger.reset()
    # custom next to get outside of function scope
    debugger.do_next(None)  # run the next command
    users_frame = (
        sys._getframe().f_back
    )  # frame where the user invoked `pyqt_set_trace()`
    debugger.interaction(users_frame, None)


#
# =============================================================================
def copy_changed_kg1_to_save(src, dst, filename):
    """

    :param src:
    :param dst:
    :param filename:
    :return: copies file from src folder to dst
    """

    src = "./" + src + "/" + filename
    dst = "./" + dst + "/" + filename

    copyfile(src, dst)


# -----------------------


def delete_files_in_folder(folder):
    try:
        for root, dirs, files in os.walk(folder):
            for f in files:
                os.unlink(os.path.join(root, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d))
        return True
    except:
        return False



def download(JPN, signalsTable,seq=0,uid='jetppf'):
    """

    :param JPN:
    :param signalsTable: read EFIT data from a signal tabe
    :param seq: sequence, generally reading from public ppf so 0 is default
    :param uid: userif, generally reading from public ppf so jetppf is default
    :return: dictionary containing downloaded signals
    """
    # pdb.set_trace()
    ppf.ppfuid(uid, rw="R")

    ier = ppf.ppfgo(JPN, seq=seq)


    # return a Dictionary of dictionary with value 'v' and time 't' of the exp signal

    signalsName = signalsTable.keys()
    signalsPath = signalsTable.values()
    # _____download signals
    dictDataExp = {}
    for signals in signalsName:
        out_str = '' + signals + ','

        dictDataExpTMP = {}
        signalsPath = signalsTable[signals]

        if signalsPath[0:3] == 'PPF':  # PPF

            dda = signalsPath[4:8]
            dtype = signalsPath[9:]

            if (dda.lower() == 'efit') | (dda.lower() == 'eftp') | (dda.lower() == 'ehtr') :
                ihdat, iwdat, data, x, t, ier = ppf.ppfget(int(JPN), dda, dtype, fix0=0, reshape=0, no_x=0, no_t=0)
                # print(' NO RESHAPE of ' + signalsPath + ' WARNING if is BPCA!!!!!!!!!!!')
            else:
                ihdat, iwdat, data, x, t, ier = ppf.ppfget(int(JPN), dda, dtype, fix0=0, reshape=1, no_x=0, no_t=0)
                # print('RESHAPE of ' + signalsPath)

            dataExp = data
            tExp = t
            dictDataExpTMP['x'] = x  # nr of columns dor a 2D array
        elif signalsPath[0:2] == 'DA' or signalsPath[0:2] == 'PF':  # JPF
            data, tvec, nwds, title, units, ier = getdat.getdat(signalsTable[signals], JPN)
            dataExp = data
            tExp = tvec
        if ier != 0:
            logger.debug('{} Error to retreive it !!'.format(signals))

        else:
            dictDataExpTMP['v'] = dataExp  # signal value
            dictDataExpTMP['t'] = tExp  # signal time
            logger.debug('{} Downloaded !!'.format(signals))


            dictDataExp[signals] = dictDataExpTMP

    return dictDataExp,ier


def readEFITFlux(expDataDictJPNobj,timeEquil):
        # given EFIT exp data it return:
        #  rC,zC: coordinates of the boundary @ t0=timeEquil ,
        # psi: flux map @ t0
        # rGrid,zGrid: coordinates of the EFIT grid where Grad Shafranov Equation is solved

        EFIT = expDataDictJPNobj
        # PSIR_v = EFIT['PSIR']['v']#data
        # PSIR_x = EFIT['PSIR']['x']#nr of elements
        # PSIR_t = EFIT['PSIR']['t']#time
        #
        # PSIZ_v = EFIT['PSIZ']['v']
        # PSIZ_x = EFIT['PSIZ']['x']
        # PSIZ_t = EFIT['PSIZ']['t']
        #
        # rPSI = PSIR_v
        # zPSI = PSIZ_v
        # rGrid,zGrid = np.meshgrid(rPSI,zPSI)


        # PSI_v = EFIT['PSI']['v']
        # PSI_x = EFIT['PSI']['x']
        # PSI_t = EFIT['PSI']['t']
        # psiEFIT =np.reshape(PSI_v,(len(PSI_t),len(PSI_x)))

        RBND_v = EFIT['RBND']['v']
        RBND_x = EFIT['RBND']['x']
        RBND_t = EFIT['RBND']['t']
        rBND = RBND_v
        rC=np.reshape(rBND,(len(RBND_t),len(RBND_x)))

        ZBND_v = EFIT['ZBND']['v']
        ZBND_x = EFIT['ZBND']['x']
        ZBND_t = EFIT['ZBND']['t']
        zBND = ZBND_v
        zC=np.reshape(zBND,(len(ZBND_t),len(ZBND_x)))

        timeEFIT = RBND_t # one of the _t variables
        iCurrentTime, = np.where(
            np.isclose(timeEFIT, timeEquil))  # floating-point
        # iCurrentTime = np.where(np.abs(timeEquil-timeEFIT)<2*min(np.diff(timeEFIT)))# twice of the min of EFIT delta time
        # print(timeEFIT[iCurrentTime])
        # pdb.set_trace()
        iTEFIT = iCurrentTime[0]

        rC0 = rC[iTEFIT,:]
        zC0 = zC[iTEFIT,:]
        # psi0 = psiEFIT[iTEFIT,:]


        return rC0,zC0,timeEFIT



