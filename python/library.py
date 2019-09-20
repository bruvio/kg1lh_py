
# ----------------------------
__author__ = "B. Viola"
# ----------------------------
from status_flag import GetSF
import numpy as np
from ppf import *
import logging
from numpy import arange,asscalar
import os

import shutil
logger = logging.getLogger(__name__)



import math
from my_flush import *
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool
from threading import Thread
from ppf import *
from signal_base import SignalBase
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

        threads = [Thread(target=target, args=[arg, i]) for i, arg in enumerate(iterable)]
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








def getdata(shot,dda,dtype,uid=None,seq=None):
    if seq is None:
        ier = ppfgo(shot, seq=0)
    else:
        ier = ppfgo(shot, seq=seq)
    if uid is None:
        ppfuid('jetppf', rw="R")
    else:
        ppfuid(uid, rw="R")
    ihdata, iwdata, data, x, time, ier = ppfget(
        shot, dda, dtype)
    pulse, seq, iwdat, comment, numdda, ddalist, ier = ppfinf(comlen=50,
                                                              numdda=50)

    name=dict()
    name['ihdata']=ihdata
    name['iwdata']=iwdata
    name['data']=data
    name['x']=x
    name['time']=time
    name['ier']=ier
    name['seq']=seq
    name['pulse']=pulse
    name['dda']=dda
    name['dtype']=dtype
    return name,seq

def plottimedata(*args):
    plt.figure()
    for arg in args:

        plt.plot(arg['time'], arg['data'], label=retrieve_name(arg))
        plt.legend(loc=0, prop={'size': 8})

def subplottimedata(*args):
    fig,ax = plt.subplots(nrows=len(args), sharex = True)
    for arg,i in enumerate(args):

            ax[arg].plot(i['time'], i['data'], label=retrieve_name(i))
            ax[arg].legend(loc=0, prop={'size': 8})


        # plt.plot(arg['time'], arg['data'], label=retrieve_name(arg))
        # plt.legend(loc=0, prop={'size': 8})


import inspect


def plot_point(point, angle, length,label):
    '''
    point - Tuple (x, y)
    angle - Angle you want your end point at in degrees.
    length - Length of the line you want to plot.

    Will plot the line on a 10 x 10 plot.
    '''

    # unpack the first point
    x, y = point

    # find the end point
    # endy = length * math.sin(math.radians(angle))
    endy = -length * math.sin(math.degrees(angle))
    # endx = length * math.cos(math.radians(angle))
    endx = -length * math.cos(math.degrees(angle))

    # plot the points
    # fig = plt.figure()
    # ax = plt.subplot(111)
    # ax.set_ylim([0, 10])  # set the bounds to be 10, 10
    # ax.set_xlim([0, 10])
    # plt.plot([x, endx], [y, endy],label=label)
    plt.plot([endx, x], [endy, y],label=label)


    # fig.show()

def retrieve_name(var):
        """
        Gets the name of var. Does it from the out most frame inner-wards.
        :param var: variable to get name from.
        :return: string
        """
        for fi in reversed(inspect.stack()):
            names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
            if len(names) > 0:
                return names[0]



    # cumsum_vec = np.cumsum(np.insert(data.KG1_data.density[1].data, 0, 0))
    # density = (cumsum_vec[rolling_mean:] - cumsum_vec[:-rolling_mean]) / rolling_mean
    #
    #
def movingaverage(interval, window_size):
        window = np.ones(int(window_size)) / float(window_size)
        return np.convolve(interval, window, 'same')
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

def test_logger():
    """
    function to test logger
    :return:
    """
    logger.info('info')
    logger.debug('debug')
    logger.warning('warn')
    logger.error('error')
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
    ax.set_ylim(min(data),
                max(data))
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
            idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(
            value - array[idx])):
        return idx - 1,array[idx - 1]
    else:
        return idx,array[idx]

def find_in_list_array(array,value):
    found=False
    array2list = np.array(array)  # numpy array
    try:
        index = list(array2list).index(value)
        found=True
        return found, index
    except ValueError:
        index =[]
        return found, index

def find_listelements_in_otherlist2(list1,list2,tstep):
    """

    :param list1:
    :param list2:
    :param tstep: minimum distance between two data points
    :return:
    """
    #
    list1=list(list1)
    list2=list(list2)
    # [i for e in list1 for i in list2 if e in i]
    found_list = []
    index_list = []
    for i, value in enumerate(list2):
        found, index = find_in_list_array(list1,value)
        if found:
            index_list.append(index)






    # def find_listelements_in_otherlist(list1,list2):
#     list1=list(list1)
#     list2=list(list2)
#
#     [item for item in list1 if any(x in item for x in list2)]


def find_within_range(array,minvalue,maxvalue):

    # idxmin = (np.abs(array - min)).argmin()
    # idxmax = (np.abs(array - max)).argmax()

    l2 = []
    l3 = []
    if array is None:
        return l3, l2
    else:
        for i,value in enumerate(array):
            if(value >= minvalue and value <= maxvalue):
                l2.append(value)
                l3.append(i)

        return l3,l2



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
    debugger.do_next(None) # run the next command
    users_frame = sys._getframe().f_back # frame where the user invoked `pyqt_set_trace()`
    debugger.interaction(users_frame, None)

def norm(data):
    """normalise data
    """
    return (data)/(max(data)-min(data))

def normalise(signal, kg1_signal, dis_time):
        """

        :param signal:  second trace
        :param kg1_signal: KG1 signal
        :param dis_time: disruption time
        :return: Use ratio of maximum of signal - kg1 as the normalisation factor. Exclude region around the disruption.
        """
        if dis_time > 0:
                ind_dis, = np.where((kg1_signal.time < dis_time - 1))

                max_kg1 = max(kg1_signal.data[ind_dis])
        else:
                max_kg1 = max(kg1_signal.data)

        max_signal = max(signal.data)

        #    print("max kg1 {} max signal {}".format(max_kg1, max_signal))

        if max_signal == 0:
            logger.warning('divide by 0 ')
            max_signal =1


        norm_factor = max_kg1 / max_signal
        dummy = np.multiply(signal.data,norm_factor)

        return dummy

def get_seq(shot_no, dda, read_uid="JETPPF"):
    """

    :param shot_no: pulse number
    :param dda:
    :param read_uid:
    :return: get sequence of a ppf
    """
    ier = ppfgo(shot_no, seq=0)
    if ier != 0:
        return None

    ppfuid(read_uid, rw="R")

    iseq, nseq, ier = ppfdda(shot_no, dda)

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
    kg1v_seq = get_seq(shot_no, dda,read_uid)
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





def check_SF(read_uid,pulse):
    """

    :param read_uid:
    :param pulse:
    :return: list of Status Flags
    """
    logging.info('\n')
    logging.info('checking status FLAGS ')

    ppfuid(read_uid, "r")

    ppfssr([0, 1, 2, 3, 4])

    channels = arange(0, 8) + 1
    SF_list = []

    pulse = int(pulse)

    for channel in channels:
            ch_text = 'lid' + str(channel)

            st_ch = GetSF(pulse, 'kg1v', ch_text)
            st_ch = asscalar(st_ch)
            SF_list.append(st_ch)
    logging.info('%s has the following SF %s', str(pulse), SF_list)

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

        with open(filename, 'r') as f_in:
           lines = f_in.readlines()
           for index, line in enumerate(lines):
            if "shot" in str(line):
                dummy = lines[index].split()
                shot = int(dummy[1])
                user = str(dummy[3])
                date = str(dummy[5])

                # #             dummy = lines[index + 1].split()
                sequence = (dummy[7])

                writtenby = (dummy[10])
                # #
                #             month =(dummy[6])
                #             day =(dummy[7])
                #             year =(dummy[9])
                #             logging.info(month,day,year)
                #             date = datetime.date(int(year),strptime(month,'%b').tm_mon , int(day))

                # logging.info(shot, user, date, sequence,writtenby)
                # return
                string_to_write = (
                    "shot: {} user: {} date: {} seq: {} by: {}\n".format(
                        str(shot).strip(),
                        user.strip(),
                        str(date).strip(),
                        str(sequence).strip(),
                        writtenby.strip()))

                if os.path.exists(outputfile):
                    if check_string_in_file(outputfile, string_to_write):
                        pass
                    else:
                        with open(outputfile, 'a+') as f_out:
                            f_out.write(string_to_write)
                        f_out.close()
                else:
                    with open(outputfile, 'a+') as f_out:
                        f_out.write(string_to_write)
                    f_out.close()

        f_in.close()
    else:
        f_in = open(filename, "w")
        f_in.close()
        string_to_write = (
            "shot: {} user: {} date: {} seq: {} by: {}\n".format(str(00000),
                                                                 'unknown',
                                                                 str('00-00-00'),
                                                                 str(000),
                                                                 'unknown'))
        if os.path.exists(outputfile):
            if check_string_in_file(outputfile, string_to_write):
                pass
            else:
                with open(outputfile, 'a+') as f_out:
                    f_out.write(string_to_write)
                f_out.close()
        else:
            with open(outputfile, 'a+') as f_out:
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
        firstFile = open(firstFile , 'rb')
        secondFile =  open(secondFile  , 'rb')
        buf1 = firstFile.read(blocksize)
        buf2 = secondFile.read(blocksize)
    while len(buf1) > 0:
        if buf1!=buf2:
            return False
        buf1, buf2 = firstFile.read(blocksize), secondFile.read(blocksize)
    return True
# =============================================================================
def pyqt_set_trace():
    '''Set a tracepoint in the Python debugger that works with Qt'''
    from PyQt4.QtCore import pyqtRemoveInputHook
    import pdb
    import sys
    pyqtRemoveInputHook()
    # set up the debugger
    debugger = pdb.Pdb()
    debugger.reset()
    # custom next to get outside of function scope
    debugger.do_next(None) # run the next command
    users_frame = sys._getframe().f_back # frame where the user invoked `pyqt_set_trace()`
    debugger.interaction(users_frame, None)
# 
# =============================================================================
def copy_changed_kg1_to_save(src,dst,filename):
    """

    :param src:
    :param dst:
    :param filename:
    :return: copies file from src folder to dst
    """

    src='./'+src+'/'+filename
    dst='./'+dst+'/'+filename
    
    copyfile(src, dst)
    
    
    
#-----------------------

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
    


#--------


def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

#--------
def map_kg1_efit_RM_pandas(arg):
    """
    new algorithm to filter kg1v/lid data using pandas rolling mean

    the sampling windown is computed as ratio between the "old fortran" rampling and the kg1v sampling
    :param arg:
    :return:
    """
    data = arg[0] # struct containing all data
    chan = arg[1] # channel to analyse
    # pdb.set_trace()
    if data.KG1_data.global_status[chan] >3:
        logger.warning('channel data is not good - skipping ch. {}!'.format(chan))
        return (data,chan)

    if data.code.lower()=='kg1l':
        ntime_efit = len(data.EFIT_data.rmag.time)
        time_efit = data.EFIT_data.rmag.time
        data_efit = data.EFIT_data.rmag.data
        data.EFIT_data.sampling_time = np.mean(np.diff(data.EFIT_data.rmag.time))
        ntkg1v = len(data.KG1_data.density[chan].time)
        tkg1v = data.KG1_data.density[chan].time
        sampling_time_kg1v = np.mean(np.diff(tkg1v))
        tsmo = data.KG1LH_data.tsmo
        rolling_mean = int(round(tsmo/sampling_time_kg1v))

    else:
        ntime_efit = len(data.EFIT_data.rmag_fast.time)
        time_efit = data.EFIT_data.rmag_fast.time
        data_efit = data.EFIT_data.rmag_fast.data
        data.EFIT_data.sampling_time = np.mean(np.diff(data.EFIT_data.rmag_fast.time))
        ntkg1v = len(data.KG1_data.density[chan].time)
        tkg1v = data.KG1_data.density[chan].time
        sampling_time_kg1v = np.mean(np.diff(tkg1v))
        tsmo = data.KG1LH_data.tsmo
        rolling_mean = int(round(sampling_time_kg1v / tsmo))



    # density = pd.rolling_mean(data.KG1_data.density[chan].data,rolling_mean)
    density2 = pd.Series(data.KG1_data.density[chan].data).rolling(window=rolling_mean).mean()
    density2.fillna(0,inplace=True)




    data.KG1LH_data.lid[chan] = SignalBase(data.constants)
    data.KG1LH_data.lid[chan].data = density2
    data.KG1LH_data.lid[chan].time = data.KG1_data.density[chan].time
    if data.interp_method == 'interp':
        dummy, dummy_time = data.KG1LH_data.lid[chan].resample_signal(
            data.interp_method, time_efit)
    if data.interp_method == 'interp_ZPS':
        dummy,dummy_time = data.KG1LH_data.lid[chan].resample_signal(data.interp_method, time_efit)

    data.KG1LH_data.lid[chan].data = np.empty(ntime_efit)
    data.KG1LH_data.lid[chan].time = np.empty(ntime_efit)

    data.KG1LH_data.lid[chan].data = dummy
    data.KG1LH_data.lid[chan].time = dummy_time








    return (data,chan)



#--------
def map_kg1_efit_RM(arg):
    """
    new algorithm to filter kg1v/lid data using rolling mean
    the sampling window is computed as ratio between the "old fortran" rampling and the kg1v sampling
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
        time_efit = data.EFIT_data.rmag.time
        data_efit = data.EFIT_data.rmag.data
        data.EFIT_data.sampling_time = np.mean(np.diff(data.EFIT_data.rmag.time))
        ntkg1v = len(data.KG1_data.density[chan].time)
        tkg1v = data.KG1_data.density[chan].time
        sampling_time_kg1v = np.mean(np.diff(tkg1v))
        tsmo = data.KG1LH_data.tsmo
        rolling_mean = int(round(tsmo/sampling_time_kg1v))

    else:
        ntime_efit = len(data.EFIT_data.rmag_fast.time)
        time_efit = data.EFIT_data.rmag_fast.time
        data_efit = data.EFIT_data.rmag_fast.data
        data.EFIT_data.sampling_time = np.mean(np.diff(data.EFIT_data.rmag_fast.time))
        ntkg1v = len(data.KG1_data.density[chan].time)
        tkg1v = data.KG1_data.density[chan].time
        sampling_time_kg1v = np.mean(np.diff(tkg1v))
        tsmo = data.KG1LH_data.tsmo
        rolling_mean = int(round(sampling_time_kg1v / tsmo))




    # pdb.set_trace()
    cumsum_vec = np.cumsum(np.insert(data.KG1_data.density[chan].data, 0, 0))
    density_cms = (cumsum_vec[rolling_mean:] - cumsum_vec[:-rolling_mean]) / rolling_mean
    density1 = movingaverage(data.KG1_data.density[chan].data, rolling_mean)

    data.KG1LH_data.lid[chan] = SignalBase(data.constants)
    data.KG1LH_data.lid[chan].data = density1
    data.KG1LH_data.lid[chan].time = data.KG1_data.density[chan].time
    # data.KG1LH_data.lid[chan].time = time_efit
    if data.interp_method == 'interp':
        dummy, dummy_time = data.KG1LH_data.lid[chan].resample_signal(
            data.interp_method, time_efit)
    if data.interp_method == 'interp_ZPS':
        dummy,dummy_time = data.KG1LH_data.lid[chan].resample_signal(data.interp_method, time_efit)

    data.KG1LH_data.lid[chan].time = np.empty(ntime_efit)

    data.KG1LH_data.lid[chan].data = dummy
    data.KG1LH_data.lid[chan].time = dummy_time




    return (data,chan)


#--------
def map_kg1_efit(arg):
    """
    original algorithm used in kg1l fortran code to filter kg1v/lid data
    :param arg:
    :return:
    """
    data = arg[0] # struct containing all data
    chan = arg[1] # channel to analyse

    if data.KG1_data.global_status[chan] >3:
        logger.warning('channel data is not good - skipping ch {}!'.format(chan))
        return (data,chan)

    if data.code.lower()=='kg1l':
        ntime_efit = len(data.EFIT_data.rmag.time)
        time_efit = data.EFIT_data.rmag.time
        data_efit = data.EFIT_data.rmag.data
        data.EFIT_data.sampling_time = np.mean(np.diff(data.EFIT_data.rmag.time))

    else:
        ntime_efit = len(data.EFIT_data.rmag_fast.time)
        time_efit = data.EFIT_data.rmag_fast.time
        data_efit = data.EFIT_data.rmag_fast.data
        data.EFIT_data.sampling_time = np.mean(np.diff(data.EFIT_data.rmag_fast.time))

    density = np.zeros(ntime_efit)
    ntkg1v = len(data.KG1_data.density[chan].time)
    tkg1v = data.KG1_data.density[chan].time
    tsmo = data.KG1LH_data.tsmo



    for it in range(0, ntime_efit):
        # pdb.set_trace()
        sum = np.zeros(8)

        nsum = 0

        tmin = 1000.0

        jmin = 1

        # in principle they can be different (?!)
        ntkg1v = len(data.KG1_data.density[chan].time)
        tkg1v = data.KG1_data.density[chan].time

        for jj in range(0, ntkg1v):
            tdif = abs(tkg1v[jj] - time_efit[it])

            if (tdif < tmin):
                tmin = tdif
                jmin = jj
            if (tkg1v[jj] >= time_efit[it] + tsmo):
                break
            if (tkg1v[jj] > time_efit[it] - tsmo):
                sum[chan - 1] = sum[chan - 1] + \
                                data.KG1_data.density[chan].data[jj]
                nsum = nsum + 1
        if nsum > 0:
            density[it] = sum[chan - 1] / float(nsum)
        else:
            density[it] = data.KG1_data.density[chan].data[jmin]






    data.KG1LH_data.lid[chan] = SignalBase(data.constants)
    data.KG1LH_data.lid[chan].data = density
    data.KG1LH_data.lid[chan].time = time_efit
    # data.KG1LH_data.lid[chan].time = time_efit
    if data.interp_method == 'interp':
        dummy, dummy_time = data.KG1LH_data.lid[chan].resample_signal(
            data.interp_method, time_efit)
    if data.interp_method == 'interp_ZPS':
        dummy,dummy_time = data.KG1LH_data.lid[chan].resample_signal(data.interp_method, time_efit)
    data.KG1LH_data.lid[chan].time = np.empty(ntime_efit)

    data.KG1LH_data.lid[chan].data = dummy
    data.KG1LH_data.lid[chan].time = dummy_time

    return (data,chan)


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


