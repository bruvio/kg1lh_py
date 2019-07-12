from multiprocessing.pool import ThreadPool
from threading import Thread
from ppf import *
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








def getdata(shot,dda,dtype,uid=None):
    ier = ppfgo(shot, seq=0)
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