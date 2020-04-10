import logging
logger = logging.getLogger(__name__)
from shapely.geometry import LineString,Polygon
import matplotlib.pyplot as plt
from types import SimpleNamespace
import numpy as np
import csv
import pdb
import math
import getpass
import os
import time
import argparse
import pathlib
from utility import *
import matplotlib.pylab as plt
from MAGTool import *  # Magnetics Tool
from consts import Consts
from ppf_write import *
from efit_data import EFITData


# ----------------------------
__author__ = "Bruno Viola"
__Name__ = "KG1LH LENs writer"
__version__ = "1"
__release__ = "2"
__maintainer__ = "Bruno Viola"
__email__ = "bruno.viola@ukaea.uk"
__status__ = "Testing"


# define figure size
GM = (math.sqrt(5)-1)/2
W =8
H = GM*W
SIZE = (W,H)

# vessel = [(x, y) for x in r_ves for y in z_ves]


def main(
    JPN,
    code,
    write_uid,
    plot,
    test=False
):
    code_start_time = time.time()
    # -------------------------------
    # 0. Init
    # -------------------------------
    logger.info('\n initializing...')
    DDA=code
    channels = np.arange(0, 8) + 1

    # type_of_ppf = 'public'
    read_uid = 'jetppf'
    type_of_ppf = read_uid

    if "USR" in os.environ:
        logger.log(5, "USR in env")
        # owner = os.getenv('USR')
        owner = os.getlogin()
    else:
        logger.log(5, "using getuser to authenticate")
        import getpass

        owner = getpass.getuser()

    cwd = os.getcwd()
    pathlib.Path(cwd + os.sep + "logFile").mkdir(parents=True, exist_ok=True)

    if DDA =='KG1L':
        EFIT = 'EFIT'
        nameSignalsTable_EFIT = 'signalsTable_EFIT'  #
    else:
        EFIT = 'EHTR'
        nameSignalsTable_EFIT = 'signalsTable_EHTR'  #


    JPNobj = MAGTool(JPN)
    data = SimpleNamespace()
    data.pulse = JPN
    # reading EFIT signal table
    nameSignals_EFIT = STJET.signalsTableJET(nameSignalsTable_EFIT)
    expDataDictJPNobj_EFIT = JPNobj.download(JPN, nameSignalsTable_EFIT, nameSignals_EFIT, 0)

    try:
        data.constants = Consts("consts.ini", __version__)
    except KeyError:
        logger.error("\n Could not read in configuration file consts.ini\n")
        sys.exit(65)
    data.EFIT_data = {}

    # -------------------------------
    # 1. check if there is already a validated public ppf
    # -------------------------------
    # if ((not force) and (write_uid.lower() == 'jetppf')):
    #     logger.info('checking SF of public KG1V ppf')
    #
    #     SF_list_public = check_SF("jetppf", shot_no, 0, dda=code.lower())
    #     if bool(set(SF_list_public) & set([1, 2, 3])):
    #         logger.warning(
    #             "\n \n there is already a saved public PPF with validated channels! \n \n "
    #         )
    #         logger.info(
    #             "\n No PPF was written. \n"
    #         )
    #
    #         logger.info("\n             Finished. \n")
    #         return 100

    # -------------------------------
    # 2. Read in EFIT data
    # -------------------------------
    logger.info('\n reading EFIT time')
    try:
        data.EFIT_data = EFITData(data.constants)
        ier = data.EFIT_data.read_data(data.pulse, 'kg1l')
    except:
        logger.error("\n could not read EFIT data \n")
    logger.info('getting EFIT seq'
    )
    try:
        logging.info('reading {} sequence '.format(EFIT))
        data.unval_seq, data.val_seq = get_min_max_seq(
            data.pulse, dda=EFIT, read_uid='jetppf'
        )
    except TypeError:
        logger.error(
            "impossible to read sequence for user {}".format(read_uid)
        )
    try:
        logging.info('reading {} version '.format(EFIT))
        data.version , dummy = getdata(JPN, "EFIT", "AREA")
        data.version = 0.1
    except:
        logger.error('failed to read {} version'.format(EFIT))

    if DDA == 'KG1L':
        time_efit = data.EFIT_data.rmag.time
    if DDA == 'KG1H':
        time_efit = data.EFIT_data.rmag_fast.time
    ntefit = len(time_efit)

    # -------------------------------
    # 3. reading line of sights
    # -------------------------------
    logging.info("\n reading line of sights")
    try:

        data.r_coord, dummy = getdata(JPN, "KG1V", "R")
        data.r_coord = data.r_coord["data"]
        data.z_coord, dummy = getdata(JPN, "KG1V", "Z")
        data.z_coord = data.z_coord["data"]
        data.a_coord, dummy = getdata(JPN, "KG1V", "A")
        data.a_coord = data.a_coord["data"]

    except:
        logger.error("\n error reading cords coordinates \n")
    # pdb.set_trace()
    LEN1 = []
    LEN2 = []
    LEN3 = []
    LEN4 = []
    LEN5 = []
    LEN6 = []
    LEN7 = []
    LEN8 = []




    print('a ',data.a_coord)
    print('r ',data.r_coord)
    print('z ',data.z_coord)

    # -------------------------------
    # 4. defining line of sigths as segments
    # -------------------------------
    logger.info('\n defining line of sigths as segments')
    LOS1 = LineString([(data.r_coord[0],-5), (data.r_coord[0],3)])
    LOS2 = LineString([(data.r_coord[1],-5), (data.r_coord[1],3)])
    LOS3 = LineString([(data.r_coord[2],-5), (data.r_coord[2],3)])
    LOS4 = LineString([(data.r_coord[3],-5), (data.r_coord[3],3)])

    # plt.figure(1, figsize=SIZE, dpi=90) #1, figsize=(10, 4), dpi=180)
    # plt.plot(r_ves, z_ves)
    #
    # plt.plot([data.r_coord[0],data.r_coord[0]], [-3,3],label='LOS1')
    # plt.plot([data.r_coord[1],data.r_coord[1]], [-3,3],label='LOS2')
    # plt.plot([data.r_coord[2],data.r_coord[2]], [-3,3],label='LOS3')
    # plt.plot([data.r_coord[3],data.r_coord[3]], [-3,3],label='LOS4')


    endx1,endx2,endy1,endy2 = plot_point([data.r_coord[4], data.z_coord[4]], math.degrees(data.a_coord[4]),2)
    LOS5 = LineString([(endx1, endy1), (endx2, endy2)])
    # plt.plot([endx1, endx2], [endy1, endy2],label='LOS5')

    endx1,endx2,endy1,endy2 = plot_point([data.r_coord[5], data.z_coord[5]], math.degrees(data.a_coord[5]),2)
    LOS6 = LineString([(endx1, endy1), (endx2, endy2)])
    # plt.plot([endx1, endx2], [endy1, endy2],label='LOS6')

    endx1,endx2,endy1,endy2 = plot_point([data.r_coord[6], data.z_coord[6]], math.degrees(data.a_coord[6]),2)
    LOS7 = LineString([(endx1, endy1), (endx2, endy2)])
    # plt.plot([endx1, endx2], [endy1, endy2],label='LOS7')

    endx1,endx2,endy1,endy2 = plot_point([data.r_coord[7], data.z_coord[7]], math.degrees(data.a_coord[7]),2)
    LOS8 =LineString([(endx1, endy1), (endx2, endy2)])
    # plt.plot([endx1, endx2], [endy1, endy2],label='LOS8')


    # -------------------------------
    # 5. time loop
    # -------------------------------
    logger.info('\n time loop')
    for IT in range(0, ntefit):

        TIMEM = time_efit[IT]
        # print(TIMEM)
        try:
            rC0,zC0 , psi0, rGrid, zGrid, iTEFIT, timeEFIT = JPNobj.readEFITFlux(expDataDictJPNobj_EFIT, TIMEM)
            BoundCoordTuple = list(zip(rC0, zC0))
            polygonBound = Polygon(BoundCoordTuple)
            x1 = polygonBound.intersection(LOS1)
            x2 = polygonBound.intersection(LOS2)
            x3 = polygonBound.intersection(LOS3)
            x4 = polygonBound.intersection(LOS4)
            x5 = polygonBound.intersection(LOS5)
            x6 = polygonBound.intersection(LOS6)
            x7 = polygonBound.intersection(LOS7)
            x8 = polygonBound.intersection(LOS8)
            for chan in channels:
                # print(chan)
                name = 'x' + str(chan)
                name_len = 'LEN'+ str(chan)
                dummy = vars()[name]
                length = vars()[name_len]
                if is_empty(dummy.bounds):
                    length.append(0)
                else:

                    r1 = dummy.xy[0][0]
                    z1 = dummy.xy[1][0]
                    r2 = dummy.xy[0][1]
                    z2 = dummy.xy[1][1]
                    length.append(np.float64(
                        math.hypot(
                            np.float64(r2) - np.float64(r1), np.float64(z2) - np.float64(z1)
                        )
                    ))
        except:
            print('skipping {}'.format(TIMEM))







    # -------------------------------
    # 6. writing PPFs
    # -------------------------------
    logging.info("\n start writing PPFs \n")
    if (write_uid != "" and not test) or (
        test and write_uid.upper() != "JETPPF" and write_uid != ""
    ):
        logger.info("\n             Writing PPF with UID {}".format(write_uid))

        err = open_ppf(data.pulse, write_uid)

        if err != 0:
            logger.error("\n failed to open ppf \n")
            return err

        itref_kg1v = -1

        for chan in channels:
            dtype_lid = "LEN{}".format(chan)
            comment = "CORD LENGTH KG1 CHANNEL {}".format(chan)

            write_err, itref_written = write_ppf(
                JPN,
                DDA,
                dtype_lid,
                vars()['LEN'+str(chan)],
                time=time_efit,
                comment=comment,
                unitd="M",
                unitt="SEC",
                itref=itref_kg1v,
                nt=len(time_efit),
                status=time_efit,
                global_status=0,
            )
            if write_err != 0:
                logger.error(
                    "Failed to write {}/{}. Errorcode {}".format(
                        DDA, dtype_lid, write_err
                    )
                )
                return write_err


        comment = "Produced by {}".format(owner)
        dtype_mode = "MODE"
        write_err, itref_written = write_ppf(
            data.pulse,
            DDA,
            dtype_mode,
            np.array([1]),
            time=np.array([0]),
            comment=comment,
            unitd=" ",
            unitt=" ",
            itref=-1,
            nt=1,
            status=None,
        )
        if write_err != 0:
            logger.error("failed to write mode ppf")
            return write_err

        comment = "EFIT source"
        if EFIT =='EFIT':

            dtype_mode = "EFIT"
        if EFIT == 'EHTR':
            dtype_mode = "EHTR"
        write_err, itref_written = write_ppf(data.pulse,DDA,dtype_mode,np.array([1]),time=np.array([0]),comment=comment,unitd=" ",unitt=" ",itref=-1,nt=1,status=None)
        if write_err != 0:
            logger.error("failed to write source ppf")
            return write_err

        # writing EFIT seq and version for data provenance
        dtype_mode = "VER"
        comment = EFIT + "version"
        write_err, itref_written = write_ppf(
            data.pulse,
            DDA,
            dtype_mode,
            np.array([data.version]),
            time=np.array([0]),
            comment=comment,
            unitd=" ",
            unitt=" ",
            itref=-1,
            nt=1,
            status=None,
        )
        if write_err != 0:
            logger.error("failed to write version ppf")
            return write_err

        dtype_mode = "SEQ"
        comment = EFIT + "sequence"
        write_err, itref_written = write_ppf(
            data.pulse,
            DDA,
            dtype_mode,
            np.array([data.val_seq]),
            time=np.array([0]),
            comment=comment,
            unitd=" ",
            unitt=" ",
            itref=-1,
            nt=1,
            status=None,
        )
        if write_err != 0:
            logger.error("failed to write version ppf")
            return write_err

        err = close_ppf(JPN, 'bviola', data.constants.code_version, DDA)

        if write_err != 0:
            logger.error("failed to close ppf")
            return err

        logger.info('\n DONE')
    else:
        logger.info(
            "No PPF was written. UID given was {}, test: {}".format(write_uid, test)
        )

    # -------------------------------
    # 7. plotting data and comparison with previous mathod
    # -------------------------------
    if plot:
        try:
            # pdb.set_trace()
            plt.figure(3, figsize=SIZE, dpi=400)  # 1, figsize=(10, 4), dpi=180)
            for chan in channels:
                kg1l_len3, dummy = getdata(JPN, DDA, "LEN" + str(chan))
                if chan == 1:
                    ax_1 = plt.subplot(8, 1, chan)
                    plt.subplot(8, 1, chan)
                    plt.plot(time_efit, vars()['LEN' + str(chan)], label='LEN' + str(chan))
                    plt.plot(
                        kg1l_len3["time"],
                        kg1l_len3["data"],
                        label="LEN_jetppf_ch" + str(chan),
                    )
                    plt.legend(loc='best', fontsize=8)
                else:
                    plt.subplot(8, 1, chan, sharex=ax_1)
                    plt.subplot(8, 1, chan)
                    plt.plot(time_efit, vars()['LEN' + str(chan)], label='LEN' + str(chan))
                    plt.plot(
                        kg1l_len3["time"],
                        kg1l_len3["data"],
                        label="LEN_jetppf_ch" + str(chan),
                    )
                    plt.legend(loc='best', fontsize=8)
            plt.savefig('./figures/overlay_LEN-{}-{}-{}.png'.format(JPN, EFIT, type_of_ppf))

            plt.figure(4, figsize=SIZE, dpi=400)  # 1, figsize=(10, 4), dpi=180)
            f = open('JPN_{}_len_{}-{}'.format(JPN, EFIT, type_of_ppf), 'w')
            for chan in channels:
                kg1l_len3, dummy = getdata(JPN, DDA, "LEN" + str(chan))
                try:
                    if chan == 1:
                        ax_1 = plt.subplot(8, 1, chan)
                        plt.plot(time_efit, abs(kg1l_len3["data"] - vars()['LEN' + str(chan)]),
                                 label='diff-LEN' + str(chan))
                        plt.legend(loc='best', fontsize=8)
                    else:
                        plt.subplot(8, 1, chan, sharex=ax_1)
                        plt.plot(time_efit, abs(kg1l_len3["data"] - vars()['LEN' + str(chan)]),
                                 label='diff-LEN' + str(chan))
                        plt.legend(loc='best', fontsize=8)

                    # print('chan {} mean difference between flush and geometry calc is {}'.format(str(chan),np.mean(abs(kg1l_len3["data"] - vars()['LEN' + str(chan)]))))
                    # print('chan {} median difference between flush and geometry calc is {} \n'.format(str(chan),np.median(abs(kg1l_len3["data"] - vars()['LEN' + str(chan)]))))

                    f.write('chan {} mean difference between flush and geometry calc is {} \n'.format(str(chan), np.mean(
                        abs(kg1l_len3["data"] - vars()['LEN' + str(chan)]))))
                    f.write('chan {} median difference between flush and geometry calc is {} \n'.format(str(chan),
                                                                                                        np.median(abs(
                                                                                                            kg1l_len3[
                                                                                                                "data"] -
                                                                                                            vars()[
                                                                                                                'LEN' + str(
                                                                                                                    chan)]))))
                except:

                    logger.warning('skipping channel {}\n'.format(chan))
                    f.write('skipping channel {}\n'.format(chan))
            f.close()
            plt.savefig('./figures/difference_LEN-{}-{}-{}.png'.format(JPN, EFIT, type_of_ppf))
        except:
            logger.error("\n could not plot data \n")
    if plot:
        plt.show(block=True)

    logger.info("--- {}s seconds ---".format((time.time() - code_start_time)))
    logger.info("\n             Finished.\n")

if __name__ == "__main__":

    # Ensure we are running python 3
    assert sys.version_info >= (3, 5), "Python version too old. Please use >= 3.5.X."

    # Parse arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--pulse", type=int, help="Pulse number to run.", required=True
    )
    parser.add_argument("-c", "--code", help="code to run.", default="KG1L")

    parser.add_argument("-u", "--uid_write", help="UID to write PPFs to.", default="")
    parser.add_argument(
        "-d",
        "--debug",
        type=int,
        help="Debug level. 0: Error, 1: Warning, 2: Info, 3: Debug, 4: Debug Plus",
        default=2,
    )
    # parser.add_argument("-fo", "--force",
    #                     help="forces code execution even when there is already a validated public pulse",
    #                     default=False)





    parser.add_argument("-pl", "--plot", help="plot data: True or False", default=False)

    parser.add_argument(
        "-t",
        "--test",
        help="""Run in test mode. In this mode code will run and if -uw=JETPPF then no PPF will be written, to avoid
                            over-writing validated data.""",
        default=False,
    )

    args = parser.parse_args(sys.argv[1:])

    # Setup the logger
    debug_map = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
        4: 5,
    }

    logging.basicConfig(level=debug_map[args.debug])

    logging.addLevelName(5, "DEBUG_PLUS")

    logger = logging.getLogger(__name__)

    if args.uid_write == "":
        logger.warning("No write UID was given: no PPF will be written.")

    # Call the main code
    main(
        args.pulse,
        args.code,
        args.uid_write,
        args.plot,
        args.test
    )