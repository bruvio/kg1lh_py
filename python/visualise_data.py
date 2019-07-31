
from library import *
from utility import *
import logging
import os
logger = logging.getLogger(__name__)

pulselist = [94314,94316,94318,94319,94320,94321,94315,94323,94324,94325,94326,94442,94446]

code = 'kg1h'

logging.info('plotting data and comparison with Fortran code')

channels = np.arange(0, 8) + 1
linewidth = 0.5
markersize = 1
cwd = os.getcwd()


# logging.info('plotting data')
dda = code
for chan in channels:
    plt.figure()
    for shot_no in pulselist:
        logger.info('pulse {}'.format(shot_no))
        # loading JETPPF data to use for comparison

        kg1v_lid3, dummy = getdata(shot_no, 'KG1V', 'LID' + str(chan))
        kg1l_lid3, dummy = getdata(shot_no, dda, 'LID' + str(chan))
        kg1l_lad3, dummy = getdata(shot_no, dda, 'LAD' + str(chan))
        kg1l_len3, dummy = getdata(shot_no, dda, 'LEN' + str(chan))
        kg1l_xtan3, dummy = getdata(shot_no, dda, 'xta' + str(chan))



        ax_1 = plt.subplot(5, 1, 1)
        plt.plot(kg1l_lid3['time'], kg1l_lid3['data'],
                 label='lid_jetppf_ch' + str(chan)+'_' +str(shot_no))


        # plt.legend(loc=0, prop={'size': 8})
        plt.legend(loc='best', prop={'size': 6}, frameon=False, ncol=2)

        plt.subplot(5, 1, 2, sharex=ax_1)
        plt.plot(kg1l_lad3['time'], kg1l_lad3['data'],
                 label='lad_jetppf_ch' + str(chan)+'_'+ str(shot_no))

        # plt.legend(loc=0, prop={'size': 8})
        plt.legend(loc='best', prop={'size': 6}, frameon=False, ncol=2)

        plt.subplot(5, 1, 3, sharex=ax_1)
        plt.plot(kg1l_xtan3['time'], kg1l_xtan3['data'],
                 label='xtan_jetppf_ch' + str(chan)+'_' +str(shot_no))

        # plt.legend(loc=0, prop={'size': 8})
        plt.legend(loc='best', prop={'size': 6}, frameon=False, ncol=2)

        plt.subplot(5, 1, 4, sharex=ax_1)
        plt.plot(kg1l_len3['time'], kg1l_len3['data'],
                 label='len_jetppf_ch' + str(chan)+'_'+ str(shot_no))

        # plt.legend(loc=0, prop={'size': 8})
        plt.legend(loc='best', prop={'size': 6}, frameon=False, ncol=2)

        plt.subplot(5, 1, 5, sharex=ax_1)
        plt.plot(kg1v_lid3['time'], kg1v_lid3['data'],
                 label='KG1V_lid_jetppf_ch' + str(chan) + '_' + str(shot_no))
        # plt.legend(loc=0, prop={'size': 8})
        plt.legend(loc='best', prop={'size': 6}, frameon=False, ncol=2)

        plt.savefig(cwd + os.sep + 'figures/' + code + '_' + str(
            shot_no) + 'ch_' + str(chan) + '_comparisons.png', dpi=300)

plt.show()