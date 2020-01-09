from library import *
from utility import *
import logging
import os

logger = logging.getLogger(__name__)

pulselist = [
    94314,
    94316,
    94318,
    94319,
    94320,
    94321,
    94315,
    94323,
    94324,
    94325,
    94326,
    94442,
    94446,
]

code = "kg1l"

logging.info("plotting data and comparison with Fortran code")

channels = np.arange(0, 8) + 1
linewidth = 0.5
markersize = 1
cwd = os.getcwd()


# logging.info('plotting data')
dda = code
# for chan in channels:
#     plt.figure()
#     for shot_no in pulselist:
#         logger.info('pulse {}'.format(shot_no))
#         # loading JETPPF data to use for comparison
#
#         kg1v_lid3, dummy = getdata(shot_no, 'KG1V', 'LID' + str(chan))
#         kg1l_lid3, dummy = getdata(shot_no, dda, 'LID' + str(chan))
#         kg1l_lad3, dummy = getdata(shot_no, dda, 'LAD' + str(chan))
#         kg1l_len3, dummy = getdata(shot_no, dda, 'LEN' + str(chan))
#         kg1l_xtan3, dummy = getdata(shot_no, dda, 'xta' + str(chan))
#
#
#
#         ax_1 = plt.subplot(5, 1, 1)
#         plt.plot(kg1l_lid3['time'], kg1l_lid3['data'],
#                  label='lid_jetppf_ch' + str(chan)+'_' +str(shot_no))
#
#
#         # plt.legend(loc=0, prop={'size': 8})
#         plt.legend(loc='best', prop={'size': 6}, frameon=False, ncol=2)
#
#         plt.subplot(5, 1, 2, sharex=ax_1)
#         plt.plot(kg1l_lad3['time'], kg1l_lad3['data'],
#                  label='lad_jetppf_ch' + str(chan)+'_'+ str(shot_no))
#
#         # plt.legend(loc=0, prop={'size': 8})
#         plt.legend(loc='best', prop={'size': 6}, frameon=False, ncol=2)
#
#         plt.subplot(5, 1, 3, sharex=ax_1)
#         plt.plot(kg1l_xtan3['time'], kg1l_xtan3['data'],
#                  label='xtan_jetppf_ch' + str(chan)+'_' +str(shot_no))
#
#         # plt.legend(loc=0, prop={'size': 8})
#         plt.legend(loc='best', prop={'size': 6}, frameon=False, ncol=2)
#
#         plt.subplot(5, 1, 4, sharex=ax_1)
#         plt.plot(kg1l_len3['time'], kg1l_len3['data'],
#                  label='len_jetppf_ch' + str(chan)+'_'+ str(shot_no))
#
#         # plt.legend(loc=0, prop={'size': 8})
#         plt.legend(loc='best', prop={'size': 6}, frameon=False, ncol=2)
#
#         plt.subplot(5, 1, 5, sharex=ax_1)
#         plt.plot(kg1v_lid3['time'], kg1v_lid3['data'],
#                  label='KG1V_lid_jetppf_ch' + str(chan) + '_' + str(shot_no))
#         # plt.legend(loc=0, prop={'size': 8})
#         plt.legend(loc='best', prop={'size': 6}, frameon=False, ncol=2)
#
#         plt.savefig(cwd + os.sep + 'figures/' + code + '_' + str(
#             shot_no) + 'ch_' + str(chan) + '_comparisons.png', dpi=300)
#
# plt.show()


#
# channels=np.arange(0, 8) + 1
#
shot_no = 88280
for chan in channels:
    plt.figure()

    kg1v_lid3_jetppf, dummy = getdata(shot_no, "KG1V", "LID" + str(chan))

    kg1l_lid3_jetppf, dummy = getdata(shot_no, dda, "LID" + str(chan))
    kg1l_lad3_jetppf, dummy = getdata(shot_no, dda, "LAD" + str(chan))
    kg1l_len3_jetppf, dummy = getdata(shot_no, dda, "LEN" + str(chan))
    kg1l_xtan3_jetppf, dummy = getdata(shot_no, dda, "xta" + str(chan))

    kg1l_lid3_RM, dummy = getdata(
        shot_no, dda, "LID" + str(chan), uid="bviola", seq=432
    )
    kg1l_lad3_RM, dummy = getdata(
        shot_no, dda, "LAD" + str(chan), uid="bviola", seq=432
    )
    kg1l_len3_RM, dummy = getdata(
        shot_no, dda, "LEN" + str(chan), uid="bviola", seq=432
    )
    kg1l_xtan3_RM, dummy = getdata(
        shot_no, dda, "xta" + str(chan), uid="bviola", seq=432
    )

    kg1l_lid3_RMP, dummy = getdata(
        shot_no, dda, "LID" + str(chan), uid="bviola", seq=433
    )
    kg1l_lad3_RMP, dummy = getdata(
        shot_no, dda, "LAD" + str(chan), uid="bviola", seq=433
    )
    kg1l_len3_RMP, dummy = getdata(
        shot_no, dda, "LEN" + str(chan), uid="bviola", seq=433
    )
    kg1l_xtan3_RMP, dummy = getdata(
        shot_no, dda, "xta" + str(chan), uid="bviola", seq=433
    )

    kg1l_lid3_F, dummy = getdata(shot_no, dda, "LID" + str(chan), uid="bviola", seq=434)
    kg1l_lad3_F, dummy = getdata(shot_no, dda, "LAD" + str(chan), uid="bviola", seq=434)
    kg1l_len3_F, dummy = getdata(shot_no, dda, "LEN" + str(chan), uid="bviola", seq=434)
    kg1l_xtan3_F, dummy = getdata(
        shot_no, dda, "xta" + str(chan), uid="bviola", seq=434
    )

    ax_1 = plt.subplot(5, 1, 1)
    plt.plot(
        kg1v_lid3_jetppf["time"],
        kg1v_lid3_jetppf["data"],
        label="lid_jetppf_ch" + str(chan) + "_" + str(shot_no),
    )
    plt.plot(
        kg1l_lid3_RM["time"],
        kg1l_lid3_RM["data"],
        label="lid_RM_ch" + str(chan) + "_" + str(shot_no),
    )
    plt.plot(
        kg1l_lid3_RMP["time"],
        kg1l_lid3_RMP["data"],
        label="lid_RMP_ch" + str(chan) + "_" + str(shot_no),
    )
    plt.plot(
        kg1l_lid3_F["time"],
        kg1l_lid3_F["data"],
        label="lid_F_ch" + str(chan) + "_" + str(shot_no),
    )

    plt.legend(loc="best", prop={"size": 6}, frameon=False, ncol=2)

    plt.subplot(5, 1, 2, sharex=ax_1)
    plt.plot(
        kg1l_lad3_jetppf["time"],
        kg1l_lad3_jetppf["data"],
        label="lad_jetppf_ch" + str(chan) + "_" + str(shot_no),
    )
    plt.plot(
        kg1l_lad3_RM["time"],
        kg1l_lad3_RM["data"],
        label="lad_RM_ch" + str(chan) + "_" + str(shot_no),
    )
    plt.plot(
        kg1l_lad3_RMP["time"],
        kg1l_lad3_RMP["data"],
        label="lad_RMP_ch" + str(chan) + "_" + str(shot_no),
    )
    plt.plot(
        kg1l_lad3_F["time"],
        kg1l_lad3_F["data"],
        label="lad_F_ch" + str(chan) + "_" + str(shot_no),
    )

    # plt.legend(loc=0, prop={'size': 8})
    plt.legend(loc="best", prop={"size": 6}, frameon=False, ncol=2)

    plt.subplot(5, 1, 3, sharex=ax_1)
    plt.plot(
        kg1l_xtan3_jetppf["time"],
        kg1l_xtan3_jetppf["data"],
        label="xtan_jetppf_ch" + str(chan) + "_" + str(shot_no),
    )
    plt.plot(
        kg1l_xtan3_RM["time"],
        kg1l_xtan3_RM["data"],
        label="xtan_RM_ch" + str(chan) + "_" + str(shot_no),
    )
    plt.plot(
        kg1l_xtan3_RMP["time"],
        kg1l_xtan3_RMP["data"],
        label="xtan_RMP_ch" + str(chan) + "_" + str(shot_no),
    )
    plt.plot(
        kg1l_xtan3_F["time"],
        kg1l_xtan3_F["data"],
        label="xtan_F_ch" + str(chan) + "_" + str(shot_no),
    )

    # plt.legend(loc=0, prop={'size': 8})
    plt.legend(loc="best", prop={"size": 6}, frameon=False, ncol=2)

    plt.subplot(5, 1, 4, sharex=ax_1)
    plt.plot(
        kg1l_len3_jetppf["time"],
        kg1l_len3_jetppf["data"],
        label="len_jetppf_ch" + str(chan) + "_" + str(shot_no),
    )
    plt.plot(
        kg1l_len3_RM["time"],
        kg1l_len3_RM["data"],
        label="len_RM_ch" + str(chan) + "_" + str(shot_no),
    )
    plt.plot(
        kg1l_len3_RMP["time"],
        kg1l_len3_RMP["data"],
        label="len_RMP_ch" + str(chan) + "_" + str(shot_no),
    )
    plt.plot(
        kg1l_len3_F["time"],
        kg1l_len3_F["data"],
        label="len_F_ch" + str(chan) + "_" + str(shot_no),
    )

    # plt.legend(loc=0, prop={'size': 8})
    plt.legend(loc="best", prop={"size": 6}, frameon=False, ncol=2)

    plt.subplot(5, 1, 5, sharex=ax_1)
    plt.plot(
        kg1v_lid3_jetppf["time"],
        kg1v_lid3_jetppf["data"],
        label="KG1V_lid_jetppf_ch" + str(chan) + "_" + str(shot_no),
    )
    # plt.legend(loc=0, prop={'size': 8})
    plt.legend(loc="best", prop={"size": 6}, frameon=False, ncol=2)

    plt.savefig(
        cwd
        + os.sep
        + "figures/"
        + code
        + "_"
        + str(shot_no)
        + "ch_"
        + str(chan)
        + "_comparison_algorithms.png",
        dpi=300,
    )

plt.show()
