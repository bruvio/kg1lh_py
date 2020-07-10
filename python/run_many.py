import os
import stat
import sys
  

# Function to add u+x permissions to files
def make_exec(files):
    cwd = os.getcwd()

    if len(files) == 0:
        return

    for filename in files:
        os.chdir(os.getcwd())
        st = os.stat(filename)
        os.chmod(filename, st.st_mode | stat.S_IEXEC)


def make_scripts(shots, code, suffix, write_uid=None):
    if write_uid == None:
        write_uid = "JETPPF"
    else:
        write_uid = write_uid
 
    cwd = os.getcwd()
    # print(cwd)
    os.chdir(os.getcwd())

    shots_list = "_".join(str(x) for x in shots)

    filename = "run_{}_{}.sh".format(code, suffix)
    filename_sub = "sub_run_{}.cmd".format(code, suffix)

    with open(filename, "w") as f_out:
        f_out.write("#!/usr/bin/env bash\n")
        # f_out.write("#!/usr/bin/ bash\n")
        #
        f_out.write(
            "export PYTHONPATH=$PYTHONPATH:/u/bviola/work/Python/KG1L-KG1H/python\n\n"
        )
        f_out.write("export PATH=$PATH:/u/bviola/work/Python/KG1L-KG1H/python\n\n")

        f_out.write("cd /u/bviola/work/Python/KG1L-KG1H/python\n\n")
        for shot in shots:
            f_out.write("echo {}\n".format(shot))
            if code.lower() == "kg1l":
                call_command = "python /u/bviola/work/Python/KG1L-KG1H/python/GO_kg1lh.py -p {} -r JETPPF -u {} -c {} -a rolling_mean \n\n".format(
                    shot, write_uid, code
                )
            else:
                call_command = "python /u/bviola/work/Python/KG1L-KG1H/python/GO_kg1lh.py -p {} -r JETPPF -u {} -c {} -a rolling_mean \n\n".format(
                    shot, write_uid, code
                )

            f_out.write(call_command)
            f_out.write("sleep 15\n")
        f_out.write("cd /u/bviola/work/Python/KG1L-KG1H\n")
    # Change file permission so can be used by batch
    make_exec([filename])

    # Write script for submitting to batch system
    filename_prefix = filename[:-3]

    with open(filename_sub, "w") as f_out:
        script_to_run = "# @ executable = " + filename + "\n"
        f_out.write(script_to_run)
        f_out.write("# @ input = /dev/null\n")
        f_out.write(
            "# @ output = /u/bviola/work/Python/KG1L-KG1H/python/ll_"
            + filename_prefix
            + ".out\n"
        )
        f_out.write(
            "# @ error = /u/bviola/work/Python/KG1L-KG1H/python/ll_"
            + filename_prefix
            + ".err\n"
        )
        f_out.write("# @ initialdir = /u/bviola/work/Python/KG1L-KG1H/python\n")
        f_out.write("# @ notify_user = bviola\n")
        f_out.write("# @ notification = complete\n")
        f_out.write("# @ queue\n")
    make_exec([filename_sub])
    # with open('suball_'+shots_list+'.sh', 'w') as f_out:
    with open("suball" + code + ".sh", "w") as f_out:
        f_out.write("#!/usr/bin/env bash\n")
        command = "llsubmit {}".format(filename_sub)
        f_out.write(command)
    make_exec(["suball" + code + ".sh"])


if __name__ == "__main__":

    # codes = ["kg1h"]
    # codes = ["kg1l"]
    codes = ["kg1l", "kg1h"]
    for code in codes:
        # make_scripts([94325,94326,94442,94446],code)
        # make_scripts([94314,94316,94318,94319,94320,94321,94315,94323,94324,94325,94326,94442,94446],code)
        # make_scripts([94565,94568,94569],code)
        # make_scripts([94315,94323,94324,94325,94326,94442,94446],code)
        # make_scripts(
        #     [88280, 88330, 88430, 88680, 94294, 94502, 94503, 94514, 94524,
        #      94525, 94526, 94527, 94530, 94565, 94568, 94569],code)
        # make_scripts(
        #   [88280, 94502, 94503, 94527],code)

        # make_scripts(
        #     [94886, 94887, 94888, 94889, 94890, 94891, 94892, 94893, 94894,
        #      94895, 94896, 94897, 95128, 95099, 95100, 87587, 81883, 93896,
        #      95089, 95090, 95091, 95092, 95093, 95094, 95097, 95098], code)
        # make_scripts([94908,94910],code)
        # make_scripts([94904,94721,94777,94721,94903,94905,94906,94907,94908,94910],code)
        # make_scripts([94726,94722,94723,94725,94727],code)
        # make_scripts([95008,95009,95010,95011,95012,95013,95014,95015],code)
        # make_scripts([94850,94845,94942,94431],code,write_uid='bviola')
        # make_scripts([95079,95270,95272],code)
        # make_scripts([95340,95369,95361,95336,95367,95360],code)
        # make_scripts([95341,95339,95364,95335,95338,95357,95334,95363,95337,95370,95331,95372,95358],code)
        # carine
        # make_scripts(
        #     [95424, 95425, 95428, 95429, 95430, 95431, 95432, 95433, 95434,
        #      95435, 95437, 95438, 95440, 95441, 95416, 95417, 95419, 95421,
        #      95422, 95423], code)
        # make_scripts([93898,95312,93874,93876],code)
        # make_scripts([95318,95317],code)
        # make_scripts([95128,94953,94954,94955,94956,94957,94958,94959,94960,95352,95353],code)
        # make_scripts([94791,94915,94964,94980],code)
        # make_scripts([95097,95098,95480,95270,91225],code)
        # make_scripts([95097,95098,95480,95270,91225],code)
        # make_scripts([95009,95010,95012,95272],code)
        # make_scripts([91221,91247,95477,95478,95479,95482,95483,95485,95268,95269,95271],code,'10oct2019')
        # make_scripts([94953,94954,94955,94956,94957,94958,94959,94960,95352,95353,95355],code,'03oct2019')
        # make_scripts([95521,95521,95521,95521],code,'testcode')
        # make_scripts([94751,94749,94753,94755,95346,95402,95403,95405,95407,95408],code,'vonthun')
        # make_scripts([94751,94749,94753,94755,95346,95402,95403,95405,95407,95408],code,'vonthun')
        # make_scripts([95294,95288,95289],code,'fernanda')
        # make_scripts([95760,95761,95762,95763,95764,95758,95759],code,'29oct2019')
        #  make_scripts([94837],code,'bart')
        # make_scripts([95854,95855,95857,95858,95859,95860,95861,95862],code,'1nov2019')
        # make_scripts([95882,95883,95884,95886,95887,95889,95890,95891],code,'groth')
        # make_scripts([95891,95892,95893,94441,94442],code,'groth')
        # make_scripts([95939,95940],code,'carine_nov05')
        # make_scripts([95941,95942,95945,95946,95947,95948,95949],code,'carine_nov06')
        # make_scripts([96177,96176,96175,96174],code,'chris')
        # make_scripts([96045,96043,96129,96136,96039],code,'27nov')
        # make_scripts([96127,96175],code,'27nov_2')
        # make_scripts([96043, 96129, 96136],code,'27nov_3')
        # make_scripts([96042,96046,96044,96038, 96137],code,'28nov')
        # make_scripts([96282,96283,96284,96285],code,'nave')
        # make_scripts([96176,96134,96138,96041,96270,96040,96046],code,'2dec')
        # make_scripts([96302,96303,96305,96306,96307,96309,96310,96311],code,'king_dec03')
        # make_scripts([96268,96269,96279,96280,96281,96286],code,'nave_dec03')
        # make_scripts([95612, 95611, 95609, 95607], code, "sergei")
        make_scripts([95616, 95606, 95612, 96759, 96983, 95149, 96916, 96914, 95611, 95152, 97117, 95609, 95153, 95145, 95103, 95615, 96760, 95605, 95610, 95607, 96855, 95141, 95143
        ],code, 'lidall_backlog')
        make_scripts([96137,93901,97128,97124,97129,97125,96912,96909,96133,97126,97139],code,'backlog')
        make_scripts([95616,
96133,
96137,
96139,
96909,
96911,
96912,
96914,
96916,
95307,
95141,
95615,
94120,
95145,
95143,
96299,
87595,
95149,
95152,
95153,
96699,
95303,
95304,
94027,
94028,
95309,
93902,
94031,
95310,
94032,
95308,
93903,
93908,
94033,
93901,
96983,
96855,
95607,
97117,
97124,
97125,
97126,
97128,
97129,
95467,
95468,
95469,
95471,
95472,
87538,
95605,
95606,
96759,
96760,
95609,
95610,
95611,
95612,
95103],code,'backlog')

        # 96137, 93901, 97128, 97124, 97129, 97125, 96912, 96909, 96133, 97126, 97139

        make_scripts([95303,
95467,
95308,
95309,
95469,
95471,
94031,
95310,
94032,
95468,
94028,
94033,
95307,
95303,
95307,
95308,
95309,
95310,
94031,
94032,
94033,
94028,
94684,
95467,
95468,
95469,
95472,
95473,
95474,
95475,
95541,
95616,
96133,
96137,
96139,
96909,
96911,
96912,
96914,
96916,
95307,
95141,
95615,
94120,
95145,
95143,
96299,
87595,
95149,
95152,
95153,
96699,
95303,
95304,
94027,
94028,
95309,
93902,
94031,
95310,
94032,
95308,
93903,
93908,
94033,
93901,
96983,
96855,
95607,
97117,
97124,
97125,
97126,
97128,
97129,
95467,
95468,
95469,
95471,
95472,
87538,
95605,
95606,
96759,
96760,
95609,
95610,
95611,
95612,
95103],code,'backlog_may13')
        make_scripts([95161,
95163,
94667,
94671,
95159,
94674,
94675,
94677,
94679,
95160,
94681,
95162,
94683,
95164,
95165],code,'may222020')
        make_scripts([93885,93888,93890,93882],code,'carine27may202011')
