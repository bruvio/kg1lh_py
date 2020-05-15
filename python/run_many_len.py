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
def make_scripts(shots, suffix, write_uid=None):
    if write_uid == None:
        write_uid = "JETPPF"
    else:
        write_uid = write_uid

    cwd = os.getcwd()
    # print(cwd)
    os.chdir(os.getcwd())

    shots_list = "_".join(str(x) for x in shots)

    filename = "run_{}_{}.sh".format('LEN', suffix)
    filename_sub = "sub_run_{}.cmd".format('LEN', suffix)

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

            call_command = "python /u/bviola/work/Python/KG1L-KG1H/python/KG1_LEN_calc.py -p {} -pl True \n\n".format(
                    shot,
                )


            f_out.write(call_command)
            f_out.write("sleep 15\n")
        f_out.write("cd /u/bviola/work/Python/KG1L-KG1H\n")
    # Change file permission so can be used by batch
    make_exec([filename])

    # Write script for submitting to batch system
    filename_prefix = filename[:-3]


if __name__ == "__main__":
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
                  95103], 'kg1_LEN')
