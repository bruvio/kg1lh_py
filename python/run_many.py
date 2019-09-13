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


def make_scripts(shots, code):
    cwd = os.getcwd()
    # print(cwd)
    os.chdir(os.getcwd())

    shots_list = '_'.join(str(x) for x in shots)

    filename = "run_{}.sh".format(code)
    filename_sub = "sub_run_{}.cmd".format(code)

    with open(filename, 'w') as f_out:
        f_out.write("#!/usr/bin/env bash\n")
        # f_out.write("#!/usr/bin/ bash\n")
        #
        f_out.write(
            "export PYTHONPATH=$PYTHONPATH:/u/bviola/work/Python/KG1L-KG1H/python\n\n")
        f_out.write(
            "export PATH=$PATH:/u/bviola/work/Python/KG1L-KG1H/python\n\n")

        f_out.write("cd /u/bviola/work/Python/KG1L-KG1H/python\n\n")
        for shot in shots:
            f_out.write("echo {}\n".format(shot))

            call_command = "python /u/bviola/work/Python/KG1L-KG1H/python/GO_kg1lh.py -p {} -r JETPPF -u JETPPF -c {} -a rolling_mean_pandas \n\n".format(
                shot, code)
            f_out.write(call_command)
            f_out.write('sleep 5\n')
        f_out.write("cd /u/bviola/work/Python/KG1L-KG1H\n")
    # Change file permission so can be used by batch
    make_exec([filename])

    # Write script for submitting to batch system
    filename_prefix = filename[:-3]

    with open(filename_sub, 'w') as f_out:
        script_to_run = "# @ executable = " + filename + "\n"
        f_out.write(script_to_run)
        f_out.write("# @ input = /dev/null\n")
        f_out.write(
            "# @ output = /u/bviola/work/Python/KG1L-KG1H/python/ll_" + filename_prefix + ".out\n")
        f_out.write(
            "# @ error = /u/bviola/work/Python/KG1L-KG1H/python/ll_" + filename_prefix + ".err\n")
        f_out.write("# @ initialdir = /u/bviola/work/Python/KG1L-KG1H/python\n")
        f_out.write("# @ notify_user = bviola\n")
        f_out.write("# @ notification = complete\n")
        f_out.write("# @ queue\n")
    make_exec([filename_sub])
    # with open('suball_'+shots_list+'.sh', 'w') as f_out:
    with open('suball' + code + '.sh', 'w') as f_out:
        f_out.write("#!/usr/bin/env bash\n")
        command = 'llsubmit {}'.format(filename_sub)
        f_out.write(command)
    make_exec(['suball' + code + '.sh'])


if __name__ == "__main__":

    #codes = ["kg1l"]
    codes = ["kg1l","kg1h"]
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
        make_scripts([94726,94722,94723,94725,94727],code)