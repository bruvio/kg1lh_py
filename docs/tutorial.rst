Tutorial
=========================================

In this section I will explain how to use the tool and what is possible to
achieve.


Installation process
-------------------------------------------
To run and use the code the user must first install it and all its dependacies.

The code is stored in a git repository

from terminal

    >> git clone https://git.ccfe.ac.uk/bviola/kg1lh_py.git -b master /your/folder



Running the code from Terminal
------------------------------------

To tun the code::
    cd /u/username/work/
    python GO_kg1lh.py -h

    usage: GO_kg1lh.py [-h] [-d DEBUG] [-r read_uid]  [-u uid_write]  [-c code]

    Run GO_kg1lh

    optional arguments:
    -h, --help            show this help message and exit
    -d DEBUG, --debug DEBUG
    Debug level. 0: Info, 1: Warning, 2: Debug, 3: Error, 4: Debug plus;
    default level is INFO




Alternatively is possible to run the code specifying the debug level to
increase verbosity and show debug/warning/error messages.

By default the debug level is **INFO**


The Main control function for the code are:

+----+-------------+----------------------------------------------------------------------------------------------+
|-p  | --pulse     | Shot number to process. Shot number must be provided.                                        |
+----+-------------+----------------------------------------------------------------------------------------------+
|-r  | --read_uid  |  UID to read KG1 PPF (EFIT by default is JETPPF) from. Default is "JETPPF".                  |
+----+-------------+----------------------------------------------------------------------------------------------+
|-u  | --uid_write | UID to write KG1 PPF data with. if this is blank then no data is written. Default = "".      |
+----+-------------+----------------------------------------------------------------------------------------------+
|-c  | --code      | Choose which ppf to compute (KG1L/KG1H). Default is KG1L.                                    |
+----+-------------+----------------------------------------------------------------------------------------------+
|-d  | --DEBUG     | Debug level.0: Error, 1: Warning, 2: Info, 3: Debug, 4: Debug Plus. Default is 2.            |
+----+-------------+----------------------------------------------------------------------------------------------+
|-ch | --ch        |      choose how many channels to use. Mostly for testing purposes. Default is 8.             |
+----+-------------+----------------------------------------------------------------------------------------------+
|-a  | --algorithm | choose which algorithm to use for filtering the LIDs. User can choose between:               |
|    |             |-fortran: is the same algorithm used in the fortran code (very slow!)                         |
|    |             |-rolling_mean: filters the data using a rolling mean windows                                  |
|    |             |-rolling_mean_pandas: same as above, using a different Python library (to test code speed)    |
|    |             | Default is rolling_mean.                                                                     |
+----+-------------+----------------------------------------------------------------------------------------------+
|-pl | --plot      | Plot data and save figs: used mostly for debugging and comparing with other version of the   |
|    |             |   code. Default is True.                                                                     |
+----+-------------+----------------------------------------------------------------------------------------------+
|-t  | --test      | Run in test mode. In this mode, the code will compute LADs, LENs and XTAs for each channel   |
|    |             |    if the KG1V/LIDx variables have already been validated. If --uid_write=JETPPF then no PPF |
|    |             |    will be written, to avoid over-writing data. Default = False.                             |
+----+-------------+----------------------------------------------------------------------------------------------+


Return codes:

+---+-----------------------------------------------+
|0  | All OK.                                       |
+---+-----------------------------------------------+
|1  | Some channels were unavailable for processing.|
+---+-----------------------------------------------+
|2  | some channles were not validated.             |
+---+-----------------------------------------------+
|5  | init error.                                   |
+---+-----------------------------------------------+
|9  | No validated LID channels in KG1V.            |
+---+-----------------------------------------------+
|11 |All available channels have validated PPFs.    |
+---+-----------------------------------------------+
|20 | No KG1V data.                                 |
+---+-----------------------------------------------+
|21 | Could not read SF for KG1V.                   |
+---+-----------------------------------------------+
|22 | Error reading KG1V line-of-sight data.        |
+---+-----------------------------------------------+
|23 | could not filter data.                        |
+---+-----------------------------------------------+
|24 | could not perform time loop.                  |
+---+-----------------------------------------------+
|25 | could not filter data.                        |
+---+-----------------------------------------------+
|30 | No EFIT data.                                 |
+---+-----------------------------------------------+
|31 | No points in EFIT.                            |
+---+-----------------------------------------------+
|65 | The initialisation file could not be read in. |
+---+-----------------------------------------------+
|66 | Problem reading the geometry file.            |
+---+-----------------------------------------------+
|67 | Failed to write PPF.                          |
+---+-----------------------------------------------+
|71 | Invalid shot number.                          |
+---+-----------------------------------------------+
|72 | No PPF exists for shot.                       |
+---+-----------------------------------------------+
|100| TEST MODE - NO PPF IS WRITTEN.                |
+---+-----------------------------------------------+
