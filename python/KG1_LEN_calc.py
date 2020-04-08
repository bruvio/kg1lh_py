import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import bisect
import numpy as np
import csv
import math
from scipy.interpolate import interp1d
from scipy.optimize import bisect
from scipy.interpolate import BSpline
import matplotlib.pylab as plt
from MAGTool import *  # Magnetics Tool



def plot_point(point, angle, length):
    '''
    point - Tuple (x, y)
    angle - Angle you want your end point at in degrees.
    length - Length of the line you want to plot.

    Will plot the line on a 10 x 10 plot.
    '''

    # unpack the first point
    x, y = point

    # find the end points
    endy1 = y-length * math.sin(math.radians(angle))
    endy2 = y+length * math.sin(math.radians(angle))
    endx1 = x-length * math.cos(math.radians(angle))
    endx2 = x+length * math.cos(math.radians(angle))
    return endx1,endx2,endy1,endy2
    # plt.plot([endx1, endx2], [endy1, endy2])

    # fig.show()

with open('./e2d_data/vessel_JET_csv.txt', 'rt') as f:
    reader = csv.reader(f, delimiter=';')
    next(reader)
    # col = list(zip(*reader))[1]
    csv_dic = []

    for row in reader:
        csv_dic.append(row);
    # print(csv_dic)
    col1 = []
    col2 = []

    for row in csv_dic:
        col1.append(row[0])
        col2.append(row[1])
    dummy=np.array(col1)
    # print(dummy)
    dummy2=np.array(col2)
    dummy2=[float(i) for i in dummy2]
    z_ves=-np.asarray(dummy2)
    dummy=[float(i) for i in dummy]
    r_ves=np.asarray(dummy)
f.close()


# vessel = [(x, y) for x in r_ves for y in z_ves]

plt.figure(1, figsize=SIZE, dpi=90) #1, figsize=(10, 4), dpi=180)
plt.plot(r_ves, z_ves)

from shapely.geometry import LineString,Polygon,MultiPoint
#definig line of sigths as segments
LOS1 = LineString([(1.885,-5), (1.885,3)])
LOS2 = LineString([(2.6969,-5), (2.6969,3)])
LOS3 = LineString([(3.034,-5), (3.034,3)])
LOS4 = LineString([(3.73,-5), (3.73,3)])

plt.plot([1.885,1.885], [-3,3],label='LOS1')
plt.plot([2.6969,2.6969], [-3,3],label='LOS2')
plt.plot([3.034,3.034], [-3,3],label='LOS3')
plt.plot([3.73,3.73], [-3,3],label='LOS4')


endx1,endx2,endy1,endy2 = plot_point([3.0, -0.6073], math.degrees(0.3901),2)
LOS5 = LineString([(endx1, endy1), (endx2, endy2)])
plt.plot([endx1, endx2], [endy1, endy2],label='LOS5')

endx1,endx2,endy1,endy2 = plot_point([3.0, -0.3727], math.degrees(0.1681),2)
LOS6 = LineString([(endx1, endy1), (endx2, endy2)])
plt.plot([endx1, endx2], [endy1, endy2],label='LOS6')

endx1,endx2,endy1,endy2 = plot_point([3.0, -0.1468], math.degrees(-0.0389),2)
LOS7 = LineString([(endx1, endy1), (endx2, endy2)])
plt.plot([endx1, endx2], [endy1, endy2],label='LOS7')

endx1,endx2,endy1,endy2 = plot_point([3.0, 0.1862], math.degrees(-0.2437),2)
LOS8 =LineString([(endx1, endy1), (endx2, endy2)])
plt.plot([endx1, endx2], [endy1, endy2],label='LOS8')



#define vessel shape
VesselCoordTuple=list(zip(r_ves, z_ves))
polygonVessel = Polygon(VesselCoordTuple)
# compute intersection with LOSx
x1 = polygonVessel.intersection(LOS1)
x2 = polygonVessel.intersection(LOS2)
x3 = polygonVessel.intersection(LOS3)
x4 = polygonVessel.intersection(LOS4)
x5 = polygonVessel.intersection(LOS5)
x6 = polygonVessel.intersection(LOS6)
x7 = polygonVessel.intersection(LOS7)
x8 = polygonVessel.intersection(LOS8)
channels = np.arange(0, 8) + 1
plt.figure(2, figsize=SIZE, dpi=90) #1, figsize=(10, 4), dpi=180)
for chan in channels:
    name = 'x'+str(chan)
    dummy = vars()[name]

    r1 = dummy.xy[0][0]
    z1 = dummy.xy[1][0]
    r2 = dummy.xy[0][1]
    z2 = dummy.xy[1][1]
    x_values = [r1, r2]

    y_values = [z1, z2]

#plotting intersection points
    plt. plot(x_values, y_values,label='LOS'+str(chan))
    # plt.plot(x1.xy[0][0],x1.xy[1][0], color='r',linestyle=' ',marker='x',markersize=5)
    # plt.plot(x1.xy[0][1],x1.xy[1][1], color='r',linestyle=' ',marker='x',markersize=5)
plt.legend(loc='best',fontsize =8)
plt.plot(r_ves, z_ves)
#computing LOS length inside vessel
# cord = np.float64(
#                     math.hypot(
#                         np.float64(r2) - np.float64(r1), np.float64(z2) - np.float64(z1)
#                     )
#                 )
# print(cord)
JPN = 95272
JPNobj = MAGTool(JPN)

nameSignalsTable_EFIT = 'signalsTable_EFIT'  #
nameSignals_EFIT = STJET.signalsTableJET(nameSignalsTable_EFIT)
expDataDictJPNobj_EFIT = JPNobj.download(JPN, nameSignalsTable_EFIT, nameSignals_EFIT, 0)


def readEFITFlux(expDataDictJPNobj, timeEquil):
    # given EFIT exp data it return:
    #  rC,zC: coordinates of the boundary @ t0=timeEquil ,
    # psi: flux map @ t0
    # rGrid,zGrid: coordinates of the EFIT grid where Grad Shafranov Equation is solved

    EFIT = expDataDictJPNobj
    PSIR_v = EFIT['PSIR']['v']  # data
    PSIR_x = EFIT['PSIR']['x']  # nr of elements
    PSIR_t = EFIT['PSIR']['t']  # time

    PSIZ_v = EFIT['PSIZ']['v']
    PSIZ_x = EFIT['PSIZ']['x']
    PSIZ_t = EFIT['PSIZ']['t']

    rPSI = PSIR_v
    zPSI = PSIZ_v
    rGrid, zGrid = numpy.meshgrid(rPSI, zPSI)

    PSI_v = EFIT['PSI']['v']
    PSI_x = EFIT['PSI']['x']
    PSI_t = EFIT['PSI']['t']
    psiEFIT = numpy.reshape(PSI_v, (len(PSI_t), len(PSI_x)))

    RBND_v = EFIT['RBND']['v']
    RBND_x = EFIT['RBND']['x']
    RBND_t = EFIT['RBND']['t']
    rBND = RBND_v
    rC = numpy.reshape(rBND, (len(RBND_t), len(RBND_x)))

    ZBND_v = EFIT['ZBND']['v']
    ZBND_x = EFIT['ZBND']['x']
    ZBND_t = EFIT['ZBND']['t']
    zBND = ZBND_v
    zC = numpy.reshape(zBND, (len(ZBND_t), len(ZBND_x)))

    timeEFIT = RBND_t  # one of the _t variables
    iCurrentTime = numpy.where(
        numpy.abs(timeEquil - timeEFIT) < 2 * min(numpy.diff(timeEFIT)))  # twice of the min of EFIT delta time
    print(timeEFIT[iCurrentTime])

    iTEFIT = iCurrentTime[0][0]

    rC0 = rC[iTEFIT, :]
    zC0 = zC[iTEFIT, :]
    psi0 = psiEFIT[iTEFIT, :]

    # ihdat,iwdat,data,x,t,ier = ppf.ppfget(JPN,'EFIT','PSIR',fix0=0,reshape=0,no_x=0,no_t=0)
    # rPSI = data
    # ihdat,iwdat,data,x,t,ier = ppf.ppfget(JPN,'EFIT','PSIZ',fix0=0,reshape=0,no_x=0,no_t=0)
    # zPSI = data
    # ihdat,iwdat,data,x,t,ier = ppf.ppfget(JPN,'EFIT','RBND',fix0=0,reshape=0,no_x=0,no_t=0)
    # rBND = data
    # rC=np.reshape(rBND,(105,989))
    # ihdat,iwdat,data,x,t,ier = ppf.ppfget(JPN,'EFIT','ZBND',fix0=0,reshape=0,no_x=0,no_t=0)
    # zBND = data
    # zC=np.reshape(zBND,(105,989))
    return rC0, zC0, psi0, rGrid, zGrid, iTEFIT, timeEFIT


LEN1 = []
LEN2 = []
LEN3 = []
LEN4 = []
LEN5 = []
LEN6 = []
LEN7 = []
LEN8 = []
data = SimpleNamespace()
data.pulse = JPN

try:
    data.constants = Consts("consts.ini", __version__)
except KeyError:
    logger.error("\n Could not read in configuration file consts.ini\n")
    sys.exit(65)
data.EFIT_data = {}

# -------------------------------
# 2. Read in EFIT data
# -------------------------------
try:
    data.EFIT_data = EFITData(data.constants)
    ier = data.EFIT_data.read_data(data.pulse, 'kg1l')
except:
    logger.error("\n could not read EFIT data \n")
    return 30

if ier != 0:
    logger.error("\n error reading EFIT data \n")
    return 30


time_efit = data.EFIT_data.rmag.time
ntefit = len(time_efit)
for IT in range(0, ntefit):

    TIMEM = time_efit[IT]
    rC0, zC0, psi0, rGrid, zGrid, iTEFIT, timeEFIT = readEFITFlux(expDataDictJPNobj, TIMEM)
    BoundCoordTuple = list(zip(r_ves, z_ves))
    polygonBound = Polygon(BoundCoordTuple)
    x1 = polygonVessel.intersection(LOS1)
    x2 = polygonVessel.intersection(LOS2)
    x3 = polygonVessel.intersection(LOS3)
    x4 = polygonVessel.intersection(LOS4)
    x5 = polygonVessel.intersection(LOS5)
    x6 = polygonVessel.intersection(LOS6)
    x7 = polygonVessel.intersection(LOS7)
    x8 = polygonVessel.intersection(LOS8)
    for chan in channels:
        name = 'x' + str(chan)
        name_len = 'LEN'+ str(chan)
        dummy = vars()[name]
        len = vars()[name_len]

        r1 = dummy.xy[0][0]
        z1 = dummy.xy[1][0]
        r2 = dummy.xy[0][1]
        z2 = dummy.xy[1][1]
        len.append(np.float64(
            math.hypot(
                np.float64(r2) - np.float64(r1), np.float64(z2) - np.float64(z1)
            )
        ))



plt.figure(3, figsize=SIZE, dpi=90) #1, figsize=(10, 4), dpi=180)
for chan in channels:
    kg1l_len3, dummy = getdata(JPN, 'KG1L', "LEN" + str(chan))
    plt.subplot(chan, 1, 1)
    plt.plot(time_efit,vars()['LEN'+str(chan)],label='LEN'+str(chan))
    plt.plot(
        kg1l_lid3["time"],
        kg1l_lid3["data"],
        label="lid_jetppf_ch" + str(chan),
    )

plt.show()




