import numpy as N
import py_c_flush as Flush


# ********************************************
#
# Wrappers for Flush routines
#
# Author : Stanislas PAMELA
# Generated by perl script, see 
# /home/flush/ITMflush/trunk/src/wrappers/python/scripts/ 
#
# Date : 7 Jun 2018
#
# See Flush manual for routine description
#
# ********************************************


def flushinit(igo, ishot, time, lunget, iseq, uid, dda, lunmsg):
 time, ier = Flush.py_flushinit(igo, ishot, time, lunget, iseq, uid, dda, lunmsg)
 return time, ier
	


def initPsi(npsi, nr, nz, r, z, psi, units):
 ier = Flush.py_initPsi(npsi, nr, nz, r, z, psi, units)
 return ier
	


def initProfiles(npsi, nr, nz, nprof, r, z, psi, fprof, qprof, pprof, psiprof, units):
 ier = Flush.py_initProfiles(npsi, nr, nz, nprof, r, z, psi, fprof, qprof, pprof, psiprof, units)
 return ier
	


def initFixedBoundary(npsi, nr, nz, nprof, r, z, psi, fprof, qprof, pprof, psiprof, rXpoint, zXpoint, psiXpoint, nXpoint, units):
 ier = Flush.py_initFixedBoundary(npsi, nr, nz, nprof, r, z, psi, fprof, qprof, pprof, psiprof, rXpoint, zXpoint, psiXpoint, nXpoint, units)
 return ier
	


def initFlush(npsi, nr, nz, nprof, r, z, psi, fprof, qprof, pprof, psiprof, rXpoint, zXpoint, psiXpoint, nXpoint, fixed, units, spline):
 ier = Flush.py_initFlush(npsi, nr, nz, nprof, r, z, psi, fprof, qprof, pprof, psiprof, rXpoint, zXpoint, psiXpoint, nXpoint, fixed, units, spline)
 return ier
	


def flushquickinit(pulse, time):
 time, ier = Flush.py_flushquickinit(pulse, time)
 return time, ier
	


def Flush_saveEfitFile(pulse, seq, uid, dda, filename, uidLen, ddaLen, filenameLen):
 ier = Flush.py_Flush_saveEfitFile(pulse, seq, uid, dda, filename, uidLen, ddaLen, filenameLen)
 return ier
	


def Flush_getMagAxisFlux():
 flux, ier = Flush.py_Flush_getMagAxisFlux()
 return flux, ier
	


def Flush_writeCoeffsToFile(filename, pulse, time, flen):
 Flush.py_Flush_writeCoeffsToFile(filename, pulse, time, flen)
 return
	


def Flush_fitCubicSpline(npoints, x, y, bcswitch, bcs):
 coef, ier = Flush.py_Flush_fitCubicSpline(npoints, x, y, bcswitch, bcs)
 return coef, ier
	


def Flush_findCubicSplineParameter(yvalue, nknots, xknots, yknots, yspline, accuracy):
 knotfound, xfound, ier = Flush.py_Flush_findCubicSplineParameter(yvalue, nknots, xknots, yknots, yspline, accuracy)
 return knotfound, xfound, ier
	


def Flush_evaluateCubicSpline(iKnot, xValue, nSpline, yKnots, ySpline):
 yValue, ier = Flush.py_Flush_evaluateCubicSpline(iKnot, xValue, nSpline, yKnots, ySpline)
 return yValue, ier
	


def Flush_evaluateFirstDerivativeCubicSpline(iKnot, xValue, nSpline, ySpline):
 deriv, ier = Flush.py_Flush_evaluateFirstDerivativeCubicSpline(iKnot, xValue, nSpline, ySpline)
 return deriv, ier
	


def Flush_evaluateSecondDerivativeCubicSpline(iKnot, xValue, nSpline, ySpline):
 deriv, ier = Flush.py_Flush_evaluateSecondDerivativeCubicSpline(iKnot, xValue, nSpline, ySpline)
 return deriv, ier
	


def Flush_getPsiGivenQ(nq, q):
 psi, ier = Flush.py_Flush_getPsiGivenQ(nq, q)
 return psi, ier
	


def Flush_getMinMaxFluxBetweenPoints(rBeg, zBeg, rEnd, zEnd, accuracy):
 rMin, zMin, fluxMin, rMax, zMax, fluxMax, nMinMax, indexMinMax, rMinMax, zMinMax, fluxMinMax, ier = Flush.py_Flush_getMinMaxFluxBetweenPoints(rBeg, zBeg, rEnd, zEnd, accuracy)
 return rMin, zMin, fluxMin, rMax, zMax, fluxMax, nMinMax, indexMinMax, rMinMax, zMinMax, fluxMinMax, ier
	


def Flush_getLineBoxIntersection(r, z, angle):
 rLeft, zLeft, rRight, zRight, ier = Flush.py_Flush_getLineBoxIntersection(r, z, angle)
 return rLeft, zLeft, rRight, zRight, ier
	


def Flush_getAllIntersections(r, z, angle, nflux, flux, accuracy):
 nInt, rInt, zInt, ier = Flush.py_Flush_getAllIntersections(r, z, angle, nflux, flux, accuracy)
 return nInt, rInt, zInt, ier
	


def Flush_getIntersectionWithBlownUpSurface(r, z, angle, psi, blow):
 rInt1, zInt1, rInt2, zInt2, ier = Flush.py_Flush_getIntersectionWithBlownUpSurface(r, z, angle, psi, blow)
 return rInt1, zInt1, rInt2, zInt2, ier
	


def Flush_getTangentsToSurfaces(r, z, angle, nflux, flux, iSide, accuracy, nBeams):
 rTan, zTan, ier = Flush.py_Flush_getTangentsToSurfaces(r, z, angle, nflux, flux, iSide, accuracy, nBeams)
 return rTan, zTan, ier
	


def Flush_getSurfaceElongation(nflux, flux):
 elongation, ier = Flush.py_Flush_getSurfaceElongation(nflux, flux)
 return elongation, ier
	


def Flush_transformCoordinatesPsiThetaToRZ(nPsi, psi, nTheta, theta):
 R, Z, dRdTheta, dZdTheta, d2RdTheta2, d2ZdTheta2, ier = Flush.py_Flush_transformCoordinatesPsiThetaToRZ(nPsi, psi, nTheta, theta)
 return R, Z, dRdTheta, dZdTheta, d2RdTheta2, d2ZdTheta2, ier
	


def Flush_blowUpSurface(psi, delta, nSurf, rSurf, zSurf):
 rSurf, zSurf, ier = Flush.py_Flush_blowUpSurface(psi, delta, nSurf, rSurf, zSurf)
 return rSurf, zSurf, ier
	


def Flush_getFluxLabelRho(nPsi, psi):
 rho, ier = Flush.py_Flush_getFluxLabelRho(nPsi, psi)
 return rho, ier
	


def Flush_getdVda(nPsi, psi):
 dVda, ier = Flush.py_Flush_getdVda(nPsi, psi)
 return dVda, ier
	


def Flush_getFluxAveragedQuantities(npsi, psi, nQuantities, quantitiesNames, quantitiesNamesLen):
 quantities, ier = Flush.py_Flush_getFluxAveragedQuantities(npsi, psi, nQuantities, quantitiesNames, quantitiesNamesLen)
 return quantities, ier
	


def Flush_getPoloidalMagneticEnergy(nPsi, psi):
 WBp, ier = Flush.py_Flush_getPoloidalMagneticEnergy(nPsi, psi)
 return WBp, ier
	


def Flush_getJB(nPsi, psi):
 JB, ier = Flush.py_Flush_getJB(nPsi, psi)
 return JB, ier
	


def Flush_getVolume(nPsi, psi):
 volume, ier = Flush.py_Flush_getVolume(nPsi, psi)
 return volume, ier
	


def Flush_getSurfaceVolume(nSurf, rSurf, zSurf):
 volume, ier = Flush.py_Flush_getSurfaceVolume(nSurf, rSurf, zSurf)
 return volume, ier
	


def Flush_quickSort(firstindex, lastindex, dblarray, index):
 dblarray, index = Flush.py_Flush_quickSort(firstindex, lastindex, dblarray, index)
 return dblarray, index
	


def Flush_intersectionBetweenSurfaceSegmentAndPolygon(nLim, rLim, zLim, psiSeg, rSeg, zSeg):
 rInt, zInt, ier = Flush.py_Flush_intersectionBetweenSurfaceSegmentAndPolygon(nLim, rLim, zLim, psiSeg, rSeg, zSeg)
 return rInt, zInt, ier
	


def Flush_writeEQDSKfile(filename, nR, nZ, filenameLen):
 ier = Flush.py_Flush_writeEQDSKfile(filename, nR, nZ, filenameLen)
 return ier
	


def Flush_e02daf(m, nrknots, nzknots, rgrid, zgrid, psigrid, w, rknots, zknots, point, npoint, nCoeff, ws, nws, eps, ifail):
 rknots, zknots, dl, coeff, sigma, rank, ifail = Flush.py_Flush_e02daf(m, nrknots, nzknots, rgrid, zgrid, psigrid, w, rknots, zknots, point, npoint, nCoeff, ws, nws, eps, ifail)
 return rknots, zknots, dl, coeff, sigma, rank, ifail
	


def Flush_e02dcf(start, nr, rgrid, nz, zgrid, psigrid, smooth, nrmax, nzmax, nrknots, rknots, nzknots, zknots, lwork, nlwork, iwork, niwork, ifail):
 nrknots, rknots, nzknots, zknots, fspline, residual, ifail = Flush.py_Flush_e02dcf(start, nr, rgrid, nz, zgrid, psigrid, smooth, nrmax, nzmax, nrknots, rknots, nzknots, zknots, lwork, nlwork, iwork, niwork, ifail)
 return nrknots, rknots, nzknots, zknots, fspline, residual, ifail
	


def Flush_e02zaf(nrknots, nzknots, rknots, zknots, m, rgrid, zgrid, npoint, adres, nadres, ifail):
 point, ifail = Flush.py_Flush_e02zaf(nrknots, nzknots, rknots, zknots, m, rgrid, zgrid, npoint, adres, nadres, ifail)
 return point, ifail
	


def Flush_getError(errnumber):
 errnumber = Flush.py_Flush_getError(errnumber)
 return errnumber
	


def Flush_getSplineDomainLimits():
 rMin, zMin, rMax, zMax, ier = Flush.py_Flush_getSplineDomainLimits()
 return rMin, zMin, rMax, zMax, ier
	


def Flush_getFlux(np, r, z):
 f, ier = Flush.py_Flush_getFlux(np, r, z)
 return f, ier
	


def Flush_getAbsoluteFlux(np, r, z):
 f, ier = Flush.py_Flush_getAbsoluteFlux(np, r, z)
 return f, ier
	


def Flush_getMagAxis():
 rAxis, zAxis, fAxis, ier = Flush.py_Flush_getMagAxis()
 return rAxis, zAxis, fAxis, ier
	


def Flush_getXpoint():
 nX, rX, zX, fX, ier = Flush.py_Flush_getXpoint()
 return nX, rX, zX, fX, ier
	


def Flush_getAllXpoints():
 nX, rX, zX, fX, ier = Flush.py_Flush_getAllXpoints()
 return nX, rX, zX, fX, ier
	


def Flush_getlcfsFlux():
 lcfsFlux, ier = Flush.py_Flush_getlcfsFlux()
 return lcfsFlux, ier
	


def Flush_getLCFSboundary(accuracy):
 nLCFS, rLCFS, zLCFS, ier = Flush.py_Flush_getLCFSboundary(accuracy)
 return nLCFS, rLCFS, zLCFS, ier
	


def Flush_getMidPlaneLCFSIntersection():
 rOuter, rInner, ier = Flush.py_Flush_getMidPlaneLCFSIntersection()
 return rOuter, rInner, ier
	


def Flush_getNormalisedMinorRadius(np, flux):
 minRad, ier = Flush.py_Flush_getNormalisedMinorRadius(np, flux)
 return minRad, ier
	


def Flush_getNormMinorRadiusProj(np, r, z):
 minRad, ier = Flush.py_Flush_getNormMinorRadiusProj(np, r, z)
 return minRad, ier
	


def Flush_getMinorRadius(np, flux):
 minRad, ier = Flush.py_Flush_getMinorRadius(np, flux)
 return minRad, ier
	


def Flush_getMidPlaneProjection(np, r, z):
 minRad, ier = Flush.py_Flush_getMidPlaneProjection(np, r, z)
 return minRad, ier
	


def Flush_getMidPlaneProjRight(np, r, z):
 minRad, ier = Flush.py_Flush_getMidPlaneProjRight(np, r, z)
 return minRad, ier
	


def Flush_getdpsidr(np, r, z):
 deriv, ier = Flush.py_Flush_getdpsidr(np, r, z)
 return deriv, ier
	


def Flush_getdpsidz(np, r, z):
 deriv, ier = Flush.py_Flush_getdpsidz(np, r, z)
 return deriv, ier
	


def Flush_getdpsidrdr(np, r, z):
 deriv, ier = Flush.py_Flush_getdpsidrdr(np, r, z)
 return deriv, ier
	


def Flush_getdpsidrdz(np, r, z):
 deriv, ier = Flush.py_Flush_getdpsidrdz(np, r, z)
 return deriv, ier
	


def Flush_getdpsidzdz(np, r, z):
 deriv, ier = Flush.py_Flush_getdpsidzdz(np, r, z)
 return deriv, ier
	


def Flush_getdpsidzdr(np, r, z):
 deriv, ier = Flush.py_Flush_getdpsidzdr(np, r, z)
 return deriv, ier
	


def Flush_getBref():
 rRef, bRef, ier = Flush.py_Flush_getBref()
 return rRef, bRef, ier
	


def Flush_getBt(np, r, z):
 Bt, ier = Flush.py_Flush_getBt(np, r, z)
 return Bt, ier
	


def Flush_getBr(np, r, z):
 Br, ier = Flush.py_Flush_getBr(np, r, z)
 return Br, ier
	


def Flush_getBz(np, r, z):
 Bz, ier = Flush.py_Flush_getBz(np, r, z)
 return Bz, ier
	


def Flush_getJt(np, r, z):
 Jt, ier = Flush.py_Flush_getJt(np, r, z)
 return Jt, ier
	


def Flush_getJr(np, r, z):
 Jr, ier = Flush.py_Flush_getJr(np, r, z)
 return Jr, ier
	


def Flush_getJz(np, r, z):
 Jz, ier = Flush.py_Flush_getJz(np, r, z)
 return Jz, ier
	


def Flush_getEverything(np, r, z):
 f, Br, Bz, Bt, Jr, Jz, Jt, ier = Flush.py_Flush_getEverything(np, r, z)
 return f, Br, Bz, Bt, Jr, Jz, Jt, ier
	


def Flush_getFfunction(np, psi):
 F, ier = Flush.py_Flush_getFfunction(np, psi)
 return F, ier
	


def Flush_getqProfile(np, psi):
 q, ier = Flush.py_Flush_getqProfile(np, psi)
 return q, ier
	


def Flush_getPProfile(np, psi):
 p, ier = Flush.py_Flush_getPProfile(np, psi)
 return p, ier
	


def Flush_getFtorProfile(np, psi):
 Ftor, ier = Flush.py_Flush_getFtorProfile(np, psi)
 return Ftor, ier
	


def Flush_getdFdPsi(np, r, z):
 deriv, ier = Flush.py_Flush_getdFdPsi(np, r, z)
 return deriv, ier
	


def Flush_getFdFdPsi(np, r, z):
 deriv, ier = Flush.py_Flush_getFdFdPsi(np, r, z)
 return deriv, ier
	


def Flush_getdPdPsi(np, r, z):
 deriv, ier = Flush.py_Flush_getdPdPsi(np, r, z)
 return deriv, ier
	


def Flush_readFirstWall():
 np, r, z, ier = Flush.py_Flush_readFirstWall()
 return np, r, z, ier
	


def Flush_insideFirstWall(np, r, z):
 inside, ier = Flush.py_Flush_insideFirstWall(np, r, z)
 return inside, ier
	


def Flush_getLimiterFile(pulse, filenameLen):
 filename, ier = Flush.py_Flush_getLimiterFile(pulse, filenameLen)
 return filename, ier
	


def Flush_readFirstWallFromFile(filename, filenameLen):
 np, r, z, ier = Flush.py_Flush_readFirstWallFromFile(filename, filenameLen)
 return np, r, z, ier
	


def Flush_getTangentFlux(r, z, tht, acc):
 rT, zT, fT, ier = Flush.py_Flush_getTangentFlux(r, z, tht, acc)
 return rT, zT, fT, ier
	


def Flush_getIntersections(r, z, tht, acc, np, f):
 nfound, r1, z1, r2, z2, r3, z3, r4, z4, ier = Flush.py_Flush_getIntersections(r, z, tht, acc, np, f)
 return nfound, r1, z1, r2, z2, r3, z3, r4, z4, ier
	


def Flush_getInputVariables(uidLen, ddaLen):
 igo, pulse, time, lunget, iseq, uid, dda, lunmsg, units, ier = Flush.py_Flush_getInputVariables(uidLen, ddaLen)
 return igo, pulse, time, lunget, iseq, uid, dda, lunmsg, units, ier
	


def Flush_getIgo():
 igo, ier = Flush.py_Flush_getIgo()
 return igo, ier
	


def Flush_getPulse():
 pulse, ier = Flush.py_Flush_getPulse()
 return pulse, ier
	


def Flush_getTime():
 time, ier = Flush.py_Flush_getTime()
 return time, ier
	


def Flush_getLunmsg():
 lunmsg, ier = Flush.py_Flush_getLunmsg()
 return lunmsg, ier
	


def Flush_getUid(uidLen):
 uid, ier = Flush.py_Flush_getUid(uidLen)
 return uid, ier
	


def Flush_getDda(ddaLen):
 dda, ier = Flush.py_Flush_getDda(ddaLen)
 return dda, ier
	


def Flush_getSmoothingParameter():
 smooth, ier = Flush.py_Flush_getSmoothingParameter()
 return smooth, ier
	


def Flush_setSmoothingParameter(smooth):
 ier = Flush.py_Flush_setSmoothingParameter(smooth)
 return ier
	


def Flush_insidePolygon(np, rBound, zBound, r, z):
 inside = Flush.py_Flush_insidePolygon(np, rBound, zBound, r, z)
 return inside
	


def Flush_cutSurfaceInsidePolygon(nPoints, rSurf, zSurf, nbound, rbound, zbound):
 nSec, nSurf, rSurf, zSurf, ierr = Flush.py_Flush_cutSurfaceInsidePolygon(nPoints, rSurf, zSurf, nbound, rbound, zbound)
 return nSec, nSurf, rSurf, zSurf, ierr
	


def Flush_cutSurfaceInsideWall(nPoints, rSurf, zSurf):
 nSec, nSurf, rSurf, zSurf, ierr = Flush.py_Flush_cutSurfaceInsideWall(nPoints, rSurf, zSurf)
 return nSec, nSurf, rSurf, zSurf, ierr
	


def get_Flush_Version(versionLen):
 version = Flush.py_get_Flush_Version(versionLen)
 return version
	


def Flush_getSvnVersion():
 version = Flush.py_Flush_getSvnVersion()
 return version
	


def Flush_getBuildDate():
 date = Flush.py_Flush_getBuildDate()
 return date
	


def flulax(lopt, np, xp, yp, work):
 fp, br, bz, bt, xpa, elong, ier = Flush.py_flulax(lopt, np, xp, yp, work)
 return fp, br, bz, bt, xpa, elong, ier
	


def fluqax(nq, q):
 nout, psi, xpai, xpao, ier = Flush.py_fluqax(nq, q)
 return nout, psi, xpai, xpao, ier
	


def flups(xpt, ypt, npsi, psi, iside, epsf, nb):
 alp, xm, ym, ier = Flush.py_flups(xpt, ypt, npsi, psi, iside, epsf, nb)
 return alp, xm, ym, ier
	


def flupso(xpt, ypt, npsi, psi, epsf, nb):
 alpr, xmr, ymr, alpl, xml, yml, ier = Flush.py_flupso(xpt, ypt, npsi, psi, epsf, nb)
 return alpr, xmr, ymr, alpl, xml, yml, ier
	


def flupsp(delta, npsi, psi, epsf, nb):
 xmr, ymr, xml, yml, ier = Flush.py_flupsp(delta, npsi, psi, epsf, nb)
 return xmr, ymr, xml, yml, ier
	


def flusur(psisur, np, lopt):
 xp, yp, br, bz, ier = Flush.py_flusur(psisur, np, lopt)
 return xp, yp, br, bz, ier
	


def flusu2(npsi, psisur, np, npdim, work, jwork, lopt):
 xp, yp, br, bz, ier = Flush.py_flusu2(npsi, psisur, np, npdim, work, jwork, lopt)
 return xp, yp, br, bz, ier
	


def Flush_getFluxSurfaces_struct(npsi, psi, accuracy):
 surfaces, ier = Flush.py_Flush_getFluxSurfaces_struct(npsi, psi, accuracy)
 return surfaces, ier
	


def Flush_wrapperForSurfaceDataAllocation(routinename, npsi, psi, rstart, zstart, npoints_single, accuracy, routinenameLen):
 npoints_single, npieces_max, npoints_max, ierr = Flush.py_Flush_wrapperForSurfaceDataAllocation(routinename, npsi, psi, rstart, zstart, npoints_single, accuracy, routinenameLen)
 return npoints_single, npieces_max, npoints_max, ierr
	


def Flush_wrapperForSurfaceDataDeallocation():
 Flush.py_Flush_wrapperForSurfaceDataDeallocation()
 return
	


def Flush_wrapperForSurfaceDataRetrieval(npsi, npieces_max, npoints_max):
 npieces, npoints, rsurf, zsurf = Flush.py_Flush_wrapperForSurfaceDataRetrieval(npsi, npieces_max, npoints_max)
 return npieces, npoints, rsurf, zsurf
	


def Flush_getParallelsToSurfaces(angle, npsi, psi, accuracy, nBeams):
 rTanRight, zTanRight, rTanLeft, zTanLeft, accuracy, ier = Flush.py_Flush_getParallelsToSurfaces(angle, npsi, psi, accuracy, nBeams)
 return rTanRight, zTanRight, rTanLeft, zTanLeft, accuracy, ier
	


def flusul(xp1, yp1, xp2, yp2, gsw, dp, np):
 np, xp, yp, ier = Flush.py_flusul(xp1, yp1, xp2, yp2, gsw, dp, np)
 return np, xp, yp, ier
	


def flseparatrix(rbound, zbound, nbound, maxpts):
 nsec, nsurf, rsurf, zsurf, ier = Flush.py_flseparatrix(rbound, zbound, nbound, maxpts)
 return nsec, nsurf, rsurf, zsurf, ier
	


def flouter(rbound, zbound, nbound, r1, z1, dl, maxpts):
 nsec, nsurf, rsurf, zsurf, ier = Flush.py_flouter(rbound, zbound, nbound, r1, z1, dl, maxpts)
 return nsec, nsurf, rsurf, zsurf, ier
	


def flugeo(nsur, asur, psisur, iunsys):
 asur, psisur, rmieq, rmaeq, zmieq, zmaeq, rzmieq, rzmaeq, eloeq, epseq, ier = Flush.py_flugeo(nsur, asur, psisur, iunsys)
 return asur, psisur, rmieq, rmaeq, zmieq, zmaeq, rzmieq, rzmaeq, eloeq, epseq, ier
	


def flusht(psisur):
 ier = Flush.py_flusht(psisur)
 return ier
	


def flut(theta, psisur):
 xpt, ypt, dxdth, dydth, d2xdth, d2ydth, dummy1, dummy2, ier = Flush.py_flut(theta, psisur)
 return xpt, ypt, dxdth, dydth, d2xdth, d2ydth, dummy1, dummy2, ier
	


def flushv(psimax):
 ier = Flush.py_flushv(psimax)
 return ier
	


def fluvol(npsi, psi):
 vpsi, ier = Flush.py_fluvol(npsi, psi)
 return vpsi, ier
	


def flblow(dela, delf):
 vol, ier = Flush.py_flblow(dela, delf)
 return vol, ier
	


def flul2b(xpt, ypt, tht):
 chord, xp1, yp1, xp2, yp2, ier = Flush.py_flul2b(xpt, ypt, tht)
 return chord, xp1, yp1, xp2, yp2, ier
	




def Flush_getClosedFluxSurfaces(npsi, psi, npoints_single):

 # Standard arguments
 local_routinename = "flush_getclosedfluxsurfaces"
 routinenameLen = len(local_routinename)
 local_npsi = npsi
 local_psi = psi
 local_rstart = 0
 local_zstart = 0
 local_npoints_single = npoints_single
 local_accuracy = 0
 local_npieces_max = 0
 local_npoints_max = 0
 local_ierr = 0

 # Initialise returned arguments
 rsurf = N.zeros(npoints_single*npsi)
 zsurf = N.zeros(npoints_single*npsi)
 ierr = 0.
 ierr = N.int32(ierr)

 # In case we are calling single-psi routines
 psi_scalar = 0
 if 'psi' in locals():
  if (N.size(psi) <= 1):
   local_psi=N.zeros(1)
   local_psi[0]=psi
   psi_scalar = 1

 # Calling data-allocation wrapper
 local_npoints_single, local_npieces_max, local_npoints_max, local_ierr = Flush_wrapperForSurfaceDataAllocation(local_routinename, local_npsi, local_psi, local_rstart, local_zstart, local_npoints_single, local_accuracy, routinenameLen)
 if(local_ierr != 0):
  ierr = local_ierr
  return rsurf, zsurf, ierr

 # Allocating python arrays
 if(local_npsi < 1):
  local_npsi = 1
 if(local_npieces_max < 1):
  local_npieces_max = 1
 if(local_npoints_max < 1):
  local_npoints_max = 1
 local_npieces = N.int32(N.zeros(local_npsi))
 local_npoints = N.int32(N.zeros(local_npsi*local_npieces_max))
 local_rsurf   = N.zeros(local_npsi*local_npieces_max*local_npoints_max)
 local_zsurf   = N.zeros(local_npsi*local_npieces_max*local_npoints_max)

 # Calling data-retrieval wrapper
 if (local_npoints_max > 0):
  local_npieces, local_npoints, local_rsurf, local_zsurf = Flush_wrapperForSurfaceDataRetrieval(local_npsi, local_npieces_max, local_npoints_max)

 # In case we are calling single-psi routines
 if (psi_scalar == 1):
  psi=local_psi[0]

 # Calling data-deallocation wrapper
 Flush_wrapperForSurfaceDataDeallocation()


 # Copy results
 npoints_single = local_npoints_single
 rsurf = local_rsurf.reshape(local_npoints_single,-1).T
 zsurf = local_zsurf.reshape(local_npoints_single,-1).T

 # Return results
 return rsurf, zsurf, ierr	


def Flush_getSingleClosedSurfaceQuick(psi, accuracy):

 # Standard arguments
 local_routinename = "flush_getsingleclosedsurfacequick"
 routinenameLen = len(local_routinename)
 local_npsi = 0
 local_psi = psi
 local_rstart = 0
 local_zstart = 0
 local_npoints_single = 0
 local_accuracy = accuracy
 local_npieces_max = 0
 local_npoints_max = 0
 local_ierr = 0

 # Initialise returned arguments
 npoints_single = 0.
 npoints_single = N.int32(npoints_single)
 rsurf = N.zeros(npoints_single)
 zsurf = N.zeros(npoints_single)
 ierr = 0.
 ierr = N.int32(ierr)

 # In case we are calling single-psi routines
 psi_scalar = 0
 if 'psi' in locals():
  if (N.size(psi) <= 1):
   local_psi=N.zeros(1)
   local_psi[0]=psi
   psi_scalar = 1

 # Calling data-allocation wrapper
 local_npoints_single, local_npieces_max, local_npoints_max, local_ierr = Flush_wrapperForSurfaceDataAllocation(local_routinename, local_npsi, local_psi, local_rstart, local_zstart, local_npoints_single, local_accuracy, routinenameLen)
 if(local_ierr != 0):
  ierr = local_ierr
  return npoints_single, rsurf, zsurf, ierr

 # Allocating python arrays
 if(local_npsi < 1):
  local_npsi = 1
 if(local_npieces_max < 1):
  local_npieces_max = 1
 if(local_npoints_max < 1):
  local_npoints_max = 1
 local_npieces = N.int32(N.zeros(local_npsi))
 local_npoints = N.int32(N.zeros(local_npsi*local_npieces_max))
 local_rsurf   = N.zeros(local_npsi*local_npieces_max*local_npoints_max)
 local_zsurf   = N.zeros(local_npsi*local_npieces_max*local_npoints_max)

 # Calling data-retrieval wrapper
 if (local_npoints_max > 0):
  local_npieces, local_npoints, local_rsurf, local_zsurf = Flush_wrapperForSurfaceDataRetrieval(local_npsi, local_npieces_max, local_npoints_max)

 # In case we are calling single-psi routines
 if (psi_scalar == 1):
  psi=local_psi[0]

 # Calling data-deallocation wrapper
 Flush_wrapperForSurfaceDataDeallocation()


 # Copy results
 npoints_single = local_npoints_single
 rsurf = local_rsurf
 zsurf = local_zsurf

 # Return results
 return npoints_single, rsurf, zsurf, ierr	


def Flush_getSingleSurfaceStartingFromPoint(rstart, zstart, accuracy):

 # Standard arguments
 local_routinename = "flush_getsinglesurfacestartingfrompoint"
 routinenameLen = len(local_routinename)
 local_npsi = 0
 local_psi = N.zeros(1)
 local_rstart = rstart
 local_zstart = zstart
 local_npoints_single = 0
 local_accuracy = accuracy
 local_npieces_max = 0
 local_npoints_max = 0
 local_ierr = 0

 # Initialise returned arguments
 npoints_single = 0.
 npoints_single = N.int32(npoints_single)
 rsurf = N.zeros(npoints_single)
 zsurf = N.zeros(npoints_single)
 ierr = 0.
 ierr = N.int32(ierr)

 # In case we are calling single-psi routines
 psi_scalar = 0
 if 'psi' in locals():
  if (N.size(psi) <= 1):
   local_psi=N.zeros(1)
   local_psi[0]=psi
   psi_scalar = 1

 # Calling data-allocation wrapper
 local_npoints_single, local_npieces_max, local_npoints_max, local_ierr = Flush_wrapperForSurfaceDataAllocation(local_routinename, local_npsi, local_psi, local_rstart, local_zstart, local_npoints_single, local_accuracy, routinenameLen)
 if(local_ierr != 0):
  ierr = local_ierr
  return npoints_single, rsurf, zsurf, ierr

 # Allocating python arrays
 if(local_npsi < 1):
  local_npsi = 1
 if(local_npieces_max < 1):
  local_npieces_max = 1
 if(local_npoints_max < 1):
  local_npoints_max = 1
 local_npieces = N.int32(N.zeros(local_npsi))
 local_npoints = N.int32(N.zeros(local_npsi*local_npieces_max))
 local_rsurf   = N.zeros(local_npsi*local_npieces_max*local_npoints_max)
 local_zsurf   = N.zeros(local_npsi*local_npieces_max*local_npoints_max)

 # Calling data-retrieval wrapper
 if (local_npoints_max > 0):
  local_npieces, local_npoints, local_rsurf, local_zsurf = Flush_wrapperForSurfaceDataRetrieval(local_npsi, local_npieces_max, local_npoints_max)

 # In case we are calling single-psi routines
 if (psi_scalar == 1):
  psi=local_psi[0]

 # Calling data-deallocation wrapper
 Flush_wrapperForSurfaceDataDeallocation()


 # Copy results
 npoints_single = local_npoints_single
 rsurf = local_rsurf
 zsurf = local_zsurf

 # Return results
 return npoints_single, rsurf, zsurf, ierr	


def Flush_getFluxSurfaces(npsi, psi, accuracy):

 # Standard arguments
 local_routinename = "flush_getfluxsurfaces"
 routinenameLen = len(local_routinename)
 local_npsi = npsi
 local_psi = psi
 local_rstart = 0
 local_zstart = 0
 local_npoints_single = 0
 local_accuracy = accuracy
 local_npieces_max = 0
 local_npoints_max = 0
 local_ierr = 0

 # Initialise returned arguments
 npieces = N.zeros(npsi)
 npieces = N.int32(npieces)
 npoints = N.zeros(npsi)
 npoints = N.int32(npoints)
 rsurf = N.zeros(npsi)
 zsurf = N.zeros(npsi)
 ierr = 0.
 ierr = N.int32(ierr)

 # In case we are calling single-psi routines
 psi_scalar = 0
 if 'psi' in locals():
  if (N.size(psi) <= 1):
   local_psi=N.zeros(1)
   local_psi[0]=psi
   psi_scalar = 1

 # Calling data-allocation wrapper
 local_npoints_single, local_npieces_max, local_npoints_max, local_ierr = Flush_wrapperForSurfaceDataAllocation(local_routinename, local_npsi, local_psi, local_rstart, local_zstart, local_npoints_single, local_accuracy, routinenameLen)
 if(local_ierr != 0):
  ierr = local_ierr
  return npieces, npoints, rsurf, zsurf, ierr

 # Allocating python arrays
 if(local_npsi < 1):
  local_npsi = 1
 if(local_npieces_max < 1):
  local_npieces_max = 1
 if(local_npoints_max < 1):
  local_npoints_max = 1
 local_npieces = N.int32(N.zeros(local_npsi))
 local_npoints = N.int32(N.zeros(local_npsi*local_npieces_max))
 local_rsurf   = N.zeros(local_npsi*local_npieces_max*local_npoints_max)
 local_zsurf   = N.zeros(local_npsi*local_npieces_max*local_npoints_max)

 # Calling data-retrieval wrapper
 if (local_npoints_max > 0):
  local_npieces, local_npoints, local_rsurf, local_zsurf = Flush_wrapperForSurfaceDataRetrieval(local_npsi, local_npieces_max, local_npoints_max)

 # In case we are calling single-psi routines
 if (psi_scalar == 1):
  psi=local_psi[0]

 # Calling data-deallocation wrapper
 Flush_wrapperForSurfaceDataDeallocation()


 # Copy results
 npoints_single = local_npoints_single
 npieces = local_npieces
 if (local_npsi <= 1):
  npoints = local_npoints
  if (local_npoints_max > 1):
   rsurf = local_rsurf.reshape(local_npoints_max,-1).T
   zsurf = local_zsurf.reshape(local_npoints_max,-1).T
  else:
   rsurf = local_rsurf
   zsurf = local_zsurf
 else:
  if (local_npieces_max < 1):
   npoints = local_npoints
   if (local_npoints_max <= 1):
    rsurf = local_rsurf
    zsurf = local_zsurf
   else:
    rsurf = local_rsurf.reshape(local_npoints_max,-1).T
    zsurf = local_zsurf.reshape(local_npoints_max,-1).T
  else:
   npoints = local_npoints.reshape(local_npieces_max,-1).T
   if (local_npoints_max <= 1):
    rsurf = local_rsurf.reshape(local_npieces_max,-1).T
    zsurf = local_zsurf.reshape(local_npieces_max,-1).T
   else:
    rsurf = local_rsurf.reshape(local_npoints_max,local_npieces_max,-1).T
    zsurf = local_zsurf.reshape(local_npoints_max,local_npieces_max,-1).T

 # Return results
 return npieces, npoints, rsurf, zsurf, ierr	


def Flush_getLCFSboundary(accuracy):

 # Standard arguments
 local_routinename = "flush_getlcfsboundary"
 routinenameLen = len(local_routinename)
 local_npsi = 0
 local_psi = N.zeros(1)
 local_rstart = 0
 local_zstart = 0
 local_npoints_single = 0
 local_accuracy = accuracy
 local_npieces_max = 0
 local_npoints_max = 0
 local_ierr = 0

 # Initialise returned arguments
 npoints_single = 0.
 npoints_single = N.int32(npoints_single)
 rsurf = N.zeros(npoints_single)
 zsurf = N.zeros(npoints_single)
 ierr = 0.
 ierr = N.int32(ierr)

 # In case we are calling single-psi routines
 psi_scalar = 0
 if 'psi' in locals():
  if (N.size(psi) <= 1):
   local_psi=N.zeros(1)
   local_psi[0]=psi
   psi_scalar = 1

 # Calling data-allocation wrapper
 local_npoints_single, local_npieces_max, local_npoints_max, local_ierr = Flush_wrapperForSurfaceDataAllocation(local_routinename, local_npsi, local_psi, local_rstart, local_zstart, local_npoints_single, local_accuracy, routinenameLen)
 if(local_ierr != 0):
  ierr = local_ierr
  return npoints_single, rsurf, zsurf, ierr

 # Allocating python arrays
 if(local_npsi < 1):
  local_npsi = 1
 if(local_npieces_max < 1):
  local_npieces_max = 1
 if(local_npoints_max < 1):
  local_npoints_max = 1
 local_npieces = N.int32(N.zeros(local_npsi))
 local_npoints = N.int32(N.zeros(local_npsi*local_npieces_max))
 local_rsurf   = N.zeros(local_npsi*local_npieces_max*local_npoints_max)
 local_zsurf   = N.zeros(local_npsi*local_npieces_max*local_npoints_max)

 # Calling data-retrieval wrapper
 if (local_npoints_max > 0):
  local_npieces, local_npoints, local_rsurf, local_zsurf = Flush_wrapperForSurfaceDataRetrieval(local_npsi, local_npieces_max, local_npoints_max)

 # In case we are calling single-psi routines
 if (psi_scalar == 1):
  psi=local_psi[0]

 # Calling data-deallocation wrapper
 Flush_wrapperForSurfaceDataDeallocation()


 # Copy results
 npoints_single = local_npoints_single
 rsurf = local_rsurf
 zsurf = local_zsurf

 # Return results
 return npoints_single, rsurf, zsurf, ierr	


def Flush_getMainXpointSurface(accuracy):

 # Standard arguments
 local_routinename = "flush_getmainxpointsurface"
 routinenameLen = len(local_routinename)
 local_npsi = 0
 local_psi = N.zeros(1)
 local_rstart = 0
 local_zstart = 0
 local_npoints_single = 0
 local_accuracy = accuracy
 local_npieces_max = 0
 local_npoints_max = 0
 local_ierr = 0

 # Initialise returned arguments
 npoints_single = 0.
 npoints_single = N.int32(npoints_single)
 rsurf = N.zeros(npoints_single)
 zsurf = N.zeros(npoints_single)
 ierr = 0.
 ierr = N.int32(ierr)

 # In case we are calling single-psi routines
 psi_scalar = 0
 if 'psi' in locals():
  if (N.size(psi) <= 1):
   local_psi=N.zeros(1)
   local_psi[0]=psi
   psi_scalar = 1

 # Calling data-allocation wrapper
 local_npoints_single, local_npieces_max, local_npoints_max, local_ierr = Flush_wrapperForSurfaceDataAllocation(local_routinename, local_npsi, local_psi, local_rstart, local_zstart, local_npoints_single, local_accuracy, routinenameLen)
 if(local_ierr != 0):
  ierr = local_ierr
  return npoints_single, rsurf, zsurf, ierr

 # Allocating python arrays
 if(local_npsi < 1):
  local_npsi = 1
 if(local_npieces_max < 1):
  local_npieces_max = 1
 if(local_npoints_max < 1):
  local_npoints_max = 1
 local_npieces = N.int32(N.zeros(local_npsi))
 local_npoints = N.int32(N.zeros(local_npsi*local_npieces_max))
 local_rsurf   = N.zeros(local_npsi*local_npieces_max*local_npoints_max)
 local_zsurf   = N.zeros(local_npsi*local_npieces_max*local_npoints_max)

 # Calling data-retrieval wrapper
 if (local_npoints_max > 0):
  local_npieces, local_npoints, local_rsurf, local_zsurf = Flush_wrapperForSurfaceDataRetrieval(local_npsi, local_npieces_max, local_npoints_max)

 # In case we are calling single-psi routines
 if (psi_scalar == 1):
  psi=local_psi[0]

 # Calling data-deallocation wrapper
 Flush_wrapperForSurfaceDataDeallocation()


 # Copy results
 npoints_single = local_npoints_single
 rsurf = local_rsurf
 zsurf = local_zsurf

 # Return results
 return npoints_single, rsurf, zsurf, ierr	


def Flush_getSecondXpointSurface(accuracy):

 # Standard arguments
 local_routinename = "flush_getsecondxpointsurface"
 routinenameLen = len(local_routinename)
 local_npsi = 0
 local_psi = N.zeros(1)
 local_rstart = 0
 local_zstart = 0
 local_npoints_single = 0
 local_accuracy = accuracy
 local_npieces_max = 0
 local_npoints_max = 0
 local_ierr = 0

 # Initialise returned arguments
 npoints_single = 0.
 npoints_single = N.int32(npoints_single)
 rsurf = N.zeros(npoints_single)
 zsurf = N.zeros(npoints_single)
 ierr = 0.
 ierr = N.int32(ierr)

 # In case we are calling single-psi routines
 psi_scalar = 0
 if 'psi' in locals():
  if (N.size(psi) <= 1):
   local_psi=N.zeros(1)
   local_psi[0]=psi
   psi_scalar = 1

 # Calling data-allocation wrapper
 local_npoints_single, local_npieces_max, local_npoints_max, local_ierr = Flush_wrapperForSurfaceDataAllocation(local_routinename, local_npsi, local_psi, local_rstart, local_zstart, local_npoints_single, local_accuracy, routinenameLen)
 if(local_ierr != 0):
  ierr = local_ierr
  return npoints_single, rsurf, zsurf, ierr

 # Allocating python arrays
 if(local_npsi < 1):
  local_npsi = 1
 if(local_npieces_max < 1):
  local_npieces_max = 1
 if(local_npoints_max < 1):
  local_npoints_max = 1
 local_npieces = N.int32(N.zeros(local_npsi))
 local_npoints = N.int32(N.zeros(local_npsi*local_npieces_max))
 local_rsurf   = N.zeros(local_npsi*local_npieces_max*local_npoints_max)
 local_zsurf   = N.zeros(local_npsi*local_npieces_max*local_npoints_max)

 # Calling data-retrieval wrapper
 if (local_npoints_max > 0):
  local_npieces, local_npoints, local_rsurf, local_zsurf = Flush_wrapperForSurfaceDataRetrieval(local_npsi, local_npieces_max, local_npoints_max)

 # In case we are calling single-psi routines
 if (psi_scalar == 1):
  psi=local_psi[0]

 # Calling data-deallocation wrapper
 Flush_wrapperForSurfaceDataDeallocation()


 # Copy results
 npoints_single = local_npoints_single
 rsurf = local_rsurf
 zsurf = local_zsurf

 # Return results
 return npoints_single, rsurf, zsurf, ierr	


def Flush_getMainXpointSurfaceLegs(accuracy):

 # Standard arguments
 local_routinename = "flush_getmainxpointsurfacelegs"
 routinenameLen = len(local_routinename)
 local_npsi = 0
 local_psi = N.zeros(1)
 local_rstart = 0
 local_zstart = 0
 local_npoints_single = 0
 local_accuracy = accuracy
 local_npieces_max = 0
 local_npoints_max = 0
 local_ierr = 0

 # Initialise returned arguments
 npoints = N.zeros(2)
 npoints = N.int32(npoints)
 rsurf = N.zeros(2)
 zsurf = N.zeros(2)
 ierr = 0.
 ierr = N.int32(ierr)

 # In case we are calling single-psi routines
 psi_scalar = 0
 if 'psi' in locals():
  if (N.size(psi) <= 1):
   local_psi=N.zeros(1)
   local_psi[0]=psi
   psi_scalar = 1

 # Calling data-allocation wrapper
 local_npoints_single, local_npieces_max, local_npoints_max, local_ierr = Flush_wrapperForSurfaceDataAllocation(local_routinename, local_npsi, local_psi, local_rstart, local_zstart, local_npoints_single, local_accuracy, routinenameLen)
 if(local_ierr != 0):
  ierr = local_ierr
  return npoints, rsurf, zsurf, ierr

 # Allocating python arrays
 if(local_npsi < 1):
  local_npsi = 1
 if(local_npieces_max < 1):
  local_npieces_max = 1
 if(local_npoints_max < 1):
  local_npoints_max = 1
 local_npieces = N.int32(N.zeros(local_npsi))
 local_npoints = N.int32(N.zeros(local_npsi*local_npieces_max))
 local_rsurf   = N.zeros(local_npsi*local_npieces_max*local_npoints_max)
 local_zsurf   = N.zeros(local_npsi*local_npieces_max*local_npoints_max)

 # Calling data-retrieval wrapper
 if (local_npoints_max > 0):
  local_npieces, local_npoints, local_rsurf, local_zsurf = Flush_wrapperForSurfaceDataRetrieval(local_npsi, local_npieces_max, local_npoints_max)

 # In case we are calling single-psi routines
 if (psi_scalar == 1):
  psi=local_psi[0]

 # Calling data-deallocation wrapper
 Flush_wrapperForSurfaceDataDeallocation()


 # Copy results
 npoints_single = local_npoints_single
 npieces = local_npieces
 if (local_npsi <= 1):
  npoints = local_npoints
  if (local_npoints_max > 1):
   rsurf = local_rsurf.reshape(local_npoints_max,-1).T
   zsurf = local_zsurf.reshape(local_npoints_max,-1).T
  else:
   rsurf = local_rsurf
   zsurf = local_zsurf
 else:
  if (local_npieces_max < 1):
   npoints = local_npoints
   if (local_npoints_max <= 1):
    rsurf = local_rsurf
    zsurf = local_zsurf
   else:
    rsurf = local_rsurf.reshape(local_npoints_max,-1).T
    zsurf = local_zsurf.reshape(local_npoints_max,-1).T
  else:
   npoints = local_npoints.reshape(local_npieces_max,-1).T
   if (local_npoints_max <= 1):
    rsurf = local_rsurf.reshape(local_npieces_max,-1).T
    zsurf = local_zsurf.reshape(local_npieces_max,-1).T
   else:
    rsurf = local_rsurf.reshape(local_npoints_max,local_npieces_max,-1).T
    zsurf = local_zsurf.reshape(local_npoints_max,local_npieces_max,-1).T

 # Return results
 return npoints, rsurf, zsurf, ierr	


def Flush_getSecondXpointSurfaceLegs(accuracy):

 # Standard arguments
 local_routinename = "flush_getsecondxpointsurfacelegs"
 routinenameLen = len(local_routinename)
 local_npsi = 0
 local_psi = N.zeros(1)
 local_rstart = 0
 local_zstart = 0
 local_npoints_single = 0
 local_accuracy = accuracy
 local_npieces_max = 0
 local_npoints_max = 0
 local_ierr = 0

 # Initialise returned arguments
 npoints = N.zeros(2)
 npoints = N.int32(npoints)
 rsurf = N.zeros(2)
 zsurf = N.zeros(2)
 ierr = 0.
 ierr = N.int32(ierr)

 # In case we are calling single-psi routines
 psi_scalar = 0
 if 'psi' in locals():
  if (N.size(psi) <= 1):
   local_psi=N.zeros(1)
   local_psi[0]=psi
   psi_scalar = 1

 # Calling data-allocation wrapper
 local_npoints_single, local_npieces_max, local_npoints_max, local_ierr = Flush_wrapperForSurfaceDataAllocation(local_routinename, local_npsi, local_psi, local_rstart, local_zstart, local_npoints_single, local_accuracy, routinenameLen)
 if(local_ierr != 0):
  ierr = local_ierr
  return npoints, rsurf, zsurf, ierr

 # Allocating python arrays
 if(local_npsi < 1):
  local_npsi = 1
 if(local_npieces_max < 1):
  local_npieces_max = 1
 if(local_npoints_max < 1):
  local_npoints_max = 1
 local_npieces = N.int32(N.zeros(local_npsi))
 local_npoints = N.int32(N.zeros(local_npsi*local_npieces_max))
 local_rsurf   = N.zeros(local_npsi*local_npieces_max*local_npoints_max)
 local_zsurf   = N.zeros(local_npsi*local_npieces_max*local_npoints_max)

 # Calling data-retrieval wrapper
 if (local_npoints_max > 0):
  local_npieces, local_npoints, local_rsurf, local_zsurf = Flush_wrapperForSurfaceDataRetrieval(local_npsi, local_npieces_max, local_npoints_max)

 # In case we are calling single-psi routines
 if (psi_scalar == 1):
  psi=local_psi[0]

 # Calling data-deallocation wrapper
 Flush_wrapperForSurfaceDataDeallocation()


 # Copy results
 npoints_single = local_npoints_single
 npieces = local_npieces
 if (local_npsi <= 1):
  npoints = local_npoints
  if (local_npoints_max > 1):
   rsurf = local_rsurf.reshape(local_npoints_max,-1).T
   zsurf = local_zsurf.reshape(local_npoints_max,-1).T
  else:
   rsurf = local_rsurf
   zsurf = local_zsurf
 else:
  if (local_npieces_max < 1):
   npoints = local_npoints
   if (local_npoints_max <= 1):
    rsurf = local_rsurf
    zsurf = local_zsurf
   else:
    rsurf = local_rsurf.reshape(local_npoints_max,-1).T
    zsurf = local_zsurf.reshape(local_npoints_max,-1).T
  else:
   npoints = local_npoints.reshape(local_npieces_max,-1).T
   if (local_npoints_max <= 1):
    rsurf = local_rsurf.reshape(local_npieces_max,-1).T
    zsurf = local_zsurf.reshape(local_npieces_max,-1).T
   else:
    rsurf = local_rsurf.reshape(local_npoints_max,local_npieces_max,-1).T
    zsurf = local_zsurf.reshape(local_npoints_max,local_npieces_max,-1).T

 # Return results
 return npoints, rsurf, zsurf, ierr	
