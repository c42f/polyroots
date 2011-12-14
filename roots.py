from __future__ import division
from pylab import *

# Iterator returning all elements in the caretesian power S^n of the set S
def setCartesianPower(S, n):
    def maketuple(*t): return t
    if n == 1:
        for i in S:
            yield (i,)
    else:
        for j in setCartesianPower(S, n-1):
            for i in S:
                yield maketuple(i, *j)

# Make domain; be careful with the offset so that it can be symmetrically
# mirrored!
def makeDomain(R, npoints):
    #x1 = linspace(-R,R,npoints)  # not making use of symmetry
    x1 = R/npoints*(arange(npoints) + 0.5)
    x,y = meshgrid(x1, x1)
    return x + 1j*y

def makePolarDomain(R, npoints):
    theta = pi/2 * (arange(npoints) + 0.5)/npoints
    npointsRad = (R - 1) * npoints / (pi/2)
    r = 1 + (R - 1)*(arange(npointsRad) + 0.5)/npointsRad
    tt,rr = meshgrid(theta,r)
    x,y = rr*cos(tt), rr*sin(tt)
    return x + 1j*y

class MinAccumulator:
    def __init__(self, z):
        self.acc = 1000*ones(z.shape)
    def accumulate(self, poly):
        fmin(self.acc, absolute(poly), self.acc)
    def result(self, z, degree):
        return self.acc*abs(z)**(-degree/2)

class SjoerdAccumulator:
    def __init__(self, z):
        self.acc = zeros(z.shape)
    def accumulate(self, poly):
        add(self.acc, absolute(poly)**-4, self.acc)
    def result(self, z, degree):
        return self.acc*abs(z)**(2*degree)

# Render roots with Sjoerd Visscher's direct evaluation algorithm
def genfrac(degree, z, accumClass=MinAccumulator, coeffSet=(-1,1)):
    N = degree + 1
    zPows = [z**i for i in range(0,N)]
    acc = accumClass(z)
    for coeffs in setCartesianPower(coeffSet, N):
        p = 0
        for zpow,c in zip(zPows,coeffs):
            p += c*zpow
        acc.accumulate(p)
    return acc.result(z, degree)

def assembleSym(F):
    F1 = concatenate((fliplr(F), F), axis=1)
    return concatenate((flipud(F1), F1), axis=0)

def evalCmap(cmap, F, vmin=None, vmax=None):
    if vmin is None:
        vmin = amin(F)
    if vmax is None:
        vmax = amax(F)
    return cmap((F-vmin)/(vmax-vmin))

#R = 1.6
#z = makeDomain(1.6,1000)
#F = genfrac(7,z)
#F = assembleSym(F)
#
#xmax = amax(real(z))
#extent = (-xmax, xmax, -xmax, xmax)
#
##imshow(exp(-20*acc**2), cmap=cm.gray)
##imshow(exp(-100*abs(4-abs(z)**2)*F**2), cmap=cm.gray, origin='lower')
##imshow(F, cmap=cm.gray, origin='lower')
##imshow(F**0.1, cmap=cm.gist_heat_r, origin='lower', vmin=0.5)
#imshow(F**0.1, cmap=cm.gist_heat_r, origin='lower', vmin=0.5,
#       extent=extent, interpolation='nearest')
#show()
