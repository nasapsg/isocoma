# --------------------------------------------------------------------------
# Module and models for computing number of molecules in a cometary atmosphere
# Based on a Haser isotropic model and employing Bessel functions and
# some approximations for non-centered and embedded measurements
# Geronimo Villanueva, NASA Goddard Space Flight Center
# Developed for the Cometary Interceptor mission - February/2021
# -------------------------------------------------------------------------
import os
import numpy as np
from scipy import special
import matplotlib.pyplot as plt

# --------------------------------------------------------
# Modules for computing number of molecules inside a beam
# --------------------------------------------------------
# Calculation number of molecules in fov
def molbeam(qgas, fov=1000.0, rh=1.0, ptau=7.7e4, dtau=0.0, offset=0, distance=1e10, diameter=7.0):
    # qgas: molecular production rate [molecules/s]
    # fov: diameter of field-of-view [km]
    # rh: heliocentric distance [AU]
    # ptau: Molecular lifetime of parent at RH=1AU [seconds]
    # dtau: Molecular lifetime of daughter at RH=1AU [seconds], if molecule is parent use dtau = 0
    # offset: offset of beam from center of comet [km]
    # distance: distance between observer and center of comet [km]
    # diameter: comet's diameter [km]
    vgas = 0.800     # Expansion velocity at 1AU [km/s, 0.8 is typical]
    evgas= -0.5      # Dependence of velocity with RH [-0.5, empirical]
    etau = 2.0       # Dependence of lifetimes with RH [2 typical]
    vexp = vgas * (rh**evgas)         # Expansion velocity [km/s]
    plam = ptau * (rh**etau ) * vexp  # Molecular lifetime of parent [km]
    dlam = dtau * (rh**etau ) * vexp  # Molecular lifetime of daughter [km]
    if dlam==0.0: Ntot = qgas * ptau * (rh**etau )
    else: Ntot = qgas * dtau * (rh**etau )

    # Calculate effect of offset
    if (offset-fov/2.0)>diameter/2.0:
        # Try to estimate offset by computing concentric rings
        fx0 = calculatefx(offset*2-fov, plam, dlam)
        fx1 = calculatefx(offset*2+fov, plam, dlam)
        afx = np.pi*((offset+fov/2.0)**2 - (offset-fov/2.0)**2)
        afov= np.pi*(fov/2.0)**2
        fx  = (fx1-fx0)*(afov/afx)
        fxh = fx*0.5 # f(x) from surface to outer space
    else:
        fn = 0.5*calculatefx(diameter, plam, dlam)
        if fov<diameter:
            fx = fn*(fov/diameter)**2; fxh=fx
        else:
            fx  = calculatefx(fov+offset, plam, dlam);
            afx = np.pi*((offset+fov)/2.0)**2
            afov= np.pi*(fov/2.0)**2
            fx *= afov/afx
            fxh = fx*0.5; fx -= fn;
        #Endelse
    #Endelse

    # See if we are inside the coma
    fx0 = calculatefx(distance*2, plam, dlam)
    fx = fx - fxh*(1.0-fx0)
    if fx<0: fx=0.0
    if fx>1: fx=1.0

    Nfov = Ntot * fx  # Number of molecules in the FOV
    return Nfov
#End molbeam

# Analytical calculation of filling factor based on Bessel functions
# Yamamoto 1981 (https://link.springer.com/article/10.1007/BF00896911)
def calculatefx(fov=1000.0, plam=7.7e4, dlam=0.0):
    # fov: diameter of field-of-view [km]
    # plam: Molecular lifetime of parent [km], if molecule is parent use plam = 0
    # dlam: Molecular lifetime of daughter [km]
    if fov==0 or plam==0.0: return 0.0
    x = 0.5*fov/plam
    if dlam==0.0:
        # Parent molecule
        gx  = 1.0/x
        gx += -special.k1(x)
        gx += np.pi/2.0 - special.iti0k0(x)[1]
        fx  = x * gx
    else:
        # Daughter molecule
        u2  = plam/dlam
        if u2==1.0: u2=1.001
        gx  = 1.0/(u2*x) - 1.0/x
        gx += -special.k1(u2*x) + special.k1(x)
        gx += -special.iti0k0(u2*x)[1] + special.iti0k0(x)[1]
        fx  = (u2*x/(1.0-u2))*gx
    #Endelse
    return fx
#End calculatefx

# -------------------------------------------------------------
# Example running the analytical functions and comparing to PSG
# ------------------------------------------------------------
# Define cometary parameters
Qgas = 1e29                        # Cometary gas activity [s-1]
rh = 1.0                           # Heliocentric distance [AU]
distance = 1e6                     # Distance to comet [km]
diameter = 7.0                     # Comet's diameter [km]
tgas1 = [1.0, 7.7e4, 0.0]          # Parameters for gas1: abundance(0 to 1.0), parent_tau[s], daugther_tau[s]
tgas2 = [1.0, 2.4e4, 1.6e5]        # Parameters for gas2: abundance(0 to 1.0), parent_tau[s], daugther_tau[s]

# Iterate across offsets
fov  = 1.0  # Size of FOV [km]
bmin = 1.0   # Minimum offset [km]
bmax = 2e5   # Maximum offset [km]
afov = np.pi*(fov*1e3/2)**2 # Area of fov
b = bmin; offs=[]; o1=[]; o2=[]; s1=[]; s2=[]
while b<bmax:
    mo1 = molbeam(Qgas*tgas1[0], fov, rh, tgas1[1], tgas1[2], b, distance, diameter)
    mo2 = molbeam(Qgas*tgas2[0], fov, rh, tgas2[1], tgas2[2], b, distance, diameter)
    offs.append(b)
    o1.append(mo1)
    o2.append(mo2)
    b=b*2
#End FOV loop

# Iterate across FOV sizes
bmin = 1.0   # Minimum FOV size [km]
bmax = 1e5   # Maximum FOV size [km]
b = bmin; fovs=[]; f1=[]; f2=[]; p1=[]; p2=[]; areas=[]
while b<bmax:
    mf1 = molbeam(Qgas*tgas1[0], b, rh, tgas1[1], tgas1[2], 0, distance, diameter)
    mf2 = molbeam(Qgas*tgas2[0], b, rh, tgas2[1], tgas2[2], 0, distance, diameter)
    bfov = np.pi*(b*1e3/2)**2 # Area of fov
    fovs.append(b)
    f1.append(mf1)
    f2.append(mf2)
    areas.append(bfov)
    b=b*2
#End FOV loop

# Plot the dependence with width
lbl  = r'$R_{h}$: %.1f AU' % rh
lbl += '\n' + r'Q($H_{2}O$): %.1e $s^{-1}$' % Qgas
lbl += '\n' + r'$v_{exp}$: %.1f km/s' % (0.8 * (rh**-0.5))
lbl += '\n' + r'$\tau_{H2O}$: %.1e s' % tgas1[1]
lbl += '\n' + r'$\tau_{OH}$: %.1e and %.1e s' % (tgas2[1],tgas2[2])
lbl += '\n' + r'diameter: %.1f km' % diameter
lbl += '\n' + r'Distance: %.1e km' % distance

# Plot the fov plots
pl,ax = plt.subplots(2,2, figsize=(12, 9))
ax[0,0].plot(fovs,np.asarray(f1)/np.asarray(areas),label=r'Water $H_{2}O$')
ax[0,0].plot(fovs,np.asarray(f2)/np.asarray(areas),label=r'Hydroxyl $OH$')
ax[0,0].set_xscale('log')
ax[0,0].set_yscale('log')
ax[0,0].autoscale(enable=True, axis='x', tight=True)
ax[0,0].set_xlabel('FOV - Diameter of integrating circle [km]')
ax[0,0].set_ylabel('Column density [molecules / m2]')
ax[0,0].set_title('Column density (centered beam)')
ax[0,0].text(0.1, 0.3, lbl, transform=ax[0,0].transAxes,fontsize='12')
ax[0,0].text(0.1, 0.05, '0.06 degrees is 1 mrad or 1 km at 1000 km', transform=ax[0,0].transAxes,fontsize='10')
ax[0,0].legend()

ax[0,1].plot(fovs,f1,label=r'Water $H_{2}O$')
ax[0,1].plot(fovs,f2,label=r'Hydroxyl $OH$')
ax[0,1].set_xscale('log')
ax[0,1].set_yscale('log')
ax[0,1].autoscale(enable=True, axis='x', tight=True)
ax[0,1].set_xlabel('FOV - Diameter of integrating circle [km]')
ax[0,1].set_ylabel('Total column number of molecules')
ax[0,1].set_title('Total number of molecules')
ax[0,1].legend()

# Plot the offset plots
ax[1,0].plot(offs,np.asarray(o1)/afov,label=r'Water $H_{2}O$')
ax[1,0].plot(offs,np.asarray(o2)/afov,label=r'Hydroxyl $OH$')
ax[1,0].set_xscale('log')
ax[1,0].set_yscale('log')
ax[1,0].autoscale(enable=True, axis='x', tight=True)
ax[1,0].set_xlabel('Offset from comet center [km]')
ax[1,0].set_ylabel('Column density [molecules / m2]')
ax[1,0].set_title('Column density (offsetted beam) - FOV %.1f km' % fov)
ax[1,0].legend()

ax[1,1].plot(offs,o1,label=r'Water $H_{2}O$')
ax[1,1].plot(offs,o2,label=r'Hydroxyl $OH$')
ax[1,1].set_xscale('log')
ax[1,1].set_yscale('log')
ax[1,1].autoscale(enable=True, axis='x', tight=True)
ax[1,1].set_xlabel('Offset from comet center [km]')
ax[1,1].set_ylabel('Total column number of molecules')
ax[1,1].set_title('Total number of molecules')
ax[1,1].legend()

plt.tight_layout()
plt.savefig("haser.png")
plt.show()
