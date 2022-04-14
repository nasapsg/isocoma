# ---------------------------------------------------
# non-LTE solver for cometary atmospheres - Time dependent solution
# NASA-GSFC Planetary Spectrum Generator (PSG, Villanueva et al. 2022)
# Based on methods as introduced in Chin & Weaver 1984, Bockelee-Morvan 1987, Biver 1997, Bensch and Bergin 2004, Zakharov et al. 2007
# Last updated April 2022
# ---------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import special

mol  = 'h2o'            # Base molecule name
Xgas = 1.0              # Abundance ratio
sym  = 1                # Desired symmetry (-1:all syms, 0:only para, 1:only ortho)

Tkin = 50.0             # Neutral gas kinetic temperature [K]
Qatm = 1e29             # Production rate [s-1] of ambient gas
matm = 18.0             # Molar mass of ambient gas [g/mol]
Batm = 1.0/77000.0      # Photodissociation rate (at rh=1 AU) of ambient gas [s-1]
rh = 1.0                # Heliocentric distance [AU]
vexp = 850.0            # Velocity [m/s]

lmax = 7                # Maximum number of levels to consider
xre = 1.0               # Scaling factor contact surface
xne = 1.0               # Scaling factor electron density
Temax = 1e4             # Maximum temperature of electrons [K]
kion = 4.1e-7           # Photoionization rate (at rh=1 AU) [s-1]
rmin = 1e4              # Minimum distance [m]
rmax = 1e9              # Maximum distance [m]
nrad = 400              # Number of shells

HP = 6.62606896e-34     # Planck's constant [J s]
KB = 1.3806505e-23      # Boltzmann's constant [J/K] or [m2 kg / (s2 K)]
C2 = 1.43877736         # Second radiation constant [K/cm-1]
CS = 2.99792458e8       # Speed of light [m/s]
AV = 6.022140857e23     # Avogadro number [molecules/mol]
ME = 9.10938e-31        # Mass of an electron [Kg]
QE = 1.602176634e-19    # Elementary charge [C]
E0 = 8.8541878128e-12   # Vacuum permittivity [F m-1] or [s4 A2 kg-1 m-3]

# Read molecular parameters
# Download if not locally available
ncoll=0; mgas=matm; Bgas=Batm; Bgas2=0.0
if not os.path.exists('%s.txt' % mol): os.system('curl -s https://psg.gsfc.nasa.gov/data/linelists/gsfc/rot/%s.txt --output %s.txt' % (mol,mol))
fr = open('%s.txt' % mol); lines = fr.readlines(); fr.close();
for l in range(len(lines)):
    line = lines[l].strip()
    if len(line)==0: continue
    if line[: 7]=='# Mass:':
        mgas=float(line[7:])
    #Endif
    if line[:11]=='# Lifetime:':
        st = line[11:].split()
        Bgas=1.0/float(st[0])
        if len(st)>1: Bgas2=1.0/float(st[1])
    #Endif
    if line[:9]=='# Levels:':
        mnlev = int(line[9:]); l+=1; j=0; Qt=0.0
        if mnlev>lmax: nlev=lmax
        else: nlev = mnlev
        ilev = np.zeros([mnlev],dtype=np.int)+lmax*2
        levs = np.zeros([3,nlev])
        Nth  = np.zeros([nlev])                  # Thermal population at each level
        A  = np.zeros([nlev,nlev])               # Einstein A-coefficients for spontaneous emission [s-1]
        BJ = np.zeros([nlev,nlev])               # CMB rate [s-1], from Bji for photon absorption [J-1 s-2 cm3] times JCMB [J s cm-3]
        v  = np.zeros([nlev,nlev])               # Transition frequency [cm-1]
        G  = np.zeros([nlev,nlev])               # Effective pumping rate [s-1]
        C  = np.zeros([nlev,nlev])               # Neutral collissional cross section [m2]
        Ce = np.zeros([nlev,nlev,nrad])          # Electron collissional excitation rate [s-1]
        Cn = np.zeros([nlev,nlev,nrad])          # Neutral collissional de-excitation rate [s-1]
        for i in range(mnlev):
            st = lines[l].split(); l+=1
            El = float(st[1])
            wl = float(st[2])
            sl = int(st[3])
            if sym>=0 and sl!=sym: continue
            Qt += wl*np.exp(-El*C2/Tkin)
            if j>=lmax: continue
            Nth[j] = wl*np.exp(-El*C2/Tkin)
            levs[0,j] = El
            levs[1,j] = wl
            levs[2,j] = sl
            ilev[i] = j
            j+=1
        #Endfor
        Nth /= Qt
    #Endif
    if line[:8]=='# Lines:':
        nlines = int(line[8:]); l+=1
        for i in range(nlines):
            st = lines[l].split(); l+=1
            iu = ilev[int(st[0])-1]
            il = ilev[int(st[1])-1]
            Aul = float(st[2])
            vul = float(st[3])
            if iu>=nlev or il>=nlev: continue
            A[iu,il]   = Aul
            v[iu,il]   = vul
            Jv         = 2.0*HP*vul**3.0 / (np.exp(vul*C2/2.725) - 1.0) # Flux of the CMB [J s cm-3] - Black Body
            BJ[il,iu]  = levs[1,iu]/levs[1,il] * Aul / (8.0*np.pi*HP*(vul**3.0)) # [J-1 s-2 cm3]
            BJ[il,iu] *= Jv # Absorption rate due CMB [s-1]
            BJ[il,il] -= BJ[il,iu]
        #Endfor
    #Endif
    if line[:11]=='# Pumpings:':
        nlines = int(line[11:]); l+=1
        for i in range(nlines):
            st = lines[l].split(); l+=1
            iu = ilev[int(st[0])-1]
            il = ilev[int(st[1])-1]
            if iu>=nlev or il>=nlev: continue
            G[iu,il] = float(st[2])/(rh*rh)
            G[iu,iu]-= G[iu,il]
        #Endfor
    #Endif
    if line[:31]=='# Collisional temperatures [K]:':
        st = line[31:].split()
        temps = [float(x) for x in st]
    #Endif
    if line[:8]=='# Rates:':
        ncoll = int(line[8:]); l+=1
        for i in range(ncoll):
            st = lines[l].split(); l+=1
            iu = ilev[int(st[0])-1]
            il = ilev[int(st[1])-1]
            if iu>=nlev or il>=nlev: continue
            vals = [float(x) for x in st[2:]]
            rct = np.interp(Tkin, temps, vals)
            C[iu,il] = np.sqrt(matm/2.0)*rct*1e-6 # Scale H2 rates by mass of atmosphere
        #Endfor
    #Endif
#Endfor

# Neutral gas component
Bgas /= rh*rh; Bgas2 /= rh*rh;
lrad = np.arange(nrad)*(np.log(rmax) - np.log(rmin))/(nrad-1) + np.log(rmin)
rad  = np.exp(lrad)                   # Distance from nucleus [m]
natm = Qatm/(4.0*np.pi*rad*rad*vexp)  # Atmosphere gas density without photodissocation decay [m-3]
lam0 = vexp*rh*rh/Bgas
if Bgas2>0:
    lam1 = vexp*rh*rh/Bgas2
    photoscl = (lam1/(lam0-lam1))*(np.exp(-rad/lam0)-np.exp(-rad/lam1))
    print(lam0, lam1)
else:
    photoscl = np.exp(-rad/lam0)
#Endelse
ngas = Xgas*natm*photoscl
natm = natm*np.exp(-Batm*rad/(vexp*rh*rh)) # Ambient gas density [m-3]

# Electron properties
Rcs = 1.125e6*xre*(Qatm/1e29)**0.75  # Contact surface [m]
Rrec = 3.2e6*xre*(Qatm/1e29)**0.50   # Recombination surface [m]
Te = np.zeros(nrad)                  # Electron temperature [K]
ne = np.zeros(nrad)                  # Electron density [m-3]
ve = np.zeros(nrad)                  # Mean velocity of electrons [m/s]
for i in range(nrad):
    if rad[i]<Rcs: Te[i] = Tkin
    elif rad[i]<=2*Rcs: Te[i] = Tkin + (Temax-Tkin)*((rad[i]/Rcs) - 1.0)
    else: Te[i] = Temax
    krec = 7e-13*(300.0/Te[i])**0.5
    ne[i] = xne*(Qatm*kion/(vexp*krec*rh*rh))**0.5
    ne[i]*= (Te[i]/300)**0.15
    ne[i]*= (Rrec/(rad[i]*rad[i]))
    ne[i]*= (1.0 - np.exp(-rad[i]/Rrec))
    ne[i]+= 5e6/(rh*rh)
    ve[i] = np.sqrt(8.0*KB*Te[i]/(np.pi*ME))
#Endfor

lg=plt.plot(rad/1e3, ngas, label='Gas')
la=plt.plot(rad/1e3, natm, label='Atm')
le=plt.plot(rad/1e3, ne, label='Electrons')
plt.xscale('log')
plt.yscale('log')
plt.xlim([rmin/1e3,rmax/1e3])
plt.ylim([1e6,1e18])
plt.ylabel('Density [m-3]')
plt.xlabel('Distance from nucleus [km]')
plt.title('%s Q:%.1e s-1, X:%.4f, Rh:%.1f AU, v:%.2f km/s' % (mol.upper(), Qatm, Xgas, rh, vexp/1e3))
ax = plt.gca().twinx()
ax.set_ylim([1e0,1e5])
tg=ax.plot(rad/1e3, Tkin+Te*0,':',color='black',label='Tgas')
te=ax.plot(rad/1e3, Te, '-.',color='black',label='Te')
ax.set_yscale('log')
ax.set_ylabel('Temperature [K]')
lns = lg+la+le+tg+te
labs = [l.get_label() for l in lns]
ax.legend(lns, labs)
plt.tight_layout()
plt.savefig('nlte_profile.png', dpi=300)
plt.close()

# Calculate radially varying parameters
vkin = np.sqrt(8.0*KB*1e3*AV*Tkin/np.pi*(1.0/mgas + 1.0/matm)) # Mean relative velocity of gas and atmosphere [m/s]
for r in range(nrad):
    for i in range(nlev):
        for j in range(nlev):
            # Collissions with neutrals
            if ncoll<=0:
                # When no collisions are provided, use an average value
                xc = 5e-18  # Average collissional cross section [m2]
                tc = natm[r]*vkin*xc
                Cn[j,i,r] = tc*Nth[i]
            elif C[i,j]>0:
                dE = levs[0,i] - levs[0,j]
                Cn[i,j,r] = natm[r]*C[i,j]
                Cn[j,i,r] = levs[1,i]/levs[1,j]*Cn[i,j,r]*np.exp(-dE*C2/Tkin)
            #Endfor

            # Collissions with electrons
            if A[i,j]>0:
                xe = (A[i,j]/v[i,j]**4.0) * ME*QE*QE / (16.0*np.pi*np.pi*E0*HP*HP*CS*1e8)
                xa = v[i,j]*C2/(2.0*Te[r])
                K0 = special.k0(xa) # Modified Bessel functions of the second kind
                ce = ne[r]*ve[r]*xe*2.0*xa*K0
                Ce[i,j,r] = ce*np.exp( xa)
                Ce[j,i,r] = ce*np.exp(-xa)*levs[1,i]/levs[1,j]
            #Endfor
        #Endfor
    #Endelse
#Endfor

# Define system of differential equations
def deriv(t, N):
    rt  = vexp*t                       # Distance [m]
    lrt = np.log(rt)                   # log of distance [m]
    ng  = np.interp(lrt, lrad, ngas)   # Gas density [m-3]
    Cer = np.zeros([nlev,nlev])        # Electron collissional excitation rate [s-1] at t
    Cnr = np.zeros([nlev,nlev])        # Neutral collissional de-excitation rate [s-1] at t
    Ar  = np.zeros([nlev,nlev])        # Corrected Einstein A coefficients [s-1] for escape probability
    for i in range(nlev):
        for j in range(nlev):
            if i==j: continue
            Cnr[i,j] = np.interp(lrt, lrad, Cn[i,j,:])
            Cer[i,j] = np.interp(lrt, lrad, Ce[i,j,:])
            Cnr[i,i]-= Cnr[i,j]
            Cer[i,i]-= Cer[i,j]

            # Compute escape probability
            if A[j,i]==0: continue
            Bji = levs[1,i]/levs[1,j] * A[j,i]*1e-6 / (8.0*np.pi*HP*(v[j,i]**3.0)) # [J-1 s-2 m3]
            tau = HP*Bji*(rt*ng/vexp)*(levs[1,j]/levs[1,i]*N[i] - N[j])
            if tau>1e-4:
                K0 = special.k0(tau/2.0) # Modified Bessel functions of the second kind
                K1 = special.k1(tau/2.0) # Modified Bessel functions of the second kind
                beta = (2.0/(3.0*tau)) - np.exp(-tau/2.0)/3.0*(K1 + tau*(K0 - K1))
            elif tau>-1e-4:
                beta = 1.0
            else:
                beta = (1.0 - np.exp(-tau))/tau
            #Endelse
            Ar[j,i] = A[j,i]*beta
            Ar[j,j]-= Ar[j,i]
        #Endfor
    #Endfor
    dNdt = np.matmul(N, Cer+Cnr+Ar+G+BJ)
    return dNdt
#Enddef

# Solve differential equation
tmin = rmin/vexp; tmax = rmax/vexp
N0 = Nth
soln = solve_ivp(deriv, [tmin,tmax], N0, method='Radau', t_eval = rad/vexp)
print('Number of forward evaluations (NFEV): ', soln.nfev)
print('Number of evaluations of the Jacobian (NJEV): ', soln.njev)
print('Number of LU decompositions (NLU): ', soln.nlu)

# Save populations
fr = open('nlte.dat', 'w')
for i in range(len(soln.t)):
    str = '%e ' % (soln.t[i]*vexp/1e3)
    for j in range(nlev): str = '%s %e' % (str, soln.y[j,i])
    fr.write('%s\n' % str)
#Endfor
fr.close()

# Plot the results
for i in range(nlev): plt.plot(soln.t*vexp/1e3, soln.y[i], color='C%d' % i)
Nt = np.zeros(len(soln.t))
for i in range(len(soln.t)): Nt[i] = np.sum(soln.y[:,i])
plt.plot(soln.t*vexp/1e3, Nt, color='black')
plt.xscale('log')
plt.yscale('log')
plt.xlim([rmin/1e3,rmax/1e3])
plt.ylim([1e-5,1.0])
plt.title('%s Q:%.1e s-1, X:%.4f, Rh:%.1f AU, v:%.2f km/s' % (mol.upper(), Qatm, Xgas, rh, vexp/1e3))
plt.ylabel('Relative level population')
plt.xlabel('Distance from nucleus [km]')
plt.tight_layout()
plt.savefig('nlte.png', dpi=300)
plt.show()
