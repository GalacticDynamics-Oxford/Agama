#!/usr/bin/python
"""
This script constructs a realistic non-axisymmetric potential of the Milky Way
including a bar and spiral arms, which was fitted to the local velocity distribution
from Gaia DR3-RVS using the backward integration method (Khalil et al. 2025).
The script is an abridged version adapted from https://github.com/yrkhalil/SPIBACK

References:  Khalil et al. 2025 (A&A/699/263); Thomas et al. 2023 (A&A/678/180).

Author: Yassin Khalil, with minor modifications by Eugene Vasiliev.

See also example_mw_potential_hunter24.py for another non-axisymmetric potential.

Note that in the present script, the Milky Way is rotating counter-clockwise
(the angular momentum of disk stars is positive), unlike the convention used in
mw_potential_hunter24 and most other literature.
"""
import agama, numpy, matplotlib, matplotlib.pyplot as plt
agama.setUnits(mass=1, length=1, velocity=1)  # 1 Msun, 1 kpc, 1 km/s

# Parameters of the non-axsiymmetric Milky Way potential (Khalil et al. 2025).
# Galactocentric left handed convention with sun at (x=R0, y=0.0).
R0 = 8.275                          # Sun's Galactocentric radius

# Bar parameters. mpars, (amplitudes and power scales), R_max, modes and ksi from Thomas et al. 2023
mpars   = numpy.array([
    [ 0.25552453,   1.80362025,   5.08270614, 0.05],
    [ 8.45479173,   4.08573902,  10.69978858, 0.025],
    [210.40997238,  5.95902218,  16.06541472, 0.05]])
R_max   = 12.0                      # Scale length
modes   = [2, 4, 6]                 # Modes
ksi     = [0.05, 0.025, 0.05]       # z_scale for each mode
phi_bar = numpy.radians(28)         # Bar angle
Omegab  = 37.0                      # Pattern speed
T_bar   = 2*numpy.pi/Omegab         # Bar rotation period
t1      = -3.2*T_bar                # Time of start for growth (also total integration time for backwards integration)
t2      = t1/2                      # Time of end for growth

# Spiral arms parameters for the m=2 Fourier component
m_2               = 2       # Mode
A_2               = 2818.1  # Amplitude
Omegas_2          = 13.071  # Pattern speed
pitch_angle_2     = 0.1415  # Pitch angle
phi0_2            = 3.9765  # Phase
R_cutoff_arms_2   = 6.6461  # Inner cutoff in radius
R_outter_cutoff_2 = 26.404  # Outer cutoff in radius
z_cutoff_2        = 1.0     # Cutoff in absolute value of height
width_R_inner_2   = 0.3     # Tanh cutoff width at inner cutoff
width_R_outer_2   = 0.3     # Tanh cutoff width at outer cutoff
hs_2              = 0.13    # Scale height
Tarms_2           = 2*numpy.pi/Omegas_2  # Spiral mode rotation period
t3_2              = -Tarms_2    # Time of start for growth
t4_2              = -Tarms_2/2  # Time of end for growth

# Spiral arms parameters for the m=3 Fourier component
m_3               = 3       # Mode
Omegas_3          = 16.406  # Pattern speed
pitch_angle_3     = 0.2386  # Pitch angle
phi0_3            = 1.4261  # Phase
A_3 = A_2 / ( (Omegas_3/Omegas_2)**2 * numpy.tan(pitch_angle_3) / numpy.tan(pitch_angle_2) ) # Amplitude relation from Hamilton 2024
R_cutoff_arms_3   = 7.9695  # Inner cutoff radius
R_outter_cutoff_3 = 19.653  # Outer cutoff radius
z_cutoff_3        = 1.0     # Cutoff in absolute value of height
width_R_inner_3   = 1.1612  # Tanh cutoff width at inner cutoff
width_R_outer_3   = 1.1612  # Tanh cutoff width at outer cutoff
hs_3              = 0.13    # Scale height
Tarms_3           = 2*numpy.pi/Omegas_3  # Spiral mode rotation period
t3_3              = -Tarms_3    # Time of start for growth
t4_3              = -Tarms_3/2  # Time of end for growth

# Setting the axisymmetric potential with density profiles parameters from Khalil et al. 2025
stellar_disk_params = dict(
    type="Disk",
    surfaceDensity=1.18814e9,
    scaleRadius=2.4,
    scaleHeight=0.3)
gas_disk_params = dict(
    type="Disk",
    surfaceDensity=0.071772e9,
    scaleRadius=4.8,
    scaleHeight=0.130)
bulge_params = dict(
    type="Spheroid",
    densityNorm=1.0834e8,
    gamma=1.30368,
    beta=2.9067,
    scaleRadius=8.1581,
    outerCutoffRadius=0.83073)
DM_halo_params = dict(
    type="Spheroid",
    densityNorm=4.56e8,
    axisRatioZ=0.8,
    gamma=0.,
    beta=0.,
    scaleRadius=1.,
    outerCutoffRadius=0.65,
    cutoffStrength=0.499)

# Bar potential
def get_bar_potential(pot_axisymmetric, bar_scaling, mpars, R_max, phi_bar, Omegab, modes, ksi):
    def bar_pot(xyz):
        R = (xyz[:,0]**2 + xyz[:,1]**2)**0.5
        z = xyz[:,2]
        phi = numpy.arctan2(xyz[:,1], xyz[:,0])
        result = numpy.zeros_like(R)
        for mode in modes:
            ind = mode//2 - 1  # 0,1,2 for m=2,4,6
            result += (mpars[ind, 0] * (R / R_max)**(mpars[ind, 1] - 1) *
                numpy.maximum(0, 1 - R / R_max)**(mpars[ind, 2] - 1) *
                numpy.cos(mode*(phi - 0*phi_bar)) / (1 + (z / (0.45*R + ksi[ind]))**2 ) )
        return numpy.nan_to_num(pot_axisymmetric.potential(numpy.column_stack((xyz[:,0:2], z*0))) * result)

    with numpy.errstate(all='ignore'):
        return agama.Potential(
        type = 'CylSpline',
        potential = bar_pot,
        Rmin = 0.1,
        Rmax = R_max+1e-6,
        zmin = 0.1,
        zmax = 10.0,
        mmax = 6,
        fixOrder  = True,
        symmetry  = 'Triaxial',
        gridSizez = 25,
        gridSizeR = 30,
        rotation = [[0, phi_bar], [1, phi_bar + Omegab]],
        scale = bar_scaling)

# Spiral arm potential
def get_spiral_potential(spiral_scaling, numberOfArms, amplitude, Omega, pitchAngle, phi0, scaleRadius, R_cutoff_inner, R_cutoff_outer, z_cutoff, width_R_inner, width_R_outer, scaleHeight):
    def densfnct(xyz):
        x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
        R = numpy.sqrt(xyz[:,0]**2 + xyz[:,1]**2)
        phi = numpy.arctan2(xyz[:,1], xyz[:,0])

        # Radial cutoff and derivatives
        tanPA   = numpy.tan(pitchAngle)
        ro      = (R_cutoff_outer - R) / width_R_outer
        ri      = (R - R_cutoff_inner) / width_R_inner
        Co      = 0.5 * (1 + numpy.tanh(ro))
        Ci      = 0.5 * (1 + numpy.tanh(ri))
        C       = Ci * Co
        dR_Co   = -0.5/width_R_outer*numpy.cosh(ro)**(-2)
        dR_Ci   = +0.5/width_R_inner*numpy.cosh(ri)**(-2)
        dR_C    = dR_Ci*Co + dR_Co*Ci
        dRR_Co  = -numpy.cosh(ro)**(-2)/width_R_outer/width_R_outer*numpy.tanh(ro)
        dRR_Ci  = -numpy.cosh(ri)**(-2)/width_R_inner/width_R_inner*numpy.tanh(ri)
        dRR_C   = dRR_Ci*Co + dRR_Co*Ci + 2*dR_Co*dR_Ci

        # 3D Spiral arm's potential
        K       = numberOfArms / (R*numpy.sin(pitchAngle))
        D       = 1 / (1 + 0.3 * K * scaleHeight) + K * scaleHeight
        B       = K * scaleHeight * ( 1 + 0.4 * K * scaleHeight)
        A       = -amplitude/(scaleRadius*K*D)
        theta   = numberOfArms*(phi + numpy.log(R/scaleRadius)/tanPA - phi0)
        u       = z*K/B
        sh      = 1/numpy.cosh(u)
        th      = numpy.tanh(u)
        h       = C*A
        f       = numpy.cos(theta)
        v       = sh**B
        Phi     = h*f*v

        # Laplacian
        dR_K    = -K/R
        dRR_K   = 2*K/R/R
        dR_D    = dR_K  * scaleHeight - (0.3 * scaleHeight * dR_K)  / (1 + 0.3 * scaleHeight*K)**2
        dRR_D   = dRR_K * scaleHeight - (0.3 * scaleHeight * dRR_K) / (1 + 0.3 * scaleHeight*K)**2 + 2 * (0.3 * scaleHeight * dR_K)**2 / (1 + 0.3 * scaleHeight * K)**3
        dR_B    = dR_K  * scaleHeight +  0.8 * scaleHeight**2 *  K * dR_K
        dRR_B   = dRR_K * scaleHeight +  0.8 * scaleHeight**2 * (K * dRR_K + dR_K**2)
        dR_A    = -A*(dR_K/K + dR_D/D)
        dRR_A   = dR_A**2/A - A*(dRR_K/K + dRR_D/D -(dR_K/K)**2 - (dR_D/D)**2)
        dR_h    = dR_C*A + C*dR_A
        dRR_h   = dRR_C*A + C*dRR_A + 2*dR_C*dR_A
        dR_f    = -numpy.sin(theta)*numberOfArms/(tanPA*R)
        dRR_f   = -numpy.cos(theta)*numberOfArms**2/((tanPA*R)**2) + numpy.sin(theta)*numberOfArms/(tanPA*R**2)
        dp_f    = -numpy.sin(theta)*numberOfArms
        dpp_f   = -numpy.cos(theta)*numberOfArms**2
        dR_v    = sh**B*(dR_B*numpy.log(sh) - z*th*(dR_K-K*dR_B/B))
        dRR_v   = dR_v**2/v + v*(dRR_B*numpy.log(sh) - dR_B/B*th*z*(dR_K-dR_B*K/B) - z**2*(dR_K-K*dR_B/B)**2*sh**2/B - z*th*(dRR_K - dR_K*dR_B/B - K/B*dRR_B + K*(dR_B/B)**2))
        dz_v    = -K*th*sh**B
        dzz_v   = K**2*sh**B*(-sh**2/B + th**2)
        dR_Phi  = dR_h*f*v + h*dR_f*v + h*f*dR_v
        dRR_Phi = dRR_h*f*v + h*dRR_f*v + h*f*dRR_v + 2*dR_h*dR_f*v + 2*dR_h*f*dR_v  + 2*h*dR_f*dR_v
        dp_Phi  = h*v*dp_f
        dpp_Phi = h*v*dpp_f
        dz_Phi  = h*f*dz_v
        dzz_Phi = h*f*dzz_v

        laplacian = dR_Phi/R + dRR_Phi + dpp_Phi/R/R + dzz_Phi

        return numpy.nan_to_num(numpy.where(abs(z) < z_cutoff, laplacian / (4*numpy.pi*agama.G), 0))

    with numpy.errstate(all='ignore'):
        return agama.Potential(type='CylSpline', density=densfnct, Rmin = 0.01*scaleRadius, Rmax = 5*scaleRadius,
        mmax = numberOfArms, zmin = 0.25*scaleHeight, zmax = 10*scaleHeight, gridSizeZ = 25, gridSizeR = 50,
        symmetry  = 'Bisymmetric' if numberOfArms % 2 == 0 else '4',  # 4 means only z-symmetry, but no full reflection symmetry
        rotation=[[0, 0], [1, Omega]], scale = spiral_scaling)

def get_scaling(t1, t2):
    # Time interval for the time-dependent amplitude
    t = numpy.linspace(t1, t2, 15)
    xi = 2 * (t - t1) / (t2 - t1) - 1
    # Modulation of amplitude in time adapted from Dehnen 2000 for the backward orbit integration.
    ampl = 3./16 * xi**5 - 5./8 * xi**3 + 15./16 * xi + 1./2
    # Time derivative of the above function
    ampl_der = 15./8 * (xi**2 - 1)**2 / (t2 - t1)
    # It is specified as a quintic function of time, but the "Scaled" potential modifier represents
    # the scaling factors by cubic splines, so we have to approximate it using several control points (t)
    return numpy.column_stack((t,
        ampl,                 # mass scaling factor
        numpy.ones_like(t),   # radius scaling factor (1= no additional scaling)
        ampl_der,             # time derivative of the mass scaling factor
        numpy.zeros_like(t))) # time derivative of the radius scaling factor

# Create the main (axisymmetric) component of the potential, which is also used to construct the bar potential
pot_axisymmetric = agama.Potential(stellar_disk_params, gas_disk_params, bulge_params, DM_halo_params)

# Create the bar potential (including the rotation and amplitude scaling)
bar_scaling = get_scaling(t1, t2)
pot_bar = get_bar_potential(pot_axisymmetric, bar_scaling, mpars, R_max, phi_bar, Omegab, modes, ksi)

# Create the m=2 spiral arms potential
spiral2_scaling = get_scaling(t3_2, t4_2)
pot_arms_2 = get_spiral_potential(spiral2_scaling, m_2, A_2, Omegas_2, pitch_angle_2, phi0_2, R0, R_cutoff_arms_2, R_outter_cutoff_2, z_cutoff_2, width_R_inner_2, width_R_outer_2, hs_2)

# Create the m=3 spiral arms potential
spiral3_scaling = get_scaling(t3_3, t4_3)
pot_arms_3 = get_spiral_potential(spiral3_scaling, m_3, A_3, Omegas_3, pitch_angle_3, phi0_3, R0, R_cutoff_arms_3, R_outter_cutoff_3, z_cutoff_3, width_R_inner_3, width_R_outer_3, hs_3)

# The total non-axisymmetric potential (stars + DM) and the part corresponding to stars only (for visualization)
pot_fiducial = agama.Potential(pot_axisymmetric, pot_bar, pot_arms_2, pot_arms_3)
pot_stars = agama.Potential(stellar_disk_params, gas_disk_params, bulge_params, pot_bar, pot_arms_2, pot_arms_3)


# Demonstrate the use of this potential to compute the local velocity distribution using the backward integration method
def get_DF_integrator(potential, integration_time):
    def DF_integrator(x0, y0, z0):
        theta = numpy.arctan2(y0, x0)
        vr = numpy.arange(120 + 1) * 2 - 120
        vphi = numpy.arange(120 + 1) * 2 + 90
        Vr, Vphi = numpy.meshgrid(vr, vphi, indexing='xy')
        Vx0 = Vr.flatten() * numpy.cos(theta) - Vphi.flatten() * numpy.sin(theta)
        Vy0 = Vr.flatten() * numpy.sin(theta) + Vphi.flatten() * numpy.cos(theta)
        Vz0 = 0.0
        xv = numpy.column_stack([x0 + Vx0*0, y0 + Vx0*0, z0 + Vx0*0, Vx0, Vy0, Vz0 + Vx0*0])
        return df(af(numpy.vstack(agama.orbit(potential=potential, ic=xv, time=integration_time, trajsize=1)[:,1]))).reshape(120 + 1, 120 + 1)
    return DF_integrator

# Setting up the distribution function
df_thin_disk  = agama.DistributionFunction(type='QuasiIsothermal', Sigma0=1.00, Rdisk=2.4, sigmaz0=20*numpy.exp(R0/10), Rsigmaz=10, sigmar0=30*numpy.exp(R0/10), Rsigmar=10, potential=pot_axisymmetric)
df_thick_disk = agama.DistributionFunction(type='QuasiIsothermal', Sigma0=0.05, Rdisk=2.4, sigmaz0=40*numpy.exp(R0/10), Rsigmaz=10, sigmar0=55*numpy.exp(R0/10), Rsigmar=10, potential=pot_axisymmetric)
df = agama.DistributionFunction(df_thin_disk, df_thick_disk)
af = agama.ActionFinder(pot_axisymmetric)

print('Computing the local velocity map')
DF = get_DF_integrator(pot_fiducial, t1)
x0, y0, z0 = 8.275, 0.0, 0.0
DF_model = DF(x0, y0, z0) * 7.29e9
plt.ion()
ax=plt.subplots(1, 2, figsize=(12,6), dpi=100)[1]

im_DF = ax[0].imshow(DF_model, interpolation='nearest', origin='lower', aspect='equal', cmap=plt.get_cmap('CMRmap_r'),
    norm=matplotlib.colors.SymLogNorm(linthresh=1, linscale=1, vmin=0, vmax=numpy.max(DF_model)), extent=[-120, 120, 90, 330])
ax[0].set_ylabel(r'$V_{\varphi}$ [km/s]')
ax[0].set_xlabel(r'$V_{R}$ [km/s]')
ax[0].text(0.5, 0.99, 'Local velocity map at t=0', ha='center', va='top', transform=ax[0].transAxes)
ax[0].set_aspect('equal')
ax[1].set_ylabel('Y')
plt.tight_layout()

# overplot the surface density contours
print('Computing the surface density of stars at different times')
rmax   = 10.0
gridr  = numpy.linspace(-rmax, rmax, 50)  # 1d grid
gridxy = numpy.column_stack((numpy.repeat(gridr, len(gridr)), numpy.tile(gridr, len(gridr)), numpy.zeros(len(gridr)**2)))
times  = numpy.linspace(-0.6, 0, 61)
for t in times:
    Sigma = pot_stars.projectedDensity(gridxy[:,0:2], t=t)
    logSigma = numpy.log10(numpy.maximum(Sigma, 1e-5))
    ax[1].cla()
    ax[1].contourf(gridr, gridr, logSigma.reshape(len(gridr), len(gridr)).T,
        levels=numpy.linspace(7, 11, 41), vmin=7, vmax=11, cmap='hell_r')
    ax[1].text(0.01, 0.99, 't=%.2f Gyr' % t, ha='left', va='top', transform=ax[1].transAxes)
    ax[1].set_aspect('equal')
    ax[1].set_xlabel('X [kpc]')
    ax[1].set_ylabel('Y [kpc]', labelpad=-5)
    plt.draw()
    plt.pause(0.1)

plt.ioff()
plt.show()
