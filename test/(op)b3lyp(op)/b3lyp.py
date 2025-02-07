from pyscf import  dft
import numpy as np
import numpy
import jax.numpy as jnp
import  math
from jax import config
config.update("jax_enable_x64", True)
delta_kali=1e-5
w_ = [0] * 5
w_[0] = 0.17457035788765415
w_[1] = 0.025429642112345774
w_[2] = 0.8803280235993975
w_[3] = 0.11967197640060245
w_[4] = 0.8


def b88(rho01,gama):
    tau1=gama**0.5
    x = tau1 / (rho01) ** (4.0 / 3.0)
    b = 0.0043906
    b88_g = -1.5 * (3.0 / 4.0 / math.pi) ** (1.0 / 3.0) - b * (x ** 2) / (1.0 + 6.0 * b * x * np.arcsinh(x))
    exc1 = rho01 ** (4.0 / 3.0 ) * b88_g

    return exc1
def lyp(rhoa, rhob, gamaa, gambb, gamab):
    a = 0.04779  # Parameters from the LYP papers
    b = 0.1022
    c = 0.3481
    d = 0.3407
    rho = rhoa + rhob
    rhom3 = jnp.power(rho, -1. / 3.)
    w = jnp.exp(-c * rhom3) / (1 + d * rhom3) * jnp.power(rho, -11. / 3.)
    dl = c * rhom3 + d * rhom3 / (1 + d * rhom3)

    fcgamaa = -a * b * w * ((1. / 9.) * rhoa * rhob * (1 - 3 * dl - (dl - 11) * rhoa / rho) - rhob * rhob)
    fcgamab = -a * b * w * ((1. / 9.) * rhoa * rhob * (47 - 7 * dl) - (4. / 3.) * rho * rho)
    fcgambb = -a * b * w * ((1. / 9.) * rhoa * rhob * (1 - 3 * dl - (dl - 11) * rhob / rho) - rhoa * rhoa)

    fc = -4 * a / (1 + d * rhom3) * rhoa * rhob / rho \
         - jnp.power(2, 11. / 3.) * 0.3 * jnp.power(3 * jnp.pi * np.pi, 2. / 3.) * a * b * w \
         * rhoa * rhob * (jnp.power(rhoa, 8. / 3.) + jnp.power(rhob, 8. / 3.)) \
         + fcgamaa * gamaa + fcgamab * gamab + fcgambb * gambb
    return fc
def vrho_1(a, b, gaa, gbb, gnn):
    delta1 = a * delta_kali
    exc = lyp(a, b, gaa, gbb, gnn)
    exc1 = (lyp(a + delta1, b, gaa, gbb, gnn))
    vrho1 = (exc1 - exc) / (delta1 + 1e-250)
    return vrho1


def vrho_2(a, b, gaa, gbb, gnn):
    delta2 = b * delta_kali
    exc = lyp(a, b, gaa, gbb, gnn)
    exc2 = lyp(a, b + delta2, gaa, gbb, gnn)
    vrho2 = (exc2 - exc) / (delta2 + 1e-250)
    return vrho2


def vgama_1(a, b, gaa, gbb, gnn):
    delta1 = (gaa) * delta_kali
    exc = lyp(a, b, gaa, gbb, gnn)
    exc1g = lyp(a, b, gaa + delta1, gbb, gnn)
    vgama1 = (exc1g - exc) / (delta1 + 1e-250)
    return vgama1


def vgama_2(a, b, gaa, gbb, gnn):
    delta2 = (gbb) * delta_kali
    exc = lyp(a, b, gaa, gbb, gnn)
    exc2g = lyp(a, b, gaa, gbb + delta2, gnn)
    vgama2 = (exc2g - exc) / (delta2 + 1e-250)
    return vgama2


def vgama_3(a, b, gaa, gbb, gnn):
    delta3 = (gnn) * delta_kali
    exc = lyp(a, b, gaa, gbb, gnn)
    exc3g = lyp(a, b, gaa, gbb, gnn + delta3)
    vgama3 = (exc3g - exc) / (delta3 + 1e-250)
    return vgama3


def dxc(a, b, gaa, gbb, gnn):
    vrho1 = vrho_1(a, b, gaa, gbb, gnn)
    vrho2 = vrho_2(a, b, gaa, gbb, gnn)

    vrhoc = [vrho1, vrho2]

    vgama1 = vgama_1(a, b, gaa, gbb, gnn)
    vgama2 = vgama_2(a, b, gaa, gbb, gnn)
    vgama3 = vgama_3(a, b, gaa, gbb, gnn)

    vgamac = [vgama1, vgama3, vgama2]

    delta1_ = (a) * delta_kali
    vrho1_ = vrho_1(a + delta1_, b, gaa, gbb, gnn)
    v2rho1 = (vrho1_ - vrho1) / (delta1_+ 1e-250)

    delta2_ = (b) * delta_kali
    vrho2_ = vrho_1(a, b + delta2_, gaa, gbb, gnn)
    v2rho21 = (vrho2_ - vrho1) / (delta2_+ 1e-250)

    delta2_ = (b) * delta_kali
    vrho2_ = vrho_2(a, b + delta2_, gaa, gbb, gnn)
    v2rho2 = (vrho2_ - vrho2) / (delta2_+ 1e-250)

    v2rhoc = [v2rho1, v2rho21, v2rho2]

    delta1 = (gaa) * delta_kali
    vgama1_ = vgama_1(a, b, gaa + delta1, gbb, gnn)
    vtautau11 = (vgama1_ - vgama1) / (delta1+ 1e-250)

    delta1 = (gaa) * delta_kali
    vgama2_ = vgama_2(a, b, gaa + delta1, gbb, gnn)
    vtautau12 = (vgama2_ - vgama2) / (delta1+ 1e-250)

    delta1 = (gaa) * delta_kali
    vgama3_ = vgama_3(a, b, gaa + delta1, gbb, gnn)
    vtautau13 = (vgama3_ - vgama3) / (delta1+ 1e-250)

    delta2 = (gbb) * delta_kali
    vgama2_ = vgama_2(a, b, gaa, gbb + delta2, gnn)
    vtautau22 = (vgama2_ - vgama2) / (delta2+ 1e-250)

    delta2 = (gbb) * delta_kali
    vgama3_ = vgama_3(a, b, gaa, gbb + delta2, gnn)
    vtautau23 = (vgama3_ - vgama3) / (delta2+ 1e-250)

    delta3 = (gnn) * delta_kali
    vgama3_ = vgama_3(a, b, gaa, gbb, gnn + delta3)
    vtautau33 = (vgama3_ - vgama3) / (delta3+ 1e-250)

    vtautauc = [vtautau11, vtautau12, vtautau13, vtautau22, vtautau23, vtautau33]

    # =================
    delta1 = (a) * delta_kali
    vgama1_ = vgama_1(a + delta1, b, gaa, gbb, gnn)
    vrhotau11 = (vgama1_ - vgama1) / (delta1+ 1e-250)

    delta1 = (a) * delta_kali
    vgama2_ = vgama_2(a + delta1, b, gaa, gbb, gnn)
    vrhotau12 = (vgama2_ - vgama2) / (delta1+ 1e-250)

    delta1 = (a) * delta_kali
    vgama3_ = vgama_3(a + delta1, b, gaa, gbb, gnn)
    vrhotau13 = (vgama3_ - vgama3) / (delta1+ 1e-250)

    delta2 = (b) * delta_kali
    vgama1_ = vgama_1(a, b + delta2, gaa, gbb, gnn)
    vrhotau21 = (vgama1_ - vgama1) / (delta2+ 1e-250)

    delta2 = (b) * delta_kali
    vgama2_ = vgama_2(a, b + delta2, gaa, gbb, gnn)
    vrhotau22 = (vgama2_ - vgama2) / (delta2+ 1e-250)

    delta2 = (b) * delta_kali
    vgama3_ = vgama_3(a, b + delta2, gaa, gbb, gnn)
    vrhotau23 = (vgama3_ - vgama3) / (delta2+ 1e-250)

    vrhotauc = [vrhotau11, vrhotau12, vrhotau13, vrhotau21, vrhotau22, vrhotau23]

    dxc = [vrhoc, vgamac, v2rhoc, vrhotauc, vtautauc]
    return dxc
def vrho(rho01, gamma1):
    delta1_ = (rho01) * delta_kali
    ex1 = b88(rho01, gamma1)
    ex1_ = b88(rho01 + delta1_, gamma1)
    vrho1 = (ex1_ - ex1) / (delta1_ + 1e-250)
    return vrho1


def vgama(rho01, gamma1):
    delta1 = (gamma1) * delta_kali
    ex1 = b88(rho01, gamma1)
    ex1_ = b88(rho01, gamma1 + delta1)
    vrho1 = (ex1_ - ex1) / (delta1 + 1e-250)
    return vrho1


def dxc_(rho01, gamma1):
    vgamma_1 = vgama(rho01, gamma1)

    delta1_ = (rho01) * delta_kali
    vrho1 = vrho(rho01, gamma1)
    vrho1_ = vrho(rho01 + delta1_, gamma1)
    vrho2 = (vrho1_ - vrho1) / (delta1_)

    delta1 = (gamma1) * delta_kali
    vgama1 = vgama(rho01, gamma1)
    vgama1_ = vgama(rho01, gamma1 + delta1)
    vtautau2 = (vgama1_ - vgama1) / (delta1)

    vgama1 = vgama(rho01, gamma1)
    vgama1_ = vgama(rho01 + delta1_, gamma1)
    vrhotau2 = (vgama1_ - vgama1) / (delta1_)
    dxc = [vrho1, vgamma_1, vrho2, vrhotau2, vtautau2]
    return dxc
def eval_xc_gga(xc_code, rho, spin, relativity=0, deriv=2, verbose=None, omega=None):
    rho1 = rho[0]
    rho2 = rho[1]

    rho01, dx1, dy1, dz1 = rho1[:4]
    rho02, dx2, dy2, dz2 = rho2[:4]

    rho01 = rho01 + 1E-250
    rho02 = rho02 + 1E-250
    w1 = rho01 / (rho01 + rho02)
    w2 = rho02 / (rho01 + rho02)
    gamma1 = dx1 ** 2 + dy1 ** 2 + dz1 ** 2
    gamma2 = dx2 ** 2 + dy2 ** 2 + dz2 ** 2

    vgamma_3 = [0] * len(rho01)
    v2rho23 = [0] * len(rho02)

    ex1 = b88(rho01, gamma1)
    ex2 = b88(rho02, gamma2)
    exc1 = (ex1 / rho01) * w1
    exc2 = (ex2 / rho02) * w2
    vrho1, vgamma_1, v2rho21, v2rhotau1, v2tau21 = dxc_((rho01), (gamma1))
    vrho2, vgamma_2, v2rho22, v2rhotau2, v2tau22 = dxc_((rho02), (gamma2))

    pbe_xc = dft.libxc.eval_xc(',0*LYP', rho, spin, relativity, 2,
                               verbose)

    fxc_ = np.array([v2rho21, v2rho23, v2rho22]) + np.array(pbe_xc[2][0]).T

    kll = np.array(pbe_xc[2][1]).T
    fxc_1 = np.array([v2rhotau1, kll[1] * 0, kll[2] * 0, kll[3] * 0, kll[4] * 0, v2rhotau2]) + np.array(pbe_xc[2][1]).T

    kll = np.array(pbe_xc[2][2]).T
    fxc_2 = np.array([v2tau21, kll[1] * 0, kll[2] * 0, kll[3] * 0, kll[4] * 0, v2tau22]) + np.array(pbe_xc[2][2]).T

    vgamma_ = np.array([vgamma_1, vgamma_3, vgamma_2])
    vgamma_ = vgamma_ + np.array(pbe_xc[1][1]).T
    vgamma = np.transpose(vgamma_)
    vrho_ = np.array([vrho1, vrho2])
    vrho_ = vrho_ + numpy.array(pbe_xc[1][0]).T

    vrho = np.transpose(vrho_)

    exc1 = np.array([exc1])
    exc2 = np.array([exc2])
    exc = np.transpose(exc1 + exc2 + pbe_xc[0])

    vxc = (vrho, vgamma, None, None)
    fxc = (np.transpose(fxc_), fxc_1.T, fxc_2.T)  # 2nd order functional derivative
    # fxc = ( blyp[2][0], blyp[2][1], blyp[2][2])  # 2nd order functional derivative
    kxc = None  # 3rd order functional derivative
    return exc, vxc, fxc, kxc


def eval_xc_lyp_b(xc_code, rho, spin, relativity=0, deriv=2, verbose=None, omega=None):
    rho1 = rho[0]
    rho2 = rho[1]

    a, dx1, dy1, dz1 = rho1[:4]
    b, dx2, dy2, dz2 = rho2[:4]
    gaa = dx1 ** 2 + dy1 ** 2 + dz1 ** 2
    gbb = dx2 ** 2 + dy2 ** 2 + dz2 ** 2
    gnn = (dx1 * dx2) + (dy1 * dy2) + (dz1 * dz2)

    exc = lyp(a, b, gaa, gbb, gnn,0)

    dx = dxc(a, b, gaa, gbb, gnn)



    al = w_[2]
    extd = dft.libxc.eval_xc('' + str("%.16f" % w_[0]) + '*HF+' + str("%.16f" % w_[1]) + '*SLATER+' + str(
        "%.16f" % 0) + '*B88  , ' + str("%.16f" % 0) + '*LYP + ' + str("%.16f" % w_[3]) + '*VWN3', rho, spin,
                             relativity, 2,
                             verbose)

    fxc_ = al * np.array(dx[2]) + np.array(extd[2][0]).T

    fxc_1 = al * np.array(dx[3]) + np.array(extd[2][1]).T

    fxc_2 = al * np.array(dx[4]) + np.array(extd[2][2]).T

    vgamma_ = al * np.array(dx[1])

    vgamma_ = vgamma_ + np.array(extd[1][1]).T

    vgamma = np.transpose(vgamma_)

    vrho_ = al * np.array(dx[0])
    vrho_ = vrho_ + numpy.array(extd[1][0]).T

    vrho = np.transpose(vrho_)

    exc = al * np.array([exc] / (a + b))

    b88_ = eval_xc_gga(xc_code, rho, spin, relativity=0, deriv=2, verbose=None, omega=None)
    bel = w_[4]

    exc = np.transpose(exc + extd[0]) + bel * b88_[0]

    vxc = (vrho + bel * b88_[1][0], vgamma + bel * b88_[1][1], None, None)
    fxc = (np.transpose(fxc_)+bel*b88_[2][0], fxc_1.T+bel*b88_[2][1], fxc_2.T+bel*b88_[2][2])

    kxc = None  # 3rd order functional derivative

    return exc, vxc, fxc, kxc
