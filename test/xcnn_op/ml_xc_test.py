import predict_xcnn as xcnn
from pyscf import  dft
import numpy as np
import numpy
import jax.numpy as jnp
import  math
from jax import config
from test.xc_deriv import x_deriv,c_deriv
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
def lyp(rhoa, rhob, gamaa, gambb, gamab,i):
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



def eval_b88(xc_code, rho, spin, relativity=0, deriv=2, verbose=None, omega=None):
    rho1 = rho[0]
    rho2 = rho[1]

    rho01, dx1, dy1, dz1 = rho1[:4]
    rho02, dx2, dy2, dz2 = rho2[:4]

    rho01 = rho01 + 1E-12
    rho02 = rho02 + 1E-12
    w1 = rho01 / (rho01 + rho02)
    w2 = rho02 / (rho01 + rho02)
    gamma1 = dx1 ** 2 + dy1 ** 2 + dz1 ** 2+ 1E-20
    gamma2 = dx2 ** 2 + dy2 ** 2 + dz2 ** 2+ 1E-20

    vgamma_3 = [0] * len(rho01)
    v2rho23 = [0] * len(rho02)

    ex1 = b88(rho01, gamma1)
    ex2 = b88(rho02, gamma2)
    exc1 = (ex1 / rho01) * w1
    exc2 = (ex2 / rho02) * w2
    vrho1, vgamma_1, v2rho21, v2rhotau1, v2tau21 = x_deriv.dxc((rho01), (gamma1),b88)
    vrho2, vgamma_2, v2rho22, v2rhotau2, v2tau22 = x_deriv.dxc((rho02), (gamma2),b88)
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

    dx = c_deriv.dxc(a, b, gaa, gbb, gnn,lyp)



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

    xcnn_=xcnn.eval_xc_gga(xc_code, rho, spin, relativity=0, deriv=2, verbose=None, omega=None)
    b88_ = eval_b88(xc_code, rho, spin, relativity=0, deriv=2, verbose=None, omega=None)
    bel = w_[4]

    exc = np.transpose(exc + extd[0]) + bel * b88_[0]+xcnn_[0]

    vxc = (vrho + bel * b88_[1][0]+xcnn_[1][0], vgamma + bel * b88_[1][1]+xcnn_[1][1], None, None)
    fxc = (np.transpose(fxc_)+bel*b88_[2][0]+xcnn_[2][0], fxc_1.T+bel*b88_[2][1]+xcnn_[2][1], fxc_2.T+bel*b88_[2][2]+xcnn_[2][2])

    kxc = None  # 3rd order functional derivative

    return exc, vxc, fxc, kxc
