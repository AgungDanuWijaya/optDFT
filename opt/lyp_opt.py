
import numpy as np
import numpy
from pyscf import gto, dft, scf, cc
from bayes_opt import BayesianOptimization
import c_deriv
from data import read_data
class opt():
    def __init__(self, ww1,ww2,ww3,ww4):
        self.ww1=ww1
        self.ww2=ww2
        self.ww3 = ww3
        self.ww4 = ww4

    def lyp(self,rhoa, rhob, gamaa, gambb, gamab):
        a = self.ww1
        b = self.ww2
        c = self.ww3
        d = self.ww4
        rho = rhoa + rhob
        rhom3 = np.power(rho, -1. / 3.)
        w = np.exp(-c * rhom3) / (1 + d * rhom3) * np.power(rho, -11. / 3.)
        dl = c * rhom3 + d * rhom3 / (1 + d * rhom3)

        fcgamaa = -a * b * w * ((1. / 9.) * rhoa * rhob * (1 - 3 * dl - (dl - 11) * rhoa / rho) - rhob * rhob)
        fcgamab = -a * b * w * ((1. / 9.) * rhoa * rhob * (47 - 7 * dl) - (4. / 3.) * rho * rho)
        fcgambb = -a * b * w * ((1. / 9.) * rhoa * rhob * (1 - 3 * dl - (dl - 11) * rhob / rho) - rhoa * rhoa)

        fc = -4 * a / (1 + d * rhom3) * rhoa * rhob / rho \
             - np.power(2, 11. / 3.) * 0.3 * np.power(3 * np.pi * np.pi, 2. / 3.) * a * b * w \
             * rhoa * rhob * (np.power(rhoa, 8. / 3.) + np.power(rhob, 8. / 3.)) \
             + fcgamaa * gamaa + fcgamab * gamab + fcgambb * gambb
        return fc

    def eval_xc_gga(self,xc_code, rho, spin, relativity=0, deriv=2, verbose=None, omega=None):
        rho1 = rho[0]
        rho2 = rho[1]
        a, dx1, dy1, dz1 = rho1[:4]
        b, dx2, dy2, dz2 = rho2[:4]
        gaa = dx1 ** 2 + dy1 ** 2 + dz1 ** 2
        gbb = dx2 ** 2 + dy2 ** 2 + dz2 ** 2
        gnn = (dx1 * dx2) + (dy1 * dy2) + (dz1 * dz2)
        exc = self.lyp(a, b, gaa, gbb, gnn)
        dx = c_deriv.dxc(a, b, gaa, gbb, gnn,self.lyp)
        al = 1
        extd = dft.xcfun.eval_xc('HF+0*b88,', rho, spin, relativity, 2,
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
        exc = np.transpose(exc + extd[0])
        vxc = (vrho, vgamma, None, None)
        fxc = (np.transpose(fxc_), fxc_1.T, fxc_2.T)
        kxc = None  # 3rd order functional derivative
        return exc, vxc, fxc, kxc

    def loss(self):
        try:
            dat1 = "He}Li}Be}C}N}O}F}Ne}Na}Mg}Al}Ar}Si}P}S}Cl"
            dat = dat1
            dat_ae = [-2.9038,
                      -7.478,
                      -14.6685,
                      -37.85474495,
                      -54.60876233,
                      -75.1082709,
                      -99.80308252,
                      -129.0467256,
                      -162.4231978,
                      -200.3044022,
                      -242.7098161,
                      -529.109435476803,
                      -289.8606704,
                      -341.9381392,
                      -399.02564351,
                      -461.36903633
                      ]

            x = dat.split("}")
            mydict = {'asa': 32}
            index_ = 0
            total_e = 0
            dir = "/home/agung/PycharmProjects/pkpt/data_/"
            index_ = 0
            for jk_ in x:
                mol = gto.Mole()
                mol.verbose = 0
                mol.atom = "" + read_data.read_g(jk_, dir) + ""
                mol.charge = 0
                mol.spin = int(read_data.read_spin(jk_, dir))
                mol.basis = "aug-cc-pvdz"
                mol.build()
                mfl = dft.UKS(mol).x2c1e()
                mfl.define_xc_(self.eval_xc_gga, xctype='GGA', hyb=1)
                mfl.xc = "HF,"
                ener = mfl.kernel()
                total_e = total_e + abs(abs(ener) - abs(dat_ae[index_])) / abs(dat_ae[index_])
                index_ = index_ + 1

            return total_e
        except:
            return 9999999999

def obj(ww1,ww2,ww3,ww4):
    a=opt(ww1,ww2,ww3,ww4).loss()
    return  1.0/a



pbounds = {'ww1': (0, 0.1), 'ww2': (0, 0.25),'ww3': (0.1, 0.35),'ww4': (0.2, 0.45)}

optimizer = BayesianOptimization(
    f=obj,
    pbounds=pbounds,
    random_state=1,
)
optimizer.maximize(
    init_points=4,
    n_iter=900,
)

