#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Thermochemistry analysis based on nuclear Hessian.
'''
from pyscf.geomopt.berny_solver import (optimize)
import mysql.connector
from pyscf import gto
from pyscf.hessian import thermo
import math
import numpy as np
import numpy
from pyscf import gto, dft, scf, cc
import jax
from opt.xc_deriv import x_deriv
from pyscf.prop.freq import uks
from data import read_data


from scipy.optimize import minimize

class opt():
    def __init__(self,w_):
        self.w_=w_
    def b88(self,rho01, gama):
        tau1 = gama ** 0.5
        x = tau1 / (rho01 + 10E-20) ** (4.0 / 3.0)
        b = self.w_[0]
        b88_g = -1.5 * (3.0 / 4.0 / math.pi) ** (1.0 / 3.0) - b * (x ** 2) / (1.0 + 6.0 * b * x * np.arcsinh(x))
        exc1 = rho01 ** (4.0 / 3.0) * b88_g
        return exc1
    def eval_xc_gga(self,xc_code, rho, spin, relativity=0, deriv=2, verbose=None, omega=None):
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
        ex1 = self.b88(rho01, gamma1)
        ex2 = self.b88(rho02, gamma2)
        exc1 = (ex1 / rho01) * w1
        exc2 = (ex2 / rho02) * w2
        vrho1, vgamma_1, v2rho21, v2rhotau1, v2tau21 = x_deriv.dxc((rho01), (gamma1),self.b88)
        vrho2, vgamma_2, v2rho22, v2rhotau2, v2tau22 =x_deriv.dxc((rho02), (gamma2),self.b88)
        pbe_xc = dft.libxc.eval_xc(',0*lyp', rho, spin, relativity, 2,
                                   verbose)
        fxc_ = np.array([v2rho21, v2rho23, v2rho22]) + np.array(pbe_xc[2][0]).T
        kll = np.array(pbe_xc[2][1]).T
        fxc_1 = np.array([v2rhotau1, kll[1] * 0, kll[2] * 0, kll[3] * 0, kll[4] * 0, v2rhotau2]) + np.array(
            pbe_xc[2][1]).T
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
        kxc = None  # 3rd order functional derivative
        return exc, vxc, fxc, kxc
    def loss(self):
            dat = "He}Li}Be}B}C}N}O}F}Ne}Na}Mg}Al}Ar}Si}P}S}Cl}H2O}HCOOH}LiH}SiH2}Si2H6"
            dat_ae = [-1.02145669820999,
                      -1.78111953087113,
                      -2.66640566799201,
                      -3.76611205989024,
                      -5.07053845883686,
                      -6.59792425003736,
                      -8.20316397641989,
                      -10.0253294216753,
                      -12.0846958335987,
                      -14.0163243748449,
                      -15.9929476193147,
                      -18.0881080898872,
                      -30.1808305185527,
                      -20.299270307024,
                      -22.6370628176094,
                      -25.0251068988656,
                      -27.5361776979597,
                      -8.93596920882512,
                      -22.3478583716952,
                      -2.14010001310563,
                      -21.003016360435,
                      -42.894484671987
                      ]

            x = dat.split("}")
            index_ = 0
            total_e = 0
            dir = "/home/agung/PycharmProjects/pkpt/data/"

            for jk_ in x:
                mol = gto.Mole()
                mol.verbose = 0
                mol.atom = "" + read_data.read_g(jk_,dir) + ""
                mol.charge = 0
                mol.spin = int(read_data.read_spin(jk_,dir))
                mol.basis = "aug-cc-pvdz"
                mol.build()
                mfl = dft.UKS(mol)
                mfl.define_xc_(self.eval_xc_gga, xctype='GGA')
                en = mfl.kernel()
                total_e = total_e + abs(abs(mfl.scf_summary.get('exc')) - abs(dat_ae[index_])) / abs(dat_ae[index_])
                index_ = index_ + 1
            print(total_e,self.w_)
            return total_e



def obj(w_):
   return  opt(w_).loss()



wi_=[0]*1
wi_[0]=0.08
res = minimize(obj,wi_,method='nelder-mead')
print(res)




