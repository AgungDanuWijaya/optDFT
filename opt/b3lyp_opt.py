from pyscf.hessian import thermo
import numpy
from pyscf import gto, dft, scf, cc
from bayes_opt import BayesianOptimization
from data import read_data
class opt():
    def __init__(self, ww1,ww3,ww5):
        self.ww1=ww1
        self.ww2=1-ww1-ww5
        self.ww3 = ww3
        self.ww4=1-ww3
        self.ww5=ww5

    def eval_xc_gga(self,xc_code, rho, spin, relativity=0, deriv=2, verbose=None, omega=None):

        exc = dft.libxc.eval_xc('' + str("%.16f" % self.ww1) + '*HF+' + str("%.16f" % self.ww2) + '*SLATER+'+ str("%.16f" % self.ww5) +'*B88  , ' + str(
            "%.16f" % self.ww3) + '*LYP + ' + str("%.16f" % self.ww4) + '*VWN3', rho, spin, relativity, 2,
                                   verbose)
        return exc

    def loss(self):
        dir = "/home/agung/PycharmProjects/pkpt/data_/"
        vf_ = [[1595, 3657, 3756], [4161.2], [999.0, 1992.8, 1995.9], [1986.9], [1183, 2615, 2626], [1556.4], [720]]
        mole = ["H2O", "H2", "SiH2", "BeH", "H2S", "O2", "S2"]
        total = 0
        hggh = [1, 1, 1, 1, 1, 1, 2]

        for kjjj in range(7):

            mol = gto.Mole()
            mol.verbose = 0
            mol.atom = "" + read_data.read_g(mole[kjjj], dir) + ""
            mol.charge = 0
            mol.spin = int(read_data.read_spin(mole[kjjj], dir))
            mol.basis = "cc-pvdz"
            mol.build()

            mf = mol.UKS()
            mf.define_xc_(self.eval_xc_gga, xctype='GGA', hyb=self.ww1)
            mf.xc = "HF,"
            mf = mf.run()

            mf.level_shift = 0.5
            mf.grids.level = hggh[kjjj]
            mf.conv_tol = 0.000000001

            hessian = mf.Hessian().kernel()

            freq_info = thermo.harmonic_analysis(mf.mol, hessian)

            vf = vf_[kjjj]
            a = 0
            for jiop in range(len(vf)):
                a = a + numpy.abs((vf[jiop] - freq_info['freq_wavenumber'][jiop]) / vf[jiop])
            total = total + numpy.sum(a) / len(vf_[kjjj])
        total = total / len(vf_)
        dat = "H2S,S,H,H}Na2,Na,Na}Si2,Si,Si}P2,P,P}S2,S,S}NaCl,Na,Cl}BeH,H,Be}HCl,H,Cl}HF,H,F}Cl2,Cl,Cl}H2,H,H}LiF,Li,F}LiH,Li,H}CH,C,H}OH,O,H}H2O,O,H,H}O2,O,O}NH,N,H}Li2,Li,Li}CO,C,O}F2,F,F}HCO,H,C,O}H2O2,H,H,O,O}H2CO,H,H,C,O}CH3,C,H,H,H}CH4,C,H,H,H,H}N2,N,N}C2H2,C,C,H,H}CH3OH,O,C,H,H,H,H}NH3,N,H,H,H}HCN,H,C,N}CN,C,N}NH2,N,H,H}C2H6,C,C,H,H,H,H,H,H}N2H4,N,N,H,H,H,H}SiH3,Si,H,H,H}SiH4,Si,H,H,H,H}PH2,P,H,H}PH3,P,H,H,H}CH3SH,C,S,H,H,H,H}SO2,S,O,O}FCl,F,Cl}CH3Cl,C,H,H,H,Cl}AlCl3,Al,Cl,Cl,Cl"

        dat_ae = [0.27601593625498,
                  0.0270916334661355,
                  0.118565737051793,
                  0.184860557768924,
                  0.160637450199203,
                  0.155219123505976,
                  0.075,
                  0.163,
                  0.215776892430279,
                  0.091,
                  0.165,
                  0.219282868525896,
                  0.089,
                  0.127,
                  0.161,
                  0.349641434262948,
                  0.188,
                  0.126,
                  0.038,
                  0.408286852589641,
                  0.059,
                  0.431394422310757,
                  0.40207171314741,
                  0.569402390438247,
                  0.460557768924303,
                  0.625338645418327,
                  0.358725099601594,
                  0.619442231075697,
                  0.766374501992032,
                  0.440956175298805,
                  0.48207171314741,
                  0.283824701195219,
                  0.271872509960159,
                  1.06151394422311,
                  0.646215139442231,
                  0.338167330677291,
                  0.482231075697211,
                  0.237768924302789,
                  0.36191235059761,
                  0.709322709163347,
                  0.404780876494024,
                  0.0941832669322709,
                  0.591394422310757,
                  0.483505976095618,
                  0.196812749003984,
                  0.229641434262948,
                  0.238884462151394,
                  0.302470119521912,
                  0.271553784860558,
                  0.0941832669322709,
                  0.810358565737052,
                  0.249083665338645,
                  0.733227091633466,
                  0.503107569721115,
                  0.672350597609562,
                  0.741832669322709,
                  0.488764940239044,
                  0.523665338645418,
                  0.6599203187251,
                  0.900557768924303,
                  0.603346613545817,
                  0.420239043824701,
                  0.298167330677291]

        x = dat.split("}")
        mydict = {'asa': 32}
        index_ = 0
        total_ae = 0
        for jk_ in x:

            x_ = jk_.split(",")
            Ae = 0
            ins = 0
            ina = 0
            for jk__ in x_:
                if (0 == 0):
                    mol = gto.Mole()
                    mol.verbose = 0
                    mol.atom = "" + read_data.read_g(jk__, dir) + ""
                    mol.charge = 0
                    mol.spin = int(read_data.read_spin(jk__, dir))
                    mol.basis = "cc-pvdz"
                    mol.build()
                    mfl = dft.UKS(mol)
                    mfl.define_xc_(self.eval_xc_gga, xctype='GGA', hyb=self.ww1)
                    mfl.xc = "HF,"
                    mfl.level_shift = 0.5
                    mfl.grids.level = 0
                    mfl.conv_tol = 0.000000001
                    if jk__ not in mydict:
                        mydict[jk__] = mfl.kernel()
                    a = mydict.get(jk__)
                    if ins == 0:
                        Ae = a
                    elif ins != 0:
                        Ae = Ae - a
                    ins = ins + 1
            total_ae = total_ae + abs((abs(Ae) - dat_ae[index_]))
            index_ = index_ + 1
        total_ae = total_ae
        dat = ("Cl2,Cl2_}P2,P2_}CO,CO_}C2H4,C2H4_}C2H2,C2H2_}"
               "HCl,HCl_}PH3,PH3_}PH2,PH2_}SiH4,SiH4_}HF,HF_}h2o,h2o_}OH,OH_}NH3,NH3_}"
               "He,He_}Li,Li_}Be,Be_}B,B_}C,C_}N,N_}O,O_}F,F_}Ne,Ne_}Na,Na_}Mg,Mg_}Al,Al_}Si,Si_}P,P_}S,S_}Cl,Cl_}Ar,Ar_}"
               "BeH,BeH_}H2O,H2O_}CH3,CH3_}C2H4O,C2H4O_}HCOOH,HCOOH_")
        dat_ae = [11.48, 10.53, 14.01, 10.51, 11.40,
                  12.74, 9.87, 9.82, 11.00, 16.03, 12.62, 13.02, 10.07,
                  24.59, 5.39, 9.32, 8.30, 11.26, 14.53, 13.62, 17.42, 21.56, 5.14, 7.65, 5.99, 8.15, 10.49, 10.36,
                  12.97,
                  15.76, 8.21, 12.62, 9.84, 10.56, 11.33, 12.61
                  ]
        x = dat.split("}")
        mydict = {'asa': 32}
        index_ = 0
        total_ip = 0
        for jk_ in x:
            x_ = jk_.split(",")
            Ae = 0
            ins = 0
            ina = 0
            for jk__ in x_:
                if (0 == 0):
                    mol = gto.Mole()
                    mol.verbose = 0
                    mol.atom = "" + read_data.read_g(jk__, dir) + ""
                    mol.charge = 0
                    if "_" in jk__:
                        mol.charge = 1
                    mol.spin = int(read_data.read_spin(jk__, dir))
                    mol.basis = "cc-pvdz"
                    mol.build()
                    mfl = dft.UKS(mol)
                    mfl.define_xc_(self.eval_xc_gga, xctype='GGA', hyb=self.ww1)
                    mfl.xc = "HF,"
                    mfl.level_shift = 0.5
                    mfl.grids.level = 0
                    mfl.conv_tol = 0.000000001
                    if jk__ not in mydict:
                        mydict[jk__] = mfl.kernel()
                    a = mydict.get(jk__)
                    if ins == 0:
                        Ae = a
                    elif ins != 0:
                        Ae = Ae - a
                    ins = ins + 1
            Ae = Ae * 27.2114
            total_ip = total_ip + abs((abs(Ae) - dat_ae[index_]))
            index_ = index_ + 1
        total_ip = total_ip / (27.2114)
        total = total + total_ae + total_ip
        return total

def obj(ww1,ww3,ww5):
    a=opt(ww1,ww3,ww5).loss()
    return  1.0/a



pbounds = {'ww1': (0.1, 0.3),'ww3': (0.7, 0.9),'ww5': (0.6, 0.8)}

optimizer = BayesianOptimization(
    f=obj,
    pbounds=pbounds,
    random_state=1,
)
optimizer.maximize(
    init_points=3,
    n_iter=900,
)
