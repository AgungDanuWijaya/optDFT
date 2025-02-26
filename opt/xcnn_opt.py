import torch
from opt.xc_deriv import c_deriv
from pyscf import gto, dft, scf, cc
import numpy as np
import numpy
from pyscf.geomopt.berny_solver import (optimize)
from pyscf.hessian import thermo
from snn_3h import SNN
from data import read_data
import nevergrad as ng
delta_kali=1e-5

scaling_factor0 = 1.0
scaling_factor1 = 1.0
scaling_factor2 = 1.0
scaling_factor3 = 0.001
input_dim = 3
output_dim = 1
depth = 4
lamda = 1e-5
beta = 1.5
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
hidden = [20, 20, 20]
s_nn = SNN(input_dim, output_dim, hidden, lamda, beta, use_cuda)
snn_para = s_nn.state_dict()
s_nn.to(device)


def lyp(rho01, rho02, gamma1, gamma2, gamma12):
    ml_in_ = np.concatenate((rho01.reshape((-1, 1)), rho02.reshape((-1, 1)), gamma1.reshape((-1, 1)),
                             gamma2.reshape((-1, 1)), gamma12.reshape((-1, 1))), axis=1)
    ml_in = torch.Tensor(ml_in_)
    ml_in.requires_grad = False
    exc_ml_out = s_nn(ml_in, is_training_data=False)
    ml_exc = exc_ml_out.detach().numpy()
    exc = (ml_exc).reshape(-1)

    return exc

def eval_xc_gga(xc_code, rho, spin, relativity=0, deriv=2, verbose=None, omega=None):
    rho1 = rho[0]
    rho2 = rho[1]

    a, dx1, dy1, dz1 = rho1[:4]
    b, dx2, dy2, dz2 = rho2[:4]
    gaa = dx1 ** 2 + dy1 ** 2 + dz1 ** 2
    gbb = dx2 ** 2 + dy2 ** 2 + dz2 ** 2
    gnn = (dx1 * dx2) + (dy1 * dy2) + (dz1 * dz2)

    exc = lyp(a, b, gaa, gbb, gnn)

    dx = c_deriv.dxc(a, b, gaa, gbb, gnn,lyp)

    al = 1
    be = 1

    extd = dft.xcfun.eval_xc('b3lyp', rho, spin, relativity, 2,
                             verbose)
    fxc_ = al * np.array(dx[2]) + be * np.array(extd[2][0]).T

    fxc_1 = al * np.array(dx[3]) + be * np.array(extd[2][1]).T

    fxc_2 = al * np.array(dx[4]) + be * np.array(extd[2][2]).T

    vgamma_ = al * np.array(dx[1])

    vgamma_ = vgamma_ + be * np.array(extd[1][1]).T

    vgamma = np.transpose(vgamma_)

    vrho_ = al * np.array(dx[0])
    vrho_ = vrho_ + be * numpy.array(extd[1][0]).T

    vrho = np.transpose(vrho_)

    exc = al * np.array([exc]) / (a + b + 1e-250)

    exc = np.transpose(exc + be * extd[0])
    vxc = (vrho, vgamma, None, None)
    fxc = (np.transpose(fxc_), fxc_1.T, fxc_2.T)

    kxc = None  # 3rd order functional derivative
    return exc, vxc, fxc, kxc


def loss1(weight, is_eval=False):
    try:

        loss = 999
        w = torch.Tensor(weight)
        # print("weight in: ","\n",w)
        k = 0
        for i00 in range(hidden[0]):
            for j in range(input_dim):
                snn_para['model.0.weight'][i00, j] = w[k] * scaling_factor0
                k += 1
        for i01 in range(hidden[0]):
            snn_para['model.0.bias'][i01] = w[k + i01] * scaling_factor0
        k = hidden[0] * (input_dim + 1)
        for i10 in range(hidden[1]):
            for j in range(hidden[0]):
                snn_para['model.2.weight'][i10, j] = w[k] * scaling_factor1
                k += 1
        for i11 in range(hidden[1]):
            snn_para['model.2.bias'][i11] = w[k + i11] * scaling_factor1
        k = hidden[0] * (input_dim + 1) + hidden[1] * (hidden[0] + 1)
        for i20 in range(hidden[2]):
            for j in range(hidden[1]):
                snn_para['model.4.weight'][i20, j] = w[k] * scaling_factor2
                k += 1
        for i21 in range(hidden[2]):
            snn_para['model.4.bias'][i21] = w[k + i21] * scaling_factor2
        k = hidden[0] * (input_dim + 1) + hidden[1] * (hidden[0] + 1) + hidden[2] * (hidden[1] + 1)
        for i3 in range(hidden[2]):
            snn_para['model.6.weight'][0, i3] = w[k + i3] * scaling_factor3
        s_nn.load_state_dict(snn_para)
        # print("weight loaded: ", snn_para)

        indicator = 0
        print("indicator before SCF: ", indicator)
        print("mol calculation starts")

        vf_ = [[1595, 3657, 3756], [4161.2], [999.0, 1992.8, 1995.9], [1986.9], [1183, 2615, 2626], [1556.4], [720]]
        mole = ["H2O", "H2", "SiH2", "BeH", "H2S", "O2", "S2"]
        total = 0
        hggh = [1, 1, 1, 1, 1, 1, 2]

        for kjjj in range(7):

            mol = gto.Mole()
            mol.verbose = 1
            mol.atom = "" + read_data.read_g(mole[kjjj],dir) + ""
            mol.charge = 0
            mol.spin = int(read_data.read_spin(mole[kjjj],dir))
            mol.basis = "cc-pvdz"
            mol.build()

            mfl = dft.UKS(mol)
            mfl.define_xc_(eval_xc_gga, xctype='GGA', hyb=w_[0])
            mfl.xc = "HF,"
            try:
                mol = optimize(mfl, maxsteps=100)
            except:
                print("err")


            mf = mol.UKS()
            mf.define_xc_(eval_xc_gga, xctype='GGA', hyb=w_[0])
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
        dat1 = "H2S,S,H,H}Na2,Na,Na}Si2,Si,Si}P2,P,P}S2,S,S}NaCl,Na,Cl}BeH,H,Be}HCl,H,Cl}HF,H,F}Cl2,Cl,Cl}H2,H,H}LiF,Li,F}LiH,Li,H}CH,C,H}OH,O,H}H2O,O,H,H}O2,O,O}NH,N,H}Li2,Li,Li}CO,C,O}F2,F,F}HCO,H,C,O}H2O2,H,H,O,O}H2CO,H,H,C,O}CH3,C,H,H,H}CH4,C,H,H,H,H}N2,N,N}C2H2,C,C,H,H}CH3OH,O,C,H,H,H,H}NH3,N,H,H,H}HCN,H,C,N}CN,C,N}NH2,N,H,H}C2H6,C,C,H,H,H,H,H,H}N2H4,N,N,H,H,H,H}SiH3,Si,H,H,H}SiH4,Si,H,H,H,H}PH2,P,H,H}PH3,P,H,H,H}CH3SH,C,S,H,H,H,H}SO2,S,O,O}FCl,F,Cl}CH3Cl,C,H,H,H,Cl}AlCl3,Al,Cl,Cl,Cl"

        dat = dat1

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
                    mol = gto.Mole()
                    mol.verbose = 1
                    mol.atom = "" + read_data.read_g(jk__,dir) + ""
                    mol.charge = 0
                    mol.spin = int(read_data.read_spin(jk__,dir))
                    mol.basis = "cc-pvdz"
                    mol.build()
                    mfl = dft.UKS(mol)
                    mfl.define_xc_(eval_xc_gga, xctype='GGA', hyb=w_[0])
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
            # print(Ae, dat_ae[index_], "aahhahssaha")
            index_ = index_ + 1
        total_ae = total_ae
        dat = ("Cl2,Cl2_}P2,P2_}CO,CO_}C2H4,C2H4_}C2H2,C2H2_}"
               "HCl,HCl_}PH3,PH3_}PH2,PH2_}SiH4,SiH4_}HF,HF_}H2O,H2O_}OH,OH_}NH3,NH3_}"
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
                    mol = gto.Mole()
                    mol.verbose = 1
                    mol.atom = "" + read_data.read_g(jk__,dir) + ""
                    mol.charge = 0
                    if "_" in jk__:
                        mol.charge = 1
                    mol.spin = int(read_data.read_spin(jk__,dir))
                    mol.basis = "cc-pvdz"
                    mol.build()
                    kk = 0
                    mfl = dft.UKS(mol)
                    mfl.define_xc_(eval_xc_gga, xctype='GGA', hyb=w_[0])
                    mfl.xc = "HF,"
                    try:
                        mol = optimize(mfl, maxsteps=100)
                    except:
                        print("hai")

                    ina = ina + 1


                    mfl = dft.UKS(mol)

                    mfl.define_xc_(eval_xc_gga, xctype='GGA', hyb=w_[0])
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

        total_ip = total_ip / 27.2114


        print(total, total_ae, total_ip, "aahhahaha", w_)
        loss = total + total_ae + total_ip
        print(loss)
        indicator = 0
        return loss, indicator
    except:
        return 9000000, 1


if __name__ == "__main__":
    w_ = [0.2]
    dir = "/data_/"

    param = ng.p.Array(
        shape=(hidden[0] * (input_dim + 1) + hidden[1] * (hidden[0] + 1) + hidden[2] * (hidden[1] + 2),)).set_bounds(
        -20, 0)

    de_opt = ng.optimization.optimizerlib.ConfiguredPSO(popsize=100, omega=0.9, phip=0.95, phig=0.9)
    de1 = de_opt(param, budget=3000, num_workers=2)

    de1.suggest(np.zeros(hidden[0] * (input_dim + 1) + hidden[1] * (hidden[0] + 1) + hidden[2] * (hidden[1] + 2)))
    indicator_scf = 0
    renda = 100;
    for i in range(3000):
        print(i, "-epoch starts")
        x1 = de1.ask()
        y1, indicator_scf = loss1(*x1.args)
        if (y1 < renda):
            file = open("ww.txt" + str(i), "w")

            file.write(str(*x1.args))
            file.close()
            file = open("ww1.txt" + str(i), "w")

            file.write(str(y1))
            file.close()
            renda = y1
        print("SCF convered? ", not bool(indicator_scf))
        if not indicator_scf:
            de1.tell(x1, y1)
            print(i, "-epoch ends \n")
        else:
            print("\n")

    recommendation = de1.recommend()
    de_best = recommendation.value
    loss_best = loss1(de_best, is_eval=True)
    print("final:", loss_best)
