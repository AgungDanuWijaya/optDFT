from pyscf import gto, dft, scf, cc

w_ = [0] * 5
w_[0] = 0.17457035788765415
w_[1] = 0.025429642112345774
w_[2] = 0.8803280235993975
w_[3] = 0.11967197640060245
w_[4] = 0.8
def eval_xc_gga(xc_code, rho, spin, relativity=0, deriv=2, verbose=None, omega=None):

    extd =  dft.libxc.eval_xc(''+str("%.16f" %w_[0])+'*HF+'+str("%.16f" %w_[1])+'*SLATER+'+str("%.16f" %w_[4])+'*B88  , '+str("%.16f" %w_[2])+'*LYP + '+str("%.16f" %w_[3])+'*VWN3', rho, spin, relativity, 2,
                               verbose)
    return extd


mol = gto.Mole()
mol.verbose = 4

mol.atom="""8            .000000     .000000     .119262
1            .000000     .763239    -.477047
1            .000000    -.763239    -.477047"""
mol.charge=0
mol.spin  =0
mol.basis = "aug-cc-pvdz"
mol.build()
mfl = dft.UKS(mol)
mfl.define_xc_(eval_xc_gga, xctype='GGA', hyb=w_[0])
mfl.xc = "HF,"
energy=mfl.kernel()
print(energy)