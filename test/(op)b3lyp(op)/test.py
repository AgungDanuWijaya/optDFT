from pyscf import gto, dft, scf, cc
import b3lyp as b3lyp
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
mfl.define_xc_(b3lyp.eval_xc_lyp_b, xctype='GGA', hyb=b3lyp.w_[0])
mfl.xc = "HF,"
energy=mfl.kernel()
print(energy)