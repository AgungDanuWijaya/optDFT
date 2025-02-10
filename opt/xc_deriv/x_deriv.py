from jax import config
config.update("jax_enable_x64", True)
delta_kali=1e-5

def vrho(rho01, gamma1,func_x):
    delta1_ = (rho01) * delta_kali
    ex1 = func_x(rho01, gamma1)
    ex1_ = func_x(rho01 + delta1_, gamma1)
    vrho1 = (ex1_ - ex1) / (delta1_ + 1e-250)
    return vrho1


def vgama(rho01, gamma1,func_x):
    delta1 = (gamma1) * delta_kali
    ex1 = func_x(rho01, gamma1)
    ex1_ = func_x(rho01, gamma1 + delta1)
    vrho1 = (ex1_ - ex1) / (delta1 + 1e-250)
    return vrho1


def dxc(rho01, gamma1,func_x):
    vgamma_1 = vgama(rho01, gamma1,func_x)

    delta1_ = (rho01) * delta_kali
    vrho1 = vrho(rho01, gamma1,func_x)
    vrho1_ = vrho(rho01 + delta1_, gamma1,func_x)
    vrho2 = (vrho1_ - vrho1) / (delta1_)

    delta1 = (gamma1) * delta_kali
    vgama1 = vgama(rho01, gamma1,func_x)
    vgama1_ = vgama(rho01, gamma1 + delta1,func_x)
    vtautau2 = (vgama1_ - vgama1) / (delta1)

    vgama1 = vgama(rho01, gamma1,func_x)
    vgama1_ = vgama(rho01 + delta1_, gamma1,func_x)
    vrhotau2 = (vgama1_ - vgama1) / (delta1_)
    dxc = [vrho1, vgamma_1, vrho2, vrhotau2, vtautau2]
    return dxc