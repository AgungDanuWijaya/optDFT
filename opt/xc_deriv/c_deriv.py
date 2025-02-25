from jax import config
config.update("jax_enable_x64", True)
delta_kali=1e-5

def vrho_1(a, b, gaa, gbb, gnn,func_corr):
    delta1 = a * delta_kali
    exc = func_corr(a, b, gaa, gbb, gnn)
    exc1 = (func_corr(a + delta1, b, gaa, gbb, gnn))
    vrho1 = (exc1 - exc) / (delta1 + 1e-250)
    return vrho1


def vrho_2(a, b, gaa, gbb, gnn,func_corr):
    delta2 = b * delta_kali
    exc = func_corr(a, b, gaa, gbb, gnn)
    exc2 = func_corr(a, b + delta2, gaa, gbb, gnn)
    vrho2 = (exc2 - exc) / (delta2 + 1e-250)
    return vrho2


def vgama_1(a, b, gaa, gbb, gnn,func_corr):
    delta1 = (gaa) * delta_kali
    exc = func_corr(a, b, gaa, gbb, gnn)
    exc1g = func_corr(a, b, gaa + delta1, gbb, gnn)
    vgama1 = (exc1g - exc) / (delta1 + 1e-250)
    return vgama1


def vgama_2(a, b, gaa, gbb, gnn,func_corr):
    delta2 = (gbb) * delta_kali
    exc = func_corr(a, b, gaa, gbb, gnn)
    exc2g = func_corr(a, b, gaa, gbb + delta2, gnn)
    vgama2 = (exc2g - exc) / (delta2 + 1e-250)
    return vgama2


def vgama_3(a, b, gaa, gbb, gnn,func_corr):
    delta3 = (gnn) * delta_kali
    exc = func_corr(a, b, gaa, gbb, gnn)
    exc3g = func_corr(a, b, gaa, gbb, gnn + delta3)
    vgama3 = (exc3g - exc) / (delta3 + 1e-250)
    return vgama3


def dxc(a, b, gaa, gbb, gnn,func_corr):
    vrho1 = vrho_1(a, b, gaa, gbb, gnn,func_corr)
    vrho2 = vrho_2(a, b, gaa, gbb, gnn,func_corr)

    vrhoc = [vrho1, vrho2]

    vgama1 = vgama_1(a, b, gaa, gbb, gnn,func_corr)
    vgama2 = vgama_2(a, b, gaa, gbb, gnn,func_corr)
    vgama3 = vgama_3(a, b, gaa, gbb, gnn,func_corr)

    vgamac = [vgama1, vgama3, vgama2]

    delta1_ = (a) * delta_kali
    vrho1_ = vrho_1(a + delta1_, b, gaa, gbb, gnn,func_corr)
    v2rho1 = (vrho1_ - vrho1) / (delta1_+ 1e-250)

    delta2_ = (b) * delta_kali
    vrho2_ = vrho_1(a, b + delta2_, gaa, gbb, gnn,func_corr)
    v2rho21 = (vrho2_ - vrho1) / (delta2_+ 1e-250)

    delta2_ = (b) * delta_kali
    vrho2_ = vrho_2(a, b + delta2_, gaa, gbb, gnn,func_corr)
    v2rho2 = (vrho2_ - vrho2) / (delta2_+ 1e-250)

    v2rhoc = [v2rho1, v2rho21, v2rho2]

    delta1 = (gaa) * delta_kali
    vgama1_ = vgama_1(a, b, gaa + delta1, gbb, gnn,func_corr)
    vtautau11 = (vgama1_ - vgama1) / (delta1+ 1e-250)

    delta1 = (gaa) * delta_kali
    vgama2_ = vgama_2(a, b, gaa + delta1, gbb, gnn,func_corr)
    vtautau12 = (vgama2_ - vgama2) / (delta1+ 1e-250)

    delta1 = (gaa) * delta_kali
    vgama3_ = vgama_3(a, b, gaa + delta1, gbb, gnn,func_corr)
    vtautau13 = (vgama3_ - vgama3) / (delta1+ 1e-250)

    delta2 = (gbb) * delta_kali
    vgama2_ = vgama_2(a, b, gaa, gbb + delta2, gnn,func_corr)
    vtautau22 = (vgama2_ - vgama2) / (delta2+ 1e-250)

    delta2 = (gbb) * delta_kali
    vgama3_ = vgama_3(a, b, gaa, gbb + delta2, gnn,func_corr)
    vtautau23 = (vgama3_ - vgama3) / (delta2+ 1e-250)

    delta3 = (gnn) * delta_kali
    vgama3_ = vgama_3(a, b, gaa, gbb, gnn + delta3,func_corr)
    vtautau33 = (vgama3_ - vgama3) / (delta3+ 1e-250)

    vtautauc = [vtautau11, vtautau12, vtautau13, vtautau22, vtautau23, vtautau33]

    # =================
    delta1 = (a) * delta_kali
    vgama1_ = vgama_1(a + delta1, b, gaa, gbb, gnn,func_corr)
    vrhotau11 = (vgama1_ - vgama1) / (delta1+ 1e-250)

    delta1 = (a) * delta_kali
    vgama2_ = vgama_2(a + delta1, b, gaa, gbb, gnn,func_corr)
    vrhotau12 = (vgama2_ - vgama2) / (delta1+ 1e-250)

    delta1 = (a) * delta_kali
    vgama3_ = vgama_3(a + delta1, b, gaa, gbb, gnn,func_corr)
    vrhotau13 = (vgama3_ - vgama3) / (delta1+ 1e-250)

    delta2 = (b) * delta_kali
    vgama1_ = vgama_1(a, b + delta2, gaa, gbb, gnn,func_corr)
    vrhotau21 = (vgama1_ - vgama1) / (delta2+ 1e-250)

    delta2 = (b) * delta_kali
    vgama2_ = vgama_2(a, b + delta2, gaa, gbb, gnn,func_corr)
    vrhotau22 = (vgama2_ - vgama2) / (delta2+ 1e-250)

    delta2 = (b) * delta_kali
    vgama3_ = vgama_3(a, b + delta2, gaa, gbb, gnn,func_corr)
    vrhotau23 = (vgama3_ - vgama3) / (delta2+ 1e-250)

    vrhotauc = [vrhotau11, vrhotau12, vrhotau13, vrhotau21, vrhotau22, vrhotau23]

    dxc = [vrhoc, vgamac, v2rhoc, vrhotauc, vtautauc]
    return dxc


