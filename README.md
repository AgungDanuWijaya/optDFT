{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNyBi/JpHllcC7O4qMhSirR",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/AgungDanuWijaya/optDFT/blob/main/opt_xchange.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "JJvycBKDMlZz"
      },
      "outputs": [],
      "source": [
        "pip install pyscf==2.4.0"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pip install numpy==1.26.4"
      ],
      "metadata": {
        "id": "3wlR4UpwMmai"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "pip install scipy==1.15.1"
      ],
      "metadata": {
        "id": "yhqniyjEM4Hq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!git clone https://github.com/AgungDanuWijaya/optDFT.git"
      ],
      "metadata": {
        "id": "XTC34dCCMoYs"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cd /content/optDFT"
      ],
      "metadata": {
        "id": "zbisQ4lDHyPX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from pyscf import gto\n",
        "import math\n",
        "import numpy as np\n",
        "import numpy\n",
        "from pyscf import gto, dft, scf, cc\n",
        "from opt.xc_deriv import x_deriv\n",
        "from data_ import read_data\n",
        "from scipy.optimize import minimize\n",
        "\n",
        "class opt():\n",
        "    def __init__(self,w_):\n",
        "        self.w_=w_\n",
        "    def b88(self,rho01, gama):\n",
        "        tau1 = gama ** 0.5\n",
        "        x = tau1 / (rho01 + 10E-20) ** (4.0 / 3.0)\n",
        "        b = self.w_[0]\n",
        "        b88_g = -1.5 * (3.0 / 4.0 / math.pi) ** (1.0 / 3.0) - b * (x ** 2) / (1.0 + 6.0 * b * x * np.arcsinh(x))\n",
        "        exc1 = rho01 ** (4.0 / 3.0) * b88_g\n",
        "        return exc1\n",
        "    def eval_xc_gga(self,xc_code, rho, spin, relativity=0, deriv=2, verbose=None, omega=None):\n",
        "        rho1 = rho[0]\n",
        "        rho2 = rho[1]\n",
        "        rho01, dx1, dy1, dz1 = rho1[:4]\n",
        "        rho02, dx2, dy2, dz2 = rho2[:4]\n",
        "        rho01 = rho01 + 1E-250\n",
        "        rho02 = rho02 + 1E-250\n",
        "        w1 = rho01 / (rho01 + rho02)\n",
        "        w2 = rho02 / (rho01 + rho02)\n",
        "        gamma1 = dx1 ** 2 + dy1 ** 2 + dz1 ** 2\n",
        "        gamma2 = dx2 ** 2 + dy2 ** 2 + dz2 ** 2\n",
        "        vgamma_3 = [0] * len(rho01)\n",
        "        v2rho23 = [0] * len(rho02)\n",
        "        ex1 = self.b88(rho01, gamma1)\n",
        "        ex2 = self.b88(rho02, gamma2)\n",
        "        exc1 = (ex1 / rho01) * w1\n",
        "        exc2 = (ex2 / rho02) * w2\n",
        "        vrho1, vgamma_1, v2rho21, v2rhotau1, v2tau21 = x_deriv.dxc((rho01), (gamma1),self.b88)\n",
        "        vrho2, vgamma_2, v2rho22, v2rhotau2, v2tau22 =x_deriv.dxc((rho02), (gamma2),self.b88)\n",
        "        pbe_xc = dft.libxc.eval_xc(',0*lyp', rho, spin, relativity, 2,\n",
        "                                   verbose)\n",
        "        fxc_ = np.array([v2rho21, v2rho23, v2rho22]) + np.array(pbe_xc[2][0]).T\n",
        "        kll = np.array(pbe_xc[2][1]).T\n",
        "        fxc_1 = np.array([v2rhotau1, kll[1] * 0, kll[2] * 0, kll[3] * 0, kll[4] * 0, v2rhotau2]) + np.array(\n",
        "            pbe_xc[2][1]).T\n",
        "        kll = np.array(pbe_xc[2][2]).T\n",
        "        fxc_2 = np.array([v2tau21, kll[1] * 0, kll[2] * 0, kll[3] * 0, kll[4] * 0, v2tau22]) + np.array(pbe_xc[2][2]).T\n",
        "        vgamma_ = np.array([vgamma_1, vgamma_3, vgamma_2])\n",
        "        vgamma_ = vgamma_ + np.array(pbe_xc[1][1]).T\n",
        "        vgamma = np.transpose(vgamma_)\n",
        "        vrho_ = np.array([vrho1, vrho2])\n",
        "        vrho_ = vrho_ + numpy.array(pbe_xc[1][0]).T\n",
        "        vrho = np.transpose(vrho_)\n",
        "        exc1 = np.array([exc1])\n",
        "        exc2 = np.array([exc2])\n",
        "        exc = np.transpose(exc1 + exc2 + pbe_xc[0])\n",
        "        vxc = (vrho, vgamma, None, None)\n",
        "        fxc = (np.transpose(fxc_), fxc_1.T, fxc_2.T)  # 2nd order functional derivative\n",
        "        kxc = None  # 3rd order functional derivative\n",
        "        return exc, vxc, fxc, kxc\n",
        "    def loss(self):\n",
        "            dat = \"He}Li}Be}B}C}N}O}F}Ne}Na}Mg}Al}Ar}Si}P}S}Cl}H2O}HCOOH}LiH}SiH2}Si2H6\"\n",
        "            dat_ae = [-1.02145669820999,\n",
        "                      -1.78111953087113,\n",
        "                      -2.66640566799201,\n",
        "                      -3.76611205989024,\n",
        "                      -5.07053845883686,\n",
        "                      -6.59792425003736,\n",
        "                      -8.20316397641989,\n",
        "                      -10.0253294216753,\n",
        "                      -12.0846958335987,\n",
        "                      -14.0163243748449,\n",
        "                      -15.9929476193147,\n",
        "                      -18.0881080898872,\n",
        "                      -30.1808305185527,\n",
        "                      -20.299270307024,\n",
        "                      -22.6370628176094,\n",
        "                      -25.0251068988656,\n",
        "                      -27.5361776979597,\n",
        "                      -8.93596920882512,\n",
        "                      -22.3478583716952,\n",
        "                      -2.14010001310563,\n",
        "                      -21.003016360435,\n",
        "                      -42.894484671987\n",
        "                      ]\n",
        "\n",
        "            x = dat.split(\"}\")\n",
        "            index_ = 0\n",
        "            total_e = 0\n",
        "            dir = \"/content/optDFT/data_/\"\n",
        "\n",
        "            for jk_ in x:\n",
        "                mol = gto.Mole()\n",
        "                mol.verbose = 0\n",
        "                mol.atom = \"\" + read_data.read_g(jk_,dir) + \"\"\n",
        "                mol.charge = 0\n",
        "                mol.spin = int(read_data.read_spin(jk_,dir))\n",
        "                mol.basis = \"aug-cc-pvdz\"\n",
        "                mol.build()\n",
        "                mfl = dft.UKS(mol)\n",
        "                mfl.define_xc_(self.eval_xc_gga, xctype='GGA')\n",
        "                en = mfl.kernel()\n",
        "                total_e = total_e + abs(abs(mfl.scf_summary.get('exc')) - abs(dat_ae[index_])) / abs(dat_ae[index_])\n",
        "                index_ = index_ + 1\n",
        "            print(\"erorr\",total_e,\"param w\",self.w_,\"==============================\")\n",
        "            return total_e\n",
        "\n",
        "\n",
        "\n",
        "def obj(w_):\n",
        "   return  opt(w_).loss()\n",
        "\n",
        "\n",
        "\n",
        "wi_=[0]*1\n",
        "wi_[0]=0.08\n",
        "res = minimize(obj,wi_,method='nelder-mead')\n",
        "print(res)\n",
        "\n",
        "\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "DQFScbSIMqyL"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
