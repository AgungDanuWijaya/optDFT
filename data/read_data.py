def read_g(mol,dir):
    f = open(dir+mol+"_g","r")
    a=f.read()
    f.close()
    return a
def read_spin(mol,dir):
    f = open(dir+mol+"_spin","r")
    a=f.read()
    f.close()
    return a