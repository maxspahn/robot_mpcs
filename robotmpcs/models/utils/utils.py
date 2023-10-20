import casadi as ca
def diagSX(val, size):
    a = ca.SX(size, size)
    for i in range(size):
        a[i, i] = val[i]
    return a

def extractVariables(z, n, nx, nu, ns):
    q = z[0: n]
    qdot = z[n: nx]
    qddot = z[nx + ns : nx + ns + nu]
    return q, qdot, qddot

def get_velocity(z, n, nx):
    return  z[n: nx]