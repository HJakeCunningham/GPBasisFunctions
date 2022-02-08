import numpy as np
import tensorflow as tf
from scipy.sparse import diags
from bisect import bisect
#from tqdm.autonotebook import tqdm


### Cubic B-Spline generation

def B3_params(u, i):
    """
    'u': a knot sequence that is used to define the B-Spline parameters
    'i': is the index of the knot

    Returns: len=4 list of parameters p where p is a tuple of len=4 cubic polynomial parameters and tuple (a,b) of basis function support

    - Included is functionality to ignore division by zeros for repeated B-Splines at the edges
    """

    """First peicewise polynomial"""
    D0 = (u[i+1] - u[i]) * (u[i+2] - u[i]) * (u[i+3] - u[i])
    S0 = (u[i], u[i+1])
    if D0 == 0:
        P0 = ([0, 0, 0, 0], S0)
    else:
        C03 = 1/D0
        C02 = 1/D0 * -3 * u[i]
        C01 = 1/D0 * 3 * u[i]**2
        C00 = 1/D0 * -u[i]**3
        P0 = ([C00, C01, C02, C03], S0)

    """Second peicewise polynomial"""
    D1 = (u[i+2] - u[i]) * (u[i+2] - u[i+1]) * (u[i+3] - u[i]) * (u[i+3] - u[i+1]) * (u[i+4] - u[i+1])
    S1 = (u[i+1], u[i+2])
    if D1 == 0:
        P1 = ([0, 0, 0, 0], S1)
    else:
        C13 =  - 1/D1 * ( ((u[i+3] - u[i+1]) * (u[i+4] - u[i+1])) + ((u[i+2] - u[i]) * (u[i+4] - u[i+1])) + ((u[i+3] - u[i]) * (u[i+2] - u[i])) )
        C12 = 1/D1 * ( (((u[i+3] - u[i+1]) * (u[i+4] - u[i+1])) * (2*u[i] + u[i+2])) + (((u[i+2] - u[i]) * (u[i+4] - u[i+1])) * (u[i] + u[i+1] + u[i+3])) + (((u[i+3] - u[i]) * (u[i+2] - u[i])) * (2*u[i+1] + u[i+4]) ))
        C11 = - 1/D1 * ( (((u[i+3] - u[i+1]) * (u[i+4] - u[i+1])) * (u[i]**2 + 2*u[i]*u[i+2])) + (((u[i+2] - u[i]) * (u[i+4] - u[i+1])) * (u[i]*u[i+1] + u[i]*u[i+3] + u[i+1]*u[i+3])) + (((u[i+3] - u[i]) * (u[i+2] - u[i])) * (u[i+1]**2 + 2*u[i+1]*u[i+4])))
        C10 =  1/D1 * ( ((u[i+3] - u[i+1]) * (u[i+4] - u[i+1]) * u[i]**2 * u[i+2]) + ((u[i+2] - u[i]) * (u[i+4] - u[i+1]) * u[i] * u[i+1] * u[i+3]) + ((u[i+3] - u[i]) * (u[i+2] - u[i]) * u[i+1]**2 * u[i+4]) )
        P1 = ([C10, C11, C12, C13], S1)

    """Third peicewise polynomial"""
    D2 = (u[i+3] - u[i]) * (u[i+3] - u[i+1]) * (u[i+3] - u[i+2]) * (u[i+4] - u[i+1]) * (u[i+4] - u[i+2])
    S2 = (u[i+2], u[i+3])
    if D2 == 0:
        P2 = ([0, 0, 0, 0], S2)
    else:
        C23 = 1/D2 * (((u[i+4] - u[i+1]) * (u[i+4] - u[i+2])) + ((u[i+3] - u[i]) * (u[i+4] - u[i+2])) + ((u[i+3] - u[i]) * (u[i+3] - u[i+1])))
        C22 = -1/D2 * (((u[i+4] - u[i+1]) * (u[i+4] - u[i+2]) * (u[i] + 2*u[i+3])) + ((u[i+3] - u[i]) * (u[i+4] - u[i+2]) * (u[i+1] + u[i+3] + u[i+4])) + ((u[i+3] - u[i]) * (u[i+3] - u[i+1]) * (u[i+2] + 2*u[i+4])))
        C21 = 1/D2 * (((u[i+4] - u[i+1]) * (u[i+4] - u[i+2]) * (u[i+3]**2 + 2*u[i]*u[i+3])) + ((u[i+3] - u[i]) * (u[i+4] - u[i+2]) * (u[i+1]*u[i+4] + u[i+1]*u[i+3] + u[i+3]*u[i+4])) + ((u[i+3] - u[i]) * (u[i+3] - u[i+1]) * (2*u[i+2]*u[i+4] + u[i+4]**2)))
        C20 = -1/D2 * (((u[i+4] - u[i+1]) * (u[i+4] - u[i+2]) * (u[i]*u[i+3]**2)) + ((u[i+3] - u[i]) * (u[i+4] - u[i+2]) * (u[i+1]*u[i+3]*u[i+4])) + ((u[i+3] - u[i]) * (u[i+3] - u[i+1]) * (u[i+2]*u[i+4]**2)))
        P2 = ([C20, C21, C22, C23], S2)

    """Fourth peicewise polynomial"""
    D3 = (u[i+4] - u[i+1]) * (u[i+4] - u[i+2]) * (u[i+4] - u[i+3])
    S3 = (u[i+3], u[i+4])
    if D3 == 0:
        P3 = ([0, 0, 0, 0], S3)
    else:
        C33 = 1/D3 * -1
        C32 = 1/D3 * 3 * u[i+4]
        C31 = 1/D3 * -3 * u[i+4]**2
        C30 = 1/D3 * u[i+4]**3
        P3 = ([C30, C31, C32, C33], S3)

    params = [P0, P1, P2, P3]

    return params

def grad_B3_params(u, i):
    """
    'u': a knot sequence that is used to define the B-Spline parameters
    'i': is the index of the basis

    Returns: list of parameters for the first derivative of the i'th B-Spline
    """
    P = B3_params(u, i)
    P_dash = []
    for i in range(4):
        P_dash.append(([P[i][0][1], 2*P[i][0][2], 3*P[i][0][3], 0], P[i][1]))
    return P_dash

def grad_grad_B3_params(u, i):
    """
    'u': a knot sequence that is used to define the B-Spline parameters
    'i': is the index of the basis

    Returns: list of parameters for the second derivative of the i'th B-Spline
    """
    P = B3_params(u, i)
    P_dash_dash = []
    for i in range(4):
        P_dash_dash.append(([2*P[i][0][2], 6*P[i][0][3], 0, 0], P[i][1]))
    return P_dash_dash 


### Product of cubic B-Splines

def __param_product(p1, p2):
    """
    'p1': a set of cubic polynomial parameters
    'p1': a set of cubic polynomial parameters

    Returns: list of parameters for a 6th order polynomial
    """

    C0 = p1[0]*p2[0]
    C1 = p1[0]*p2[1] + p1[1]*p2[0]
    C2 = p1[0]*p2[2] + p1[1]*p2[1] + p1[2]*p2[0]
    C3 = p1[0]*p2[3] + p1[1]*p2[2] + p1[2]*p2[1] + p1[3]*p2[0]
    C4 = p1[1]*p2[3] + p1[2]*p2[2] + p1[3]*p2[1]
    C5 = p1[2]*p2[3] + p1[3]*p2[2]
    C6 = p1[3]*p2[3]

    return [C0, C1, C2, C3, C4, C5, C6]

def B3_product(knots, i, j):
    """
    'knots': a knot sequence that is used to define the B-Spline parameters
    'i' and 'j': indicies of the basis functions

    Returns: list of parameters for the product of two B-Spline basis functions
    """

    params_1 = B3_params(knots, i)
    params_2 = B3_params(knots, j)

    # idx_1 = np.where(knots == params_1[-1][1][1])[0][0]
    # idx_2 = np.where(knots == params_2[-1][1][1])[0][0]
    # offset = idx_2 - idx_1

    offset = j-i

    if abs(offset) > 3:
        return 0
    else:
        product = []
        for i in range(4 - abs(int(offset))):
            if offset < 0:
                p1 = params_1[i][0]
                p2 = params_2[i-int(offset)][0]
                s = params_1[i][1]
            else:
                p1 = params_1[i + int(offset)][0]
                p2 = params_2[i][0]
                s = params_2[i][1]
            product.append((__param_product(p1, p2), s))

        return product

def grad_B3_product(knots, i, j):
    """
    'knots': a knot sequence that is used to define the B-Spline parameters
    'i' and 'j': indicies of the basis functions

    Returns: list of parameters for the product of the derivative of two B-Spline basis functions
    """
    
    params_1 = grad_B3_params(knots, i)
    params_2 = grad_B3_params(knots, j)
    offset = j-i
    if abs(offset) > 3:
        return 0
    else:
        product = []
        for i in range(4 - abs(int(offset))):
            if offset < 0:
                p1 = params_1[i][0]
                p2 = params_2[i-int(offset)][0]
                s = params_1[i][1]
            else:
                p1 = params_1[i + int(offset)][0]
                p2 = params_2[i][0]
                s = params_2[i][1]
            product.append((__param_product(p1, p2), s))

        return product

### Integration of Cubic B-Splines

def param_integrate(params):
    """
    params: parameters for a cubic B-Spline
    
    Returns: The exact definite integral of a cubic B-Spline over it's support
    """

    if params == 0:
        return 0
    else:
        integral = 0

        for param in params:
            a = param[1][0]
            b = param[1][1]

            i_constants = [1/x for x in range(1, len(param[0])+1)]
            i_param = [0] + [a*b for a,b in zip(i_constants,param[0])]

            a_p = [a**i for i, a in enumerate([a]*len(i_param))]
            b_p = [b**i for i, b in enumerate([b]*len(i_param))]

            d = [b-a for a,b in zip(a_p, b_p)]

            integral += sum(i*d for i,d in zip(i_param, d))

        return integral


### Computing the Gram Matrix

def Matern12_RKHS_IP(knots, i, j, l, sigma):
    """
    'knots': knot sequence that is used to define the B-Spline parameters
    'i' and 'j': indicies of the basis functions
    'l': kernel lengthscale
    'sigma': kernel noise parameter

    Returns: the RKHS inner product between the i'th and j'th Cubic B-Splines
    """

    p = B3_product(knots, i, j)
    p_dash = grad_B3_product(knots, i, j)

    integral = param_integrate(p)
    integral_dash = param_integrate(p_dash)

    term_1 = l/(2*sigma**2) * integral_dash
    term_2 = 1/(2*l*sigma**2) * integral

    return term_1 + term_2


def __convert_sparse_matrix_to_sparse_tensor(X):
    """
    Converts scipy sparse matrix to sparse tensor
    """
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.sparse.reorder(tf.SparseTensor(indices, coo.data, coo.shape))


def Matern12_Kuu(knots, l, sigma):
    """
    'knots': a knot sequence that is used to define the B-Spline parameters
    'l': kernel lengthscale
    'sigma': kernel noise parameter

    Returns: the Gram Matrix for the Matern12 RKHS inner product
    """

    elements = []
    for i in range(7):
        elements.append(Matern12_RKHS_IP(knots, i, 3, l=l, sigma=sigma))
    Kuu_sparse = diags(elements, [-3, -2, -1, 0, 1, 2, 3], shape=(len(knots)-1-3, len(knots)-1-3))
    Kuu_sparse_tensor = __convert_sparse_matrix_to_sparse_tensor(Kuu_sparse)
    return tf.sparse.to_dense(Kuu_sparse_tensor)


def __evaluate_basis_function(knots, B, x, i):
    params = B3_params(knots, B)[i]

    if x>params[1][0] and x<params[1][1]:
        f = 0
        for i, p in enumerate(params[0]):
            f += p * x**i
        return f
    else:
        print('SUPPORTS DO NOT MATCH')


def Matern12_Kuf(knots, X):
    Kuf = np.zeros((len(knots)-4, len(X)))
    for j, x in enumerate(X):
        i0 = bisect(knots, x)
        # i = parameter index, B = Basis function index
        for i, B in enumerate(np.arange(max(0, i0-4), i0)[::-1]):
            if B < 7:
                Kuf[B, j] = __evaluate_basis_function(knots, B, x, i)
    return tf.convert_to_tensor(Kuf)






