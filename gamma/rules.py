from numba import jit


@jit(nopython=True, parallel=True)
def rules_gamma_operator(train, test, teta):
    retorno_1 = 1
    retorno_0 = 0
    if abs(test - train) <= teta:
        return retorno_1
    else:
        return retorno_0