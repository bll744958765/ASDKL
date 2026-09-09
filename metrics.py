from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import numpy as np

def RMSE(y,yhat):
    
    return np.sqrt(

        mean_squared_error(
            y,
            yhat
        )
    )

def MAE(y,yhat):
    
    return mean_absolute_error(
        y,
        yhat
    )


def R2(y,yhat):
    
    return r2_score(
        y,
        yhat
    )


def NLL(

        y,

        mean,

        var):

    return np.mean(

        0.5*np.log(
            2*np.pi*var
        )

        +

        (y-mean)**2
        /
        (2*var)
    )


def Coverage95(

        y,

        mean,

        var):

    std = np.sqrt(var)

    lower = (

        mean
        -
        1.96*std
    )

    upper = (

        mean
        +
        1.96*std
    )

    return np.mean(

        (y>=lower)
        &
        (y<=upper)

    )

##Mean Prediction Interval Width

def MPIW(var):
    
    return np.mean(

        3.92*np.sqrt(var)

    )

## Moran's I
from sklearn.neighbors import KDTree
def MoranI(

        coordinates,

        residuals,

        k=8):

    tree = KDTree(
        coordinates
    )

    _, idx = tree.query(
        coordinates,
        k=k+1
    )

    idx = idx[:,1:]

    N = len(residuals)

    z = residuals - residuals.mean()

    numerator = 0
    denominator = (z**2).sum()

    W = 0

    for i in range(N):

        for j in idx[i]:

            numerator += (
                    z[i]
                    *
                    z[j]
            )

            W += 1

    I = (

        N
        *
        numerator

        /

        (
                W
                *
                denominator
        )
    )

    return I