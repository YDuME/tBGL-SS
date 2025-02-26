import numpy as np
import itertools
import operator
import math
import copy

from numpy.linalg import norm as Norm
from numpy.linalg import solve as Solve
from scipy.linalg import block_diag
from parametric_pde_find import *

import matplotlib.pyplot as plt
import seaborn as sns

def movingaverage(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

def groupError(coef, m):
    variance = np.var(coef, ddof=1, axis=1).reshape(m,-1).T
    mean = np.mean(coef, axis=1).reshape(m,-1).T
    rate = np.sum(variance, axis=1)/np.sum(mean**2, axis=1)
    return rate

def totalError(coef, m):
    return np.nansum(groupError(coef, m))

def bootstrapMedian(s, alpha = 0.05, rep = 5000, show_hist = False, bins = 20, kde = False):
    n = len(s)
    sb = np.random.choice(s, (n, rep), replace = True)
    mb = np.median(sb, axis = 0)
    mb.sort()
    if show_hist:
        sns.distplot(mb, bins=bins, kde=kde)
    return np.percentile(mb, [alpha*100/2, 100-alpha*100/2]), np.mean(mb)

def getCI(coef, m, alpha = 0.05, rep = 5000):
    lower = []
    upper = []
    means = []
    for i in range(coef.shape[0]):
        ci, mean = bootstrapMedian(coef[i, :], alpha = alpha, rep = rep)
        lower.append(ci[0])
        upper.append(ci[1])
        means.append(mean)
    lower = np.array(lower).reshape(m,-1).T
    upper = np.array(upper).reshape(m,-1).T
    return (lower, upper), means

def getMSEs(X, Remain, xi_true):
    MSE = []
    d, m = xi_true.shape
    for i, xi in enumerate(X):
        remain = Remain[i]
        xi_fullsize = np.zeros(xi_true.shape)

        for j in range(xi.shape[0]): 
            if np.linalg.norm(xi[j,:]) > 10**-10:
                xi_fullsize[remain[j], :] = xi[j,:]
        MSE.append(Norm(xi_fullsize-xi_true)**2/(m*d))
    return MSE

def TrainBGLSS_findt1(As, bs, niter = 500, burnin = 0, a = 1, 
    b = 1, num_update = 10, niter_update = 50, verbose = True, 
    alpha = 0.1, gamma = 0.1, pi_prior = True, pi = 0.5, update_tau = True,
    lambda2_update = np.array([]), normalize = 0, loss_normalize = 2,
    num_threshold = 21, max_threshold = 0.1, min_threshold = 0):

    np.random.seed(0) # for consistancy

    m = len(As)
    n,D = As[0].shape
    
    X = []
    Losses = []
    Lam = []
    Coef = []
    Threshold = []
    TE = []
    Remain = []

    Threshold = [t for t in np.linspace(min_threshold, max_threshold, num_threshold)]
    print(Threshold)

    invalid = []

    for t in Threshold:
        if verbose:
            print("threshold:")
            print(t)
        x, lam, loss, path, coef, remain = thresholdBGLSS(As, bs, threshold = t, niter = niter, burnin = burnin, a = a, 
            b = b, num_update = num_update, niter_update = niter_update, verbose = verbose, 
            alpha = alpha, gamma = gamma, pi_prior = pi_prior, pi = pi, update_tau = update_tau,
            normalize = normalize, loss_normalize = loss_normalize)

        if len(remain) != 0:
            X.append(x)
            Lam.append(lam)
            Losses.append(loss)
            Coef.append(coef)
            TE.append(totalError(coef, m))
            Remain.append(remain)
            if verbose:
                print("totalError:")
                print(TE[-1])
                print()
        else:
            invalid.append(t)

    for t in invalid:
        Threshold.remove(t)
            
    return X, Lam, Losses, Coef, Remain, TE, Threshold

def TrainBGLSS_findt2(As, bs, niter = 500, burnin = 0, a = 1, 
    b = 1, num_update = 10, niter_update = 50, verbose = True, 
    alpha = 0.1, gamma = 0.1, pi_prior = True, pi = 0.5, update_tau = True,
    lambda2_update = np.array([]), normalize = 0, loss_normalize = 2, threshold1 = 0.5,
    num_threshold = 21, max_threshold = 2.5, min_threshold = 0.5):

    np.random.seed(0) # for consistancy

    m = len(As)
    n,D = As[0].shape
    
    X = []
    Losses = []
    Lam = []
    Coef = []
    Threshold = []
    TE = []
    Remain = []

    Threshold = [t for t in np.linspace(min_threshold, max_threshold, num_threshold)]
    print(Threshold)

    invalid = []

    for t in Threshold:
        if verbose:
            print("threshold:")
            print(t)
        x, lam, loss, path, coef, remain = thresholdBGLSS_combined(As, bs, threshold1 = threshold1, threshold2 = t, niter = niter, burnin = burnin, a = a, 
            b = b, num_update = num_update, niter_update = niter_update, verbose = verbose, 
            alpha = alpha, gamma = gamma, pi_prior = pi_prior, pi = pi, update_tau = update_tau,
            normalize = normalize, loss_normalize = loss_normalize)

        if len(remain) != 0:
            X.append(x)
            Lam.append(lam)
            Losses.append(loss)
            Coef.append(coef)
            TE.append(totalError(coef, m))
            Remain.append(remain)
            if verbose:
                print("totalError:")
                print(TE[-1])
                print()
        else:
            invalid.append(t)

    for t in invalid:
        Threshold.remove(t)
            
    return X, Lam, Losses, Coef, Remain, TE, Threshold


def TrainBGLSS(As, bs, niter = 5000, burnin = 1000, a = 1, 
    b = 1, num_update = 20, niter_update = 100, verbose = True, 
    alpha = 0.1, gamma = 0.1, pi_prior = True, pi = 0.5, update_tau = True,
    lambda2_update = np.array([]), normalize = 0, loss_normalize = 2):

    np.random.seed(0) # for consistancy

    m = len(As)
    n,D = As[0].shape
    
    # Normalize
    if normalize != 0:

        # get norm of each column
        candidate_norms = np.zeros(D)
        for i in range(D):
            candidate_norms[i] = Norm(np.vstack(A[:,i] for A in As), normalize)

        norm_bs = [m*Norm(b, normalize) for b in bs]

        # normalize 
        for i in range(m):
            As[i] = As[i].dot(np.diag(candidate_norms**-1))
            bs[i] = bs[i]/norm_bs[i]
        print('Normalized!')
    
    loss = 0
    coef = np.array([])

    x, lambda2, lambda2_path, coef = BGLSS(As, bs, niter = niter, burnin = burnin, a = a, 
        b = b, num_update = num_update, niter_update = niter_update, verbose = verbose, 
        alpha = alpha, gamma = gamma, pi_prior = pi_prior, pi = pi, update_tau = update_tau, 
        lambda2_update = lambda2_update)
    
    if loss_normalize != 0:
        candidate_norms = np.zeros(D)
        for i in range(D):
            candidate_norms[i] = Norm(np.vstack(A[:,i] for A in As), loss_normalize)
        norm_bs = [m*Norm(b, loss_normalize) for b in bs]

        Asn = []
        bsn = []
        xn = np.zeros(x.shape)
        for i in range(m):
            Asn.append(As[i].dot(np.diag(candidate_norms**-1)))
            bsn.append(bs[i]/norm_bs[i])
        for i in range(D):
            for j in range(m):
                xn[i,j] = x[i,j]*candidate_norms[i]/norm_bs[j]
        loss = PDE_FIND_Loss(Asn, bsn, xn)
    else:
        loss = PDE_FIND_Loss(As, bs, x)

    if normalize != 0:
        for i in range(D):
            for j in range(m):
                x[i,j] = x[i,j]/candidate_norms[i]*norm_bs[j]
        for i in range(m):
            As[i] = As[i].dot(np.diag(candidate_norms))
            bs[i] = bs[i]*norm_bs[i]
            
    return x, lambda2, loss, lambda2_path, coef


def BGLSS(Xs, ys, niter = 5000, burnin = 1000, a = 1, 
    b = 1, num_update = 20, niter_update = 100, verbose = True, 
    alpha = 0.1, gamma = 0.1, pi_prior = True, pi = 0.5, update_tau = True, 
    lambda2_update = np.array([])):

    
    if len(Xs) != len(ys): raise Exception('Number of Xs and ys mismatch')
    if len(set([X.shape[1] for X in Xs])) != 1: 
        raise Exception('Number of coefficients inconsistent across timesteps')
        
    d = Xs[0].shape[1]
    n = Xs[0].shape[0]
    m = len(Xs)
    
    N = m*n
    P = m*d

    tau2 = np.ones(d)
    sigma2 = 1
    l = np.zeros(d)
    beta = np.zeros((d, m))
    Z = np.zeros(d)

    lambda2_path = np.array([])
    print(lambda2_update.size)
    print(d)
    if (lambda2_update.size != d):
        fit_for_lambda2 = BGLSS_EM_lambda(Xs, ys, num_update = num_update, 
            niter = niter_update, verbose = verbose)
        lambda2 = fit_for_lambda2[-1, :]
        lambda2_path = fit_for_lambda2[:, 0]
    else:
        lambda2 = lambda2_update
        lambda2_path = np.array([lambda2_update[0]])
        print(lambda2)
    
    
    YtY = 0
    for i in range(m):
        YtY += np.dot(ys[i].T, ys[i])
    XtY = np.vstack([np.dot(Xs[i].T, ys[i]) for i in range(m)])
    XtX = [np.dot(Xs[i].T, Xs[i]) for i in range(m)]

    XktY = [np.vstack([np.dot(Xs[i][:,j], ys[i]) for i in range(m)]) for j in range(d)]
    XktXk = [[np.dot(Xs[i][:,j], Xs[i][:,j]) for i in range(m)] for j in range(d)]
    XktXmk = [[np.dot(Xs[i][:,j], np.delete(Xs[i], j, axis = 1)) for i in range(m)] for j in range(d)]
    
    coef = np.array([]).reshape(m*d, 0)
    coef_tau = []

    for itera in range(niter):
        if (verbose):
            print(itera)
        for i in range(d):
            bmk = np.delete(beta, i, axis = 0)
            f1 = XktY[i] - np.vstack([np.dot(XktXmk[i][j], bmk[:, j]) for j in range(m)])
            f2 = np.array([XktXk[i][j] + 1/tau2[i] for j in range(m)])
            f2_inverse = np.array([1/(XktXk[i][j] + 1/tau2[i]) for j in range(m)])
            mu = np.multiply(f2_inverse, f1.T[0])

            maxf = np.max(f2)
            trythis = (-m/2) * math.log(tau2[i]) + (-1/2) * np.sum(np.log(f2/maxf)) + (-m/2)*math.log(maxf) + np.dot(f1.T[0], mu)/(2 * sigma2)
            if trythis < -50:
                l[i] = 1
            elif trythis > 50:
                l[i] = 0
            else:
                l[i] = pi/(pi + (1-pi)*math.exp(trythis))

            if np.random.uniform() < l[i]:
                beta[i, :] = np.zeros(m)
                Z[i] = 0
            else:
                beta[i, :] = np.random.multivariate_normal(mu, sigma2*np.diag(f2_inverse))
                Z[i] = 1
        if (update_tau):
            for i in range(d):
                if Z[i] == 0:
                    tau2[i] = np.random.gamma((m+1)/2, scale = 2/lambda2[i])
                else:
                    tau2[i] = 1/np.random.wald(math.sqrt(lambda2[i]*sigma2/np.sum(beta[i,:]**2)), lambda2[i])

        s = 0
        for i in range(d):
            s += np.sum(beta[i,:]**2)/tau2[i]
        beta_vec = beta.T.reshape(m*d, 1)
        if itera > burnin:
            coef = np.hstack([coef, beta_vec])
        
        gamma_shape = (N-1)/2+np.sum(Z*m)/2+alpha
        btXtXb = 0
        for i in range(m):
            Xb = np.dot(Xs[i], beta[:,i].reshape(d,1))
            btXtXb += np.dot(Xb.T, Xb)
        gamma_scale = (YtY - 2*np.dot(beta_vec.T, XtY)+btXtXb+s)/2 + gamma
        sigma2 = 1/np.random.gamma(gamma_shape, scale = 1/gamma_scale)
        if (pi_prior):
            pi = np.random.beta(a+d-np.sum(Z),b+np.sum(Z))

    pos_mean = np.mean(coef, axis=1)
    pos_median = np.median(coef, axis=1)
    W = pos_median.reshape(m,d).T
                    
    return W, lambda2, lambda2_path, coef

def BGLSS_EM_lambda(Xs, ys, num_update = 20, niter = 100, a = 1, 
    b = 1, verbose = False, delta = 0.001, alpha = 0.1, gamma = 0.1, 
    pi_prior = True, pi = 0.5):

    d = Xs[0].shape[1]
    n = Xs[0].shape[0]
    m = len(Xs)
    
    N = m*n
    P = m*d

    tau2 = np.ones(d)
    sigma2 = 1
    lambda2 = 1
    matlambda2 = np.ones(d)
    lambda2_path = np.repeat(-1, num_update)
    matlambda2_path = np.repeat(-1.0, d*num_update).reshape(num_update, d)
    l = np.zeros(d)
    beta = np.zeros((d, m))
    Z = np.zeros(d)

    YtY = 0
    for i in range(m):
        YtY += np.dot(ys[i].T, ys[i])
    XtY = np.vstack([np.dot(Xs[i].T, ys[i]) for i in range(m)])
    XtX = [np.dot(Xs[i].T, Xs[i]) for i in range(m)]

    XktY = [np.vstack([np.dot(Xs[i][:,j], ys[i]) for i in range(m)]) for j in range(d)]
    XktXk = [[np.dot(Xs[i][:,j], Xs[i][:,j]) for i in range(m)] for j in range(d)]
    XktXmk = [[np.dot(Xs[i][:,j], np.delete(Xs[i], j, axis = 1)) for i in range(m)] for j in range(d)]

    for update in range(num_update):
        # coef = np.array([]).reshape(m*d, 0)
        tau2_each_update = np.array([]).reshape(d, 0)
        for itera in range(niter):
            
            for i in range(d):
                bmk = np.delete(beta, i, axis = 0)
                f1 = XktY[i] - np.vstack([np.dot(XktXmk[i][j], bmk[:, j]) for j in range(m)])
                f2 = np.array([XktXk[i][j] + 1/tau2[i] for j in range(m)])
                f2_inverse = np.array([1/(XktXk[i][j] + 1/tau2[i]) for j in range(m)])
                mu = np.multiply(f2_inverse, f1.T[0])

                maxf = np.max(f2)
                trythis = (-m/2) * math.log(tau2[i]) + (-1/2) * np.sum(np.log(f2/maxf)) + (-m/2)*math.log(maxf) + np.dot(f1.T[0], mu)/(2 * sigma2)
                if trythis < -50:
                    l[i] = 1
                elif trythis > 50:
                    l[i] = 0
                else:
                    l[i] = pi/(pi + (1-pi)*math.exp(trythis))

                if np.random.uniform() < l[i]:
                    beta[i, :] = np.zeros(m)
                    Z[i] = 0
                else:
                    beta[i, :] = np.random.multivariate_normal(mu, sigma2*np.diag(f2_inverse))
                    Z[i] = 1
          
            for i in range(d):
                if Z[i] == 0:
                    tau2[i] = np.random.gamma((m+1)/2, scale = 2/matlambda2[i])
                else:
                    tau2[i] = 1/np.random.wald(math.sqrt(matlambda2[i]*sigma2/np.sum(beta[i,:]**2)), matlambda2[i])
            tau2_each_update = np.hstack([tau2_each_update, tau2.reshape(d,1)])
            s = 0
            for i in range(d):
                s += np.sum(beta[i,:]**2)/tau2[i]
            beta_vec = beta.T.reshape(m*d, 1)
            # coef = np.hstack([coef, beta_vec])

            gamma_shape = (N-1)/2+np.sum(Z*m)/2+alpha
            btXtXb = 0
            for i in range(m):
                Xb = np.dot(Xs[i], beta[:,i].reshape(d,1))
                btXtXb += np.dot(Xb.T, Xb)
            gamma_scale = (YtY - 2*np.dot(beta_vec.T, XtY)+btXtXb+s)/2 + gamma
            sigma2 = 1/np.random.gamma(gamma_shape, scale = 1/gamma_scale)

            if (pi_prior):
                pi = np.random.beta(a+d-np.sum(Z),b+np.sum(Z))

        tau2_mean = np.mean(tau2_each_update, axis = 1)
        lambda2 = (m*d+d)/np.sum(tau2_mean)
        matlambda2 = np.repeat(lambda2,d)
        matlambda2_path[update, :] = matlambda2
        
        if (verbose):
            print('Update:')
            print(update)
            print('Lambda2:')
            print(matlambda2[0])
            print()
    
    return matlambda2_path

def thresholdBGLSS(Xs, ys, threshold = 10**-2, niter = 600, burnin = 100, a = 1, 
    b = 1, num_update = 10, niter_update = 50, verbose = True, 
    alpha = 0.1, gamma = 0.1, pi_prior = True, pi = 0.5, update_tau = True,
    lambda2_update = np.array([]), normalize = 0, loss_normalize = 2):
    
    d = Xs[0].shape[1]
    n = Xs[0].shape[0]
    m = len(Xs)

    Xs_n = copy.deepcopy(Xs)
    l2 = np.copy(lambda2_update)
    remain = list(range(d))
    usel2 = (lambda2_update.size == d)
    
    while True:
        noChange = True
        xi, lam, loss, path, coef = TrainBGLSS(Xs_n, ys, niter = niter, burnin = burnin, a = a, b = b,
                                           num_update = num_update, niter_update = niter_update, verbose = verbose, 
                                           alpha = alpha, gamma = gamma, pi_prior = pi_prior, pi = pi, update_tau = update_tau, 
                                           lambda2_update = l2, normalize = normalize, loss_normalize = loss_normalize)
        deleted = []
        for j in range(xi.shape[0]):
            rms = np.linalg.norm(xi[j,:])/math.sqrt(np.size(xi[j,:]))
            if rms < threshold or rms < 10**-10:
                noChange = False
                deleted.append(j)
        if noChange:
            break
        for j in sorted(deleted, reverse=True):
            pop = remain.pop(j)
            if usel2:
                l2 = np.delete(l2, j)
            for k in range(len(Xs_n)):
                Xs_n[k] = np.delete(Xs_n[k], j, axis = 1)
        print(remain)
        if len(remain) == 0:
            print("No coefficient left in the model!")
            break
    
    return xi, lam, loss, path, coef, remain
    
    
def thresholdBGLSS_combined(Xs, ys, threshold1 = 0.05, threshold2 = 1, niter = 600, burnin = 100, a = 1, 
    b = 1, num_update = 10, niter_update = 50, verbose = True, 
    alpha = 0.1, gamma = 0.1, pi_prior = True, pi = 0.5, update_tau = True,
    lambda2_update = np.array([]), normalize = 0, loss_normalize = 2):
    
    d = Xs[0].shape[1]
    n = Xs[0].shape[0]
    m = len(Xs)

    Xs_n = copy.deepcopy(Xs)
    l2 = np.copy(lambda2_update)
    remain = list(range(d))
    usel2 = (lambda2_update.size == d)
    
    while True:
        noChange = True
        xi, lam, loss, path, coef = TrainBGLSS(Xs_n, ys, niter = niter, burnin = burnin, a = a, b = b,
                                           num_update = num_update, niter_update = niter_update, verbose = verbose, 
                                           alpha = alpha, gamma = gamma, pi_prior = pi_prior, pi = pi, update_tau = update_tau, 
                                           lambda2_update = l2, normalize = normalize, loss_normalize = loss_normalize)
        deleted = []
        ge = groupError(coef,m)
        print("GE: ")
        print(ge)
        for j in range(xi.shape[0]):
            rms = np.linalg.norm(xi[j,:])/math.sqrt(np.size(xi[j,:]))
            if  rms < threshold1 or rms < 10**-10 or ge[j] > threshold2 or np.isnan(ge[j]):
                noChange = False
                deleted.append(j)
        if noChange:
            break
        for j in sorted(deleted, reverse=True):
            pop = remain.pop(j)  
            if usel2:
                l2 = np.delete(l2, j)
            for k in range(len(Xs_n)):
                Xs_n[k] = np.delete(Xs_n[k], j, axis = 1)
        print(remain)
        if len(remain) == 0:
            print("No coefficient left in the model!")
            break
    
    return xi, lam, loss, path, coef, remain
    