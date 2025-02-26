def BGLSS(Y, X, group_size, niter = 10000, burnin = 5000, a = 1, 
    b = 1, num_update = 100, niter_update = 100, verbose = False, 
    alpha = 0.1, gamma = 0.1, pi_prior = True, pi = 0.5, update_tau = True, 
    lambda2_update = np.array([])):

    n = X.shape[0]
    p = X.shape[1]
    ngroup = group_size.size
    tau2 = np.ones(ngroup)
    sigma2 = 1
    l = np.zeros(ngroup)
    beta = []
    for i in range(ngroup):
        beta.append(np.zeros(group_size[i]))
    Z = np.zeros(ngroup)
    if (lambda2_update.size == 0):
        fit_for_lambda2 = BGLSS_EM_lambda(Y, X, group_size, num_update = num_update, 
            niter = niter_update, verbose = verbose)
        lambda2 = fit_for_lambda2[-1, :]
    
    else:
        lambda2 = lambda2_update

    YtY = np.dot(Y.T, Y)
    XtY = np.dot(X.T, Y)
    XtX = np.dot(X.T, X)
    XktY = []
    XktXk = []
    XktXmk = []

    begin_idx = 0
    for i in range(ngroup):
        end_idx = begin_idx + group_size[i]
        Xk = X[:, begin_idx:end_idx]
        Xmk = np.hstack([X[:, 0:begin_idx], X[:, end_idx:]])
        XktY.append(np.dot(Xk.T, Y))
        XktXk.append(np.dot(Xk.T, Xk))
        XktXmk.append(np.dot(Xk.T, Xmk))
        begin_idx = end_idx

    coef = np.array([]).reshape(p, 0)
    coef_tau = []
    
    for itera in range(niter):
        if (verbose):
            print(itera)
        for i in range(ngroup):
            bmk = np.array([])
            for j in range(ngroup):
                if j!=i:
                    bmk = np.append(bmk, beta[j])

            bmk = bmk.reshape(-1, 1)

            f1 = XktY[i] - np.dot(XktXmk[i], bmk)
            f2 = XktXk[i] + 1/tau2[i] * np.eye(group_size[i])
            f2_inverse = np.linalg.inv(f2)
            mu = np.dot(f2_inverse, f1)

            maxf = np.max(f2)
            trythis = (-group_size[i]/2) * math.log(tau2[i]) + (-1/2) * np.log(np.linalg.det(f2/maxf)) + (-group_size[i]/2)*math.log(maxf) + np.dot(f1.T, mu)/(2 * sigma2)
            if trythis < -50:
                l[i] = 1
            elif trythis > 50:
                l[i] = 0
            else:
                l[i] = pi/(pi + (1-pi)*math.exp(trythis))

            if np.random.uniform() < l[i]:
                beta[i] = np.zeros(group_size[i])
                Z[i] = 0
            else:
                beta[i] = np.random.multivariate_normal(mu.T[0], sigma2*f2_inverse)
                Z[i] = 1
          
        for i in range(ngroup):
            if Z[i] == 0:
                tau2[i] = np.random.gamma((group_size[i]+1)/2, scale = 2/lambda2[i])
            else:
                tau2[i] = 1/np.random.wald(math.sqrt(lambda2[i]*sigma2/np.sum(beta[i]**2)), lambda2[i])
        
        s = 0
        for i in range(ngroup):
            s += np.sum(beta[i]**2)/tau2[i]
            
        beta_vec = np.array([])
        for j in range(ngroup):
            beta_vec = np.append(beta_vec, beta[j])
        beta_vec = beta_vec.reshape(-1, 1)
            
        if itera > burnin:
            coef = np.hstack([coef, beta_vec])
        
        gamma_shape = (n-1)/2+np.sum(Z*group_size)/2+alpha
        gamma_scale = (YtY - 2*np.dot(beta_vec.T, XtY)+np.dot(np.dot(beta_vec.T, XtX), beta_vec)+s)/2 + gamma
        sigma2 = 1/np.random.gamma(gamma_shape, scale = 1/gamma_scale)

        if (pi_prior):
            pi = np.random.beta(a+ngroup-np.sum(Z),b+np.sum(Z))

    pos_mean = np.mean(coef, axis=1)
    pos_median = np.median(coef, axis=1)
    
    return pos_median, lambda2