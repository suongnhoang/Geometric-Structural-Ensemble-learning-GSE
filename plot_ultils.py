import numpy as np


def create_hyperplane_plot(w, phi, start=-1, end=1):
    assert len(w.shape)<=2 or len(phi.shape)<=1
    hyperplan = lambda x: -(w[0]*x +phi)/w[1]
    hyper_x = np.linspace(start, end, num=1000)
    hyper_y = hyperplan(hyper_x)
    return hyper_x, hyper_y

def create_domain_plot(gse_learner, start=-1, end=1):
    X = np.mgrid[start:end:0.01, start:end:0.01].reshape(2,-1).T
    labels = gse_learner.predict(X)
    X_mask = labels > 0
    maj_domain, min_domain = X[~X_mask], X[X_mask]
    return maj_domain, min_domain

