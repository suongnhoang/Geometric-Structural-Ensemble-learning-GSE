import numpy as np
from gse_utils import get_classification_sign


def create_hyperplane_plot(w, phi, start=-1, end=1):
    assert len(w.shape)<=2 or len(phi.shape)<=1
    hyperplan = lambda x: -(w[0]*x +phi)/w[1]
    hyper_x = np.linspace(start, end, num=1000)
    hyper_y = hyperplan(hyper_x)
    return hyper_x, hyper_y

def create_domain_plot(hyperplanes, start=-1, end=1):
    assert any([len(w.shape)<=2 for hyperplane in hyperplanes for w,_ in hyperplane])
    X = np.mgrid[start:end:0.01, start:end:0.01].reshape(2,-1).T
    X_mask = None
    for hyperplane in hyperplanes:
        X_mask_i = None
        
        for w_phi in hyperplane:
            if X_mask_i is None:
                X_mask_i = (~get_classification_sign(w_phi, X, True))*1
            else:
                X_mask_i += (~get_classification_sign(w_phi, X, True))*1
                X_mask_i = (X_mask_i>=2)*1
        if X_mask is None:
            X_mask = X_mask_i
        else:
            X_mask += X_mask_i
            X_mask = (X_mask>=1)*1
    X_mask = X_mask.astype('bool_')
    maj_domain, min_domain = X[X_mask], X[~X_mask]
    return maj_domain, min_domain

