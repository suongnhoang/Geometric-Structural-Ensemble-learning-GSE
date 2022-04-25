import numpy as np
import copy


def euclid_dist(x, y):
    x, y = x.squeeze(), y.squeeze()
    assert x.shape == y.shape and len(x.shape)==1
    return np.sqrt(((x-y)**2).sum())

def get_classification_sign(w_phi, X, soft_class=False, poolean_out=True):
    w, phi = w_phi
    if poolean_out:
        return np.sign(X.dot(w.T)+phi)>=0 if soft_class else np.sign(X.dot(w.T)+phi)>0
    return np.sign(X.dot(w.T)+phi)

def calculate_dist_Xmaj_xmp(X_maj, x_mp):
    dists = []
    for i in range(X_maj.shape[0]):
        dists.append(euclid_dist(X_maj[i,:], x_mp))
    return np.array(dists)

def find_xrc(X_maj, x_mp):    
    dists_maj_xmp = calculate_dist_Xmaj_xmp(X_maj, x_mp)
    x_rv = X_maj[np.argmax(dists_maj_xmp),:]
    return x_rv

def find_xcv(X_min, x_rv):
    dists_min_xmp = calculate_dist_Xmaj_xmp(X_min, x_rv)
    x_cv = X_min[np.argmin(dists_min_xmp),:]
    return x_cv

def find_Xin_smaples(X_maj, x_rv, radius):
    dist = calculate_dist_Xmaj_xmp(X_maj, x_rv)
    mask = dist <= radius
    return X_maj[mask], X_maj[~mask]

def cal_max_projecttion_point(X_in, x_rv, w):
    max_projection = X_in[np.argmax((X_in - x_rv).dot(w.T)), :]
    return max_projection

def get_batch(P:int,samples:np.array):
    n = samples.shape[0]
    assert P >= 1 and n>0
    samples_ = copy.deepcopy(samples)
    np.random.shuffle(samples_)
    samples_list, res, k, i_n = [], n, 0, 0
    for i in range(P,0,-1):
        p_n = i_n
        i_n = res//i
        res -= i_n
        samples_list.append(samples_[k*p_n:(k+1)*i_n,...])
        k += 1 
    return samples_list
