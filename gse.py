from gse_utils import cal_max_projecttion_point, euclid_dist, find_Xin_smaples,\
                        find_xcv, find_xrc, get_classification_sign, get_batch
import numpy as np

class GSE:
    def __init__(self, P = 5, R = 1.0):
        self.P = P
        self.R = R
        self.hyperplanes = []

    def basic_classifier(self, X_in, x_cv, x_rv, move_rate=0.5):
        w = x_cv - x_rv
        x_tp = cal_max_projecttion_point(X_in,x_rv,w)
        move_vetor = move_rate*(x_cv - x_tp)
        phi = -np.dot(x_tp+move_vetor, w.T)
        return w, phi

    def CHG(self, X_maj_star, x_rv, x_cv, move_rate=0.2):
        maj_sample_radius = euclid_dist(x_cv, x_rv)
        X_in, X_maj_star = find_Xin_smaples(X_maj_star, x_rv, maj_sample_radius)
        w, phi = self.basic_classifier(X_in, x_cv, x_rv, move_rate)
        return (w,phi)

    def BCG(self, x_mp, X_min, X_maj_star):
        x_rv = find_xrc(X_maj_star, x_mp)
        X_min_star = X_min
        hyperplanes, X_del_maj_masks = [], None
        while len(X_min_star):
            x_cv = find_xcv(X_min_star, x_rv)
            w_phi = self.CHG(X_maj_star, x_rv, x_cv)
            if X_del_maj_masks is None:
                X_del_maj_masks = ~get_classification_sign(w_phi, X_maj_star,True)*1
            else:
                X_del_maj_masks += ~get_classification_sign(w_phi, X_maj_star,True)*1
                X_del_maj_masks = (X_del_maj_masks>=2)*1
            X_del_min_mask = get_classification_sign(w_phi, X_min_star,True)
            X_min_star = X_min_star[~X_del_min_mask]
            hyperplanes.append(w_phi)
        return hyperplanes, X_del_maj_masks.astype('bool_')

    def fit(self, X_maj, X_min):
        X_del, self.hyperplanes = [],[]
        x_mp = X_min.mean(0)
        IR = X_maj.shape[0] / (self.P * X_min.shape[0])
        for X_maj_star in get_batch(self.P, X_maj):
            while X_maj_star.shape[0]:
                hyperplanes_i, X_del_maj_masks = self.BCG(x_mp, X_min, X_maj_star)
                X_del_maj = X_maj_star[X_del_maj_masks]
                X_del.append(X_del_maj)
                if X_del_maj.shape[0] >= self.R*IR:
                    self.hyperplanes.append(hyperplanes_i)
                X_maj_star = X_maj_star[~X_del_maj_masks]
        return self.hyperplanes
    
    def predict(self, X):
        X_mask = None
        for hyperplane in self.hyperplanes:
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
        X_mask = ~X_mask.astype('bool_')*1
        return X_mask