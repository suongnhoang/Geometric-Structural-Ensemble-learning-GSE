from gse_utils import cal_max_projecttion_point, euclid_dist, find_Xin_smaples,\
                        find_xcv, find_xrc, get_classification_sign, get_batch, vector_module
import numpy as np

class GSE:
    def __init__(self, P = 5, R = 1.0):
        """
        subset_classifiers in this class is a list of all subset training set basic_classifiers,
        one basic_classifier can is contruct from at least one or many hyperplanes
        """
        self.P = P
        self.R = R
        self.subset_classifiers = []

    def basic_classifier(self, X_in, x_cv, x_rv, move_rate=0.0):
        w = x_cv - x_rv
        if vector_module(w) == 0:
            w = np.zeros_like(x_cv)
            w = w + 1e-10
        x_tp = cal_max_projecttion_point(X_in,x_rv,w)
        move_vetor = move_rate*(x_cv - x_tp)
        phi = -np.dot(x_tp+move_vetor, w.T)
        return w, phi

    def CHG(self, X_maj_star, x_rv, x_cv, move_rate=0.05):
        maj_sample_radius = euclid_dist(x_cv, x_rv)
        X_in, X_maj_star = find_Xin_smaples(X_maj_star, x_rv, maj_sample_radius)
        w, phi = self.basic_classifier(X_in, x_cv, x_rv, move_rate)
        return (w,phi)

    def BCG(self, x_mp, X_min, X_maj_star):
        x_rv = find_xrc(X_maj_star, x_mp)
        X_min_star = X_min
        hyperplanes, X_del_maj_masks = [], np.ones(X_maj_star.shape[0])
        while len(X_min_star):
            x_cv = find_xcv(X_min_star, x_rv)
            w_phi = self.CHG(X_maj_star, x_rv, x_cv)
            X_del_maj_masks = np.multiply(X_del_maj_masks, (get_classification_sign(w_phi, X_maj_star,True, False)<=0)*1.0)
            X_del_min_mask = (get_classification_sign(w_phi, X_min_star, True, False) >= 0)
            X_min_star = X_min_star[~X_del_min_mask]
            hyperplanes.append(w_phi)
        return hyperplanes, X_del_maj_masks.astype('bool_')

    def fit(self, X_maj, X_min):
        X_del, self.subset_classifiers = [],[]
        x_mp = X_min.mean(0)
        IR = X_maj.shape[0] / (self.P * X_min.shape[0])
        for X_maj_star in get_batch(self.P, X_maj):
            hyperplanes_subset = []
            while X_maj_star.shape[0]:
                hyperplanes_i, X_del_maj_masks = self.BCG(x_mp, X_min, X_maj_star)
                X_del_maj = X_maj_star[X_del_maj_masks]
                X_del.append([X_del_maj])
                if X_del_maj.shape[0] >= self.R*IR:
                    hyperplanes_subset.append(hyperplanes_i)
                X_maj_star = X_maj_star[~X_del_maj_masks]
            self.subset_classifiers.append(hyperplanes_subset)
        return X_del
    
    def predict(self, X):
        C_sum = np.zeros(X.shape[0])
        for i, subset_i in enumerate(self.subset_classifiers):
            C_i = np.ones(X.shape[0])
            for j, basic_clasifier_j in enumerate(subset_i):
                H_ij = np.zeros(X.shape[0])
                for k, hyperplaners in enumerate(basic_clasifier_j):
                    # if len(basic_clasifier_j)
                    H_ij = H_ij + get_classification_sign(hyperplaners, X, False , False)
                H_ij = np.sign((H_ij + len(basic_clasifier_j)))
                C_i = np.multiply(C_i,H_ij)
            C_sum = C_sum + C_i
        F = np.sign(-0.5 + C_sum)
        return (F>0)*1
