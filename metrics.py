from sklearn.metrics import confusion_matrix
import numpy as np

def cal_gmean(y_true, y_pred):
    cf_metric = confusion_matrix(y_true, y_pred)
    TP, FN, FP, TN = cf_metric[0,0], cf_metric[0,1], cf_metric[1,0], cf_metric[1,1]
    return np.sqrt((TP/(TP+FN))*(TN/(TN+FP)))
