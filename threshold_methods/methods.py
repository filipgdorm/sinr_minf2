import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve

def upper_b_threshold(preds, species_locs):
    y_test = np.zeros(preds.shape, int)
    y_test[species_locs] = 1 

    precision, recall, thresholds = precision_recall_curve(y_test, preds)
    p1 = (2 * precision * recall)
    p2 = (precision + recall)
    out = np.zeros( (len(p1)) )
    fscore = np.divide(p1,p2, out=out, where=p2!=0)
    index = np.argmax(fscore)
    thres = thresholds[index]
    max_fscore = fscore[index]

    return thres, max_fscore
