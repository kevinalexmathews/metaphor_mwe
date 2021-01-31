import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

class Evaluate():
    """define evaluation metrics (acc, precision, recall, and f1-score)"""
    def __init__(self, out, labels):
        self.out = np.argmax(out, axis=1).numpy().flatten()
        self.labels = labels.numpy().flatten()
    def accuracy(self):
        nb_correct = sum(y_t==y_p for y_t, y_p in zip(self.labels, self.out))
        nb_true = len(self.labels)
        score = nb_correct / nb_true
        return score
    def precision_recall_fscore(self, tag_list=[0,1], average='macro'):
        return precision_recall_fscore_support(self.labels, self.out, average=average,labels=tag_list)[:-1]
    def precision_recall_fscore_coarse(self, tag_list=[0,1], average='binary'):
        # return metrics corresponding to the pos_label ONLY for coarse experimental setting (MET or LIT)
        return precision_recall_fscore_support(self.labels, self.out, average=average, pos_label=1, labels=tag_list)[:-1]