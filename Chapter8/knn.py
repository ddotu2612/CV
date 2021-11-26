import numpy as np

# Tao mot lop phan loai dua vao KNN
class KnnClassifier(object):

    def __init__(self, labels, samples) -> None:
        super().__init__()
        self.labels = labels
        self.samples = samples
    
    def classify(self, point, k=3):
        """ Classify a point against k nearest
            in the training data, return label. """

        # compute distance to all training points
        dist = np.array([L2dist(point, s) for s in self.samples])

        # sort them
        ndx = dist.argsort()

        # use disctionary to store the k nearest
        votes = {}
        for i in range(k):
            label = self.labels[ndx[i]]
            votes.setdefault(label, 0)
            votes[label] += 1
        
        return max(votes)

def L2dist(p1, p2):
    return np.sqrt(np.sum(p1 - p2)**2)


        
