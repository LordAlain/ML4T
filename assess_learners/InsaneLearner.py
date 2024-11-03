import BagLearner as bl
import LinRegLearner as lrl
class InsaneLearner(object):
    def author(self):
        return "akommi3"
    def study_group(self):
        return "akommi3"
    def gtid(self):
        return 903135337
    def __init__(self, verbose=False):
        self.bag = bl.BagLearner(learner=bl.BagLearner, kwargs={"learner": lrl.LinRegLearner,"kwargs": {}}, bags=20,verbose=verbose)
    def add_evidence(self, data_x, data_y):
        self.bag.add_evidence(data_x, data_y)
    def query(self, points):
        return self.bag.query(points)
if __name__ == "__main__":
    verbose = False
    if verbose: print("the secret clue is 'zzyzx'")