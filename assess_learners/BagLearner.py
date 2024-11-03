import numpy as np

class BagLearner:

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "akommi3"  # replace tb34 with your Georgia Tech username.

    def study_group(self):
        """
        :return: A comma separated string of GT_Name of each member of your study group
        :rtype: str
        # Example:"gburdell3, jdoe77, tbalch7" or "gburdell3" if a single individual working alone
        """
        return "akommi3"

    def gtid(self):
        """
        :return: The GT ID of the student
        :rtype: int
        """
        return 903135337  # replace with your GT ID number

    def __init__(self, learner, kwargs, bags=20, boost=False, verbose=False):
        """
        Initialize the Bag Learner with specified number of bags (learners).

        Parameters:
            learner (class): Learner class used in bagging.
            kwargs (dict): Arguments passed to the learner's constructor.
            bags (int): Number of learners to train in the bagging ensemble.
            boost (bool): If true, implements boosting (optional).
            verbose (bool): If True, outputs debugging information.
        """
        self.learner = learner
        self.bags = bags
        self.verbose = verbose
        learners = []
        self.learners = [learner(**kwargs) for i in range(0, self.bags)]  # Create multiple learners

    def add_evidence(self, data_x, data_y):
        """
        Train the Bag Learner by training each learner on random samples of the data.

        Parameters:
            data_x (numpy.ndarray): Feature values to train the learner.
            data_y (numpy.ndarray): Target values corresponding to the features.
        """
        for learner in self.learners:
            indices = np.random.choice(data_x.shape[0], size=data_x.shape[0], replace=True)
            learner.add_evidence(data_x[indices], data_y[indices])  # Train on bootstrap samples

    def query(self, points):
        """
        Predict target values by aggregating the predictions from all learners.

        Parameters:
            points (numpy.ndarray): Feature values of the points to query.

        Returns:
            numpy.ndarray: Aggregated predicted target values.
        """
        predictions = np.array([learner.query(points) for learner in self.learners])
        return np.mean(predictions, axis=0)  # Average predictions from all learners

if __name__ == "__main__":
    verbose = False
    if verbose: print("the secret clue is 'zzyzx'")
