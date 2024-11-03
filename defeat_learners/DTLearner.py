import numpy as np
# import numpy.ma as ma

class DTLearner(object):
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    This is a Decision Tree Learner.

    :param leaf_size: hyperparameter that influences the decision to form a leaf in the decision tree.
        Specifically, it defines the minimum number of samples required for a potential split.
        If a node has fewer than or equal to leaf_size samples, it becomes a leaf node.
        However, an exception is made if all samples at a node have the same value of Y, in which case the node is
         immediately turned into a leaf regardless of the number of samples.
    :type leaf_size: int
  		  	   		 	   		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	   		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		 	   		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    def __init__(self, leaf_size = 1, verbose=False):
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        Constructor method

        Initialize the Decision Tree Learner with a specified leaf size.

        :param leaf_size:: Maximum number of samples to form a leaf.
        :type leaf_size: int
        :param verbose: If True, outputs debugging information.
        :type verbose: bool
        """
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None
        pass  # move along, these aren't the drones you're looking for  		  	   		 	   		  		  		    	 		 		   		 		  

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

    def build_tree(self, data):
        """
        Recursively build the Decision Tree using highest feature correlation to split with a specified leaf size.

        :param data: Maximum number of samples to form a leaf.
        :type data: numpy.ndarray
        :return: Decision Tree node
        :rtype: numpy.ndarray
        """
        data_x = data[:, 0:-1]
        data_y = data[:, -1]
        # if data.shape[0] ==  1 (leaf size) : return [leaf, data.y, NA, NA]
        if data.shape[0] <= self.leaf_size:
            return np.array(['leaf', np.mean(data_y), "NA", "NA"])

        # if all data.y same: return ['leaf', data.y, NA, NA]
        elif np.all(data_y == data_y[0]):
            return np.array(['leaf', data_y[0], "NA", "NA"])

        else:
            # Determine the best feature i to split on by highest correlation
            correlations = np.abs([np.corrcoef(data[:, n], data[:, -1])[0, 1] for n in range(data.shape[1] - 1)])
            best_feat = np.argmax(correlations)
            split_val = np.median(data[:, best_feat])

            # Split data based on the median value of the best feature
            left_data = data[data[:, best_feat] <= split_val]
            right_data = data[data[:, best_feat] > split_val]

            # If the data cannot be split further, return a leaf
            if left_data.shape[0] == 0 or right_data.shape[0] == 0:
                return np.array(['leaf', np.mean(data_y), "NA", "NA"])

            # Use Recursion to build tree branches
            left_tree = self.build_tree(left_data)
            right_tree = self.build_tree(right_data)

            # Create root and append the subtrees to create the tree
            root = np.array([best_feat, split_val, 1, left_tree.shape[0] + 1])

            # print(left_tree.shape)
            # Recursion edge case to prevent skipping right_tree leafs when left_tree.shape == (4,):
            if left_tree.ndim == 1: root = np.array([best_feat, split_val, 1, 2])

            # tree = np.append(np.append(root, left_tree, axis=0), right_tree, axis=0)
            tree = np.vstack((root, left_tree, right_tree))
            # print("tree")
            return tree

    def add_evidence(self, data_x, data_y):  		  	   		 	   		  		  		    	 		 		   		 		  
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        Add training data to learner

        :param data_x: A set of feature values used to train the learner  		  	   		 	   		  		  		    	 		 		   		 		  
        :type data_x: numpy.ndarray  		  	   		 	   		  		  		    	 		 		   		 		  
        :param data_y: The value we are attempting to predict given the X data  		  	   		 	   		  		  		    	 		 		   		 		  
        :type data_y: numpy.ndarray  		  	   		 	   		  		  		    	 		 		   		 		  
        """
        # y = data_y.reshape(-1, 1)
        # data1 = np.hstack((data_x, data_y.reshape(-1, 1)))
        # print("hello")
        #data = np.append(data_x, data_y, axis=1)
        data = np.append(data_x, data_y.reshape(-1,1), axis=1)
        # if data.all() == data1.all(): print("true")
        # print("hello")
        self.tree = self.build_tree(data)
        # print("tree built")

    def query(self, points):  		  	   		 	   		  		  		    	 		 		   		 		  
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        Estimate a set of test points given the model we built.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		 	   		  		  		    	 		 		   		 		  
        :type points: numpy.ndarray  		  	   		 	   		  		  		    	 		 		   		 		  
        :return: The predicted result of the input data according to the trained model  		  	   		 	   		  		  		    	 		 		   		 		  
        :rtype: numpy.ndarray  		  	   		 	   		  		  		    	 		 		   		 		  
        """
        # Predict values for each point by traversing the tree
        predictions = []
        for point in points:
            index = 0  # Start at the root node
            # Traverse the tree until a leaf node is reached
            while self.tree[index, 0] != 'leaf':
                feat = int(float(self.tree[index, 0]))  # Feature index to split on
                if point[feat] <= float(self.tree[index, 1]):  # Compare point's feature value with split val
                    index += int(float(self.tree[index, 2]))  # Move to the left subtree
                else:
                    # if index + int(float(self.tree[index, 3])) >= self.tree.shape[0]:
                    #     print(index, int(float(self.tree[index, 3])))
                    index += int(float(self.tree[index, 3]))  # Move to the right subtree
            predictions.append(float(self.tree[index, 1]))  # Return the predicted value at the leaf
        return np.array(predictions)

        # return np.array([self.query_point(row) for row in points])

  		  	   		 	   		  		  		    	 		 		   		 		  
if __name__ == "__main__":
    verbose = False
    if verbose: print("the secret clue is 'zzyzx'")
