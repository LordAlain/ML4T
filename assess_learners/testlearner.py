""""""  		  	   		 	   		  		  		    	 		 		   		 		  
"""  		  	   		 	   		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	   		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		 	   		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	   		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		 	   		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		 	   		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		 	   		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	   		  		  		    	 		 		   		 		  
or edited.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		 	   		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		 	   		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	   		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		 	   		  		  		    	 		 		   		 		  
"""  		  	   		 	   		  		  		    	 		 		   		 		  

import math
import sys
import numpy as np
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
# import InsaneLearner as it
import matplotlib.pyplot as plt
import time

def load_data(datapath):
    read_data = np.genfromtxt(datapath, delimiter=",")
    if datapath == "Data/Istanbul.csv":
        data = read_data[1:, 1:] # removing header row and date column
    else: data = read_data
    return data

def train_test_split(data, split_ratio = 0.6):

    # Generate a random permutation of indices for shuffling the data
    shuffled_indices = np.random.permutation(data.shape[0])


    # compute how much of the data is training and testing
    row_split = int(split_ratio * data.shape[0])
    #test_rows = data.shape[0] - row_split


    # # separate out training and testing data
    # train_x = data[:row_split, 0:-1]
    # train_y = data[:row_split, -1]
    # test_x = data[row_split:, 0:-1]
    # test_y = data[row_split:, -1]

    # Use the shuffled indices to create training and testing sets
    train_idx, test_idx = shuffled_indices[:row_split], shuffled_indices[row_split:]
    train_x, test_x = data[train_idx, 0:-1], data[test_idx, 0:-1]
    train_y, test_y = data[train_idx, -1], data[test_idx, -1]

    ## # print(f"{test_x.shape}")
    ## # print(f"{test_y.shape}")

    return train_x, test_x, train_y, test_y

# Function to calculate Root Mean Squared Error (RMSE)
def get_rmse(pred, y):
    error = math.sqrt(((y - pred) ** 2).sum() / y.shape[0])
    return error

def experiment_1(data):
    """
    Experiment 1: Investigates overfitting with DTLearner by varying leaf_size.
    """
    leaf_sizes = range(1, 51)
    in_sample = []
    out_sample = []

    # Split the data
    train_x, test_x, train_y, test_y = train_test_split(data)

    for leaf_size in leaf_sizes:
        learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)

        # Train the learner and query on the test set
        learner.add_evidence(train_x, train_y)

        # Calculate RMSE for in_sample
        pred_y = learner.query(train_x)
        err = get_rmse(pred_y, train_y)
        in_sample.append(err)

        # Calculate RMSE for out_sample
        pred_y = learner.query(test_x)
        err = get_rmse(pred_y, test_y)
        out_sample.append(err)

    # in_sample = np.array(in_sample).reshape((-1, 1))
    # out_sample = np.array(out_sample).reshape((-1, 1))
    #
    # plot_data = np.append(in_sample, out_sample, axis=1)

    diff_error = np.asarray(out_sample) - np.asarray(in_sample)
    plt.figure(1)
    plt.plot(leaf_sizes, in_sample, label="DT in-sample errors")
    plt.plot(leaf_sizes, out_sample, label="DT out-sample errors")
    plt.plot(leaf_sizes, diff_error, label="DT out-sample - in-sample")
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.title('Experiment 1: RMSE vs Leaf Size (DTLearner)')
    plt.grid(True)
    plt.legend()
    plt.savefig('images/experiment_1_DT.png')
    plt.clf()


    leaf_sizes = range(1, 51)
    in_sample = []
    out_sample = []

    for leaf_size in leaf_sizes:
        learner = rt.RTLearner(leaf_size=leaf_size, verbose=False)

        # Train the learner and query on the test set
        learner.add_evidence(train_x, train_y)

        # Calculate RMSE for in_sample
        pred_y = learner.query(train_x)
        err = get_rmse(pred_y, train_y)
        in_sample.append(err)

        # Calculate RMSE for out_sample
        pred_y = learner.query(test_x)
        err = get_rmse(pred_y, test_y)
        out_sample.append(err)


    # in_sample = np.array(in_sample).reshape((-1, 1))
    # out_sample = np.array(out_sample).reshape((-1, 1))
    #
    # plot_data = np.append(in_sample, out_sample, axis=1)

    diff_error = np.asarray(out_sample) - np.asarray(in_sample)
    plt.figure(2)
    plt.plot(leaf_sizes, in_sample, label="RT in-sample errors")
    plt.plot(leaf_sizes, out_sample, label="RT out-sample errors")
    plt.plot(leaf_sizes, diff_error, label="RT out-sample - in-sample")
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.title('Experiment 1: RMSE vs Leaf Size (RTLearner)')
    plt.grid(True)
    plt.legend()
    plt.savefig('images/experiment_1_RT.png')
    plt.clf()


def experiment_2(data):
    """
    Experiment 2: Investigates the effect of bagging on overfitting with DTLearner.
    """
    leaf_sizes = range(1, 51)
    in_sample = []
    out_sample = []

    # Split the data
    train_x, test_x, train_y, test_y = train_test_split(data)

    for leaf_size in leaf_sizes:
        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": leaf_size}, bags=20, verbose=False)

        # Train the learner and query on the test set
        learner.add_evidence(train_x, train_y)

        # Calculate RMSE for in_sample
        pred_y = learner.query(train_x)
        in_sample.append(get_rmse(pred_y, train_y))

        # Calculate RMSE for out_sample
        pred_y = learner.query(test_x)
        out_sample.append(get_rmse(pred_y, test_y))

    diff_error = np.asarray(out_sample) - np.asarray(in_sample)
    plt.figure(3)
    plt.plot(leaf_sizes, in_sample, label="DT in-sample errors")
    plt.plot(leaf_sizes, out_sample, label="DT out-sample errors")
    plt.plot(leaf_sizes, diff_error, label="DT out-sample - in-sample")
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.title('Experiment 2: RMSE vs Leaf Size (Bagging with DTLearner)')
    plt.grid(True)
    plt.legend()
    plt.savefig('images/experiment_2_DT.png')
    plt.clf()


    leaf_sizes = range(1, 51)
    in_sample = []
    out_sample = []

    # Split the data
    train_x, test_x, train_y, test_y = train_test_split(data)

    for leaf_size in leaf_sizes:
        learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": leaf_size}, bags=20, verbose=False)

        # Train the learner and query on the test set
        learner.add_evidence(train_x, train_y)

        # Calculate RMSE for in_sample
        pred_y = learner.query(train_x)
        in_sample.append(get_rmse(pred_y, train_y))

        # Calculate RMSE for out_sample
        pred_y = learner.query(test_x)
        out_sample.append(get_rmse(pred_y, test_y))

    diff_error = np.asarray(out_sample) - np.asarray(in_sample)
    plt.figure(4)
    plt.plot(leaf_sizes, in_sample, label="RT in-sample errors")
    plt.plot(leaf_sizes, out_sample, label="RT out-sample errors")
    plt.plot(leaf_sizes, diff_error, label="DT out-sample - in-sample")
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.title('Experiment 2: RMSE vs Leaf Size (Bagging with RTLearner)')
    plt.grid(True)
    plt.legend()
    plt.savefig('images/experiment_2_RT.png')
    plt.clf()


# Function to calculate Mean Absolute Error (MAE)
def mean_absolute_error(pred, y):
    return np.mean(np.abs(y - pred))

# Function to calculate Coefficient of Determination (R-Squared)
def r_squared(pred, y):
    ss_res = np.sum((y - pred) ** 2) # sum squared regression (SSR)
    ss_tot = np.sum((y - np.mean(y)) ** 2) # total sum of squares (SST)
    return 1 - (ss_res / ss_tot)

# Function to calculate Mean Absolute Percentage Error (MAPE)
def get_mape(pred, y):
    return np.mean(np.abs((y - pred) / y)) * 100

# Function to calculate Maximum Error (ME)
def max_error(pred, y):
    return np.max(np.abs(y - pred))

def experiment_3(data):

    # Split the data
    train_x, test_x, train_y, test_y = train_test_split(data)

    ## DTLearner
    leaf_sizes = range(1, 51)
    train_times_dt = []  # Store training times for DTLearner

    # in-sample arrays
    mae_is = []
    r2_is = []
    mape_is = []
    me_is = []

    # out-sample arrays
    mae_os = []
    r2_os = []
    mape_os = []
    me_os = []

    for leaf_size in leaf_sizes:
        learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)

        # Train the learner and query on the test set
        start_time = time.time()
        learner.add_evidence(train_x, train_y)
        train_times_dt.append(time.time() - start_time)

        # Calculate stats for in_sample
        pred_y = learner.query(train_x)
        mae_is.append(mean_absolute_error(pred_y, train_y))
        r2_is.append(r_squared(pred_y, train_y))
        mape_is.append(get_mape(pred_y, train_y))
        me_is.append(max_error(pred_y, train_y))

        # Calculate stats for out_sample
        pred_y = learner.query(test_x)
        mae_os.append(mean_absolute_error(pred_y, test_y))
        r2_os.append(r_squared(pred_y, test_y))
        mape_os.append(get_mape(pred_y, test_y))
        me_os.append(max_error(pred_y, test_y))

    mae_diff = np.asarray(mae_os) - np.asarray(mae_is)
    r2_diff = np.asarray(r2_os) - np.asarray(r2_is)
    mape_diff = np.asarray(mape_os) - np.asarray(mape_is)
    me_diff = np.asarray(me_os) - np.asarray(me_is)

    plt.figure(5)
    plt.plot(leaf_sizes, mae_is, label="DT in-sample MAE")
    plt.plot(leaf_sizes, mae_os, label="DT out-sample MAE")
    plt.plot(leaf_sizes, mae_diff, label="DT MAE Difference")

    plt.figure(6)
    plt.plot(leaf_sizes, r2_is, label="DT in-sample R2")
    plt.plot(leaf_sizes, r2_os, label="DT out-sample R2")
    #plt.plot(leaf_sizes, r2_diff, label="DT R2 Difference")

    plt.figure(7)
    plt.plot(leaf_sizes, mape_is, label="DT in-sample MAPE")
    plt.plot(leaf_sizes, mape_os, label="DT out-sample MAPE")
    # plt.plot(leaf_sizes, mape_diff, label="DT MAPE Difference")

    plt.figure(8)
    plt.plot(leaf_sizes, me_is, label="DT in-sample ME")
    plt.plot(leaf_sizes, me_os, label="DT out-sample ME")
    #plt.plot(leaf_sizes, me_diff, label="DT ME Difference")

    ## RTLearner
    leaf_sizes = range(1, 51)
    train_times_rt = []  # Store training times for RTLearner

    # in-sample arrays
    mae_is = []
    r2_is = []
    mape_is = []
    me_is = []

    # out-sample arrays
    mae_os = []
    r2_os = []
    mape_os = []
    me_os = []

    for leaf_size in leaf_sizes:
        learner = rt.RTLearner(leaf_size=leaf_size, verbose=False)

        # Train the learner and query on the test set
        start_time = time.time()
        learner.add_evidence(train_x, train_y)
        train_times_rt.append(time.time() - start_time)

        # Calculate stats for in_sample
        pred_y = learner.query(train_x)
        mae_is.append(mean_absolute_error(pred_y, train_y))
        r2_is.append(r_squared(pred_y, train_y))
        mape_is.append(get_mape(pred_y, train_y))
        me_is.append(max_error(pred_y, train_y))

        # Calculate stats for out_sample
        pred_y = learner.query(test_x)
        mae_os.append(mean_absolute_error(pred_y, test_y))
        r2_os.append(r_squared(pred_y, test_y))
        mape_os.append(get_mape(pred_y, test_y))
        me_os.append(max_error(pred_y, test_y))

    mae_diff = np.asarray(mae_os) - np.asarray(mae_is)
    r2_diff = np.asarray(r2_os) - np.asarray(r2_is)
    mape_diff = np.asarray(mape_os) - np.asarray(mape_is)
    me_diff = np.asarray(me_os) - np.asarray(me_is)

    plt.figure(5)
    plt.plot(leaf_sizes, mae_is, label="RT in-sample MAE")
    plt.plot(leaf_sizes, mae_os, label="RT out-sample MAE")
    plt.plot(leaf_sizes, mae_diff, label="RT MAE Difference")
    plt.xlabel('Leaf Size')
    plt.ylabel('MAE')
    plt.title('Experiment 3: Mean Absolute Error (MAE) (DT vs RT)')
    plt.grid(True)
    plt.legend()
    plt.savefig('images/experiment_3_MAE.png')
    plt.clf()

    plt.figure(6)
    plt.plot(leaf_sizes, r2_is, label="RT in-sample R2")
    plt.plot(leaf_sizes, r2_os, label="RT out-sample R2")
    #plt.plot(leaf_sizes, r2_diff, label="RT R2 Difference")
    plt.xlabel('Leaf Size')
    plt.ylabel('R-Squared')
    plt.title('Experiment 3: Coefficient of Determination (R-Squared) (DT vs RT)')
    plt.grid(True)
    plt.legend()
    plt.savefig('images/experiment_3_R2.png')
    plt.clf()

    plt.figure(7)
    plt.plot(leaf_sizes, mape_is, label="RT in-sample MAPE")
    plt.plot(leaf_sizes, mape_os, label="RT out-sample MAPE")
    #plt.plot(leaf_sizes, mape_diff, label="RT MAPE Difference")
    plt.xlabel('Leaf Size')
    plt.ylabel('MAPE')
    plt.title('Experiment 3: Mean Absolute Percentage Error (MAPE) (DT vs RT)')
    plt.grid(True)
    plt.legend()
    plt.savefig('images/experiment_3_MAPE.png')
    plt.clf()

    plt.figure(8)
    plt.plot(leaf_sizes, me_is, label="RT in-sample ME")
    plt.plot(leaf_sizes, me_os, label="RT out-sample ME")
    # plt.plot(leaf_sizes, me_diff, label="RT ME Difference")
    plt.xlabel('Leaf Size')
    plt.ylabel('Mean Error')
    plt.title('Experiment 3: Maximum Error (ME) (DT vs RT)')
    plt.grid(True)
    plt.legend()
    plt.savefig('images/experiment_3_ME.png')
    plt.clf()

    # Plot the training times
    plt.figure(9)
    plt.plot(leaf_sizes, train_times_dt, label="DT Learner Training Time")
    plt.plot(leaf_sizes, train_times_rt, label="RT Learner Training Time")
    plt.xlabel('Leaf Size')
    plt.ylabel('Training Time (seconds)')
    plt.title('Experiment 3: Training Time (DT vs RT)')
    plt.grid(True)
    plt.legend()
    plt.savefig('images/experiment_3_time.png')
    plt.clf()

def run_experiments(data):
    # # print("Running Experiment 1: Overfitting with DTLearner")
    experiment_1(data)
    # # print("Running Experiment 2: Bagging with DTLearner")
    experiment_2(data)
    # # print("Running Experiment 3: Comparison between DTLearner and RTLearner")
    experiment_3(data)

def initial_testing(data):
    # # print("Initial Testing")

    # Randomly split the data into training and testing sets
    train_x, test_x, train_y, test_y = train_test_split(data)

    # create a learner and train it
    learner = lrl.LinRegLearner()  # create a LinRegLearner
    learner.add_evidence(train_x, train_y)  # train it
    # # print(learner.author())

    learner = dt.DTLearner(leaf_size=1, verbose=False)  # constructor
    learner.add_evidence(train_x, train_y)  # training step

    # evaluate in sample
    pred_y = learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    # # print("DT Leaf 1")
    # # print("In sample results")
    # # print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    # # print(f"corr: {c[0, 1]}")
    # # print()


    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    # # print("Out of sample results")
    # # print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    # # print(f"corr: {c[0, 1]}")
    # # print()

    learner = dt.DTLearner(leaf_size=50, verbose=False)  # constructor
    learner.add_evidence(train_x, train_y)  # training step

    # evaluate in sample
    pred_y = learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    # # print("DT Learner Leaf 50")
    # # print("In sample results")
    # # print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    # # print(f"corr: {c[0, 1]}")
    # # print()

    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    # # print()
    # # print("Out of sample results")
    # # print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    # # print(f"corr: {c[0, 1]}")
    # # print()

    learner = rt.RTLearner(leaf_size=1, verbose=False)  # constructor
    learner.add_evidence(train_x, train_y)  # training step

    # evaluate in sample
    pred_y = learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    # # print("RT Learner Leaf 1")
    # # print("In sample results")
    # # print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    # # print(f"corr: {c[0, 1]}")
    # # print()

    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    # # print("Out of sample results")
    # # print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    # # print(f"corr: {c[0, 1]}")
    # # print()

    learner = rt.RTLearner(leaf_size=50, verbose=False)  # constructor
    learner.add_evidence(train_x, train_y)  # training step

    # evaluate in sample
    pred_y = learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    # # print("RT Learner Leaf 50")
    # # print("In sample results")
    # # print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    # # print(f"corr: {c[0, 1]}")
    # # print()

    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    # # print("Out of sample results")
    # # print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    # # print(f"corr: {c[0, 1]}")
    # # print()


if __name__ == "__main__":
    if len(sys.argv) != 2:  		  	   		 	   		  		  		    	 		 		   		 		  
        # # print("Usage: python testlearner.py <filename>")  		  	   		 	   		  		  		    	 		 		   		 		  
        sys.exit(1)

    datapath = sys.argv[1]
    # inf = open(sys.argv[1])
    # data = np.array(
    #     [list(map(float, s.strip().split(","))) for s in inf.readlines()]
    # )
    data = load_data(datapath)
    np.random.seed(903135337)

    initial_testing(data)
    run_experiments(data)