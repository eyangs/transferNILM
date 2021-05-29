import argparse
import datetime
from remove_space import remove_space
from model_test import Tester

# Allows a model to be tested from the terminal.

dataset = 'REDD'
# dataset = 'UKDALE'
# dataset = 'REFIT'
# You need to input your test data directory
appliances = ['microwave','fridge','dishwasher','washingmachine']
# appliances = ['fridge']
# trained_model_dir = "experiments/exp210304/REFIT/"
trained_model_dir = "experiments/exp210304/{}/retrain/".format(dataset)
test_dir = "experiments/exp210304/data/{}/".format(dataset)
fine_tuning = True
experiment_dir = "experiments/exp210304/"

if fine_tuning:
    experiment_dir = experiment_dir + "{}/transfer/tuning/".format(dataset)
else:
    experiment_dir = experiment_dir + "{}/transfer/".format(dataset)
# test_dir = "/mnt/user/seq2point-nilm/data/REDD123/"
# test_dir = "/mnt/user/seq2point-nilm/data/UKDALE12/"
#parameters: crop and algorithm modified by parser argument
for applianc_name in appliances:

    test_directory= test_dir + "{}_test_.csv".format(applianc_name)
    parser = argparse.ArgumentParser(description="Train a prun;ed neural network for energy disaggregation. ")
    parser.add_argument("--appliance_name", type=remove_space, default=applianc_name, help="The name of the appliance to perform disaggregation with. Default is kettle. Available are: kettle, fridge, dishwasher, microwave. ")
    parser.add_argument("--batch_size", type=int, default="1000", help="The batch size to use when training the network. Default is 1000. ")
    parser.add_argument("--crop", type=int, default=None, help="all testsize")
    # parser.add_argument("--crop", type=int, default="1200000", help="ukdale testsize")
    # parser.add_argument("--crop", type=int, default="200000", help="redd testsize")
    # parser.add_argument("--crop", type=int, default="6000000", help="refit testsize")
    # parser.add_argument("--crop", type=int, default="1200", help="The number of rows of the dataset to take training data from. Default is 10000. ")
    # parser.add_argument("--algorithm", type=remove_space, default="seq2point", help="The pruning algorithm of the model to test. Default is none. ")
    parser.add_argument("--algorithm", type=remove_space, default="t2vattention", help="The pruning algorithm of the model to test. Default is none. ")
    parser.add_argument("--network_type", type=remove_space, default="", help="The seq2point architecture to use. Only use if you do not want to use the standard architecture. Available are: default, dropout, reduced, and reduced_dropout. ")
    parser.add_argument("--input_window_length", type=int, default="599", help="Number of input data points to network. Default is 599. ")
    parser.add_argument("--test_directory", type=str, default=test_directory, help="The dir for training data. ")

    arguments = parser.parse_args()

    # You need to provide the trained model
    saved_model_dir = trained_model_dir + arguments.appliance_name + "_" + arguments.algorithm + "_model.h5"

    # The logs including results will be recorded to this log file
    log_file_dir = experiment_dir  + arguments.algorithm + ".log"
    # log_file_dir = "experiments/exp210215/{}_{}.log".format(arguments.algorithm, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    fig_path = experiment_dir
    tester = Tester(arguments.appliance_name, arguments.algorithm, arguments.crop, 
                    arguments.batch_size, arguments.network_type,
                    arguments.test_directory, saved_model_dir, log_file_dir,fig_path,
                    arguments.input_window_length
                    )
    tester.test_model()

