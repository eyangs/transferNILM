import argparse
from remove_space import remove_space
from model_train import Trainer
# Allows a model to be trained from the terminal.
# import pysnooper

def main(appliance_name,experiment_directory, data_dir, crop, isTransfer=False, preTrained_model_dir = ''):

    training_directory= data_dir + "{}_training_.csv".format(appliance_name)
    validation_directory= data_dir + "{}_validation_.csv".format(appliance_name)



    parser = argparse.ArgumentParser(description="Train sequence-to-point learning for energy disaggregation. ")

    parser.add_argument("--appliance_name", type=remove_space, default=appliance_name, help="The name of the appliance to train the network with. Default is kettle. Available are: kettle, fridge, washing machine, dishwasher, and microwave. ")
    parser.add_argument("--batch_size", type=int, default="1000", help="The batch size to use when training the network. Default is 1000. ")
    parser.add_argument("--crop", type=int, default=crop, help="parameter")
    # parser.add_argument("--crop", type=int, default="3000000", help="REFIT 3M")
    # parser.add_argument("--crop", type=int, default="1200000", help="UKDALE 1.2M")
    # parser.add_argument("--crop", type=int, default="200000", help="REDD 0.2M")
    #parser.add_argument("--pruning_algorithm", type=remove_space, default="default", help="The pruning algorithm that the network will train with. Default is none. Available are: spp, entropic, threshold. ")
    # parser.add_argument("--network_type", type=remove_space, default="seq2point", help="The seq2point architecture to use. ")
    # parser.add_argument("--network_type", type=remove_space, default="attention", help="The attention architecture to use. ")
    parser.add_argument("--network_type", type=remove_space, default="t2vattention", help="The time2vec_attention architecture to use. ")
    parser.add_argument("--epochs", type=int, default="1000", help="Number of epochs. Default is 10. ")
    parser.add_argument("--input_window_length", type=int, default="599", help="Number of input data points to network. Default is 599.")
    parser.add_argument("--validation_frequency", type=int, default="1", help="How often to validate model. Default is 1. ")
    parser.add_argument("--experiment_directory", type=str, default=experiment_directory, help="The dir for experiment result. ")
    parser.add_argument("--training_directory", type=str, default=training_directory, help="The dir for training data. ")
    parser.add_argument("--validation_directory", type=str, default=validation_directory, help="The dir for validation data. ")

    arguments = parser.parse_args()

    # Need to provide the trained model
    save_model_dir = experiment_directory + arguments.appliance_name + "_" + arguments.network_type + "_model.h5"
    if isTransfer:
        save_model_dir = experiment_directory + '/retrain/' + arguments.appliance_name + "_" + arguments.network_type + "_model.h5"
        preTrained_model_dir = preTrained_model_dir + arguments.appliance_name + "_" + arguments.network_type + "_model.h5"

    trainer = Trainer(arguments.appliance_name, arguments.batch_size, arguments.crop, arguments.network_type,
                    arguments.training_directory, arguments.validation_directory,arguments.experiment_directory, 
                    save_model_dir,
                    epochs = arguments.epochs, input_window_length = arguments.input_window_length,
                    patience = 5,
                    validation_frequency = arguments.validation_frequency)
    # trainer.train_model()
    trainer.train_model(isTransfer=isTransfer, preTrained_model_dir = preTrained_model_dir)

if __name__ == '__main__':
    # appliance_name = 'dishwasher'
    exp_dic = {
        'REDD': {'exp_dir':'experiments/exp210304/REDD/', 'data_dir':'experiments/exp210304/data/REDD/', 'crop':None}, #'crop':'300000'
        # 'UKDALE':{'exp_dir':'experiments/exp210304/UKDALE/', 'data_dir':'experiments/exp210304/data/UKDALE/', 'crop':None}, #crop:737344
        # 'REFIT':{'exp_dir':'experiments/exp210304/REFIT/','data_dir':'experiments/exp210304/data/REFIT/', 'crop':'18000000'} #所有设备中训练集crop：最小18M,最大40M
        # 'REFIT':{'exp_dir':'experiments/exp210304/REFIT/','data_dir':'experiments/exp210304/data/REFIT/', 'crop':None} #所有设备中训练集crop：最小18M,最大40M
    }
    # data_dir = 'data/UKDALE12/'
    appliances = ['microwave','fridge','dishwasher','washingmachine']
    # appliances = ['fridge']
    for key in exp_dic:
        for appName in appliances:
            # main(appName,exp_dic[key]['exp_dir'],exp_dic[key]['data_dir'], exp_dic[key]['crop'])                    
            main(appName,exp_dic[key]['exp_dir'],exp_dic[key]['data_dir'], exp_dic[key]['crop'], isTransfer=True, preTrained_model_dir='experiments/exp210304/REFIT/')                    

