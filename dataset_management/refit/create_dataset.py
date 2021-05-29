from dataset_infos import *
import pandas as pd
import time
import os
import re
import argparse

DATA_DIRECTORY = '/mnt/user/nilm/data/CLEAN_REFIT/'
SAVE_PATH = '/mnt/user/seq2point-nilm/experiments/exp210304/data/REFIT/'

#dincecco_transfer_2019使用的参数，main线路上使用了统一的均值和标准差,REDD，REFIT，UKDALE都使用同一值
AGG_MEAN = 522
AGG_STD = 814
# APPLIANCE_NAME = 'washingmachine'
# APPLIANCE_NAME = 'fridge'
# APPLIANCE_NAME = 'dishwasher'
APPLIANCE_NAME = 'microwave'

def get_arguments():
    parser = argparse.ArgumentParser(description='sequence to point learning \
                                     example for NILM')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                          help='The directory containing the CLEAN REFIT data')
    parser.add_argument('--appliance_name', type=str, default=APPLIANCE_NAME,
                          help='which appliance you want to train: kettle,\
                          microwave,fridge,dishwasher,washingmachine')
    parser.add_argument('--aggregate_mean',type=int,default=AGG_MEAN,
                        help='Mean value of aggregated reading (mains)')
    parser.add_argument('--aggregate_std',type=int,default=AGG_STD,
                        help='Std value of aggregated reading (mains)')
    parser.add_argument('--save_path', type=str, default=SAVE_PATH,
                          help='The directory to store the training data')
    return parser.parse_args()


def load(path, building, appliance, channel):

    # load csv
    file_name = path + 'CLEAN_House' + str(building) + '.csv'
    single_csv = pd.read_csv(file_name,
                             header=0,
                             names=['aggregate', appliance],
                             usecols=[2, channel+2],
                             na_filter=False,
                             parse_dates=True,
                             infer_datetime_format=True,
                             memory_map=True
                             )

    return single_csv


def main():
    start_time = time.time()        
    # test path
    # path = '../../../data/refit/CLEAN_REFIT_081116/'
    # save_path = 'refitdata/'
    
    args = get_arguments()
    
    appliance_name = args.appliance_name
    print(appliance_name)
    
    path = args.data_dir
    save_path = args.save_path
    # if not os.path.exists(appliance_name):
    #     os.makedirs(appliance_name)
    # save_path = appliance_name + '/'
    print(path)
    aggregate_mean = args.aggregate_mean#522
    aggregate_std = args.aggregate_std#814  
    
    total_length = 0
    print("Starting creating dataset...")
    # Looking for proper files
    for idx, filename in enumerate(os.listdir(path)):
        single_step_time = time.time()
    
        if filename == 'CLEAN_House' + str(params_appliance[appliance_name]['test_house']) + '.csv':
            print('File: ' + filename + ' test set')
            # Loading
            test = load(path,
                 params_appliance[appliance_name]['test_house'],
                 appliance_name,
                 params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses']
                        .index(params_appliance[appliance_name]['test_house'])]
                 )
    
            # Normalization
            test['aggregate'] = (test['aggregate'] - aggregate_mean) / aggregate_std
            test[appliance_name] = \
                (test[appliance_name] - params_appliance[appliance_name]['mean']) / params_appliance[appliance_name]['std']
    
            # Save
            test.to_csv(save_path + appliance_name + '_test_' + 'H' + str(params_appliance[appliance_name]['test_house'])
                        + '.csv', index=False)
    
            print("Size of test set is {:.3f} M rows (House {:d})."
                  .format(test.shape[0] / 10 ** 6, params_appliance[appliance_name]['test_house']))
            del test
    
        elif filename == 'CLEAN_House' + str(params_appliance[appliance_name]['validation_house']) + '.csv':
            print('File: ' + filename + ' validation set')
            # Loading
            val = load(path,
                 params_appliance[appliance_name]['validation_house'],
                 appliance_name,
                 params_appliance[appliance_name]['channels']
                 [params_appliance[appliance_name]['houses']
                        .index(params_appliance[appliance_name]['validation_house'])]
                 )
    
            # Normalization
            val['aggregate'] = (val['aggregate'] - aggregate_mean) / aggregate_std
            val[appliance_name] = \
                (val[appliance_name] - params_appliance[appliance_name]['mean']) / params_appliance[appliance_name]['std']
    
            # Save
            val.to_csv(save_path + appliance_name + '_validation_' + 'H' + str(params_appliance[appliance_name]['validation_house'])
                       + '.csv', index=False)
    
            print("Size of validation set is {:.3f} M rows (House {:d})."
                  .format(val.shape[0] / 10 ** 6, params_appliance[appliance_name]['validation_house']))
            del val
    
        elif int(re.search(r'\d+', filename).group()) in params_appliance[appliance_name]['houses']:
            print('File: ' + filename)
            print('    House: ' + re.search(r'\d+', filename).group())
    
            # Loading
            try:
                csv = load(path,
                           int(re.search(r'\d+', filename).group()),
                           appliance_name,
                           params_appliance[appliance_name]['channels']
                           [params_appliance[appliance_name]['houses']
                                  .index(int(re.search(r'\d+', filename).group()))]
                           )
    
                # Normalization
                csv['aggregate'] = (csv['aggregate'] - aggregate_mean) / aggregate_std
                csv[appliance_name] = \
                    (csv[appliance_name] - params_appliance[appliance_name]['mean']) / params_appliance[appliance_name][
                        'std']
    
                rows, columns = csv.shape
                total_length += rows
    
                if filename == 'CLEAN_House' + str(params_appliance[appliance_name]['test_on_train_house']) + '.csv':
                    csv.to_csv(save_path + appliance_name + '_test_on_train_' + 'H' + str(
                        params_appliance[appliance_name]['test_on_train_house'])
                               + '.csv', index=False)
                    print("Size of test on train set is {:.3f} M rows (House {:d})."
                          .format(csv.shape[0] / 10 ** 6, params_appliance[appliance_name]['test_on_train_house']))
    
                # saving the whole merged file
                csv.to_csv(save_path + appliance_name + '_training_.csv', mode='a', index=False, header=False)
    
                del csv
    
            except:
                pass
    
            #print('    total_partial length: {}'.format(total_length / 10 ** 6))
    
    print("Size of training set is {:.3f} M rows.".format(total_length / 10 ** 6))
    print("\nNormalization parameters: ")
    print("Mean and standard deviation values USED for AGGREGATE are:")
    print("    Mean = {:d}, STD = {:d}".format(aggregate_mean, aggregate_std))
    print('Mean and standard deviation values USED for ' + appliance_name + ' are:')
    print("    Mean = {:d}, STD = {:d}"
          .format(params_appliance[appliance_name]['mean'], params_appliance[appliance_name]['std']))
    print("\nTraining, validation and test sets are  in: " + save_path)
    print("Total elapsed time: {:.2f} min.".format((time.time() - start_time) / 60))
    
if __name__ == '__main__':
    main()
