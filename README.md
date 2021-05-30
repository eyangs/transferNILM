# transferNILM
the code for the paper "Nonintrusive Residential Electricity Load Decomposition Based on Transfer Learning"

redd, ukdale, refit dataset should be download to data path.
The scripts for data pre-processing are in the data_management folder
model is defined in model_structure.py
steps for experiment:
- 1. prepare dataset for training and testing.
- 2. train or fine-tuning model by running the script file: train_main.py
- 3. test model by running the script file: test_main.py
- 4. using the pre-trained model to another dataset by running the script file: transfer.py  



reference:
- https://github.com/MingjunZhong/seq2point-nilm
- Time2Vec https://arxiv.org/abs/1907.05321"
