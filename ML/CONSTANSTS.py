import os

directory_base_path = os.getcwd()
PATH_ADDER = '/' if '/' in directory_base_path else '\\'
ML_PATH = directory_base_path + PATH_ADDER + 'ML'
HP_TUNER_RESULT_PATH = ML_PATH + PATH_ADDER + 'HP_Tuner_Results'
MODEL_EVAL_RESULTS_PATH = ML_PATH + PATH_ADDER + 'Model_Eval_Results'