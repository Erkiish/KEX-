import os

DIRECTORY_BASE_PATH = os.getcwd()
PATH_ADDER = '/' if '/' in DIRECTORY_BASE_PATH else '\\'
ML_PATH = DIRECTORY_BASE_PATH + PATH_ADDER + 'ML'
HP_TUNER_RESULT_PATH = ML_PATH + PATH_ADDER + 'HP_Tuner_Results'
MODEL_EVAL_RESULTS_PATH = ML_PATH + PATH_ADDER + 'Model_Eval_Results'
TENSORBOARD_LOG_FIT_PATH = ML_PATH + PATH_ADDER + 'TensorBoard_logs' + PATH_ADDER + 'fit_logs' + PATH_ADDER