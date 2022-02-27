import tensorflow as tf
from dataclasses import dataclass
from typing import List, Union
from ML.Helper_Functions import compute_result_info, view_false_positives, view_false_negatives
import numpy as np
import json
from ML.CONSTANSTS import MODEL_EVAL_RESULTS_PATH, PATH_ADDER, TENSORBOARD_LOG_FIT_PATH
import datetime
@dataclass
class MLPipeline:
    """
    Pipeline for training, evaluationg and visualizing results from a ML model.
    Args: 
        ML_model (tf.keras.Model): A compiled ML model.
        data_dict (dict): A dictionary with the following keys and values:
                        {
                            'X_train': np.ndarray,
                            'Y_train': np.ndarray,
                            'X_val': np.ndarray,
                            'Y_val': np.ndarray,
                            'X_unresampled_unscaled': np.ndarray,
                            'X_resample_pred': np.ndarray,
                            'Y_resample_pred': np.ndarray,
                            'X_unresample_pred': np.ndarray,
                            'Y_unresample_pred': np.ndarray
                        }
                        The keys 'X_train', 'Y_train', 'X_val', 'Y_val' and 'X_unresampled_unscaled' are required.
    """
    ML_model: tf.keras.Model
    model_name: str
    data_dict: dict
    _fit: bool=False

    def fit_model(self, epochs: int, callbacks = None, tensorboard: bool=False, tensorboard_kwargs: dict=None):
        """
        Method for fitting model.

        Args:
            epochs (int): Number of epochs to train on.
            callbacks ([tf.keras.callbacks], optional): List of tf.keras.callbacks. Defaults to [tf.keras.callbacks.EarlyStopping(patience=10)].
        """
        self._fit = True

        if callbacks is None:
            callbacks = [tf.keras.callbacks.EarlyStopping(patience=10)]
            if tensorboard:
                log_dir = TENSORBOARD_LOG_FIT_PATH + self.model_name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                if tensorboard_kwargs is None:
                    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))
                else:
                    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, **tensorboard_kwargs))

        self.ML_model.fit(
            x=self.data_dict['X_train'],
            y=self.data_dict['Y_train'],
            epochs=epochs,
            validation_data=(self.data_dict['X_val'], self.data_dict['Y_val']),
            callbacks=callbacks
        )

        self.ML_model_json_config = self.ML_model.to_json(indent=4)

    def evaluate_model(self, view_false_positives: bool=True, view_false_negatives: bool=True, save_file: Union[str, bool]=False, view_pn_kwargs: dict=None):
        """
        Method for evaluating the trained model and visualizing false positives and false negatives for a deeper understanding of the models performance.

        Args:
            view_false_positives (bool, optional): If false positive examples are to be displayed. Defaults to True.
            view_false_negatives (bool, optional): If false negative examples are to be displayed. Defaults to True.
            save_file (Union[str, bool], optional): If the model network is to be saved as a JSON file. Defaults to False.
            view_pn_kwargs (dict, optional): Dict of plotting information to be passed to view_false_positives and view_false_negatives. Defaults to:
                                            view_pn_kwargs = {
                                                                'col_y':5,
                                                                'n_views':3,
                                                                'plot_cols':{
                                                                            'RSI_14':5,
                                                                            'close':3
                                                                            }
                                                                }
        """

        if not self._fit:
            print('Model first has to be trained/fit.')
            return
        
        if view_pn_kwargs is None:
            view_pn_kwargs = {
                'col_y':5,
                'n_views':3,
                'plot_cols':{
                    5:'RSI_14',
                    3:'close'
                }
            }
        
        self._evaluate_model(data_name='Resampled', dict_name='resample_pred')
        self._evaluate_model(data_name='Non resampled', dict_name='unresample_pred', view_false_positives_bool=view_false_positives, view_false_negatives_bool=view_false_negatives, view_pn_kwargs=view_pn_kwargs)

        if isinstance(save_file, str):

            self.save_model(save_file=save_file)
    
    def save_model(self, save_file: str):
        """
        Method for saving the model in a JSON format.

        Args:
            save_directory (str): The name of the JSON file. The directory where it gets saved is ML/Model_Eval_Results.
        """

        if not self._fit:
            print('Model first has to be trained/fit.')
            return
        
        with open(f'{MODEL_EVAL_RESULTS_PATH}{PATH_ADDER}{save_file}.json', 'w') as model_json_file:
                model_json_file.write(self.ML_model_json_config)

        
    
    def _evaluate_model(self, data_name: str, dict_name: str, view_false_positives_bool: bool=False, view_false_negatives_bool: bool=False, view_pn_kwargs: dict=None):

        if f'X_{dict_name}' in self.data_dict and f'Y_{dict_name}' in self.data_dict:

            print('\n\n')
            print(f'{data_name} data test metrics: ')
            y_pred = self.predict(self.data_dict[f'X_{dict_name}'])
            y_correct = self.data_dict[f'Y_{dict_name}']
            compute_result_info(
                y_pred=y_pred,
                y_correct=y_correct
            )
            if view_false_positives_bool:
                view_false_positives(y_pred=y_pred, y_correct=y_correct, data=self.data_dict['X_unresampled_unscaled'], **view_pn_kwargs)
            
            if view_false_negatives_bool:
                view_false_negatives(y_pred=y_pred, y_correct=y_correct, data=self.data_dict['X_unresampled_unscaled'], **view_pn_kwargs)



    
    def predict(self, X_pred: np.ndarray) -> np.ndarray:

        if not self._fit:
            print('Model first has to be trained/fit.')
            return

        return np.rint(self.ML_model.predict(X_pred))[:, 0]



