import tensorflow as tf
import numpy as np
from SFN.SFNBase import SFN


class SFHPS(SFN):
    """
    Selective Forgettable Hard Parameter Sharing MTL

    Caruana, R. "Multitask learning: A knowledge-based source of inductive bias."
    International Conference on Machine Learning. 1993.
    """

    def __init__(self, config):
        super(SFHPS, self).__init__(config)

    def train_hps(self):
        pass

    def predict_only_after_training(self) -> list:
        pass

    def reconstruct_model(self):
        pass

    def recover_params(self, idx):
        pass

    def selective_forget(self, task_to_forget, number_of_neurons, policy) -> tuple:
        pass

    def get_importance_vector(self, task_id, importance_criteria: str, layer_separate=False) -> tuple or np.ndarray:
        pass

    def _retrain_at_task(self, task_id, data, retrain_flags, is_verbose):
        pass

    def _assign_retrained_value_to_tensor(self, task_id):
        pass

    def assign_new_session(self):
        pass
