from ..exceptions import IllegalArgument, ModelException, PredictionException, AbstractMethodException


class ModelTemplate(object):
    """
    Notes
    -----
    contain data structure ; specify what need to fill from abstract to turn a model concrete
    """
    def __init__(self, response_col='y', date_col='ds', **kwargs):
        # general fields passed into Base Template
        self.response_col = response_col
        self.date_col = date_col

        # basic response fields
        # mainly set by ._set_training_df_meta() and ._set_dynamic_attributes()
        self.response = None
        self.date_array = None
        self.num_of_observations = None
        self.training_start = None
        self.training_end = None
        self._model_data_input = None

        # basic estimator fields
        self.estimator_type = estimator_type
        self.estimator = self.estimator_type(**kwargs)
        self.with_mcmc = None
        # set by ._set_init_values
        # this is ONLY used by stan which by default used 'random'
        self._init_values = None

        # set by _set_model_param_names()
        self._model_param_names = list()

    def predict(self, posterior_estimates, df, include_error=False, **kwargs):
        """Predict interface for users"""
        raise AbstractMethodException("Abstract method.  Model should implement concrete .predict().")
