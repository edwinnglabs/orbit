from ..exceptions import IllegalArgument, ModelException, PredictionException, AbstractMethodException
from ..models.model_template import ModelTemplate
import custom_inherit as ci
from copy import copy, deepcopy
import numpy as np
import pandas as pd

from ..constants.constants import PredictMethod, PredictionKeys
from ..estimators.stan_estimator import StanEstimatorMCMC
from ..utils.docstring_style import merge_numpy_docs_dedup
from ..utils.predictions import prepend_date_column, compute_percentiles
from ..exceptions import IllegalArgument, ModelException, PredictionException, AbstractMethodException
from ..utils.general import is_ordered_datetime


class Forecaster(object):
    def __init__(self, response_col='y', date_col='ds', estimator_type=StanEstimatorMCMC, **kwargs):
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

        self._validate_supported_estimator_type()
        self._set_with_mcmc()

        # set by _set_model_param_names()
        self._model_param_names = list()

        # set by `fit()`
        self._posterior_samples = dict()
        # init aggregate posteriors
        self._aggregate_posteriors = dict()

        # for full Bayesian, user can store full prediction array if requested
        self.prediction_array = None
        self.prediction_input_meta = dict()

        # storing metrics in training result meta
        self._training_metrics = dict()

    def fit(self, df):
        pass

    def predict(self, df, **kwargs):
        """Predict interface for users"""
        raise AbstractMethodException("Abstract method.  Model should implement concrete .predict().")


def MAPForecasterBuilder(model):
    assert isinstance(model, ModelTemplate)
    forecaster = Forecaster()
    # n_bootstrap_draws here only to provide empirical prediction percentiles;
    # mid-point estimate is always replaced
    forecaster.n_bootstrap_draws = n_bootstrap_draws
    forecaster.prediction_percentiles = prediction_percentiles
    forecaster._prediction_percentiles = None

    # unlike full prediction, it does not take negative number of bootstrap draw
    # if self.n_bootstrap_draws < 2:
    #     raise IllegalArgument("Error: Number of bootstrap draws must be at least 2")
    if forecaster.prediction_percentiles is None:
        forecaster._prediction_percentiles = [5, 95]
    else:
        forecaster._prediction_percentiles = copy(forecaster.prediction_percentiles)

    forecaster._prediction_percentiles += [50]  # always find median
    forecaster._prediction_percentiles = list(set(forecaster._prediction_percentiles))  # unique set
    forecaster._prediction_percentiles.sort()

    # override init aggregate posteriors
    forecaster._aggregate_posteriors = {PredictMethod.MAP.value: dict()}
