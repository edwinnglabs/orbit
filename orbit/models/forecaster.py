from ..models.model_template import ModelTemplate
from copy import copy, deepcopy
import numpy as np
import pandas as pd

from ..constants.constants import PredictMethod, PredictionKeys
from ..utils.predictions import prepend_date_column, compute_percentiles
from ..exceptions import IllegalArgument, ModelException, PredictionException
from ..utils.general import is_ordered_datetime


class MAPForecaster(object):
    def __init__(self,
                 model,
                 estimator_type,
                 response_col='y',
                 date_col='ds',
                 n_bootstrap_draws=1e4,
                 prediction_percentiles=None,
                 **kwargs):

        # general fields passed into Base Template
        assert isinstance(model, ModelTemplate)
        self._model = model
        self.response_col = response_col
        self.date_col = date_col
        self.estimator_type = estimator_type
        self.estimator = self.estimator_type(**kwargs)

        self.n_bootstrap_draws = n_bootstrap_draws
        self.prediction_percentiles = prediction_percentiles
        self._prediction_percentiles = None
        if self.prediction_percentiles is None:
            self._prediction_percentiles = [5, 95]
        else:
            self._prediction_percentiles = copy(self.prediction_percentiles)

        self._prediction_percentiles += [50]  # always find median
        self._prediction_percentiles = list(set(self._prediction_percentiles))  # unique set
        self._prediction_percentiles.sort()

        # basic response fields
        # mainly set by ._set_training_df_meta() and ._set_dynamic_attributes()
        self._training_meta = dict()
        self._training_data_input = dict()

        self._posterior_samples = dict()
        self._point_posteriors = {PredictMethod.MAP.value: dict()}
        self._training_metrics = None

        self._prediction_meta = dict()

    # TODO: get this back later
    def _validate_training_df(self, df):
        pass

    def _set_training_meta(self, df):
        training_meta = dict()
        response = df[self.response_col].values
        training_meta['response'] = response
        training_meta['date_array'] = pd.to_datetime(df[self.date_col]).reset_index(drop=True)
        training_meta['num_of_observations'] = len(response)
        training_meta['response_sd'] = np.nanstd(response)
        training_meta['training_start'] = df[self.date_col].iloc[0]
        training_meta['training_end'] = df[self.date_col].iloc[-1]
        self._training_meta = training_meta

    def get_training_meta(self):
        return deepcopy(self._training_meta)

    def set_training_data_input(self):
        """Collects data attributes into a dict for sampling/optimization api"""
        # refresh a clean dict
        data_input_mapper = self._model.get_data_input_mapper()
        if not data_input_mapper:
            raise ModelException('Empty or invalid data_input_mapper')

        # always get standard input from training
        training_meta = self.get_training_meta()
        training_data_input = {
            'RESPONSE': training_meta['response'],
            'RESPONSE_SD': training_meta['response_sd'],
            'NUM_OF_OBS': training_meta['num_of_observations'],
            'WITH_MCMC': 0,
        }

        for key in data_input_mapper:
            # mapper keys in upper case; inputs in lower case
            key_lower = key.name.lower()
            input_value = getattr(self._model, key_lower, None)
            if input_value is None:
                raise ModelException('{} is missing from data input'.format(key_lower))
            if isinstance(input_value, bool):
                # stan accepts bool as int only
                input_value = int(input_value)
            training_data_input[key.value] = input_value

        self._training_data_input = training_data_input

    def is_fitted(self):
        # if either aggregate posterior and posterior_samples are non-empty, claim it as fitted model (true),
        # else false.
        return bool(self._posterior_samples) or bool(self._point_posteriors)

    def fit(self, df):
        estimator = self.estimator
        model_name = self._model.get_model_name()
        df = df.copy()

        # default set and validation of input data frame
        # self._validate_training_df(df)
        # extract standard training metadata
        self._set_training_meta(df)
        # based on the model and df, set training input
        self.set_training_data_input()
        # if model provide initial values, set it
        self._model.set_init_values()

        # estimator inputs
        data_input = self.get_training_data_input()
        init_values = self._model.get_init_values()
        model_param_names = self._model.get_model_param_names()

        # note that estimator will search for the .stan, .pyro model file based on the
        # estimator type and model_name provided
        _posterior_samples, training_metrics = estimator.fit(
            model_name=model_name,
            model_param_names=model_param_names,
            data_input=data_input,
            fitter=None,
            init_values=init_values
        )

        self._posterior_samples = _posterior_samples
        self._training_metrics = training_metrics

        posterior_samples = self._posterior_samples
        map_posterior = {}
        for param_name in self._model.get_model_param_names():
            param_array = posterior_samples[param_name]
            # add dimension so it works with vector math in `_predict`
            param_array = np.expand_dims(param_array, axis=0)
            map_posterior.update({param_name: param_array})

        self._point_posteriors[PredictMethod.MAP.value] = map_posterior

    def _set_prediction_meta(self, df):
        # remove reference from original input
        df = df.copy()

        # get prediction df meta
        prediction_meta = {
            'date_array': pd.to_datetime(df[self.date_col]).reset_index(drop=True),
            'df_length': len(df.index),
            'prediction_start': df[self.date_col].iloc[0],
            'prediction_end': df[self.date_col].iloc[-1],
        }

        if not is_ordered_datetime(prediction_meta['date_array']):
            raise IllegalArgument('Datetime index must be ordered and not repeat')

        # TODO: validate that all regressor columns are present, if any

        if prediction_meta['prediction_start'] < self._training_meta['training_start']:
            raise PredictionException('Prediction start must be after training start.')

        trained_len = self._training_meta['num_of_observations']

        # If we cannot find a match of prediction range, assume prediction starts right after train
        # end
        if prediction_meta['prediction_start'] > self._training_meta['training_end']:
            forecast_dates = set(prediction_meta['date_array'])
            n_forecast_steps = len(forecast_dates)
            # time index for prediction start
            start = trained_len
        else:
            # compute how many steps to forecast
            forecast_dates = \
                set(prediction_meta['date_array']) - set(self._training_meta['date_array'])
            # check if prediction df is a subset of training df
            # e.g. "negative" forecast steps
            n_forecast_steps = len(forecast_dates) or - (
                len(set(self._training_meta['date_array']) - set(prediction_meta['date_array']))
            )
            # time index for prediction start
            start = pd.Index(
                self._training_meta['date_array']).get_loc(prediction_meta['prediction_start'])

        prediction_meta.update({
            'start': start,
            'n_forecast_steps': n_forecast_steps,
        })

        self._prediction_meta = prediction_meta

    def get_prediction_meta(self):
        return deepcopy(self._prediction_meta)

    def predict(self, df, decompose=False, **kwargs):
        # raise if model is not fitted
        if not self.is_fitted():
            raise PredictionException("Model is not fitted yet.")
        # obtain basic meta data from input df
        self._set_prediction_meta(df)
        prediction_meta = self.get_prediction_meta()
        training_meta = self.get_training_meta()

        # perform point prediction
        point_posteriors = self._point_posteriors.get(PredictMethod.MAP.value)
        point_predicted_dict = self._model.predict(
            posterior_estimates=point_posteriors,
            df=df,
            training_meta=training_meta,
            prediction_meta=prediction_meta,
            # false for point estimate
            include_error=False,
            **kwargs
        )
        for k, v in point_predicted_dict.items():
            point_predicted_dict[k] = np.squeeze(v, 0)

        # to derive confidence interval; the condition should be sufficient since we add [50] by default
        if self.n_bootstrap_draws > 0 and len(self._prediction_percentiles) > 1:
            # perform bootstrap; we don't have posterior samples. hence, we just repeat the draw here.
            posterior_samples = {}
            for k, v in point_posteriors.items():
                posterior_samples[k] = np.repeat(v, self.n_bootstrap_draws, axis=0)
            predicted_dict = self._model.predict(
                posterior_estimates=posterior_samples,
                df=df,
                training_meta=training_meta,
                prediction_meta=prediction_meta,
                include_error=True,
                **kwargs
            )
            percentiles_dict = compute_percentiles(predicted_dict, self._prediction_percentiles)
            # replace mid point prediction by point estimate
            percentiles_dict.update(point_predicted_dict)

            if PredictionKeys.PREDICTION.value not in percentiles_dict.keys():
                raise PredictionException("cannot find the key:'{}' from return of _predict()".format(
                    PredictionKeys.PREDICTION.value))

            # reduce to prediction only if decompose is not requested
            if not decompose:
                k = PredictionKeys.PREDICTION.value
                reduced_keys = [k + "_" + str(p) if p != 50 else k for p in self._prediction_percentiles]
                percentiles_dict = {k: v for k, v in percentiles_dict.items() if k in reduced_keys}
            predicted_df = pd.DataFrame(percentiles_dict)
        else:
            predicted_df = pd.DataFrame(point_predicted_dict)

        predicted_df = prepend_date_column(predicted_df, df, self.date_col)
        return predicted_df

    def get_training_data_input(self):
        return self._training_data_input

