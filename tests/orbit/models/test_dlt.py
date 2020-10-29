import pytest
from orbit.models.dlt import BaseDLT, DLTFull, DLTAggregated, DLTMAP
from orbit.estimators.stan_estimator import StanEstimatorMCMC, StanEstimatorVI, StanEstimatorMAP
from orbit.utils.features import make_fourier_series_df
import numpy as np


def test_base_dlt_init():
    dlt = BaseDLT()

    is_fitted = dlt.is_fitted()

    model_data_input = dlt._get_model_data_input()
    model_param_names = dlt._get_model_param_names()
    init_values = dlt._get_init_values()

    assert not is_fitted  # model is not yet fitted
    assert not model_data_input  # should only be initialized and not set
    assert model_param_names  # model param names should already be set
    # todo: change when init_values callable is implemented
    assert not init_values


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC, StanEstimatorVI])
def test_dlt_full_univariate(synthetic_data, estimator_type):
    train_df, test_df, coef = synthetic_data

    dlt = DLTFull(
        response_col='response',
        date_col='week',
        prediction_percentiles=[5, 95],
        seasonality=52,
        num_warmup=50,
        verbose=False,
        estimator_type=estimator_type
    )

    dlt.fit(train_df)
    predict_df = dlt.predict(test_df)

    expected_columns = ['week', 'prediction_lower', 'prediction', 'prediction_upper']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 13

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(dlt._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC, StanEstimatorVI])
def test_dlt_aggregated_univariate(synthetic_data, estimator_type):
    train_df, test_df, coef = synthetic_data

    dlt = DLTAggregated(
        response_col='response',
        date_col='week',
        seasonality=52,
        num_warmup=50,
        verbose=False,
        estimator_type=estimator_type
    )

    dlt.fit(train_df)
    predict_df = dlt.predict(test_df)

    expected_columns = ['week', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 13

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(dlt._posterior_samples) == expected_num_parameters


def test_dlt_map_univariate(synthetic_data):
    train_df, test_df, coef = synthetic_data

    dlt = DLTMAP(
        response_col='response',
        date_col='week',
        seasonality=52,
        num_warmup=50,
        verbose=False,
    )

    dlt.fit(train_df)
    predict_df = dlt.predict(test_df)

    expected_columns = ['week', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 12  # no `lp__` parameter in optimizing()

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(dlt._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC, StanEstimatorVI])
def test_dlt_non_seasonal_fit(synthetic_data, estimator_type):
    train_df, test_df, coef = synthetic_data

    dlt = DLTFull(
        response_col='response',
        date_col='week',
        estimator_type=estimator_type
    )

    dlt.fit(train_df)
    predict_df = dlt.predict(test_df)

    expected_columns = ['week', 'prediction_lower', 'prediction', 'prediction_upper']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 11

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(dlt._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC, StanEstimatorVI])
@pytest.mark.parametrize(
    "regressor_signs",
    [
        ["+", "+", "+", "+", "+", "+"],
        ["=", "=", "=", "=", "=", "="],
        ["+", "=", "+", "=", "+", "+"]
    ],
    ids=['positive_only', 'regular_only', 'mixed_signs']
)
def test_dlt_full_with_regression(synthetic_data, estimator_type, regressor_signs):
    train_df, test_df, coef = synthetic_data

    dlt = DLTFull(
        response_col='response',
        date_col='week',
        regressor_col=train_df.columns.tolist()[2:],
        regressor_sign=regressor_signs,
        prediction_percentiles=[5, 95],
        seasonality=52,
        num_warmup=50,
        verbose=False,
        estimator_type=estimator_type
    )

    dlt.fit(train_df)
    predict_df = dlt.predict(test_df)

    regression_out = dlt.get_regression_coefs()
    num_regressors = regression_out.shape[0]

    expected_columns = ['week', 'prediction_lower', 'prediction', 'prediction_upper']
    expected_shape = (51, len(expected_columns))
    expected_regression_shape = (6, 3)

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert regression_out.shape == expected_regression_shape
    assert num_regressors == len(train_df.columns.tolist()[2:])


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC, StanEstimatorVI])
@pytest.mark.parametrize(
    "regressor_signs",
    [
        ["+", "+", "+", "+", "+", "+"],
        ["=", "=", "=", "=", "=", "="],
        ["+", "=", "+", "=", "+", "+"]
    ],
    ids=['positive_only', 'regular_only', 'mixed_signs']
)
def test_dlt_aggregated_with_regression(synthetic_data, estimator_type, regressor_signs):
    train_df, test_df, coef = synthetic_data

    dlt = DLTAggregated(
        response_col='response',
        date_col='week',
        regressor_col=train_df.columns.tolist()[2:],
        regressor_sign=regressor_signs,
        seasonality=52,
        num_warmup=50,
        verbose=False,
        estimator_type=estimator_type
    )

    dlt.fit(train_df)
    predict_df = dlt.predict(test_df)

    regression_out = dlt.get_regression_coefs()
    num_regressors = regression_out.shape[0]

    expected_columns = ['week', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_regression_shape = (6, 3)

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert regression_out.shape == expected_regression_shape
    assert num_regressors == len(train_df.columns.tolist()[2:])


@pytest.mark.parametrize("global_trend_option", ["linear", "loglinear", "logistic", "flat"])
def test_dlt_map_global_trend(synthetic_data, global_trend_option):
    train_df, test_df, coef = synthetic_data

    dlt = DLTMAP(
        response_col='response',
        date_col='week',
        seasonality=52,
        global_trend_option=global_trend_option,
    )

    dlt.fit(train_df)
    predict_df = dlt.predict(test_df)

    expected_columns = ['week', 'prediction']
    expected_shape = (51, len(expected_columns))
    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns


@pytest.mark.parametrize("global_trend_option", ["linear", "loglinear", "logistic", "flat"])
def test_dlt_map_global_trend(synthetic_data, global_trend_option):
    train_df, test_df, coef = synthetic_data

    dlt = DLTMAP(
        response_col='response',
        date_col='week',
        seasonality=52,
        global_trend_option=global_trend_option,
    )

    dlt.fit(train_df)
    predict_df = dlt.predict(test_df)

    expected_columns = ['week', 'prediction']
    expected_shape = (51, len(expected_columns))
    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns


def test_dlt_predict_all_positive_reg(iclaims_training_data):
    df = iclaims_training_data

    dlt = DLTMAP(
        response_col='claims',
        date_col='week',
        regressor_col=['trend.unemploy', 'trend.filling', 'trend.job'],
        regressor_sign=['+', '+', '+'],
        seasonality=52,
        seed=8888,
    )

    dlt.fit(df)
    predicted_df = dlt.predict(df, decompose=True)

    assert any(predicted_df['regression'].values)


def test_dlt_complex_seasonality(m5_agg_data):
    df = m5_agg_data[-1000:]
    regressor_col = ["Christmas", "Halloween", "LaborDay", "Thanksgiving", "Mother's day", "PresidentsDay", "NewYear"]
    df, fs_cols = make_fourier_series_df(df, 'date', period=365.25, order=3)
    m = df['sales'][0]
    df['y'] = np.log(df['sales'] / m)

    dlt = DLTMAP(
        response_col='y',
        date_col='date',
        seasonality=7,
        seed=2020,
        regressor_col=fs_cols + regressor_col,
    )

    dlt.fit(df)
    predicted_df = dlt.predict(df)
    predicted_df['prediction'] = np.exp(predicted_df['prediction']) * m
    err = np.mean(np.abs(predicted_df['prediction'] - df['sales'])/df['sales'])
    expected_shape = (len(df.index), 2)
    assert predicted_df.shape == expected_shape
    # in-sample mape less than 50%
    assert err < 0.5


