# Copyright (c) 2016 - present
# QuantumBlack Visual Analytics Ltd (a McKinsey company).
# All rights reserved.
#
# This software framework contains the confidential and proprietary information
# of QuantumBlack, its affiliates, and its licensors. Your use of these
# materials is governed by the terms of the Agreement between your organisation
# and QuantumBlack, and any unauthorised use is forbidden. Except as otherwise
# stated in the Agreement, this software framework is for your internal use
# only and may only be shared outside your organisation with the prior written
# permission of QuantumBlack.
"""
Nodes of the model training pipeline.
"""
import logging
import datetime
from copy import deepcopy
from typing import Any, Dict
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline as SklearnPipeline
from xgboost import XGBRegressor

from project_clisham.optimus_core import utils
from project_clisham.optimus_core.model_helpers.performance import (
    generate_prediction_metrics,
    tree_feature_importance,
)
from project_clisham.optimus_core.model_helpers.tuning import sklearn_tune, xgb_tune
from project_clisham.optimus_core.reporting_html.nodes import create_html_report

from project_clisham.optimus_core.tag_management import TagDict
from project_clisham.optimus_core.transformers import NumExprEval, SelectColumns

logger = logging.getLogger(__name__)

SUPPORTED_MODEL_FEATURE_IMPORTANCE = {"tree": tree_feature_importance}


def load_regressor(params: dict):
    """
    Loads a regressor object based on given parameters.
    Args:
        params: dictionary of parameters
    Returns:
        sklearn compatible model
    """
    model_class = params["regressor"]["class"]
    model_kwargs = params["regressor"]["kwargs"]
    regressor = utils.load_obj(model_class)(**model_kwargs)
    assert hasattr(regressor, "fit"), "Model object must have a .fit method"
    assert hasattr(regressor, "predict"), "Model object must have a .predict method"
    return regressor


def add_transformers(params: dict, td: TagDict, regressor: Any):
    """
    Creates a sklearn model pipeline based on the regressor and adds
    the desired transformers. This is where things like imputation,
    scaling, feature selection, and dynamic feature generation should plug in.
    Args:
        params: dictionary of parameters
        td: tag dictionary
        regressor: regressor object
    Returns:
        sklearn model pipeline with transformers
    """

    # Transformer which reduces the model input to the
    # relevant features
    model_feature = params["dict_model_feature"]
    feat_cols = td.select(model_feature)
    column_selector = SelectColumns(feat_cols)

    model = SklearnPipeline(
        [
            ("select_columns", column_selector),
            ("regressor", regressor),
        ]
    )

    return model


def train_tree_model(
    params: dict, td: TagDict, data: pd.DataFrame, model: SklearnPipeline
) -> Dict[str, Any]:
    """
    Args:
        params: dictionary of parameters
        td: tag dictionary
        data: input data
        model: sklearn pipeline with regressor and transformers
    Returns:
        Trained sklearn compatible model
        Hyper-parameter Tuning Search Results
        Feature importance
    """
    return _train_model(params, td, data, model, "tree")


def _train_model(
    params: dict,
    td: TagDict,
    data: pd.DataFrame,
    model: SklearnPipeline,
    model_type: str,
) -> Dict[str, Any]:
    """
    Args:
        params: dictionary of parameters
        td: tag dictionary
        data: input data
        model: sklearn pipeline with regressor and transformers
        model_type: string used for determining feature importance
            Supported values: ["tree"]
    Returns:
        Trained sklearn compatible model
        Hyper-parameter Tuning Search Results
        Feature importance
    """
    model_target = params["dict_model_target"]
    model_feature = params["dict_model_feature"]
    target_col = td.select("target", model_target)[0]
    target = data[target_col]

    # strictly speaking, selection features should not be necessary
    # as this is done by the model transformer. However, we do it
    # regardless to reduce our memory footprint and protect against
    # accidental removal of the transformer
    feat_cols = td.select(model_feature)
    feature_df = data[feat_cols]

    regressor = model.named_steps["regressor"]
    if isinstance(regressor, XGBRegressor):
        logger.info("Tuning using `xgb_tune`.")
        tuned_model, cv_results_df = xgb_tune(params, feature_df, target, model)
    else:
        logger.info("Tuning using `sklearn_tune`.")
        tuned_model, cv_results_df = sklearn_tune(params, feature_df, target, model)

    feature_importance = SUPPORTED_MODEL_FEATURE_IMPORTANCE.get(model_type)
    importances = feature_importance(tuned_model)

    # print(importances)
    # exit()

    return dict(
        model=tuned_model, cv_results=cv_results_df, feature_importance=importances
    )


def create_predictions(
    params: dict, td: TagDict, data: pd.DataFrame, model: SklearnPipeline
) -> Dict[str, pd.DataFrame]:
    """
    Creates model predictions for a given data set
    Args:
        params: dictionary of parameters
        td: tag dictionary
        data: input data
        model: sklearn pipeline with regressor and transformers
    Returns:
        predictions, metrics
    """
    model_target = params["dict_model_target"]
    prediction_col = "prediction"
    predictions = model.predict(data)
    target_col = td.select("target", model_target)[0]
    res_df = data.copy()
    if predictions.shape[0] != data.shape[0]:
        missing_rows = data.shape[0] - predictions.shape[0]
        res_df = res_df.iloc[missing_rows:, :]
    res_df[prediction_col] = predictions
    prediction_metrics_df = pd.DataFrame()
    prediction_metrics_df["opt_perf_metrics"] = generate_prediction_metrics(
        res_df, target_col, prediction_col
    )
    print(prediction_metrics_df)
    return dict(predictions=res_df, metrics=prediction_metrics_df)


def retrain_tree_model(
    td: TagDict, model: SklearnPipeline, data: pd.DataFrame
) -> SklearnPipeline:
    """
    Retraining the model object with the new dataset.
    Args:
        td: tag dictionary
        model: sklearn pipeline with regressor and transformers
        data: input data

    Returns:
        retrained SklearnPipeline model

    """
    target_col = td.select("target", model_target)[0]
    target = data[target_col]

    retrain_model = deepcopy(model)
    retrain_model.fit(data, target)
    return retrain_model


def generate_performance_report(
    params: Dict,
    kedro_env: str,
    test_predictions,  # to force node order
):  # pylint:disable=unused-argument
    namespace = params["namespace"]
    template_path = Path(params["report"])
    reporting_dir = Path(params["report_dir"])
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    html_report_filename = f"{namespace}_training_perf_report_{timestamp_str}.html"
    report_params = {
        "template_path": template_path,
        "output_dir": reporting_dir,
        "report_name": html_report_filename,
        "namespace": namespace,
        "remove_code": True,
    }
    try:
        create_html_report(
            report_params,
            kedro_env=kedro_env,
        )
    except RuntimeError as err:
        # keep going if the report fails
        logger.warning("Could not generate performance report: {}".format(str(err)))


########################################################################################################################
############################ MODELO SAG COMIENZO DE EXPORTACIÓN ########################################################
########################################################################################################################

# TRAIN SET: split train test using features
def split_X_y(df_train: pd.DataFrame, df_test: pd.DataFrame, data_dict_obj, model_output=None) -> dict:
    """ split data into train and test with X Y separated, also returns X and Y dataframes. Filters the columns
        depending on the model

      Args:
          df_train: train test
          df_test: test set
          data_dict_obj: dictionary of the variables
          parameters: parameters to train with
          model_output: model to be traned

      Returns:
          dict:
              df_X_train:
              df_X_test:
              df_y_train:
              df_y_test:
              df_X:
              df_y:
      """

    if model_output == 'All':
        tag_X_list = list(data_dict_obj.get_all_model_tags())
        tag_y_list = data_dict_obj.get_targets()
    else:
        tag_X_list = data_dict_obj.get_submodel_features(model_output)
        tag_y_list = data_dict_obj.get_submodel_targets(model_output)

    # make sure these features/targets are still in the dataset, hasn't been dropped by poor quality filter yet
    tag_X_list = [x for x in tag_X_list if x in df_train.columns.tolist()]
    tag_y_list = [x for x in tag_y_list if x in df_train.columns.tolist()]

    # import ipdb; ipdb.set_trace();
    # assert len(tag_y_list) == 1, 'There should be just a single target, Dictionary has {}'.format(len(tag_y_list))

    df_X_train = df_train[tag_X_list]
    df_X_test = df_test[tag_X_list]
    df_y_train = df_train[tag_y_list]
    df_y_test = df_test[tag_y_list]

    if model_output == "SAG1":
        df_y_train = df_y_train[df_y_train[tag_y_list[0]] > np.percentile(df_y_train[tag_y_list[0]], 10)]
        df_X_train = df_X_train.loc[df_y_train.index]

    if model_output == "SAG2":
        df_y_train = pd.concat([df_y_train.loc[:dt.date(2019, 2, 1)], df_y_train.loc[dt.date(2019, 6, 1):]], axis=0)
        df_X_train = df_X_train.loc[df_y_train.index]

    df_X = df_X_train.append(df_X_test)
    df_X = df_X[~df_X.index.duplicated(keep='last')]
    df_y = df_y_train.append(df_y_test)
    df_y = df_y[~df_y.index.duplicated(keep='last')]

    return dict(
        df_X_train=df_X_train,
        df_X_test=df_X_test,
        df_y_train=df_y_train,
        df_y_test=df_y_test,
        df_X=df_X,
        df_y=df_y
    )

def update_model_parameter_values(parameters, model_output="Throughput"):
    """
        updates the parameters to be used for each model

        Args:
            parameters: tph model
            model_output: recovery model
        """
    if model_output == "Recovery":
        parameters["model"]["xgb_search_space_random_search"]["max_depth"] = [2]
        parameters["model"]["xgb_search_space_random_search"]["n_estimators"] = [300]  # 120
        parameters["model"]["xgb_search_space_random_search"]["min_child_weight"] = [5]
        # parameters["model"]["xgb_search_space_random_search"]["gamma"] = [0]
        parameters["model"]["xgb_search_space_random_search"]["learning_rate"] = [0.09]
        parameters["model"]["xgb_search_space_random_search"]["subsample"] = [0.9]
        parameters["model"]["xgb_search_space_random_search"]["colsample_bytree"] = [0.95]
        # parameters["model"]["xgb_search_space_random_search"]["tree_method"] = ['exact']
    if model_output == "SAG1":
        parameters["model"]["xgb_search_space_random_search"]["max_depth"] = [2]
        parameters["model"]["xgb_search_space_random_search"]["n_estimators"] = [700]
        parameters["model"]["xgb_search_space_random_search"]["subsample"] = [0.8]
        parameters["model"]["xgb_search_space_random_search"]["learning_rate"] = [0.12]
        # parameters["model"]["xgb_search_space_random_search"]["min_child_weight"] = [20]
    if model_output == "SAG2":
        parameters["model"]["xgb_search_space_random_search"]["max_depth"] = [3]
        parameters["model"]["xgb_search_space_random_search"]["n_estimators"] = [200]
        # parameters["model"]["xgb_search_space_random_search"]["subsample"] = [0.9]
        parameters["model"]["xgb_search_space_random_search"]["learning_rate"] = [0.2]
        # parameters["model"]["xgb_search_space_random_search"]["min_child_weight"] = [1]
        # parameters["model"]["xgb_search_space_random_search"]["colsample_bytree"] = [0.9]
    return parameters

# TRAIN SET: fit model
def make_boosted_model(parameters, model='Recovery'):
    """
    Wraps the boosted model
    :param parameters:
    :return:
    """
    logger.info("Starting execution of make_quick_model")

    bayes_search_hyper_tune = utils.get_params_key(parameters, 'model', 'bayes_search_hyper_tune')
    random_search_hyper_tune = utils.get_params_key(parameters, 'model', 'random_search_hyper_tune')
    quick_boosted_model_args = utils.get_params_key(parameters, 'model', 'quick_boosted_model_args')

    n_iter = utils.get_params_key(parameters, 'model', 'tune_n_iter')
    cv_splits = utils.get_params_key(parameters, 'model', 'tune_cv_splits')

    if model == 'SAG1':
        model_inst = XGBoostFixer(objective=myobj_sag1)
    else:
        if model == 'SAG2':
            model_inst = XGBoostFixer(objective=myobj_sag2)
        else:
            model_inst = XGBoostFixer()
    # model_inst = XGBoostFixer()

    # build
    if bayes_search_hyper_tune:
        search_space_dict_list = utils.get_params_key(parameters, 'model', 'xgb_search_space_bayes_search')
        search_space_dict_tuple = dict()
        for key, value in search_space_dict_list.items():
            search_space_dict_tuple[key] = tuple(value)

        model_inst = bayes_search_cv(model_inst, search_space=search_space_dict_tuple,
                                     n_iter=n_iter, cv_splits=cv_splits)
    elif random_search_hyper_tune:
        search_space_dict_list = utils.get_params_key(parameters, 'model', 'xgb_search_space_random_search')
        search_space_dict_tuple = dict()
        for key, value in search_space_dict_list.items():
            search_space_dict_tuple[key] = tuple(value)

        model_inst = random_search_cv(model_inst, search_space=search_space_dict_tuple,
                                      n_iter=n_iter, cv_splits=cv_splits)

    elif (bayes_search_hyper_tune + random_search_hyper_tune) in [0, 2]:
        model_inst = XGBoostFixer(**quick_boosted_model_args)

    return model_inst

def fit(model_inst, X, y, parameters):
    """
    Wraps around XGBQuick Boost, BayesearchCV and RandomizedSearchCV fit methods
    :param model_inst:
    :param X:
    :param y:
    :param parameters:
    :return:
    """

    assert y.shape[1] == 1, 'y must be single col DataFrame; Received {} columns'.format(y.shape[1])

    bayes_search_hyper_tune = utils.get_params_key(parameters, 'model', 'bayes_search_hyper_tune')
    random_search_hyper_tune = utils.get_params_key(parameters, 'model', 'random_search_hyper_tune')
    tune = False
    if bayes_search_hyper_tune|random_search_hyper_tune:
        tune = True

    if tune:
        model_inst.fit(X, y)
        cv_results = pandas.DataFrame(model_inst.cv_results_)
        charts_folder1 = 'results/modelling'
        if not os.path.exists(charts_folder1):
            os.makedirs(charts_folder1)
        cv_results.to_csv('results/modelling/cv_results_recovery.csv')
#
    else:
        # Split the data into 80% - 20% : Helps generate learning curves
        split = X.shape[0] // 5
        X_train, X_test = X[:-split], X[-split:]
        y_train, y_test = y[:-split], y[-split:]
        y_train = y_train.values.flatten()
        y_test = y_test.values.flatten()


        model_inst.fit(X_train, y_train,
                       early_stopping_rounds=10,
                       eval_set=[(X_train, y_train),
                                 (X_test, y_test)],
                       verbose=False)
    print(model_inst.get_params())
    print(model_inst.best_params_)
    return model_inst

# TRAIN & TEST SET: predict and evaluate model
def predict(model, X, data_dict_obj, model_output=None):
    #import ipdb;
    #ipdb.set_trace();
    """
    Wraps around XGBQuick Boost, BayesearchCV and RandomizedSearchCV predict methods
    :return:
    """
    tag_y_list = data_dict_obj.get_submodel_targets(model_output)
    # tag_y_list = data_dict_obj.get_targets()
    assert len(tag_y_list)==1, 'Single target should be present in DataDictionary; ' \
                               'Received {}'.format(len(tag_y_list))

    p = pandas.DataFrame(model.predict(X), index=X.index,
                         columns=[tag_y_list[0]])
    return p

# MODEL EVALUATION
def my_assess_regression_single_model_perf(y_train, y_test, dataset_train, dataset_test, train_preds, test_preds,
                                           parameters,
                                           model_name=None) -> None:
    """Assess a regression model, by calculating:
         - MSE
         - RMSE
         - MAE
         - Explained variance
         - R squared
         - MAPE
    Assumes that y vaJlues have multiple columns, ie, that this is a multi-output
    prediciton problem, and will take the uniform average of all evalaution metrics.
    :param y_train:
    :param y_test:
    :param train_preds:
    :param test_preds:
    :param model_name: denotes the folder name (mod~$
el object which called this)
    :param parameters:
    :return: A dataframe, with rows of metrics and colums of [train, test]
    """
    if model_name is None:
        raise TypeError('Pass name of model in model_name')

    output_fs = utils.make_results_osfs(parameters)
    model_output_fs = output_fs.makedirs('modelling/model_performance/{}'.format(model_name), recreate=True)

    metric_tasks = {
        'mean of output': mean_output,
        'std of output': std_output,
        'mean squared error': metrics.mean_squared_error,
        'root mean squared error': rmse,
        'mean absolute error': metrics.mean_absolute_error,
        'explaned variance score': metrics.explained_variance_score,
        'r2': metrics.r2_score,
        'mean absolute percentage error': mean_absolute_percentage_error,
    }

    train_scores = {}
    test_scores = {}

    for col in y_train.columns:
        for name, metric_fn in metric_tasks.items():
            train_scores[name] = metric_fn(
                y_train[col], train_preds[col])
            test_scores[name] = metric_fn(
                y_test[col], test_preds[col])

        train_scores = pd.Series(train_scores, name='train')
        test_scores = pd.Series(test_scores, name='test')
        all_scores = pd.concat([train_scores, test_scores], 1)

        name = 'model_performance_{}.csv'.format(col)
        with model_output_fs.open(name, 'w') as f:
            all_scores.to_csv(f)

    if model_name == 'XGB_recovery':

        train_scores = {}
        test_scores = {}

        target_rec_tag = train_preds.columns[0]

        ley_alim_train = dataset_train['LEY_ALIMENTACION_CALCULATED']
        ley_conc_train = dataset_train['LAB-Q:LEY_CU_CONC_FINAL_SAG.TURNO']
        if target_rec_tag == 'LAB-Q:LEY_CU_COLA_SAG.TURNO_delta':
            ley_cola_pred_train = dataset_train['LAB-Q:LEY_CU_COLA_SAG.TURNO_lag1'] + \
                                  train_preds['LAB-Q:LEY_CU_COLA_SAG.TURNO_delta']
        else:
            ley_cola_pred_train = train_preds[target_rec_tag]
        recovery_pred_train = ((ley_alim_train - ley_cola_pred_train) * ley_conc_train) / (
                (ley_conc_train - ley_cola_pred_train) * ley_alim_train)

        ley_alim_test = dataset_test['LEY_ALIMENTACION_CALCULATED']
        ley_conc_test = dataset_test['LAB-Q:LEY_CU_CONC_FINAL_SAG.TURNO']
        if target_rec_tag == 'LAB-Q:LEY_CU_COLA_SAG.TURNO_delta':
            ley_cola_pred_test = dataset_test['LAB-Q:LEY_CU_COLA_SAG.TURNO_lag1'] + \
                                 test_preds['LAB-Q:LEY_CU_COLA_SAG.TURNO_delta']
        else:
            ley_cola_pred_test = test_preds[target_rec_tag]

        recovery_pred_test = ((ley_alim_test - ley_cola_pred_test) * ley_conc_test) / (
                (ley_conc_test - ley_cola_pred_test) * ley_alim_test)

        for name, metric_fn in metric_tasks.items():
            train_scores[name] = metric_fn(
                dataset_train['recovery_FLOTATION'], recovery_pred_train)
            test_scores[name] = metric_fn(
                dataset_test['recovery_FLOTATION'], recovery_pred_test)

        train_scores = pd.Series(train_scores, name='train')
        test_scores = pd.Series(test_scores, name='test')
        all_scores = pd.concat([train_scores, test_scores], 1)

        name = 'model_performance_recovery_FLOTATION.csv'
        with model_output_fs.open(name, 'w') as f:
            all_scores.to_csv(f)

def visualize_residuals(y_train, y_test, train_preds, test_preds, parameters, model_name=None) -> None:
    """Make residual plots, and output to the the results_base_dir
    :param y_train:
    :param y_test:
    :param train_preds:
    :param test_preds:
    :param model_name: denotes the folder name (model object which called this)
    :param parameters: Requires the following items:
            - results_base_dir: the base location for results
            - model_viz.resid_plot_size: residual figure size, passed to figure
    """
    logger.info("Starting execution of visualize_residuals")
    import matplotlib.pyplot as plt

    if model_name is None:
        raise TypeError('Pass name of model in model_name')

    output_fs = utils.make_results_osfs(parameters)
    model_output_fs = output_fs.makedirs('modelling/residuals/{}'.format(model_name), recreate=True)

    fig_opts = {
        'figsize': utils.get_params_key(parameters, 'plot_size'),
    }

    if isinstance(y_train, pandas.Series):
        fig = _plot_resid(y_train, y_test, train_preds, test_preds, fig_opts)
        with model_output_fs.open('residuals_plot.png', 'wb') as f:
            fig.savefig(f, type='png')
        fig.close()
    else:
        for col in y_train.columns:
            fig = _plot_resid(y_train[col], y_test[col], train_preds[col], test_preds[col], fig_opts)
            name = 'residuals_plot_{}.png'.format(col)
            with model_output_fs.open(name, 'wb') as f:
                fig.savefig(f, type='png')
    plt.close()

def visualize_data_predictions(y_train, y_test, train_preds, test_preds, parameters, model_name=None) -> None:
    #import ipdb; ipdb.set_trace()
    """Make residual plots, and output to the the results_base_dir
    :param y_train:
    :param y_test:
    :param train_preds:
    :param test_preds:
    :param model_name: denotes the folder name (model object which called this)
    :param parameters: Requires the following items:
            - results_base_dir: the base location for results
            - model_viz.resid_plot_size: residual figure size, passed to figure
    """
    logger.info("Starting execution of visualize_data_predictions")
    import matplotlib.pyplot as plt

    if model_name is None:
        raise TypeError('Pass name of model in model_name')

    output_fs = utils.make_results_osfs(parameters)
    model_output_fs = output_fs.makedirs('modelling/predictions/'.format(model_name), recreate=True)

    fig_opts = {
        'figsize': utils.get_params_key(parameters, 'plot_size'),
    }
    for col in y_train.columns:
        fig_train = _plot_pred(y_train[col], train_preds[col], fig_opts)
        name_tr = 'predictions_plot_train_{}.png'.format(col)
        plt.tight_layout()
        with model_output_fs.open(name_tr, 'wb') as f:
            fig_train.savefig(f, type='png', bbox_inches='tight')
        plt.close()
        fig_test = _plot_pred_test(y_test[col], test_preds[col], fig_opts)
        name_te = 'predictions_plot_test_{}.png'.format(col)
        plt.tight_layout()
        with model_output_fs.open(name_te, 'wb') as f:
            fig_test.savefig(f, type='png', bbox_inches='tight')
        plt.close()

def visualize_qq(y_train, y_test, train_preds, test_preds, parameters, model_name=None):
    """Make Q-Q plots, and output to the the results_base_dir
    :param y_train:
    :param y_test:
    :param train_preds:
    :param test_preds:
    :param model_name: denotes the folder name (model object which called this)
    :param parameters: Requires the following items:
              - results_base_dir: the base location for results
              - model_viz.resid_plot_size: residual figure size, passed to figure
      """
    import matplotlib.pyplot as plt

    if model_name is None:
        raise TypeError('Pass name of model in model_name')

    output_fs = OSFS(parameters['results_base_dir'])
    model_output_fs = output_fs.makedirs('modelling/QQ Plots/{}'.format(model_name), recreate=True)

    fig_opts = {
        'figsize': utils.get_params_key(parameters, 'plot_size'),
    }
    if isinstance(y_train, pandas.Series):
        fig = _plot_qq(y_train, y_test, train_preds, test_preds, fig_opts)
        with model_output_fs.open('qq_plot.png', 'wb') as f:
            fig.savefig(f, type='png')
            plt.close()
    else:
        for col in y_train.columns:
            fig = _plot_qq(y_train[col], y_test[col], train_preds[col], test_preds[col], fig_opts)
            name = 'qq_plot_{}.png'.format(col)
            with model_output_fs.open(name, 'wb') as f:
                fig.savefig(f, type='png')
            plt.close()

def get_feature_importance(model, X, parameters: dict, data_dict_obj, model_name=None):
    """

    :param model: model object
    :param X:
    :param parameters:
    :param data_dict_obj: data dictionary carries target name
    :param model_name:
    :return:
    """
    logger.info("Starting execution of plot_xgboost_feature_importances")
    import matplotlib.pyplot as plt
    if model_name is None:
        raise TypeError('Pass name of model in model_name')

    output_fs = utils.make_results_osfs(parameters)
    model_output_fs = output_fs.makedirs('modelling/importances/{}'.format(model_name), recreate=True)

    # Extract the target name
    tag_y_list = data_dict_obj.get_targets()

    # Check for hyper_tune; then obj.best_estimator.feature_imp_
    bayes_search_hyper_tune = utils.get_params_key(parameters, 'model', 'bayes_search_hyper_tune')
    random_search_hyper_tune = utils.get_params_key(parameters, 'model', 'random_search_hyper_tune')
    tune = False
    if bayes_search_hyper_tune | random_search_hyper_tune:
        tune = True

    features = X.columns.tolist()
    if tune:
        importances = model.best_estimator_.feature_importances_
    else:
        importances = model.feature_importances_

    # Create the feature importance  data
    indices = np.argsort(importances)
    feat_imp_dt = pandas.DataFrame({'features': features,
                                    'scores': importances,
                                    'indices': indices}).sort_values('scores', ascending=False)
    feat_imp_dt = feat_imp_dt.iloc[:20, :]
    n = feat_imp_dt.shape[0]
    fname = 'feat_importance_{}.png'.format(tag_y_list[0])
    with model_output_fs.open(fname, 'wb') as f:
        plt.figure(figsize=utils.get_params_key(parameters, 'plot_size'))

        plt.title('{} most important features for prediction of {}'.format(n, tag_y_list[0]))
        plt.barh(range(n), feat_imp_dt.scores[::-1], color='C0', align='center')
        plt.yticks(range(n), feat_imp_dt.features[::-1])

        plt.gca().get_xaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(f, format='png')
        plt.close()

    return feat_imp_dt

def call_plot_learning_curve(model, parameters, data_dict_obj, model_name=None):
    """
    Only to check if model is a BayesSeacrhCV / RandomizedSearchCV object :
    If yes, then dont plot learning curve since no eval_result will be present.. While fitting eval_set was not passed
    :param model: model object
    :param data_dict_obj: carries the target name
    :param parameters:
    :return:
    """

    # Check for hyper_tune; then obj.best_estimator.feature_imp_
    bayes_search_hyper_tune = utils.get_params_key(parameters, 'model', 'bayes_search_hyper_tune')
    random_search_hyper_tune = utils.get_params_key(parameters, 'model', 'random_search_hyper_tune')

    if bayes_search_hyper_tune + random_search_hyper_tune == 0:
        plot_learning_curve(model, parameters, data_dict_obj, model_name)
    else:
        logger.info('Tuned model objects has no eval results(passed while fitting)')
