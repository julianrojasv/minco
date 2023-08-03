# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.


from kedro.pipeline import Pipeline, node

from .split import split_data
from .model_input_nodes import filter_target, remove_shut_downs, select_data_clusters, select_data_crisishidrica


def create_pipeline():
    return Pipeline(
        [
            node(
                func=filter_target,
                inputs=dict(
                    params=f"params:model_input", td="td", data="data_all_features"
                ),  # features all data es igual para todos
                outputs="master_target_filtered",
                name="filter_target",
            ),
            node(
                func=remove_shut_downs,
                inputs=dict(
                    params=f"params:model_input", data="master_target_filtered"
                ),
                outputs="master_filtered",
                name="remove_shut_down",
            ),
            node(
                func=split_data,
                inputs=dict(params="params:model_input", data="master_filtered"),
                outputs=dict(train="train_set", test="test_set"),
                name="split_data",
            ),
        ],
        tags="model_input",
    )


def create_pipeline_select_cluster():
    return Pipeline(
        [
            node(
                func=select_data_clusters,
                inputs=dict(
                    params=f"params:model_input", data="data_all_features", data_cluster="data_cluster"
                ),
                outputs="master_target_cluster",
                name="select_data_clusters",
            )
        ],
        tags="model_input_clusters",
    )

def create_pipeline_select_crisishidrica():
    return Pipeline(
        [
            node(
                func=select_data_crisishidrica,
                inputs=dict(
                    params=f"params:model_input", data="data_all_features"
                ),
                outputs="master_target_crisis_hidrica",
                name="select_data_crisis_hidrica",
            )
        ],
        tags="model_input_crisis_hidrica",
    )


########################################################################################################################
############################ MODELO SAG COMIENZO DE EXPORTACIÓN ########################################################
########################################################################################################################

def sag_create_pipeline():
    return Pipeline(
        [
            # TRAIN SET: split train test using features
            node(named_partial(my_engineer.split_X_y, model_output='SAG1'),
                 ['df_no_outliers_imputed_train_th', 'df_no_outliers_imputed_test_th', 'tag_dict_validated_th'],
                 dict(df_X_train='df_X_train_th_line1', df_X_test='df_X_test_th_line1',
                      df_y_train='df_y_train_th_line1', df_y_test='df_y_test_th_line1',
                      df_X='df_X_th_line1', df_y='df_y_th_line1')),
            node(named_partial(my_optimization.update_model_parameter_values, model_output="SAG1"),
                 ['parameters'], 'parameters_th_line1'),

            # TRAIN SET: fit model
            node(named_partial(my_xgb_model.make_boosted_model, model="SAG1"),
                 ['parameters_th_line1'], 'xgb_model_inst_th_line1'),
            node(my_xgb_model.fit, ['xgb_model_inst_th_line1', 'df_X_train_th_line1', 'df_y_train_th_line1',
                                    'parameters_th_line1'], 'xgb_trained_model_th_line1'),

            # TRAIN SET: predict and evaluate model
            node(named_partial(my_xgb_model.predict, model_output='SAG1'),
                 ['xgb_trained_model_th_line1', 'df_X_test_th_line1', 'tag_dict_validated_th'], 'xgb_y_test_pred_th_line1'),
            node(named_partial(my_xgb_model.predict, model_output='SAG1'),
                 ['xgb_trained_model_th_line1', 'df_X_train_th_line1', 'tag_dict_validated_th'], 'xgb_y_train_pred_th_line1'),

            # MODEL EVALUATION
            node(named_partial(my_features.my_assess_regression_single_model_perf, model_name="XGB_throughput_line1"),
                 ['df_y_train_th_line1', 'df_y_test_th_line1', 'df_no_outliers_imputed_train_th',
                  'df_no_outliers_imputed_test_th',
                  'xgb_y_train_pred_th_line1', 'xgb_y_test_pred_th_line1', 'parameters_th_line1'],
                 'xgb_df_model_performance_th_line1'),
            node(named_partial(model_eval.visualize_residuals, model_name="XGB_throughput_line1"),
                 ['df_y_train_th_line1', 'df_y_test_th_line1', 'xgb_y_train_pred_th_line1', 'xgb_y_test_pred_th_line1',
                  'parameters_th_line1'], None),
            node(named_partial(model_eval.visualize_data_predictions, model_name="XGB_throughput_line1"),
                 ['df_y_train_th_line1', 'df_y_test_th_line1', 'xgb_y_train_pred_th_line1', 'xgb_y_test_pred_th_line1',
                  'parameters_th_line1'], None),
            node(named_partial(model_eval.visualize_qq, model_name="XGB_throughput_line1"),
                 ['df_y_train_th_line1', 'df_y_test_th_line1', 'xgb_y_train_pred_th_line1', 'xgb_y_test_pred_th_line1',
                  'parameters_th_line1'], None),
            node(named_partial(my_model_eval.get_feature_importance, model_name="XGB_throughput_line1", model_output='SAG1'),
                 ['xgb_trained_model_th_line1', 'df_X_train_th_line1', 'parameters_th_line1', 'tag_dict_validated_th'],
                 'xgb_feature_importance_th_line1'),
            node(named_partial(my_model_eval.call_plot_learning_curve, model_name="XGB_throughput_line1", model_output='SAG1'),
                 ['xgb_trained_model_th_line1', 'parameters', 'tag_dict_validated_th'], None)
        ],
        tags="model_input",
    )