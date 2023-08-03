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

from .features_nodes import (
    add_across_features_by_hour,
    add_across_features_by_shift,
    create_tag_dict,
    group_by_shift,
    create_target_counts,
    create_target_lags,
)

from .feature_nodes_A2 import add_sag_features_by_hour, add_sag_features_by_shift


def create_pipeline():
    return Pipeline(
        [
            node(
                func=create_tag_dict,
                inputs="tag_dict_master",
                outputs="td",
                tags=["dict"],
            ),
            node(
                func=add_sag_features_by_hour,
                inputs=["parameters", "data_primary"],
                outputs="data_sag_features_by_hour",
            ),
            node(
                func=add_across_features_by_hour,
                inputs=["parameters", "data_sag_features_by_hour"],
                outputs="data_general_features",
            ),
            node(
                func=group_by_shift,
                inputs=["parameters", "data_general_features"],
                outputs="data_aggregated",
            ),
            node(
                func=add_sag_features_by_shift,
                inputs=["parameters", "data_aggregated"],
                outputs="data_sag_features_by_shift",
            ),
            node(
                func=add_across_features_by_shift,
                inputs=["parameters", "data_sag_features_by_shift"],
                outputs="data_features_by_shift",
            ),
            node(
                func=create_target_counts,  # TODO: ML move to features_by_shift
                inputs=[
                    "parameters",
                    "td",
                    "data_sag_features_by_hour",
                    "data_features_by_shift",
                ],
                outputs="data_aggregated_counts",
            ),
            node(
                func=create_target_lags,  # TODO: ML move to features_by_shift
                inputs=["td", "data_aggregated_counts", "parameters"],
                outputs= ["data_all_features","data_all_features_csv"]
            ),
        ],
        tags=["features", "de"],
    )
