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

from .features_esp_nodes import *

def create_pipeline():
    return Pipeline(
        [
            node(
                func=parse_column_names,
                inputs="data_det",
                #inputs="data_corrected_csv",
                outputs="data_norm",
                name="parse_columns_esp",
            ),
            node(
                func=remove_outliers,
                inputs="data_norm",
                outputs="data_filtrada",
                name="remove_outliers",
            ),
            node(
                func=data_imputation,
                inputs="data_filtrada",
                outputs="data_general",
                name="data_imputation",
            ),
            node(
                func=create_features,
                inputs=dict(
                    params="params:general_tags_esp", dataset="data_general"
                ),  # features all data es igual para todos
                outputs="data_features_general",
                name="create_features",
            ),
            node(
                func=data_group_per_hour,
                inputs="data_features_general",
                outputs="data_all_features_grouped",
                name="group_per_hour",
            ),
            node(
                func=add_on_off_features_by_hour,
                inputs=["parameters", "data_all_features_grouped"],
                outputs="data_all_features_esp",
            ),
        ],
        tags="engineering_esp",
    )
