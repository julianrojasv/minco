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
# NON-INFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
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


from .raw import raw_pipeline as rp
from .primary import primary_pipeline as pp
from .intermediate import intermediate_pipeline as ip
from .features import features_pipeline as fp
from .features.features_nodes import create_tag_dict
from .features_esp import features_esp_pipeline as fep
from .curation import curation_pipelines as dcp

from kedro.pipeline import Pipeline, node


def create_pipeline():

#    raw_pipeline = rp.create_pipeline()
#    intermediate_pipeline = ip.create_pipeline()
    data_curation_pipeline = dcp.create_pipeline()
#    primary_pipeline = pp.create_pipeline()
#    features_pipeline = fp.create_pipeline()
    features_esp_pipeline = fep.create_pipeline()

#    all_pipelines = (
#         raw_pipeline
#         + intermediate_pipeline
#         + data_curation_pipeline
#         + primary_pipeline
#         + features_pipeline
#    )

    all_pipelines = (
#       data_curation_pipeline
     #  + primary_pipeline
     #  + features_pipeline
         features_esp_pipeline
    )
    return all_pipelines


def create_pipeline_dict():
    return Pipeline(
        [
            node(
                func=create_tag_dict,
                inputs="tag_dict_master",
                outputs="td",
                tags=["dict"],
            )])
