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

from .model_input import model_input_pipeline as mip
from .train_model import train_model_pipeline as tmp
from kedro.pipeline import pipeline, Pipeline, node

########################################################################################################################
############################ MODELO ESP COMIENZO DE EXPORTACIÓN ########################################################
########################################################################################################################
def create_r2_pipeline():
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    r2_model = pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "data_all_features_esp", "td": "td"},
        parameters={"params:model_input": "params:r2_esp.model_input"},
        namespace="r2_esp",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:r2_esp.train_model"},
        namespace="r2_esp",
    )
    r2_model = r2_model.tag(["esp", "r2"])
    return r2_model

def create_r3_pipeline():
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    r3_model = pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "data_all_features_esp", "td": "td"},
        parameters={"params:model_input": "params:r3_esp.model_input"},
        namespace="r3_esp",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:r3_esp.train_model"},
        namespace="r3_esp",
    )
    r3_model = r3_model.tag(["esp", "r3"])
    return r3_model

def create_r4_pipeline():
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    r4_model = pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "data_all_features_esp", "td": "td"},
        parameters={"params:model_input": "params:r4_esp.model_input"},
        namespace="r4_esp",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:r4_esp.train_model"},
        namespace="r4_esp",
    )
    r4_model = r4_model.tag(["esp", "r4"])
    return r4_model

def create_r5_pipeline():
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    r5_model = pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "data_all_features_esp", "td": "td"},
        parameters={"params:model_input": "params:r5_esp.model_input"},
        namespace="r5_esp",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:r5_esp.train_model"},
        namespace="r5_esp",
    )
    r5_model = r5_model.tag(["esp", "r5"])
    return r5_model

def create_r6_pipeline():
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    r6_model = pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "data_all_features_esp", "td": "td"},
        parameters={"params:model_input": "params:r6_esp.model_input"},
        namespace="r6_esp",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:r6_esp.train_model"},
        namespace="r6_esp",
    )
    r6_model = r6_model.tag(["esp", "r6"])
    return r6_model

def create_r7_pipeline():
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    r7_model = pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "data_all_features_esp", "td": "td"},
        parameters={"params:model_input": "params:r7_esp.model_input"},
        namespace="r7_esp",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:r7_esp.train_model"},
        namespace="r7_esp",
    )
    r7_model = r7_model.tag(["esp", "r7"])
    return r7_model

def create_r9_pipeline():
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    r9_model = pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "data_all_features_esp", "td": "td"},
        parameters={"params:model_input": "params:r9_esp.model_input"},
        namespace="r9_esp",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:r9_esp.train_model"},
        namespace="r9_esp",
    )
    r9_model = r9_model.tag(["esp", "r9"])
    return r9_model

def create_r10_pipeline():

    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    r10_model = pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "data_all_features_esp", "td": "td"},
        parameters={"params:model_input": "params:r10_esp.model_input"},
        namespace="r10_esp",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:r10_esp.train_model"},
        namespace="r10_esp",
    )
    r10_model = r10_model.tag(["esp", "r10"])
    return r10_model

################ Pipelines de crisis hidrica #####################
## La crisis hidrica se define con un valor el cual debe de ser superado.

def create_r10_pipeline_crisis_hidrica():
    select_crisis_hidrica = mip.create_pipeline_select_crisishidrica()
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    r10_model_crisis_hidrica = pipeline(
        select_crisis_hidrica,
        inputs={"data_all_features": "data_all_features_esp"},
        parameters={"params:model_input": "params:r10_esp_ch.model_input"},
        namespace="r10_esp_ch",
    ) + pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "r10_esp_ch.master_target_crisis_hidrica", "td": "td"},
        parameters={"params:model_input": "params:r10_esp_ch.model_input"},
        namespace="r10_esp_ch",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:r10_esp_ch.train_model"},
        namespace="r10_esp_ch",
    )
    r10_model_crisis_hidrica = r10_model_crisis_hidrica.tag(["r10_esp_ch", "r10", "esp"])
    return r10_model_crisis_hidrica

def create_r9_pipeline_crisis_hidrica():
    select_crisis_hidrica = mip.create_pipeline_select_crisishidrica()
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    r9_model_crisis_hidrica = pipeline(
        select_crisis_hidrica,
        inputs={"data_all_features": "data_all_features_esp"},
        parameters={"params:model_input": "params:r9_esp_ch.model_input"},
        namespace="r9_esp_ch",
    ) + pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "r9_esp_ch.master_target_crisis_hidrica", "td": "td"},
        parameters={"params:model_input": "params:r9_esp_ch.model_input"},
        namespace="r9_esp_ch",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:r9_esp_ch.train_model"},
        namespace="r9_esp_ch",
    )
    r9_model_crisis_hidrica = r9_model_crisis_hidrica.tag(["r9_esp_ch", "r9", "esp"])
    return r9_model_crisis_hidrica


def create_r7_pipeline_crisis_hidrica():
    select_crisis_hidrica = mip.create_pipeline_select_crisishidrica()
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    r7_model_crisis_hidrica = pipeline(
        select_crisis_hidrica,
        inputs={"data_all_features": "data_all_features_esp"},
        parameters={"params:model_input": "params:r7_esp_ch.model_input"},
        namespace="r7_esp_ch",
    ) + pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "r7_esp_ch.master_target_crisis_hidrica", "td": "td"},
        parameters={"params:model_input": "params:r7_esp_ch.model_input"},
        namespace="r7_esp_ch",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:r7_esp_ch.train_model"},
        namespace="r7_esp_ch",
    )
    r7_model_crisis_hidrica = r7_model_crisis_hidrica.tag(["r7_esp_ch", "r7", "esp"])
    return r7_model_crisis_hidrica


def create_r6_pipeline_crisis_hidrica():
    select_crisis_hidrica = mip.create_pipeline_select_crisishidrica()
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    r6_model_crisis_hidrica = pipeline(
        select_crisis_hidrica,
        inputs={"data_all_features": "data_all_features_esp"},
        parameters={"params:model_input": "params:r6_esp_ch.model_input"},
        namespace="r6_esp_ch",
    ) + pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "r6_esp_ch.master_target_crisis_hidrica", "td": "td"},
        parameters={"params:model_input": "params:r6_esp_ch.model_input"},
        namespace="r6_esp_ch",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:r6_esp_ch.train_model"},
        namespace="r6_esp_ch",
    )
    r6_model_crisis_hidrica = r6_model_crisis_hidrica.tag(["r6_esp_ch", "r6", "esp"])
    return r6_model_crisis_hidrica


def create_r5_pipeline_crisis_hidrica():
    select_crisis_hidrica = mip.create_pipeline_select_crisishidrica()
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    r5_model_crisis_hidrica = pipeline(
        select_crisis_hidrica,
        inputs={"data_all_features": "data_all_features_esp"},
        parameters={"params:model_input": "params:r5_esp_ch.model_input"},
        namespace="r5_esp_ch",
    ) + pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "r5_esp_ch.master_target_crisis_hidrica", "td": "td"},
        parameters={"params:model_input": "params:r5_esp_ch.model_input"},
        namespace="r5_esp_ch",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:r5_esp_ch.train_model"},
        namespace="r5_esp_ch",
    )
    r5_model_crisis_hidrica = r5_model_crisis_hidrica.tag(["r5_esp_ch", "r5", "esp"])
    return r5_model_crisis_hidrica


def create_r4_pipeline_crisis_hidrica():
    select_crisis_hidrica = mip.create_pipeline_select_crisishidrica()
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    r4_model_crisis_hidrica = pipeline(
        select_crisis_hidrica,
        inputs={"data_all_features": "data_all_features_esp"},
        parameters={"params:model_input": "params:r4_esp_ch.model_input"},
        namespace="r4_esp_ch",
    ) + pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "r4_esp_ch.master_target_crisis_hidrica", "td": "td"},
        parameters={"params:model_input": "params:r4_esp_ch.model_input"},
        namespace="r4_esp_ch",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:r4_esp_ch.train_model"},
        namespace="r4_esp_ch",
    )
    r4_model_crisis_hidrica = r4_model_crisis_hidrica.tag(["r4_esp_ch", "r4", "esp"])
    return r4_model_crisis_hidrica


def create_r3_pipeline_crisis_hidrica():
    select_crisis_hidrica = mip.create_pipeline_select_crisishidrica()
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    r3_model_crisis_hidrica = pipeline(
        select_crisis_hidrica,
        inputs={"data_all_features": "data_all_features_esp"},
        parameters={"params:model_input": "params:r3_esp_ch.model_input"},
        namespace="r3_esp_ch",
    ) + pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "r3_esp_ch.master_target_crisis_hidrica", "td": "td"},
        parameters={"params:model_input": "params:r3_esp_ch.model_input"},
        namespace="r3_esp_ch",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:r3_esp_ch.train_model"},
        namespace="r3_esp_ch",
    )
    r3_model_crisis_hidrica = r3_model_crisis_hidrica.tag(["r3_esp_ch", "r3", "esp"])
    return r3_model_crisis_hidrica


def create_r2_pipeline_crisis_hidrica():
    select_crisis_hidrica = mip.create_pipeline_select_crisishidrica()
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    r2_model_crisis_hidrica = pipeline(
        select_crisis_hidrica,
        inputs={"data_all_features": "data_all_features_esp"},
        parameters={"params:model_input": "params:r2_esp_ch.model_input"},
        namespace="r2_esp_ch",
    ) + pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "r2_esp_ch.master_target_crisis_hidrica", "td": "td"},
        parameters={"params:model_input": "params:r2_esp_ch.model_input"},
        namespace="r2_esp_ch",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:r2_esp_ch.train_model"},
        namespace="r2_esp_ch",
    )
    r2_model_crisis_hidrica = r2_model_crisis_hidrica.tag(["r2_esp_ch", "r2", "esp"])
    return r2_model_crisis_hidrica



