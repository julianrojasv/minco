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
from kedro.pipeline import pipeline


def create_s13_pipeline():

    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    s13_model = pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "data_all_features", "td": "td"},
        parameters={"params:model_input": "params:s13.model_input"},
        namespace="s13",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:s13.train_model"},
        namespace="s13",
    )
    s13_model = s13_model.tag(["s13", "ma1", "a1"])
    return s13_model


def create_s14_pipeline():

    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    s14_model = pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "data_all_features", "td": "td"},
        parameters={"params:model_input": "params:s14.model_input"},
        namespace="s14",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:s14.train_model"},
        namespace="s14",
    )
    s14_model = s14_model.tag(["s14", "ma1", "a1"])
    return s14_model


def create_s15_pipeline():
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()
    s15_model = pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "data_all_features", "td": "td"},
        parameters={"params:model_input": "params:s15.model_input"},
        namespace="s15",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:s15.train_model"},
        namespace="s15",
    )
    s15_model = s15_model.tag(["s15", "ma1", "a1"])
    return s15_model


def create_fa1l1_pipeline():
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()
    fa1l1_model = pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "data_all_features", "td": "td"},
        parameters={"params:model_input": "params:fa1l1.model_input"},
        namespace="fa1l1",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:fa1l1.train_model"},
        namespace="fa1l1",
    )
    fa1l1_model = fa1l1_model.tag(["fa1l1", "a1"])
    return fa1l1_model


def create_fa1l2_pipeline():
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()
    fa1l2_model = pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "data_all_features", "td": "td"},
        parameters={"params:model_input": "params:fa1l2.model_input"},
        namespace="fa1l2",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:fa1l2.train_model"},
        namespace="fa1l2",
    )
    fa1l2_model = fa1l2_model.tag(["fa1l2", "a1"])
    return fa1l2_model
