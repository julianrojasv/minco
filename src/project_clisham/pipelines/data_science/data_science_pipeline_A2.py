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
from .clusterization_model import clusterization_model_pipeline as cls

def create_clusterization_pipeline():

    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()
    clusterization_model_pipeline = cls.create_pipeline()

    return clusterization_model_pipeline


def create_s16_pipeline():
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    s16_model = pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "data_all_features", "td": "td"},
        parameters={"params:model_input": "params:s16.model_input"},
        namespace="s16",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:s16.train_model"},
        namespace="s16",
    )
    s16_model = s16_model.tag(["s16", "ma2", "a2"])
    return s16_model


def create_s17_pipeline():
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()
    s17_model = pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "data_all_features", "td": "td"},
        parameters={"params:model_input": "params:s17.model_input"},
        namespace="s17",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:s17.train_model"},
        namespace="s17",
    )
    s17_model = s17_model.tag(["s17", "ma2", "a2"])
    return s17_model


def create_fa2l1_pipeline():
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()
    fa2l1_model = pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "data_all_features", "td": "td"},
        parameters={"params:model_input": "params:fa2l1.model_input"},
        namespace="fa2l1",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:fa2l1.train_model"},
        namespace="fa2l1",
    )
    fa2l1_model = fa2l1_model.tag(["fa2l1", "a2"])
    return fa2l1_model


def create_fa2l2_pipeline():
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()
    fa2l2_model = pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "data_all_features", "td": "td"},
        parameters={"params:model_input": "params:fa2l2.model_input"},
        namespace="fa2l2",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:fa2l2.train_model"},
        namespace="fa2l2",
    )
    fa2l2_model = fa2l2_model.tag(["fa2l2", "a2"])
    return fa2l2_model


def create_fa2l3_pipeline():
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()
    fa2l3_model = pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "data_all_features", "td": "td"},
        parameters={"params:model_input": "params:fa2l3.model_input"},
        namespace="fa2l3",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:fa2l3.train_model"},
        namespace="fa2l3",
    )
    fa2l3_model = fa2l3_model.tag(["fa2l3", "a2"])
    return fa2l3_model


def create_fa2lg_pipeline():
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()
    fa2lg_model = pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "data_all_features", "td": "td"},
        parameters={"params:model_input": "params:fa2lg.model_input"},
        namespace="fa2lg",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:fa2lg.train_model"},
        namespace="fa2lg",
    )
    fa2lg_model = fa2lg_model.tag(["fa2lg", "a2"])

    return fa2lg_model


########################################################################################################################
############################ MODELO SAG COMIENZO DE EXPORTACIÓN ########################################################
########################################################################################################################
def create_sag1_pipeline():
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    sag1_model = pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "data_all_features", "td": "td"},
        parameters={"params:model_input": "params:sag1.model_input"},
        namespace="sag1",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:sag1.train_model"},
        namespace="sag1",
    )
    sag1_model = sag1_model.tag(["sag1", "msag", "sag"])
    return sag1_model


def create_sag2_pipeline():
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    sag2_model = pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "data_all_features", "td": "td"},
        parameters={"params:model_input": "params:sag2.model_input"},
        namespace="sag2",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:sag2.train_model"},
        namespace="sag2",
    )
    sag2_model = sag2_model.tag(["sag2", "msag", "sag"])
    return sag2_model


def create_fsag_pipeline():
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()
    fsag_model = pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "data_all_features", "td": "td"},
        parameters={"params:model_input": "params:fsag.model_input"},
        namespace="fsag",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:fsag.train_model"},
        namespace="fsag",
    )
    fsag_model = fsag_model.tag(["fsag", "sag"])

    return fsag_model

########################################################################################################################
############################         MODELO SAG CLUSTERS        ########################################################
########################################################################################################################

def create_sag1_pipeline_cluster_3():
    select_cluster = mip.create_pipeline_select_cluster()
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    sag1_model_cluster = pipeline(
        select_cluster,
        inputs={"data_all_features": "data_all_features", "data_cluster": "data_cluster3"},
        parameters={"params:model_input": "params:sag1_cluster3.model_input"},
        namespace="sag1_cluster3",
    ) + pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "sag1_cluster3.master_target_cluster", "td": "td"},
        parameters={"params:model_input": "params:sag1_cluster3.model_input"},
        namespace="sag1_cluster3",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:sag1_cluster3.train_model"},
        namespace="sag1_cluster3",
    )
    sag1_model_cluster = sag1_model_cluster.tag(["sag1_cluster3", "msag", "sag"])
    return sag1_model_cluster


def create_sag1_pipeline_cluster_2():
    select_cluster = mip.create_pipeline_select_cluster()
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    sag1_model_cluster = pipeline(
        select_cluster,
        inputs={"data_all_features": "data_all_features", "data_cluster": "data_cluster2"},
        parameters={"params:model_input": "params:sag1_cluster2.model_input"},
        namespace="sag1_cluster2",
    ) + pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "sag1_cluster2.master_target_cluster", "td": "td"},
        parameters={"params:model_input": "params:sag1_cluster2.model_input"},
        namespace="sag1_cluster2",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:sag1_cluster2.train_model"},
        namespace="sag1_cluster2",
    )
    sag1_model_cluster = sag1_model_cluster.tag(["sag1_cluster2", "msag", "sag"])
    return sag1_model_cluster


def create_sag1_pipeline_cluster_1():
    select_cluster = mip.create_pipeline_select_cluster()
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    sag1_model_cluster = pipeline(
        select_cluster,
        inputs={"data_all_features": "data_all_features", "data_cluster": "data_cluster1"},
        parameters={"params:model_input": "params:sag1_cluster1.model_input"},
        namespace="sag1_cluster1",
    ) + pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "sag1_cluster1.master_target_cluster", "td": "td"},
        parameters={"params:model_input": "params:sag1_cluster1.model_input"},
        namespace="sag1_cluster1",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:sag1_cluster1.train_model"},
        namespace="sag1_cluster1",
    )
    sag1_model_cluster = sag1_model_cluster.tag(["sag1_cluster1", "msag", "sag"])
    return sag1_model_cluster


def create_sag1_pipeline_cluster_0():
    select_cluster = mip.create_pipeline_select_cluster()
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    sag1_model_cluster = pipeline(
        select_cluster,
        inputs={"data_all_features": "data_all_features", "data_cluster": "data_cluster0"},
        parameters={"params:model_input": "params:sag1_cluster0.model_input"},
        namespace="sag1_cluster0",
    ) + pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "sag1_cluster0.master_target_cluster", "td": "td"},
        parameters={"params:model_input": "params:sag1_cluster0.model_input"},
        namespace="sag1_cluster0",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:sag1_cluster0.train_model"},
        namespace="sag1_cluster0",
    )
    sag1_model_cluster = sag1_model_cluster.tag(["sag1_cluster0", "msag", "sag"])
    return sag1_model_cluster


def create_sag2_pipeline_cluster_3():
    select_cluster = mip.create_pipeline_select_cluster()
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    sag2_model_cluster = pipeline(
        select_cluster,
        inputs={"data_all_features": "data_all_features", "data_cluster": "data_cluster3"},
        parameters={"params:model_input": "params:sag2_cluster3.model_input"},
        namespace="sag2_cluster3",
    ) + pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "sag2_cluster3.master_target_cluster", "td": "td"},
        parameters={"params:model_input": "params:sag2_cluster3.model_input"},
        namespace="sag2_cluster3",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:sag2_cluster3.train_model"},
        namespace="sag2_cluster3",
    )
    sag2_model_cluster = sag2_model_cluster.tag(["sag2_cluster3", "msag", "sag"])
    return sag2_model_cluster


def create_sag2_pipeline_cluster_2():
    select_cluster = mip.create_pipeline_select_cluster()
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    sag2_model_cluster = pipeline(
        select_cluster,
        inputs={"data_all_features": "data_all_features", "data_cluster": "data_cluster2"},
        parameters={"params:model_input": "params:sag2_cluster2.model_input"},
        namespace="sag2_cluster2",
    ) + pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "sag2_cluster2.master_target_cluster", "td": "td"},
        parameters={"params:model_input": "params:sag2_cluster2.model_input"},
        namespace="sag2_cluster2",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:sag2_cluster2.train_model"},
        namespace="sag2_cluster2",
    )
    sag2_model_cluster = sag2_model_cluster.tag(["sag2_cluster2", "msag", "sag"])
    return sag2_model_cluster


def create_sag2_pipeline_cluster_1():
    select_cluster = mip.create_pipeline_select_cluster()
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    sag2_model_cluster = pipeline(
        select_cluster,
        inputs={"data_all_features": "data_all_features", "data_cluster": "data_cluster1"},
        parameters={"params:model_input": "params:sag2_cluster1.model_input"},
        namespace="sag2_cluster1",
    ) + pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "sag2_cluster1.master_target_cluster", "td": "td"},
        parameters={"params:model_input": "params:sag2_cluster1.model_input"},
        namespace="sag2_cluster1",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:sag2_cluster1.train_model"},
        namespace="sag2_cluster1",
    )
    sag2_model_cluster = sag2_model_cluster.tag(["sag2_cluster1", "msag", "sag"])
    return sag2_model_cluster


def create_sag2_pipeline_cluster_0():
    select_cluster = mip.create_pipeline_select_cluster()
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    sag2_model_cluster = pipeline(
        select_cluster,
        inputs={"data_all_features": "data_all_features", "data_cluster": "data_cluster0"},
        parameters={"params:model_input": "params:sag2_cluster0.model_input"},
        namespace="sag2_cluster0",
    ) + pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "sag2_cluster0.master_target_cluster", "td": "td"},
        parameters={"params:model_input": "params:sag2_cluster0.model_input"},
        namespace="sag2_cluster0",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:sag2_cluster0.train_model"},
        namespace="sag2_cluster0",
    )
    sag2_model_cluster = sag2_model_cluster.tag(["sag2_cluster0", "msag", "sag"])
    return sag2_model_cluster


def create_fsag_pipeline_cluster_3():
    select_cluster = mip.create_pipeline_select_cluster()
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    fsag_model_cluster = pipeline(
        select_cluster,
        inputs={"data_all_features": "data_all_features", "data_cluster": "data_cluster3"},
        parameters={"params:model_input": "params:fsag_cluster3.model_input"},
        namespace="fsag_cluster3",
    ) + pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "fsag_cluster3.master_target_cluster", "td": "td"},
        parameters={"params:model_input": "params:fsag_cluster3.model_input"},
        namespace="fsag_cluster3",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:fsag_cluster3.train_model"},
        namespace="fsag_cluster3",
    )
    fsag_model_cluster = fsag_model_cluster.tag(["fsag_cluster3", "sag"])
    return fsag_model_cluster


def create_fsag_pipeline_cluster_2():
    select_cluster = mip.create_pipeline_select_cluster()
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    fsag_model_cluster = pipeline(
        select_cluster,
        inputs={"data_all_features": "data_all_features", "data_cluster": "data_cluster2"},
        parameters={"params:model_input": "params:fsag_cluster2.model_input"},
        namespace="fsag_cluster2",
    ) + pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "fsag_cluster2.master_target_cluster", "td": "td"},
        parameters={"params:model_input": "params:fsag_cluster2.model_input"},
        namespace="fsag_cluster2",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:fsag_cluster2.train_model"},
        namespace="fsag_cluster2",
    )
    fsag_model_cluster = fsag_model_cluster.tag(["fsag_cluster2", "sag"])
    return fsag_model_cluster


def create_fsag_pipeline_cluster_1():
    select_cluster = mip.create_pipeline_select_cluster()
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    fsag_model_cluster = pipeline(
        select_cluster,
        inputs={"data_all_features": "data_all_features", "data_cluster": "data_cluster1"},
        parameters={"params:model_input": "params:fsag_cluster1.model_input"},
        namespace="fsag_cluster1",
    ) + pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "fsag_cluster1.master_target_cluster", "td": "td"},
        parameters={"params:model_input": "params:fsag_cluster1.model_input"},
        namespace="fsag_cluster1",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:fsag_cluster1.train_model"},
        namespace="fsag_cluster1",
    )
    fsag_model_cluster = fsag_model_cluster.tag(["fsag_cluster1", "sag"])
    return fsag_model_cluster


def create_fsag_pipeline_cluster_0():
    select_cluster = mip.create_pipeline_select_cluster()
    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    fsag_model_cluster = pipeline(
        select_cluster,
        inputs={"data_all_features": "data_all_features", "data_cluster": "data_cluster0"},
        parameters={"params:model_input": "params:fsag_cluster0.model_input"},
        namespace="fsag_cluster0",
    ) + pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "fsag_cluster0.master_target_cluster", "td": "td"},
        parameters={"params:model_input": "params:fsag_cluster0.model_input"},
        namespace="fsag_cluster0",
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": "params:fsag_cluster0.train_model"},
        namespace="fsag_cluster0",
    )
    fsag_model_cluster = fsag_model_cluster.tag(["fsag_cluster0", "sag"])
    return fsag_model_cluster
