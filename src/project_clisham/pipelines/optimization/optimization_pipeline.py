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
from kedro.pipeline import pipeline, Pipeline, node
from .transform_models import transform_models_pipeline as tmp
from .recommendation import recommendation_pipeline as rp
from .sensitivity import sensitivity_pipeline as sp

from .transform_models.transform_models_nodes import create_models_dictionary


def create_pipeline():

    tph_models_pipeline = tmp.create_tph_pipeline()
    cuf_a2_models_pipeline = tmp.create_cuf_pipeline_a2()
    recommendation_pipeline = rp.create_pipeline()
    sensitivity_pipeline = sp.create_pipeline()

    # Area SAG - molienda
    sag_tph_recommendations = (
        Pipeline(
            [
                node(
                    func=create_models_dictionary,
                    inputs=dict(td="td", sag1="sag1.train_model", sag2="sag2.train_model"),
                    outputs="sag.tph_dict",
                )
            ]
        )
        + pipeline(
            tph_models_pipeline,
            inputs={},
            parameters={},
            namespace="sag",
        )
        + pipeline(
            recommendation_pipeline,
            inputs={
                "data_all_features": "data_all_features",
                "td": "td",
                "models_dict": "sag.tph_dict",  # actualizar con el output de create models_dictionary
            },
            parameters={
                "params:recommend": "params:sag.recommend",
                "params:uplift_report": "params:sag.recommend",
            },
            namespace="sag",
        )
        + pipeline(
            sensitivity_pipeline,
            inputs={"td": "td"},
            parameters={"params:recommend_sensitivity": "params:sag.recommend"},
            namespace="sag",
        )
    )
    sag_tph_recommendations = sag_tph_recommendations.tag(["sag", "sag", "sag_optim"])

    # ## Area SAG - CuF por linea
    # ca2xl_tph_recommendations = (
    #     Pipeline(
    #         [
    #             node(
    #                 func=create_models_dictionary,
    #                 inputs=dict(
    #                     td="td",
    #                     sag1="sag1.train_model",
    #                     sag2="sag2.train_model",
    #                     fsag="fsag.train_model"
    #                 ),
    #                 outputs="ca2xl.cuf_dict",
    #             )
    #         ]
    #     )
    #     + pipeline(
    #         cuf_a2_models_pipeline,
    #         inputs={},
    #         parameters={"params": "parameters"},
    #         namespace="ca2xl",
    #     )
    #     + pipeline(
    #         recommendation_pipeline,
    #         inputs={
    #             "data_all_features": "data_all_features",
    #             "td": "td",
    #             "models_dict": "ca2xl.cuf_dict",  # actualizar con el output de create models_dictionary
    #         },
    #         parameters={
    #             "params:recommend": "params:ca2xl.recommend",
    #             "params:uplift_report": "params:ca2xl.recommend",
    #         },
    #         namespace="ca2xl",
    #     )
    #     + pipeline(
    #         sensitivity_pipeline,
    #         inputs={"td": "td"},
    #         parameters={"params:recommend_sensitivity": "params:ca2xl.recommend"},
    #         namespace="ca2xl",
    #     )
    # )
    # ca2xl_tph_recommendations = ca2xl_tph_recommendations.tag(
    #     ["ca2xl", "a2", "cuf_a2_optim"]
    # )

    # BASURA DE A1 QUE SOLO SIRVE DE RESPALDO!
    # # Area A1 - CuF por linea
    # ca1xl_tph_recommendations = (
    #     Pipeline(
    #         [
    #             node(
    #                 func=create_models_dictionary,
    #                 inputs=dict(
    #                     td="td",
    #                     s13="s13.train_model",
    #                     s14="s14.train_model",
    #                     s15="s15.train_model",
    #                     fa1l1="fa1l1.train_model",
    #                     fa1l2="fa1l2.train_model",
    #                 ),
    #                 outputs="ca1xl.cuf_dict",
    #             )
    #         ]
    #     )
    #     + pipeline(
    #         cuf_a1_models_pipeline,
    #         inputs={},
    #         parameters={"params": "parameters"},
    #         namespace="ca1xl",
    #     )
    #     + pipeline(
    #         recommendation_pipeline,
    #         inputs={
    #             "data_all_features": "data_all_features",
    #             "td": "td",
    #             "models_dict": "ca1xl.cuf_dict",  # actualizar con el output de create models_dictionary
    #         },
    #         parameters={
    #             "params:recommend": "params:ca1xl.recommend",
    #             "params:uplift_report": "params:ca1xl.recommend",
    #         },
    #         namespace="ca1xl",
    #     )
    #     + pipeline(
    #         sensitivity_pipeline,
    #         inputs={"td": "td"},
    #         parameters={"params:recommend_sensitivity": "params:ca1xl.recommend"},
    #         namespace="ca1xl",
    #     )
    # )
    # ca1xl_tph_recommendations = ca1xl_tph_recommendations.tag(
    #     ["ca1xl", "a1", "cuf_a1_optim"]
    # )
    #
    # all_pipeline = (
    #     ma2_tph_recommendations + ca2xl_tph_recommendations + ca1xl_tph_recommendations
    # )

    all_pipeline = (
        sag_tph_recommendations
    )


    return all_pipeline


def create_pipeline_espesadores():

    torque_models_pipeline = tmp.create_torque_pipeline()
    recommendation_pipeline = rp.create_pipeline()
    sensitivity_pipeline = sp.create_pipeline()

    # Area Espesadores - Torque
    esp_torque_recommendations = (
        Pipeline(
            [
                node(
                    func=create_models_dictionary,
                    inputs=dict(td="td",
                                r10_esp="r10_esp.train_model",
                                r9_esp="r9_esp.train_model",
                                r7_esp="r7_esp.train_model",
                                r6_esp="r6_esp.train_model",
                                r5_esp="r5_esp.train_model",
                                r4_esp="r4_esp.train_model",
                                r3_esp="r3_esp.train_model",
                                r2_esp="r2_esp.train_model"
        ),
                    outputs="esp.torque_dict",
                )
            ]
        )
        + pipeline(
            torque_models_pipeline,
            inputs={},
            parameters={"params:general_tags_esp": "params:general_tags_esp"},
            namespace="esp",
        )
        + pipeline(
            recommendation_pipeline,
            inputs={
                "data_all_features": "data_all_features_esp",
                "td": "td",
                "models_dict": "esp.torque_dict",  # actualizar con el output de create models_dictionary
            },
            parameters={
                "params:recommend": "params:esp.recommend",
                "params:uplift_report": "params:esp.recommend",
            },
            namespace="esp",
        )
        + pipeline(
            sensitivity_pipeline,
            inputs={"td": "td"},
            parameters={"params:recommend_sensitivity": "params:esp.recommend"},
            namespace="esp",
        )
    )
    esp_torque_recommendations = esp_torque_recommendations.tag(["esp","esp_optim"])
    return (esp_torque_recommendations)