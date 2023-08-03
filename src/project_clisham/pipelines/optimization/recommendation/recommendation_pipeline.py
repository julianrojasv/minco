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
Optimization Pipeline
"""

from kedro.pipeline import Pipeline, node


from .recommendation_nodes import bulk_optimize
from project_clisham.optimus_core.reporting_html.nodes import create_html_report
from .support_nodes import (
    filter_timestamp_optimization,
    generate_recommendation_csv,
    generate_uuid,
    create_bulk_result_tables,
)


def create_pipeline(**kwargs):  # pylint:disable=unused-argument
    return Pipeline(
        [
            node(
                func=filter_timestamp_optimization,
                inputs=dict(
                    params="params:recommend",
                    data="data_all_features",
                ),
                outputs="data_input_optim",
                name="filter_optimize",
            ),
            node(
                func=generate_uuid,
                inputs=dict(
                    data="data_input_optim",
                ),
                outputs="data_input_optim_uuid",
            ),
            node(
                func=bulk_optimize,
                inputs=dict(
                    params="params:recommend",
                    td="td",
                    data="data_input_optim_uuid",
                    model="optimization_function",
                    model_dict="models_dict",
                ),
                outputs=dict(
                    recommendations="recommendations",
                    recommended_controls="recommended_controls",
                    projected_optimization="projected_optimization",
                ),
                name="bulk_optimize",
            ),
            # node(
            #     func=generate_recommendation_csv,
            #     inputs=dict(
            #         td="td", params="params:recommend", recomm="recommendations"
            #     ),
            #     outputs=dict(
            #         all_objective="rep_optim_objective",
            #         all_controls="rep_recommendations",
            #     ),
            # ),
            node(
                create_bulk_result_tables,
                dict(
                    params="params:uplift_report",
                    td="td",
                    recommendations="recommendations",
                    opt_df="data_input_optim_uuid",
                ),
                dict(
                    states="bulk_state",
                    controls="bulk_ctrl",
                    outcomes="bulk_output",
                ),
                name="create_bulk_result_tables",
            ),
            # node(
            #    create_html_report,
            #    [
            #        "params:uplift_report",
            #        "params:KEDRO_ENV",
            #        *kwargs.get("wait_on", ["bulk_output"]),
            #    ],
            #    None,
            #    name="create_html_report",
            # ),
        ]
    )
