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
Core nodes performing the optimization.
"""
import logging
import uuid

from functools import partial
from itertools import product
from multiprocessing import Pool
from typing import Dict
import pandas as pd
from project_clisham.optimus_core.tag_management import TagDict


logger = logging.getLogger(__name__)


def generate_uuid(data: pd.DataFrame, col_name: str = "run_id") -> pd.DataFrame:
    """Extract the parameters saved in conf
    Args:
        data: original DataFrame
        col_name: name for column for UUID
    Returns:
        DataFrame with UUID added
    Raises:
        AttributeError: When the param does not exist
    """
    columns = data.columns
    data[col_name] = [str(uuid.uuid4()) for _ in range(len(data.index))]
    data = data[[col_name, *columns]]
    return data


def filter_timestamp_optimization(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Filters the dataframe to only keep the timestamps needed for optimization

    Args:
        params: dict of pipeline parameters
        data: dataframe to process
    Returns:
        dataframe for optimization
    """
    datetime_col = params["datetime_col"]
    data.set_index(datetime_col, inplace=True)
    begin_date = params["filter_timestamps"]["begin_date"]
    end_date = params["filter_timestamps"]["end_date"]
    data = data[begin_date:end_date]

    selection_type = params["filter_timestamps"]["type"]
    if selection_type not in ["beginning", "end", "date"]:
        raise RuntimeError(
            """
            Valid arguments are "beginning", "end" or "date". Please, fix it.
            """
        )
    values = params["filter_timestamps"][selection_type]
    if (selection_type in ["beginning", "end"]) & (not isinstance(values, int)):
        raise RuntimeError(
            """
            Valid arguments are integers
            """
        )
    if (selection_type in ["date"]) & (not isinstance(values, list)):
        raise RuntimeError(
            """
            Valid arguments is a list of timestamps with format %Y-%m-%d %H:%M:%S
            """
        )
    if selection_type == "beginning":
        data = data[:values]
    elif selection_type == "end":
        data = data[-values:]
    else:
        index_list = data.index.strftime(date_format="%Y-%m-%d %H:%M:%S").tolist()
        # logger.info(index_list)
        if values in index_list:
            data = data.loc[pd.to_datetime(values)]
        else:
            in_there = list(set(values).intersection(set(index_list)))
            if len(in_there) == 0:
                raise RuntimeError(
                    """
                    None of the selected timestamps are in the test set
                    """
                )
            else:
                data = data.loc[pd.to_datetime(in_there)]
        data.index = data.index.set_names([datetime_col])
    data.reset_index(inplace=True)
    return data


def _filter_stable_conditions(data, conditions):

    for condition in conditions:
        min_val = conditions[condition][0]
        max_val = conditions[condition][1]
        if min_val > max_val:
            raise RuntimeError(
                """
            The min value is above the max value
            """
            )
        else:
            cond = (data[condition] > min_val) & (data[condition] < max_val)
            data = data[cond]
    return data


def generate_recommendation_csv(td: TagDict, params: dict, recomm):
    """
    Process the JSON with recommendations to have nice csv

    Args:
        td: Tag dictionary.
        recomm: recommendations
    Returns:
        objective df
        controls df
    """
    context_cols = params["context_variables"]
    tag_dict = td.to_frame().set_index("tag")
    controls_tags = list(
        set([con for x in range(len(recomm)) for con in recomm["controls"][x].keys()])
    )
    all_controls = pd.DataFrame(index=controls_tags + context_cols)
    all_controls = all_controls.merge(
        tag_dict[["description", "area", "tag_type"]],
        how="left",
        right_index=True,
        left_index=True,
    )
    all_objective = pd.DataFrame()
    for row in recomm.index:
        # TODO: DM - hardcoded name
        timestamp = recomm["Fecha"][row]
        controls = pd.DataFrame.from_dict(recomm["controls"][row]).T
        context = pd.DataFrame.from_dict(
            recomm["context"][row], columns=[timestamp], orient="index"
        )
        context.columns = ["current"]
        context = context.loc[context_cols, :]
        controls = controls.append(context)
        controls.columns = [
            col + "_" + str(timestamp).replace(" ", "_") for col in controls.columns
        ]
        all_controls = all_controls.merge(
            controls, how="left", right_index=True, left_index=True
        )
        objective = pd.DataFrame.from_dict(
            recomm["outputs"][row], columns=[timestamp], orient="index"
        ).T
        for model in recomm["models"][row]:
            model_info = pd.DataFrame.from_dict(
                recomm["models"][row][model],
                columns=[timestamp],
                orient="index",
            ).T
            objective = objective.merge(
                model_info, how="left", right_index=True, left_index=True
            )
        all_objective = all_objective.append(objective, sort=True)

    all_objective["uplift_pred"] = 100 * (
        (all_objective["target_pred_optimized"] - all_objective["target_pred_current"])
        / all_objective["target_pred_current"]
    )
    all_controls = all_controls.sort_values("description").T

    return dict(all_objective=all_objective, all_controls=all_controls)


def _extract_state_row(row: dict) -> pd.DataFrame:
    """
    extracts state data from optimization result.
    """
    index = pd.Index([row["run_id"]], name="run_id")
    state_columns = row["state"].keys()

    state_row = pd.DataFrame(index=index, columns=state_columns, dtype=float)
    for col in state_columns:
        state_row.loc[row["run_id"], col] = row["state"][col]
    return state_row


def _extract_ctrl_row(row: dict) -> pd.DataFrame:
    """
    extracts controls data from optimization result.
    """
    index = pd.Index([row["run_id"]], name="run_id")
    ctrl_columns = list(
        product(row["controls"].keys(), ["current", "suggested", "delta"])
    )
    ctrl_row = pd.DataFrame(
        index=index, columns=pd.MultiIndex.from_tuples(ctrl_columns), dtype=float
    )
    for ctrl, status in ctrl_columns:
        ctrl_row.loc[row["run_id"], (ctrl, status)] = row["controls"][ctrl][status]
    return ctrl_row


def _extract_output_row(row: dict, target: str) -> pd.DataFrame:
    """
    extracts output data from optimization result.
    """
    index = pd.Index([row["run_id"]], name="run_id")
    output_columns = list(product([target], row["outputs"].keys()))
    output_row = pd.DataFrame(
        index=index, columns=pd.MultiIndex.from_tuples(output_columns), dtype=float
    )
    for _, outp in output_columns:
        output_row.loc[row["run_id"], (target, outp)] = row["outputs"][outp]
    return output_row


def create_bulk_result_tables(
    params: dict, td: TagDict, recommendations: pd.DataFrame, opt_df: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Creates a bulk-optimization output table.

    Args:
        params: dict of pipeline parameters
        td: tag dictionary
        recommendations: output of bulk optimization
        opt_df: opt data
    Returns:
        states, controls, outcomes
    """
    n_jobs = params["n_jobs"]
    opt_target = params["opt_target"]
    target = td.select("target", opt_target)[0]

    recommendations = recommendations.to_dict(orient="records")

    with Pool(n_jobs) as pool:
        all_states = list(pool.map(_extract_state_row, recommendations))
        all_ctrls = list(pool.map(_extract_ctrl_row, recommendations))
        all_outputs = list(
            pool.map(partial(_extract_output_row, target=target), recommendations)
        )

    output_df = pd.concat(all_outputs)
    state_df = pd.concat(all_states)
    ctrl_df = pd.concat(all_ctrls)

    # append true opt set outcomes to output
    actual_df = pd.DataFrame(
        data=opt_df[target].values,
        columns=pd.MultiIndex.from_tuples([(target, "actual")]),
        index=pd.Index(name="run_id", data=opt_df["run_id"]),
    )
    output_df = pd.merge(
        output_df, actual_df, how="left", left_index=True, right_index=True
    )

    # calculate deltas
    output_df[(target, "optimized_vs_predicted")] = (
        output_df[(target, "target_pred_optimized")]
        - output_df[(target, "target_pred_current")]
    )
    output_df[(target, "optimized_vs_predicted_pct")] = (
        output_df[(target, "optimized_vs_predicted")]
        / output_df[(target, "target_pred_current")]
        * 100
    )
    output_df[(target, "optimized_vs_actual")] = (
        output_df[(target, "target_pred_optimized")] - output_df[(target, "actual")]
    )
    output_df[(target, "optimized_vs_actual_pct")] = (
        output_df[(target, "optimized_vs_actual")] / output_df[(target, "actual")] * 100
    )

    return {"states": state_df, "controls": ctrl_df, "outcomes": output_df}
