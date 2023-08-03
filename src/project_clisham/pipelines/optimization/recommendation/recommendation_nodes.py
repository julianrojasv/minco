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
from ast import literal_eval
from copy import deepcopy
from datetime import datetime
from multiprocessing import Pool
from typing import Any, List, Tuple, Dict

import pandas as pd
import numpy as np
from tqdm import tqdm

from optimizer.constraint import Repair, repair
from optimizer.problem import (
    OptimizationProblem,
    StatefulOptimizationProblem,
    StatefulContextualOptimizationProblem,
)
from optimizer.solvers import Solver
from optimizer.stoppers.base import BaseStopper

from project_clisham.optimus_core.tag_management import TagDict
from project_clisham.optimus_core.utils import load_obj

logger = logging.getLogger(__name__)


def optimize(  # pylint:disable=too-many-locals
    timestamp: datetime,
    row: pd.DataFrame,
    ui_states: List[str],
    controls: List[str],
    on_controls: List[str],
    target: str,
    problem: OptimizationProblem,
    solver: Solver,
    stopper: BaseStopper,
    model_dict: Dict,
) -> Tuple:
    """
    Optimizes a single row and returns the expected json object.
    Args:
        timestamp: timestamp key
        row: row of data to score
        ui_states: list of state columns to include in the results
        controls: list of controllable features
        on_controls: list of controllable features that are on
        target: target to optimize objective function
        problem: optimization problem
        solver: solver
        stopper: stopper
    Returns:
        Tuple of JSON results
    """
    # score with current and with optimal controls
    scores = pd.concat([row] * 2, ignore_index=True)
    scores.index = ["curr", "opt"]

    if solver:
        stop_condition = False
        while not stop_condition:
            parameters = solver.ask()
            obj_vals, parameters = problem(parameters)
            solver.tell(parameters, obj_vals)
            stop_condition = solver.stop()
            if stopper:
                stopper.update(solver)
                stop_condition |= stopper.stop()

        best_controls, _ = solver.best()
        scores.loc["opt", on_controls] = best_controls

    # note the subtle difference between re-scoring with the objective
    # which translates to a model.predict vs calling the problem, directly
    # which includes constraint penalties

    scores["_pred"] = problem.objective(scores)
    for model in model_dict:
        scores[model + "_pred"] = model_dict[model].predict(scores)

    # assemble result
    states_ = {state: float(row[state].values[0]) for state in ui_states}
    tags = row.columns.drop("run_id")
    context_ = {state: float(row[state].values[0]) for state in tags}
    controls_ = {
        ctrl: {
            "current": float(scores.loc["curr", ctrl]),
            "suggested": float(scores.loc["opt", ctrl]),
            "delta": float(scores.loc["opt", ctrl] - scores.loc["curr", ctrl]),
        }
        for ctrl in controls
    }
    models_ = {
        model: {
            f"{model}_current": float(row[model]),
            f"{model}_predicted": float(scores.loc["curr", model + "_pred"]),
            f"{model}_optimized": float(scores.loc["opt", model + "_pred"]),
            f"{model}_delta": float(
                scores.loc["opt", model + "_pred"] - scores.loc["curr", model + "_pred"]
            ),
        }
        for model in model_dict
    }
    outputs_ = {
        f"target_{target}_current": float(row[target]),
        "target_pred_current": float(scores.loc["curr", "_pred"]),
        "target_pred_optimized": float(scores.loc["opt", "_pred"]),
    }

    uplift_report_dict = {
        "run_id": row["run_id"].values[0],
        "Fecha": str(timestamp),
        "context": context_,
        "state": states_,
        "controls": controls_,
        "outputs": outputs_,
        "models": models_,
    }

    control_recommendations = []
    for control in on_controls:
        control_recommendations.append(
            {
                "tag_id": control,
                "run_id": row["run_id"].values[0],
                "value": scores.loc["opt", control],
            }
        )

    output_recommendation = {
        "run_id": row["run_id"].values[0],
        "tag_id": target,
        "baseline": float(scores.loc["curr", "_pred"]),
        "optimized": float(scores.loc["opt", "_pred"]),
    }
    return uplift_report_dict, control_recommendations, output_recommendation


def _optimize_dict(kwargs):
    return optimize(**kwargs)


def _get_features(td: TagDict, params: dict) -> Tuple[List[str], List[str]]:
    """ return features and optimizable features """
    feat_columns = params["model_features"]
    features = list(set([tag for col in feat_columns for tag in td.select(col)]))
    all_controls = td.select("tag_type", "control")
    controls = [f for f in features if f in all_controls]

    return features, controls


def _get_target(td: TagDict, params: dict) -> str:
    """ return target feature """
    opt_target = params["opt_target"]
    target = td.select("target", opt_target)[0]
    return target


def _get_on_features(
    current_value: pd.DataFrame, td: TagDict, controls: List[str]
) -> List[str]:
    """
    return controls that are on by checking current state
    """
    on_controls = []
    for feature in controls:
        on_flag = all(
            [current_value[d].iloc[0] > 0.5 for d in td.dependencies(feature)]
        )
        if on_flag:
            if np.isnan(current_value[feature].values[0]):
                logger.warning(f"Current Value for feature {feature} is NaN ")
            else:
                on_controls.append(feature)
        else:
            logger.warning(f"Current feature {feature} is OFF")
    return on_controls


def make_solver(params: dict, bounds: Tuple) -> Solver:
    """
    Creates an ask-tell solver from the tag dict

    Args:
        params: dict of pipeline parameters
        bounds: Tuple of Bounds of Solver
    Returns:
        optimization solver object
    """
    solver_class = load_obj(params["solver"]["class"])
    solver_kwargs = params["solver"]["kwargs"]

    solver_kwargs.update({"bounds": bounds})

    return solver_class(**solver_kwargs)


def get_solver_bounds(
    current_value: pd.DataFrame, td: TagDict, controls: List[str]
) -> List[Tuple]:
    """
    Add more appropriate bounds to controls, applying max_deltas
    if available

    Args:
        current_value: DataFrame of current value for optimization,
        td: tag dictionary
        controls: List of strings indicating controls to optimize
    Returns:
        bounded solver instance
    """
    solver_bounds = []
    for control in controls:
        control_entry = td[control]
        op_min = control_entry["op_min"]
        op_max = control_entry["op_max"]
        lower_bounds = [op_min]
        upper_bounds = [op_max]

        if not pd.isna(control_entry["max_delta"]):
            current_val = current_value[control].iloc[0]
            if (current_val < op_min) or (current_val > op_max):
                logger.warning(
                    f"Current Value for Control {control} {current_val} "
                    f"is outside of range [{op_min}, {op_max}]"
                )
            lower_bounds.append(current_val - control_entry["max_delta"])
            upper_bounds.append(current_val + control_entry["max_delta"])
        solver_bounds.append((max(lower_bounds), min(upper_bounds)))
    return solver_bounds


def make_stopper(params: dict) -> BaseStopper:
    """
    Creates a stopper using configured params

    Args:
        params: dict of pipeline parameters
    Returns:
        optimization stopper object
    """
    if params["stopper"]:
        stopper_class = load_obj(params["stopper"]["class"])
        stopper_kwargs = params["stopper"]["kwargs"]
        return stopper_class(**stopper_kwargs)
    return None


def _make_set_repair(td: TagDict, column: str) -> Repair:
    """ Creates a new constraint set repair for a given column """
    constraint_set = literal_eval(td[column]["constraint_set"])
    return repair(column, "in", constraint_set)


def bulk_optimize(  # pylint:disable=too-many-locals
    params: dict, td: TagDict, data: pd.DataFrame, model: Any, model_dict: Dict
) -> Dict:
    """
    Create recommendations for a whole dataframe in row by row.

    Args:
        params: dict of pipeline parameters
        td: tag dictionary
        data: dataframe to process
        model: model object. Needs to have a `.predict` method
    Returns:
        recommendations, recommended_controls, projected_optimization
    """
    if not hasattr(model, "predict"):
        raise ValueError("`model` must have a `predict` method.")

    # do not use parallel processing in the model
    # since we are parallelizing over rows in the dataframe
    try:
        model.set_params(regressor__n_jobs=1)
    except (ValueError, AttributeError):
        pass

    n_jobs = params["n_jobs"]
    datetime_col = params["datetime_col"]
    stopper = None
    # stopper = params["stopper"]

    if params["stopper"]:
        stopper = make_stopper(params)

    features, controls = _get_features(td, params)
    target = _get_target(td, params)

    for feature in controls:
        if pd.isna(td[feature]["op_min"]) or pd.isna(td[feature]["op_max"]):
            raise ValueError(f"Operating Ranges for f{feature} must be specified.")

    # for now, we show all model-states in the UI
    ui_states = [f for f in features if f not in controls]

    controls_with_constraints = td.select("constraint_set", pd.notnull)

    def yield_dicts():
        # we iterate over rows as single-row dataframes
        # instead of pd.Series in order to preserve dtypes
        for idx in data.index:
            row = data.loc[[idx], :]
            on_controls = _get_on_features(row, td, controls)

            if on_controls:
                # the normal case: we have at least one control variable
                # that we want to optimize
                row_solver_bounds = get_solver_bounds(row, td, on_controls)
                row_solver = make_solver(params, row_solver_bounds)

                repairs = [
                    _make_set_repair(td, col)
                    for col in (set(on_controls) & set(controls_with_constraints))
                ] or None

                problem = StatefulOptimizationProblem(
                    model,
                    state=row,
                    optimizable_columns=on_controls,
                    repairs=repairs,
                    sense="maximize",
                )
            else:
                # if all machines are off, there is no recommendation to be
                # produced and we simply create a dummy problem
                row_solver = None
                problem = OptimizationProblem(model, sense="maximize")

            yield dict(
                timestamp=row.at[idx, datetime_col],
                row=row,
                ui_states=ui_states,
                controls=controls,
                on_controls=on_controls,
                target=target,
                problem=deepcopy(problem),
                solver=row_solver,
                stopper=deepcopy(stopper),
                model_dict=deepcopy(model_dict),
            )

    # we use imap (lazy pool.map) here to make tqdm work
    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            results = list(
                tqdm(pool.imap(_optimize_dict, yield_dicts()), total=len(data))
            )
            pool.close()
            pool.join()
    else:
        results = tqdm(
            [_optimize_dict(kwargs) for kwargs in yield_dicts()], total=len(data)
        )
    uplift_results, control_results, output_results = list(zip(*results))
    uplift_results = pd.DataFrame(list(uplift_results))
    control_results = pd.DataFrame(list(list(cr for cr in control_results)))
    output_results = pd.DataFrame(list(output_results))
    return {
        "recommendations": uplift_results,
        "recommended_controls": control_results,
        "projected_optimization": output_results,
    }
