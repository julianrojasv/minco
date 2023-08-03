import itertools as it
import logging
import time
from datetime import datetime
from typing import Any, Dict
from uuid import uuid4

import mlflow
import mlflow.sklearn
import requests
import simplejson as json
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.pipeline.node import Node

from project_clisham.utils import alarms

logger = logging.getLogger(__name__)


class MlFlowHooks:

    namespaces = [
        "s0",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
        "s7",
        "s8",
        "s9",
        "s10",
        "s11",
        "s12",
        "s13",
        "s14",
        "s15",
        "s16",
        "s17",
        "fa0l1",
        "fa0l2",
        "fa1l1",
        "fa1l2",
        "fa2l1",
        "fa2l2",
        "fa2l3",
    ]

    def __init__(self):
        self.target_runid = {}

    @hook_impl
    def after_node_run(
        self, node: Node, inputs: Dict[str, Any], outputs: Dict[str, Any]
    ):
        if node.namespace in self.namespaces:
            target = node.namespace
            run_id = self.target_runid.get(target)
            with mlflow.start_run(run_id=run_id) as run:
                self.target_runid[target] = run.info.run_id
                mlflow.set_tag("model", node.namespace)
                if node.name.endswith("load_regressor"):
                    mlflow.log_params(
                        {
                            "train_model_params": inputs[
                                f"params:{target}.train_model"
                            ]["regressor"]
                        }
                    )

                elif node.name.endswith("train_tree_model"):
                    model = outputs[f"{target}.train_model"]
                    mlflow.sklearn.log_model(model, f"{target}.model")
                    mlflow.xgboost.autolog(model)

                elif node.name.endswith("create_predictions"):
                    metrics = outputs[f"{target}.test_set_metrics"].to_dict()
                    met_values = metrics["opt_perf_metrics"]
                    for key in met_values:
                        mlflow.log_metrics({key: met_values[key]})


class SendToAPIHook:
    namespaces = ["esp"]
    nodes_names = ["generate_sensitivity_data"]

    @staticmethod
    @hook_impl
    def after_node_run(
        node: Node,
        catalog: DataCatalog,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
    ):
        envs = [
            "base",
        ]
        KEDRO_ENV = catalog.load("params:KEDRO_ENV")
        if KEDRO_ENV in envs:
            # Debug prints
            logger.info(f"ENTER API HOOK")
            logger.info(f"NODE NAME: {node.name}")
            logger.info(f"NODE NAMESPACE: {node.namespace}")
            alarm_params = catalog.load("params:alarm_config")
            ns_list = SendToAPIHook.namespaces
            nodes = SendToAPIHook.nodes_names
            full_name = [".".join(x) for x in list(it.product(ns_list, nodes))]
            if node.namespace in ns_list and node.name in full_name:
                # Set the namespace
                ns = node.namespace
                # Retrieve parameters from the catalog
                params = catalog.load("params:backend_api")

                # Create the URLS for the API
                analytics_url = params["REACT_APP_OPTIMUS_ANALYTICS_API"]
                data_insight_url = params["REACT_APP_OPTIMUS_DATA_INSIGHT_API"]
                # Map the correct endpoint for each model.
                analytics_url += params["model_to_api_map"][ns]
                #data_insight_url += params["model_to_api_map"][ns]
                # Debug info
                logger.info(f"Analytics URL: {analytics_url}")
                logger.info(f"Data Insight URL: {data_insight_url}")
                # Prepare the data to send
                recomm = catalog.load(f"{ns}.recommendations")
                #sensi = outputs[f"{ns}.sensitivity_plot_df"]
                # retrieve the run_id to get the sens data.
                run_id = recomm["run_id"].values[0]

                json_data = [recomm.loc[0].to_dict()]
                # This is to remove the context
                json_data[0].pop("context")
                # let get the controls tags to filter on the sensitivity data.
                control_tags = list(json_data[0]["controls"].keys())
                # sensi data treatment
                #sensi_data = sensi.query("run_id == @run_id").copy()
                #sensi_data_grp = sensi_data.groupby("control_tag_id")
                # add sensi data to a new dict
                #new_dict = dict()
                #for name, group in sensi_data_grp:
                #    if name in control_tags:
                #        group["target_value"].to_list()
                #        new_dict[name] = {
                #            "target_value": group["target_value"].to_list(),
                #            "control_value": group["control_value"].to_list(),
                #        }
                # paste sensi data to json
                #json_data[0]["sensitivity"] = new_dict
                # dump it into a string
                fixed_data = json.dumps(json_data, ignore_nan=True)
                logger.info(fixed_data)
                # Send the Data
                try:
                    headers = {
                        "Content-type": "application/json",
                        "Accept": "text/plain",
                    }
                    logger.debug(fixed_data)
                    response = requests.post(
                        analytics_url, data=fixed_data, headers=headers
                    )
                    logger.info(f"Analytics API Response:\t{response.content}")
                    if response.status_code != 201:
                        alarms.create_json_msg(
                            msg=response.content, group="data_team", params=alarm_params
                        )
                    # Wait a little.
                    time.sleep(10)
                    #response = requests.post(
                    #    data_insight_url, data=fixed_data, headers=headers
                    #)
                    logger.info(f"Data Insight API Response:\t{response.content}")
                    if response.status_code != 201:
                        alarms.create_json_msg(
                            msg=response.content, group="data_team", params=alarm_params
                        )
                except ConnectionRefusedError as c:
                    logger.error("Could not send the recommendation to the Backend.")
                    logger.error(c)
                except Exception as e:
                    logger.warning(e)
                    raise e


class GeneralAlarm:
    @staticmethod
    @hook_impl
    def on_pipeline_error(
        error: Exception, run_params: Dict, catalog: DataCatalog
    ) -> None:
        """TODO: Docstring
        """
        allowed_envs = [
            "cloud_recommend_dev",
            "cloud_recommend_mvp",
            "cloud_recommend_test",
        ]
        KEDRO_ENV = catalog.load("params:KEDRO_ENV")
        if KEDRO_ENV in allowed_envs:
            msg = f"Pipeline Error: {error}\n"
            msg += f"RUN: {run_params['run_id']}\n"
            msg += f"ENV: {run_params['env']}\n"
            msg += f"Pipeline: {run_params['pipeline_name']}\n"

            alarm_params = catalog.load("params:alarm_config")

            alarms.create_json_msg(
                msg=msg, group="data_team", params=alarm_params, severity="High"
            )

