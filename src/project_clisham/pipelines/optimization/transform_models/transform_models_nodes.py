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
"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

PLEASE DELETE THIS FILE ONCE YOU START WORKING ON YOUR OWN PROJECT!
"""

from typing import Any, Dict

import pandas as pd
import numpy as np

from project_clisham.pipelines.data_engineering.features.features_nodes import (
    create_target_flot,
    create_cuf_feature,
)


def create_models_dictionary(**dict_models) -> Dict:
    """Creates  a dictionary with the model objects

    Returns:
        [type]: [description]
    """
    td = dict_models["td"]
    dict_models.pop("td")
    new_dict = {}
    for key in dict_models:
        target_col = td.select("target", key + "_target")[0]
        new_dict[target_col] = dict_models[key]

    return new_dict


class ThroughputOptimization:
    def __init__(self, dict_models):
        self._tph = dict_models

    def predict(self, x):
        predictions = []
        for model in self._tph.values():
            predictions.append(model.predict(x))
        predictions = np.array(predictions)
        # Final prediciton is sum of all the predictions
        fin_pred = np.sum(predictions, 0)

        return fin_pred


class TorqueOptimization:
    def __init__(self, dict_models, params_esp):
        self._torque = dict_models
        self._params_esp = params_esp

    def predict(self, x):
        predictions = []
        for model in self._torque.values():
            predictions.append(model.predict(x))
        predictions = np.array(predictions)

        # Final prediciton is the mean of all the predictions
        fin_pred = np.mean(predictions, 0)

        if self._params_esp['opt_solido']:
            solido = []
            for r in range(2, 11):
                if r == 8:
                    continue
                solido.append(x[self._params_esp['esp_target_name_solido_r' + str(r)]].mean(axis=1))
            solido = np.array(solido)
            solido = np.mean(solido, 0)

            fin_pred = fin_pred - float(self._params_esp['alpha_solido'])*solido

        return fin_pred


def create_throughput_optimization(dict_models):
    """Creates a ThoughputOptimization Object

    Returns:
        [type]: [description]
    """
    optim_fn = ThroughputOptimization(dict_models)

    return optim_fn


def create_torque_optimization_esp(dict_models, params_esp):
    """Creates a TorqueOptimzation Object

    Returns:
        [type]: [description]
    """
    optim_fn = TorqueOptimization(dict_models, params_esp)

    return optim_fn


class CufOptimizationA2:
    def __init__(self, dict_models, params):
        self.models = dict_models
        self.params = params
        self.model_tph_name = "ma2"
        self.model_rec_name = "fa2"

    def predict(self, x):
        """Combine tph and recovery models in a nested way"""
        params = self.params
        params_t = params[f"{self.model_rec_name}_flotation_target"]
        leyes_cola = params_t["ley_cola_tf_tags"]
        mod_tph_tags = params_t["tph_tags"]
        sum_tph_name = params[f"{self.model_tph_name}_target_name"]
        new_x = x.copy()

        # Update cosas molienda
        for key in mod_tph_tags:
            if key not in self.models:
                raise RuntimeError(f"No model has been provided for {key}")
            else:
                new_x[key] = self.models[key].predict(x)
        # TODO: avoid making hardcoded updates
        new_x["calc_tph_s16_over_s17"] = new_x[mod_tph_tags[0]] / new_x[mod_tph_tags[1]]
        new_x[sum_tph_name] = new_x[mod_tph_tags].sum(axis=1)

        # Update cosas flotacion
        for key in leyes_cola:
            if key not in self.models:
                raise RuntimeError(f"No model has been provided for {key}")
            else:
                new_x[key] = self.models[key].predict(new_x)

        # Update all recovery variables (by line and weighted)
        rec_features = create_target_flot(params, new_x, self.model_rec_name)
        new_x[rec_features.columns] = rec_features.values

        # Update CuF value
        cuf_feature = create_cuf_feature(params, new_x, self.model_rec_name)

        # Return CuF
        return cuf_feature[params_t["cuf_obj_name"]]


class CufOptimizationA1:  # TODO: make modular class
    def __init__(self, dict_models, params):
        self.models = dict_models
        self.params = params
        self.model_tph_name = "ma1"
        self.model_rec_name = "fa1"

    def predict(self, x):
        """Combine tph and recovery models in a nested way"""
        params = self.params
        params_t = params[f"{self.model_rec_name}_flotation_target"]
        leyes_cola = params_t["ley_cola_tf_tags"]
        mod_tph_tags = params_t["tph_tags"]
        sum_tph_name = params[f"{self.model_tph_name}_target_name"]
        new_x = x.copy()

        # Update cosas molienda
        for key in mod_tph_tags:
            if key not in self.models:
                raise RuntimeError(f"No model has been provided for {key}")
            else:
                new_x[key] = self.models[key].predict(x)
        # TODO: avoid making hardcoded updates
        new_x[sum_tph_name] = new_x[mod_tph_tags].sum(axis=1)

        # Update cosas flotacion
        for key in leyes_cola:
            if key not in self.models:
                raise RuntimeError(f"No model has been provided for {key}")
            else:
                new_x[key] = self.models[key].predict(new_x)

        # Update all recovery variables (by line and weighted)
        rec_features = create_target_flot(params, new_x, self.model_rec_name)
        new_x[rec_features.columns] = rec_features.values

        # Update CuF value
        cuf_feature = create_cuf_feature(params, new_x, self.model_rec_name)

        # Return CuF
        return cuf_feature[params_t["cuf_obj_name"]]


class CufOptimizationSAG:
    def __init__(self, dict_models, params):
        self.models = dict_models
        self.params = params
        self.model_tph_name = "sag"
        self.model_rec_name = "fsag"

    def predict(self, x):
        """Combine tph and recovery models in a nested way"""
        params = self.params
        params_t = params[f"{self.model_rec_name}_flotation_target"]
        leyes_cola = params_t["ley_cola_tf_tags"]
        mod_tph_tags = params_t["tph_tags"]
        sum_tph_name = params[f"{self.model_tph_name}_target_name"]
        new_x = x.copy()

        # Update cosas molienda
        for key in mod_tph_tags:
            if key not in self.models:
                raise RuntimeError(f"No model has been provided for {key}")
            else:
                new_x[key] = self.models[key].predict(x)
        # TODO: avoid making hardcoded updates
        new_x["calc_tph_sag1_sag2"] = new_x[mod_tph_tags[0]] / new_x[mod_tph_tags[1]]
        new_x[sum_tph_name] = new_x[mod_tph_tags].sum(axis=1)

        # Update cosas flotacion
        for key in leyes_cola:
            if key not in self.models:
                raise RuntimeError(f"No model has been provided for {key}")
            else:
                new_x[key] = self.models[key].predict(new_x)

        # Update all recovery variables (by line and weighted)
        rec_features = create_target_flot(params, new_x, self.model_rec_name)
        new_x[rec_features.columns] = rec_features.values

        # Update CuF value
        cuf_feature = create_cuf_feature(params, new_x, self.model_rec_name)

        # Return CuF
        return cuf_feature[params_t["cuf_obj_name"]]


def create_cuf_optimization_a2(dict_models, params):
    """Creates a CufOptimization Object

    Returns:
        [type]: [description]
    """
    optim_fn = CufOptimizationA2(dict_models, params)

    return optim_fn


def create_cuf_optimization_a1(dict_models, params):  # TODO: make modular function
    """Creates a CufOptimization Object

    Returns:
        [type]: [description]
    """
    optim_fn = CufOptimizationA1(dict_models, params)

    return optim_fn


def create_cuf_optimization_sag(dict_models, params):
    """Creates a CufOptimization Object

    Returns:
        [type]: [description]
    """
    optim_fn = CufOptimizationSAG(dict_models, params)

    return optim_fn


