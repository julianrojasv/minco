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

import project_clisham.pipelines.data_science.data_science_pipeline_A2 as a2
import project_clisham.pipelines.data_science.data_science_pipeline_esp as esp

def create_pipeline():
    # Espesadores
    r2_model = esp.create_r2_pipeline()
    r3_model = esp.create_r3_pipeline()
    r4_model = esp.create_r4_pipeline()
    r5_model = esp.create_r5_pipeline()
    r6_model = esp.create_r6_pipeline()
    r7_model = esp.create_r7_pipeline()
    r9_model = esp.create_r9_pipeline()
    r10_model = esp.create_r10_pipeline()

    all_pipeline = (r2_model +
                    r3_model +
                    r4_model +
                    r5_model +
                    r6_model +
                    r7_model +
                    r9_model +
                    r10_model)

    return all_pipeline

def create_pipeline_1esp():
    # espesador = input('Ingresa el espesador: ')
    espesador = 'r10'
    model = getattr(esp, f'create_{espesador}_pipeline')
    model_espesador = model()
    return (model_espesador)


def create_pipeline_1esp_crisis_hidrica():
    # espesador = input('Ingresa el espesador: ')
    espesador = 'r9'
    model = getattr(esp, f'create_{espesador}_pipeline_crisis_hidrica')
    model_espesador = model()
    return (model_espesador)
