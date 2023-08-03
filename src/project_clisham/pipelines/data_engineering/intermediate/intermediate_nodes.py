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

# from typing import Any, Dict

import pandas as pd
import numpy as np


def add_duracion_revestimiento(
    parameters, df: pd.DataFrame, revestimiento: pd.DataFrame
) -> pd.DataFrame:
    """Transform the column names into standardized names.

    Args:
        df: Raw data.
        # TODO: JA - complete docstring (better description and some func args are missing)

    Returns:
        pd.DataFrame: data with standardized names
    """
    
    # Added by MV, to fix file with duplicated
    df = df.drop_duplicates('Fecha', keep= 'last')

    equip = [
        "sag1",
        "sag2",
    ]
    col_timestamp = parameters["timestamp_col_name"]
    feature_prefix = parameters["rev_cal"]["tag_prefix"]
    df = df.set_index(col_timestamp)
    for eq in equip:
        df[feature_prefix + eq + "_ucf_dias"] = None
        df[feature_prefix + eq + "_ucf_dias"] = df[
            feature_prefix + eq + "_ucf_dias"
        ].astype(float)
        dates_rev_change = np.array(
            sorted([pd.to_datetime(d) for d in revestimiento[eq + "_ucf"].values])
        )
        for dias in df.index:  # TODO: JA - inefficient loop, try to improve
            try:
                last_change = sorted(
                    dates_rev_change[dates_rev_change <= dias], key=lambda x: dias - x
                )[0]
                delta = dias - last_change
                df.loc[dias, feature_prefix + eq + "_ucf_dias"] = delta.days
            except IndexError:
                pass  # TODO: JA - consideracion DM a JA de log

    return df.reset_index()



def replace_duracion_revestimiento(
    parameters, df: pd.DataFrame
) -> pd.DataFrame:
    """Transform the column names into standardized names.

    Args:
        df: Raw data.
        # TODO: JA - complete docstring (better description and some func args are missing)

    Returns:
        pd.DataFrame: data with standardized names
    """
    #import ipdb;ipdb.set_trace();

    df = df.drop_duplicates('fecha', keep= 'last')
    col_timestamp = parameters["timestamp_col_name"]
#    df = df.set_index(col_timestamp)
    df = df.set_index(['fecha'])
    df.index = pd.to_datetime(df.index, format = '%Y-%m-%d %H:%M:%S')

    return df.reset_index()