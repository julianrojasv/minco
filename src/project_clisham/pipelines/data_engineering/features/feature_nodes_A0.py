import pandas as pd

from project_clisham.optimus_core.utils import merge_tables_on_timestamp

from .features_nodes import (
    create_target_flot,
)


def add_a0_features_by_hour(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Add calculated features to the hourly master data table.

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing source data.

    Returns:
        data_feat: Master table with additional calculated features.

    """

    # Create features
    list_features = [data]
    target_flot = create_target_flot(parameters, data, model_name="fa0")
    list_features.append(target_flot)

    # Merge data and created features
    data_feat = merge_tables_on_timestamp(parameters=parameters, df_list=list_features)

    return data_feat
