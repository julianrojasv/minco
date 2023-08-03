import pandas as pd
import numpy as np

from project_clisham.optimus_core.tag_management import TagDict
from project_clisham.optimus_core.utils import (
    cut_values_from_dict,
    cut_values_from_list,
)
from project_clisham.optimus_core.utils import merge_tables_on_timestamp


def create_tag_dict(tag_dict_csv) -> TagDict:
    return TagDict(tag_dict_csv)


def add_across_features_by_hour(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Add calculated features to the hourly master data table.

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing source data.

    Returns:
        data_feat: Master table with additional calculated features.

    """

    # Create features
    list_features = [data]
    #group_mean = create_group_mean(parameters, data)
    #list_features.append(group_mean)
    #group_sum = create_group_sum(parameters, data)
    #list_features.append(group_sum)
    #group_sol = create_group_sol(parameters, data)
    #list_features.append(group_sol)
    camera_level = calculate_camera_level(parameters, data)
    list_features.append(camera_level)
    specific_power = calculate_specific_power(parameters, data)
    list_features.append(specific_power)

    # Merge data and created features
    data_feat = merge_tables_on_timestamp(parameters=parameters, df_list=list_features)

    return data_feat


def group_by_shift(parameters, df) -> pd.DataFrame:
    """group by shift

    Args:
        parameters: dictionary with parameters
        df: all data

    Returns:
        pd.DataFrame: df grouped
    """
    freq_group_rule = parameters["group_shift_freq"]
    col_timestamp = parameters["timestamp_col_name"]
    offset = parameters["grouping_offset"]
    agg_df = df.resample(
        rule=freq_group_rule,
        on=col_timestamp,
        offset=offset,
        closed="right",
        label="right",
    ).mean()

    return agg_df.reset_index()


def create_target_counts(parameters, td, df_raw, df_agg) -> pd.DataFrame:
    """Append counts of target to aggregate dataframe

    Args:
        parameters: dictionary with parameters
        td: Tag dictionary.
        df_raw: all data
        df_agg: grouped dataframe

    Returns:
        pd.DataFrame: df grouped
    """
    freq_group_rule = parameters["group_shift_freq"]
    col_timestamp = parameters["timestamp_col_name"]
    offset = parameters["grouping_offset"]
    targets = td.get_targets()
    # Get raw target name
    raw_tar = [
        target if len(target.split("lag_")) == 1 else target.split("lag_")[1]
        for target in targets
    ]
    dict_agg = {}
    for var in raw_tar:
        dict_agg[var] = "count"
    agg_counts = df_raw.resample(
        rule=freq_group_rule,
        offset=offset,
        on=col_timestamp,
        closed="right",
        label="right",
    ).agg(dict_agg)

    agg_counts.columns = ["calc_count_" + col for col in agg_counts.columns]
    agg_counts.reset_index(inplace=True)

    df_agg = df_agg.merge(
        agg_counts,
        how="left",
        right_on=col_timestamp,
        left_on=col_timestamp,
        validate="1:1",
    )
    return df_agg


def create_target_lags(
    td: TagDict, df_agg: pd.DataFrame, parameters: dict
) -> pd.DataFrame:
    """Append lags of targets to aggregated dataframe.

    Args:
        td: Tag dictionary.
        df_agg: Grouped dataframe.
        parameters: Dictionary of parameters.

    Returns:
        pd.DataFrame: df grouped
    """
    col_timestamp = parameters["timestamp_col_name"]
    freq = parameters["group_shift_freq"]
    targets = td.get_targets()
    # Get raw target name
    raw_tar = [
        target if len(target.split("lag_")) == 1 else target.split("lag_")[1]
        for target in targets
    ]

    df_agg.set_index(col_timestamp, inplace=True)
    shifts = df_agg[raw_tar].shift(freq="-" + freq)
    shifts.columns = ["calc_p1_lag_" + col for col in shifts.columns]
    df_agg = df_agg.merge(
        shifts, how="left", right_index=True, left_index=True, validate="1:1"
    )
    shifts_back = df_agg[raw_tar].shift(freq=freq)
    shifts_back.columns = ["calc_m1_lag_" + col for col in shifts_back.columns]
    df_agg = df_agg.merge(
        shifts_back, how="left", right_index=True, left_index=True, validate="1:1"
    )
    df_agg.reset_index(inplace=True)

    return [df_agg, df_agg]


def create_diff_celdas(
    parameters: dict, data: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    """Add differences between celdas/bancos for flotation variables.

    This is a generic function meant to be used for each flotation model. It gets the plant's configuration from the
    parameters and creates the following features:
    - 'Dif flujo de aire': difference in 'flujo de aire' between consecutive cells for each line
    - 'Dif vel burbujas': difference in 'flujo de aire' between consecutive cells for each line
    - 'Dif nivel espuma': difference in 'nivel de espuma' between consecutive cells (disabled for lack of data)
    - 'Dif nivel pulpa': difference in 'nivel de pulpa' between consecutive cells for each line

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing flotation variables.
        model_name: Name of the model.

    Returns:
        data: Master table with additional calculated flotation features.

    """
    flot_tags = parameters[f"{model_name}_flotation_tags"]

    # Create sub-df containing only flotation features to be aggregated
    tag_list = [parameters["timestamp_col_name"]]
    for group in flot_tags:
        tag_list.extend(flot_tags[group])
    df = data[tag_list].copy()
    tag_list.pop(0)  # Remove timestamp

    # Create number of lines
    plant_params = parameters[f"{model_name}_flotation_config"]
    lines_list = ["l" + str(i + 1) for i in range(plant_params["n_lines"])]
    n_celdas = plant_params["n_celdas"]

    # Iterate over lineas, celdas and bancos
    for lin in lines_list:

        for cel in range(n_celdas - 1):
            # Flujo aire
            calc_diff_flujo_aire_name = (
                f"calc_{model_name}_diff_flujo_aire_{lin}_c{cel+2}c{cel+1}"
            )
            df[calc_diff_flujo_aire_name] = (
                df[flot_tags[f"flujo_aire_{lin}_tags"][cel + 1]]
                - df[flot_tags[f"flujo_aire_{lin}_tags"][cel]]
            )
            # Velocidad burbujas
            calc_diff_vel_burbujas_name = (
                f"calc_{model_name}_diff_vel_burbuja_{lin}_c{cel+2}c{cel+1}"
            )
            df[calc_diff_vel_burbujas_name] = (
                df[flot_tags[f"vel_burbujas_{lin}_tags"][cel + 1]]
                - df[flot_tags[f"vel_burbujas_{lin}_tags"][cel]]
            )
            # # Nivel espuma  # TODO: ML -  disabled bc' data is missing, might need to remove commented text and parameters
            # calc_diff_nivel_espuma_name = (
            #         f"calc_{model_name}_diff_nivel_espuma_{lin}_c{cel+2}c{cel+1}"
            # )
            # df[calc_diff_nivel_espuma_name] = (
            #         df[flot_tags[f"nivel_espuma_{lin}_tags"][cel+1]]
            #         - df[flot_tags[f"nivel_espuma_{lin}_tags"][cel]]
            # )

    # Select and sort features
    return df[sorted(df.columns.difference(tag_list))]


def create_diff_bancos(
    parameters: dict, data: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    """Add differences between celdas/bancos for flotation variables.

    This is a generic function meant to be used for each flotation model. It gets the plant's configuration from the
    parameters and creates the following features:
    - 'Dif flujo de aire': difference in 'flujo de aire' between consecutive cells for each line
    - 'Dif vel burbujas': difference in 'flujo de aire' between consecutive cells for each line
    - 'Dif nivel espuma': difference in 'nivel de espuma' between consecutive cells (disabled for lack of data)
    - 'Dif nivel pulpa': difference in 'nivel de pulpa' between consecutive cells for each line

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing flotation variables.
        model_name: Name of the model.

    Returns:
        data: Master table with additional calculated flotation features.

    """
    #import ipdb; ipdb.set_trace()
    flot_tags = parameters[f"{model_name}_flotation_tags"]

    # Create sub-df containing only flotation features to be aggregated
    tag_list = [parameters["timestamp_col_name"]]
    for group in flot_tags:
        tag_list.extend(flot_tags[group])
    df = data[tag_list].copy()
    tag_list.pop(0)  # Remove timestamp

    # Create number of lines
    plant_params = parameters[f"{model_name}_flotation_config"]
    lines_list = ["l" + str(i + 1) for i in range(plant_params["n_lines"])]
    n_bancos = plant_params["n_bancos"]

    # Create bancos
    for lin in lines_list:

        for ban in range(n_bancos):
            # Velocidad burbujas
            calc_vel_burbujas_name = f"calc_{model_name}_vel_burbujas_{lin}_b{ban+1}"
            df[calc_vel_burbujas_name] = (
                df[flot_tags[f"vel_burbujas_{lin}_tags"][2 * ban]]
                + df[flot_tags[f"vel_burbujas_{lin}_tags"][2 * ban + 1]]
            ) / 2

    # Create diff between bancos
    for lin in lines_list:
        for ban in range(n_bancos - 1):
            # Velocidad burbujas
            calc_diff_vel_burbujas_name = (
                f"calc_{model_name}_diff_vel_burbujas_{lin}_b{ban+2}b{ban+1}"
            )
            df[calc_diff_vel_burbujas_name] = (
                df[f"calc_{model_name}_vel_burbujas_{lin}_b{ban+2}"]
                - df[f"calc_{model_name}_vel_burbujas_{lin}_b{ban+1}"]
            )

    # Select and sort features
    return df[sorted(df.columns.difference(tag_list))]


def create_mean_by_line(
    parameters: dict, data: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    """Create mean across celdas for these flotation variables:
    - Flujo aire
    - Velocidad burbujas
    - Nivel pulpa
    - Tamaño burbujas

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing flotation variables.
        model_name: Name of the model.

    Return:
        df: DataFrame with new features.
    """

    flot_tags = parameters[f"{model_name}_flotation_tags"]
    timestamp_col_name = parameters["timestamp_col_name"]

    # Create sub-df containing only flotation features to be aggregated
    tag_list = []
    for group in flot_tags:
        tag_list.extend(flot_tags[group])
    df = data[[timestamp_col_name] + tag_list].copy()

    # Create number of lines
    plant_params = parameters[f"{model_name}_flotation_config"]
    lines_list = ["l" + str(i + 1) for i in range(plant_params["n_lines"])]

    # Remove outliers # TODO:  DM - this process and hard-coded values below should be placed somewhere else
    for lin in lines_list:
        df = cut_values_from_list(df, flot_tags[f"vel_burbujas_{lin}_tags"], [1, 50])

    # Create mean by linea
    for lin in lines_list:
        df[f"calc_{model_name}_mean_vel_burbujas_{lin}"] = df[
            flot_tags[f"vel_burbujas_{lin}_tags"]
        ].mean(axis=1)

    # Select and sort features
    return df[sorted(df.columns.difference(tag_list))]


def create_delta_aire_by_line(
    parameters: dict, data: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    """Create delta between C1 and C7 for flujo de aire.

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing flotation variables.
        model_name: Name of the model.
    """

    flot_tags = parameters[f"{model_name}_flotation_tags"]

    # Create sub-df containing only flotation features to be aggregated
    timestamp_col_name = parameters["timestamp_col_name"]
    tag_list = []
    for group in flot_tags:
        tag_list.extend(flot_tags[group])
    df = data[[timestamp_col_name] + tag_list].copy()

    # Create number of lines
    plant_params = parameters[f"{model_name}_flotation_config"]
    lines_list = ["l" + str(i + 1) for i in range(plant_params["n_lines"])]
    n_celdas = plant_params["n_celdas"]

    # Create delta C(N-1)-C(1) by linea
    for lin in lines_list:
        calc_delta_flujo_aire_name = (
            f"calc_{model_name}_delta_flujo_aire_{lin}_c{n_celdas-1}c1"
        )
        df[calc_delta_flujo_aire_name] = (
            df[flot_tags[f"flujo_aire_{lin}_tags"][n_celdas - 2]]
            - df[flot_tags[f"flujo_aire_{lin}_tags"][0]]
        )

    # Select and sort features
    return df[sorted(df.columns.difference(tag_list))]


def create_fe_over_cu(
    parameters: dict, data: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    """Create Fe/Cu ratio.

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing flotation variables.
        model_name: Name of the model.
    """

    params = parameters[f"{model_name}_fe_over_cu"]
    timestamp_col_name = parameters["timestamp_col_name"]
    ley_fe = params["ley_fe_alim"]
    ley_cu = params["ley_cu_alim"]

    tag_list = ley_fe + ley_cu
    df = data[[timestamp_col_name] + tag_list].copy()

    # Create Fe/Cu
    fe_over_cu_tag_name = f"calc_{model_name}_fe_over_cu"
    df[fe_over_cu_tag_name] = df[ley_fe].mean(axis = 1) / df[ley_cu].mean(axis = 1)
    # Remove outliers  # TODO: DM - move this somewhere else
    df = cut_values_from_dict(df, {fe_over_cu_tag_name: [0, 5]})

    return df[df.columns.difference(tag_list)]


def create_target_flot(
    parameters: dict, data: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    """Create calculated recovery based on feed, concentrate and tailings grade.

    Assumptions:
    - There are several tailings lines, the final tailings grade is calculated as the average of the lines
    - Data might have time gaps (e.g. every 2/4 h), so back-fill and forward-fill are applied
    - Data is noisy so temporary clips are introduced here

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing flotation variables.
        model_name: Name of the model, used to name the calculated features.

    Returns:
        data: Master table with additional calculated flotation features.

    """
    # Select target columns
    timestamp_col_name = parameters["timestamp_col_name"]
    params_target = parameters[f"{model_name}_flotation_target"]
    ley_conc_tag = params_target["ley_conc_tag"]  # Individual
    ley_alim_tag = params_target["ley_alim_tag"]  # Individual
    ley_cola_rx_tags = params_target["ley_cola_rx_tags"]  # Leyes RX
    target_tags = ley_conc_tag + ley_alim_tag + ley_cola_rx_tags

    # Select features
    if timestamp_col_name in data.columns:
        df = data[[timestamp_col_name] + target_tags].copy()
    else:
        df = data[target_tags].copy()

    # Fill na for all targets # TODO: ML - temporary! This process should be performed somewhere else
    df[target_tags] = (
        df[target_tags].fillna(method="ffill", limit=1).fillna(method="bfill", limit=2))

    # Filter values  # ToDO: DM - temporary! This filter should be somewhere else and not hard-coded values
    df = cut_values_from_dict(df, {ley_alim_tag[0]: [0.5, 2]})  # Alimentacion
    df = cut_values_from_list(
        df, ley_cola_rx_tags, [0.02, 0.4]
    )  # Colas

    # Calculate recup en peso using colas RX  # TODO ML - split into independent functions
    new_feat_list = []
    for line, ley_cola in enumerate(ley_cola_rx_tags):
        recup_peso_name = f"calc_{model_name}_recup_peso_l{line+1}"
        df[recup_peso_name] = (df[target_tags[1]] - df[ley_cola]) / (
            df[target_tags[0]] - df[ley_cola]
        )
        new_feat_list.append(recup_peso_name)

    # Calculate recovery by line using cola TF and alim/conc RX
    weights = params_target["weights_colas"]
    ley_cola_pond = 0
    for linea, ley_cola in enumerate(ley_cola_rx_tags):
        linea_name = str(linea + 1)
        recup_name = f"calc_{model_name}_recup_l{linea_name}"
        new_feat_list.append(recup_name)
        # Calculate recup by line
        df[recup_name] = (
            df[target_tags[0]]
            * (df[target_tags[1]] - df[ley_cola])
            / (df[target_tags[1]] * (df[target_tags[0]] - df[ley_cola]))
        )
        # Update weighted ley cola
        ley_cola_pond += df[ley_cola] * weights[linea]
    # Global recovery (weighted ley colas)
    recup_wt_name = params_target["recup_wt_name"]
    df[recup_wt_name] = (
        df[target_tags[0]]
        * (df[target_tags[1]] - ley_cola_pond)
        / (df[target_tags[1]] * (df[target_tags[0]] - ley_cola_pond))
    )
    new_feat_list.append(recup_wt_name)

    df = cut_values_from_list(
        df, new_feat_list, [0, 1]
    )  # ToDO: DM - temporary! Where do we clip created features?

    if timestamp_col_name in data.columns:
        return df[[timestamp_col_name] + new_feat_list]
    else:
        return df[new_feat_list]


def create_group_mean(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Create the average of group of variables.

    Assumptions:
    - Each group has more than one tag and we want the average of them

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing variables.

    Returns:
        data: df with new variables.

    """
    # Select mill columns
    timestamp_col_name = parameters["timestamp_col_name"]
    feature_prefix = parameters["mean_grouping"]["tag_prefix"]
    groups = parameters["mean_grouping"]["groups"]
    tags = [tag for mill in groups for tag in parameters[mill]]

    # Select features
    df = data[[timestamp_col_name] + tags].copy()

    new_var_names = []
    for mill in groups:
        name = feature_prefix + mill
        new_var_names.append(name)
        tags_mill = parameters[mill]
        df[name] = df[tags_mill].mean(axis=1)

    return df[[timestamp_col_name] + new_var_names]


def create_group_sum(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Create the average of group of variables.

    Assumptions:
    - Each group has more than one tag and we want the average of them

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing variables.

    Returns:
        data: df with new variables.

    """
    # Select mill columns
    timestamp_col_name = parameters["timestamp_col_name"]
    feature_prefix = parameters["sum_grouping"]["tag_prefix"]
    groups = parameters["sum_grouping"]["groups"]
    tags = [tag for mill in groups for tag in parameters[mill]]

    # Select features
    df = data[[timestamp_col_name] + tags].copy()

    new_var_names = []
    for mill in groups:
        name = feature_prefix + mill
        new_var_names.append(name)
        tags_mill = parameters[mill]
        df[name] = df[tags_mill].sum(axis=1)

    return df[[timestamp_col_name] + new_var_names]


def create_group_sol(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Creates the % of solid of group of variables

    Assumptions:
    - Each group has more than one tag and we want the % solid of them

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing variables.

    Returns:
        data: df with new variables.

    """
    # Select mill columns
    timestamp_col_name = parameters["timestamp_col_name"]
    feature_prefix = parameters["sol_grouping"]["tag_prefix"]
    groups = parameters["sol_grouping"]["groups"]
    tags = [timestamp_col_name]
    for group in groups:
        tags_group = [
            tag for mill in parameters[group] for tag in parameters[group][mill]
        ]
        tags = tags + tags_group

    # Select features
    df = data[tags].copy()

    new_var_names = []
    for group in groups:
        tag_name = feature_prefix + group
        tags_water = parameters[group]["water"]
        tags_sol = parameters[group]["sol"]
        sum_all_water = df[tags_water].sum(axis=1)
        sum_all_solid = df[tags_sol].sum(axis=1)
        df[tag_name] = sum_all_solid / (sum_all_water + sum_all_solid)
        df[tag_name] = df[tag_name].replace([np.inf, -np.inf], 0)
        new_var_names.append(tag_name)

    return df[[timestamp_col_name] + new_var_names]


def create_on_off(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Create binary on off variables

    Assumptions:
    - Each variable represents an equiment/area being on/off above or below a value

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing variables.

    Returns:
        data: df with new variables.

    """
    # Select filter columns
    timestamp_col_name = parameters["timestamp_col_name"]
    feature_prefix = parameters["on_off"]["tag_prefix"]
    groups = parameters["on_off"]["groups"]

    # Select features
    df = data.copy()

    new_var_names = []
    for group in groups:
        for filter_col in parameters[group]:
            name = feature_prefix + filter_col
            new_var_names.append(name)
            tags_filter_col = parameters[group][filter_col]
            tag = tags_filter_col["tag"]
            off_when = tags_filter_col["off_when"]
            value = tags_filter_col["value"]
            df[name] = 1
            if off_when == "less_than":
                df.loc[df[tag] < value, name] = 0
            elif off_when == "greater_than":
                df.loc[df[tag] > value, name] = 0
            df.loc[df[tag].isna(), name] = 0

    return df[[timestamp_col_name] + new_var_names]


def create_feature_lags(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Create lagged variables grouped in parameters.

    Assumptions:
    - It is assumed that in parameters_features there is entry 'lag_grouping', which contains different groups of
    variables (e.g. ["group of variables", n_shifts]). Each group of variables in the list will be shifted 'n_shifts'
    number of periods.

    Args:
        parameters: Dictionary of parameters.
        data: Dataset grouped by shift containing all tags to be lagged.

    Returns:
        df: Dataframe of all lagged variables.

    """
    timestamp_col_name = parameters["timestamp_col_name"]
    tag_groups = parameters["lag_grouping"]["groups"]

    # Select all tags to be lagged
    tag_list = [timestamp_col_name]
    for group, n_shifts in tag_groups:
        tag_list.extend(parameters[group])

    tag_list = list(set(tag_list))
    df = data[tag_list].set_index(timestamp_col_name)
    # Create new DF to create lags for same group
    df_lags = pd.DataFrame(index=df.index)

    # Create lagged features
    for group, n_shifts in tag_groups:
        tags = parameters[group]
        type_shift = "p" * (n_shifts <= 0) + "m" * (n_shifts > 0) + str(n_shifts)
        prefix = f"calc_{type_shift}_lag_"
        df_lags[tags] = df[tags].shift(periods=n_shifts)
        df_lags.rename(columns={tag: prefix + tag for tag in tags}, inplace=True)

    return df_lags.reset_index()


def create_cuf_feature(
    parameters: dict, data: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    """Create CuF features.

     Two alternatives are provided:
    - Using weighted average of individual recovery lines
    - Using global recovery line

    Args:
    parameters: Dictionary of parameters.
    data: Dataset grouped by shift containing all tags to be lagged.
    model_name: Name of the model for which tags will be selected and tagged.

    Returns:
        df: Dataframe of all lagged variables.

    """
    # Select features
    timestamp_col_name = parameters["timestamp_col_name"]
    params = parameters[f"{model_name}_flotation_target"]
    ley_alim = params["ley_alim_tag"][0]
    recup_wt_name = params["recup_wt_name"]
    tph_tags = params["tph_tags"]
    cuf_name = params["cuf_obj_name"]
    target_tags = [ley_alim, recup_wt_name] + tph_tags

    if timestamp_col_name in data.columns:
        df = data[[timestamp_col_name] + target_tags].copy()
    else:
        df = data[target_tags].copy()

    df[cuf_name] = (
        df[tph_tags].sum(axis=1) * df[recup_wt_name] * df[ley_alim] / 100
    )  # TODO: ML split into separate fun

    if timestamp_col_name in data.columns:
        return df[[timestamp_col_name, cuf_name]].replace([np.inf, -np.inf], np.nan)
    else:
        return df[[cuf_name]].replace([np.inf, -np.inf], np.nan)


def calculate_specific_power(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Calculates the specific power

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing variables.

    Returns:
        data: df with new variables.

    """

    # Select mill columns
    timestamp_col_name = parameters["timestamp_col_name"]
    feature_prefix = parameters["specific_power_grouping"]["tag_prefix"]
    groups = parameters["specific_power_grouping"]["groups"]
    factor = parameters["specific_power_grouping"]["factor"]
    tags = []
    for group in groups:
        tags_group = [
            tag for var in parameters[group] for tag in parameters[group][var]
        ]
        tags = tags + tags_group

    # Select features
    df = data[[timestamp_col_name] + tags].copy()

    new_var_names = []
    for group in groups:
        tag_name = feature_prefix + group
        new_var_names.append(tag_name)
        tags_target = parameters[group]
        if "tonelaje" in tags_target and "potencia" in tags_target:
            tons = tags_target["tonelaje"]
            pot = tags_target["potencia"][0]
            total_tons = df[tons].sum(axis=1)
            df[tag_name] = total_tons / (df[pot] * factor[group])
            df[tag_name].replace([np.inf, -np.inf], 0, inplace=True)
        else:
            raise ValueError(
                f"{group} needs to have tag for potencia and tonelaje to calculate the specific power"
            )
    return df[[timestamp_col_name] + new_var_names]

def calculate_camera_level(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Calculates the specific power

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing variables.

    Returns:
        data: df with new variables.

    """

    # Select mill columns
    timestamp_col_name = parameters["timestamp_col_name"]
    base_tag = parameters["sag_camera"]["base_tag"]
    result_tag = parameters["sag_camera"]["result_tag"]
    tags = []
    for tag in base_tag:
        tags = tags + [tag]
    # Select features
    df = data[[timestamp_col_name] + tags].copy()

    new_var_names = []
    for tag, new_tag in zip(tags * 2,result_tag):
        tag_name = new_tag
        new_var_names.append(tag_name)
        df[tag_name] = df[tag]

    return df[[timestamp_col_name] + new_var_names]

def add_across_features_by_shift(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Add calculated features to the master data table by shift.

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing source data.

    Returns:
        data_feat: Master table with additional calculated features.

    """

    # Create features
    list_features = [data]
    lag_sol_colas = create_feature_lags(parameters, data)
    list_features.append(lag_sol_colas)

    # Merge data and created features
    data_feat = merge_tables_on_timestamp(parameters=parameters, df_list=list_features)

    return data_feat


def create_dosif_reactives(
    parameters: dict, data: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    """Create additional features transforming units from cc/min to gr/ton.

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing source data.
        model_name: Name of the model for which tags will be transformed.

    Returns:
        data_feat: Master table with additional calculated features.

    """
    # Get parameters
    timestamp_col_name = parameters["timestamp_col_name"]
    parameters_r = parameters["conversion_reactivos"]
    dry_load = parameters_r["carga_seca"]
    suffix = parameters_r["suffix"]
    parameters_m = parameters[parameters_r["tags"][model_name]]
    tph_tags = parameters_m["tph"]

    # Select all tags
    tags_used = tph_tags + list(parameters_m["react"].values())
    df = data[[timestamp_col_name] + tags_used].copy()
    # Calculate total tph
    df["tph_total"] = df[tph_tags].sum(axis=1)

    # Calculate dosif
    new_vars = []
    for reactive in parameters_m["react"]:
        tag = parameters_m["react"][reactive]
        dosif_name = f"calc_{tag}_{suffix}"
        df[dosif_name] = (df[tag] * parameters_r["densidad"][reactive] * 60).divide(
            df.tph_total * dry_load, axis=0
        )
        # Cut values out of range  # TODO: ML should this be here?
        df = cut_values_from_dict(df, {dosif_name: parameters_r["min_max"][reactive]})
        new_vars.append(dosif_name)

    return df[[timestamp_col_name] + new_vars]
