import pandas as pd
import numpy as np
import math as math

from project_clisham.optimus_core.utils import merge_tables_on_timestamp

from .features_nodes import (
    create_target_flot,
    create_diff_bancos,
    create_mean_by_line,
    create_delta_aire_by_line,
    create_fe_over_cu,
    create_cuf_feature,
    create_dosif_reactives,
)


def add_sag_features_by_hour(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Add calculated features to the hourly master data table.

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing source data.

    Returns:
        data_feat: Master table with additional calculated features.

    """

    # Create features
    list_features = [data]
    #diff_bancos = create_diff_bancos(parameters, data, model_name="fsag")
    #list_features.append(diff_bancos)
    target_flot = create_target_flot(parameters, data, model_name="fsag")
    list_features.append(target_flot)
    #mean_lines = create_mean_by_line(parameters, data, model_name="fsag")
    #list_features.append(mean_lines)
    fe_cu = create_fe_over_cu(parameters, data, model_name="fsag")
    list_features.append(fe_cu)
    #delta_lines = create_delta_aire_by_line(parameters, data, model_name="fsag")
    #list_features.append(delta_lines)
    #ganancia_feeder = _normalize_ganancia_feeder(parameters, data)
    #list_features.append(ganancia_feeder)
    tph_input = _calculate_tph_targets_sag(parameters, data)
    list_features.append(tph_input)
    perc_pebbles = _calculate_percentage_pebbles(parameters, data)
    list_features.append(perc_pebbles)
    #dosif = create_dosif_reactives(parameters, data, model_name="fsag")
    #list_features.append(dosif)

    # Merge data and created features
    data_feat = merge_tables_on_timestamp(parameters=parameters, df_list=list_features)

    return data_feat


def _normalize_ganancia_feeder(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Normalizes the ganacia feeder so they add up to one

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing variables.

    Returns:
        data: df with new variables.

    """

    # Select mill columns
    timestamp_col_name = parameters["timestamp_col_name"]
    feeder_parameters = parameters["ma2_feeders"]
    tag_prefix = feeder_parameters["tag_prefix"]
    feeder_parameters.pop("tag_prefix")
    tags = [tag for feeder in feeder_parameters for tag in feeder_parameters[feeder]]

    # Select features
    df = data[[timestamp_col_name] + tags].copy()

    new_var_names = []
    for feeder in feeder_parameters:
        tags_feeder = feeder_parameters[feeder]
        sum_all = df[tags_feeder].sum(axis=1)
        for tag in tags_feeder:
            tag_name = tag_prefix + feeder + tag
            new_var_names.append(tag_name)
            df[tag_name] = df[tag] / sum_all
            df[tag_name].replace([np.inf, -np.inf], 0, inplace=True)

    return df[[timestamp_col_name] + new_var_names]


def _calculate_tph_targets_sag(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Subtracts pebble recirculation from TPH to have a more accurate TPH measurement

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing variables.

    Returns:
        data: df with new variables.

    """

    # Select mill columns
    timestamp_col_name = parameters["timestamp_col_name"]
    target_parameters = parameters["sag_tph_pebbles"]
    tag_prefix = target_parameters["tag_prefix_target"]
    target_parameters = {m_key: target_parameters[m_key] for m_key in ["sag1", "sag2"]}
    tags = [
        tag
        for target in target_parameters
        for tag in list(target_parameters[target].values())
    ]

    # Select features
    df = data[[timestamp_col_name] + tags].copy()

    new_var_names = []
    for target in target_parameters:
        tag_name = tag_prefix + target
        new_var_names.append(tag_name)
        tags_target = target_parameters[target]
        df[tag_name] = (
            df[tags_target["feed_rate"]]
        )

    # Add ratio tph s16/s17
    tag_name = tag_prefix + "sag1_over_sag2"
    df[tag_name] = df[new_var_names[0]] / df[new_var_names[1]]
    new_var_names.append(tag_name)

    # Add total TPH
    tag_name = tag_prefix + "sag_total"
    df[tag_name] = df[new_var_names].sum(axis=1)
    new_var_names.append(tag_name)

    return df[[timestamp_col_name] + new_var_names]


def _calculate_percentage_pebbles(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Subtracts pebble recirculation from TPH to have a more accurate TPH measurement

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing variables.

    Returns:
        data: df with new variables.

    """

    # Select mill columns
    timestamp_col_name = parameters["timestamp_col_name"]
    target_parameters = parameters["sag_tph_pebbles"]
    tag_prefix = target_parameters["tag_prefix_percentage"]
    target_parameters = {m_key: target_parameters[m_key] for m_key in ["sag1", "sag2"]}
    tags = [
        tag
        for target in target_parameters
        for tag in list(target_parameters[target].values())
    ]

    # Select features
    df = data[[timestamp_col_name] + tags].copy()

    new_var_names = []
    for target in target_parameters:
        tag_name = tag_prefix + target
        new_var_names.append(tag_name)
        tags_target = target_parameters[target]

        sum_pebbles = df[tags_target["pebbles_producidos"]]
        df[tag_name] = sum_pebbles / df[tags_target["feed_rate"]]
        df[tag_name].replace([np.inf, -np.inf], 0, inplace=True)
        df.loc[df[tag_name] > 1, tag_name] = 0

    return df[[timestamp_col_name] + new_var_names]


def _calculate_jb_jc(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Create the JC and JB level of group of variables
    JB: ball filling level
    JC: total load filling level
    Assumptions:
    - With control variables, jb and jc are estimated

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing variables.

    Returns:
        data: df with new variables.
    """

    # Select mill columns
    timestamp_col_name = parameters["timestamp_col_name"]
    const = parameters["ma2_jb_jc_const"]
    jb_jc_parameters = parameters["ma2_jb_jc"]
    feature_prefix = jb_jc_parameters["tag_prefix"]
    jb_jc_parameters.pop("tag_prefix")

    # Select tags
    tags = [timestamp_col_name]
    for group in jb_jc_parameters:
        for var in jb_jc_parameters[group]:
            tags_group = jb_jc_parameters[group][var]
            tags.append(tags_group)

    # Select features
    df = data[tags].copy()
    new_var_names = []
    d_mol = const["k1"] / const["k2"]  # Diametro molino, convertido pies
    l = const["k3"] / const["k2"]  # Proporcion dimensiones molino
    jc_ideal = const["jc_ideal"]
    m_jc = const[
        "m_JC"
    ]  # Relacion entre JC y Presion de descansos, pendiente 1.2 Domingo, 0.33 SIP
    jfill = const[
        "Jfill"
    ]  # Llenado de mineral en interticios entre bolas y rocas lo que se llena de k17
    k6 = const["k6"]  # Convesion de % de velocidad critica
    k2 = const["k2"]  # pies a metros
    k7 = const["k7"]  # Constante de modelo de potencias
    k8 = const["k8"]  # Constante del modelo de potencias
    k9 = const["k9"]  # Para convertir a %
    min_vel = const["min_vel"]
    min_sol = const["min_sol"]
    max_sol = const["max_sol"]
    min_pres = const["min_pres"]
    max_pres = const["max_pres"]
    min_pot = const["min_pot"]
    k10 = const["k10"]  # Densidad del mineral
    k11 = const["k11"]  # Dendidad del acero/bolas
    min_jb = const["min_jb"]  # rango de validez de modelo
    k12 = const["k12"]  # rango de validez de modelo
    k13 = const["k13"]  # rango de validez de modelo
    k14 = const["k14"]  # rango de validez de modelo
    k15 = const["k15"]  # rango de validez de modelo
    k16 = const["k16"]  # rango de validez de modelo
    k17 = const["k17"]  # rango de validez de modelo
    k18 = const["k18"]  # rango de validez de modelo

    for group in jb_jc_parameters:
        tag_name = feature_prefix + group
        velocidad = jb_jc_parameters[group]["velocidad"]
        presion = jb_jc_parameters[group]["presion"]
        presion_hl = jb_jc_parameters[group]["presion_hl"]
        cp = jb_jc_parameters[group]["cp"]
        potencia = jb_jc_parameters[group]["potencia"]
        df["Ro_pulpa"] = 1 / (df[cp] / k10 / k9 + (1 - df[cp] / k9))
        df["Nc"] = df[velocidad] / k6 * math.sqrt(d_mol * k2) * k9

        for i in range(len(m_jc)):
            df["bt" + str(i)] = jc_ideal - m_jc[i] * df[presion_hl]
            df["JC" + str(i)] = m_jc[i] * df[presion] + df["bt" + str(i)]
            df["cte" + str(i)] = (
                k7
                * (d_mol * k2) ** k12
                * l
                * k2
                * df["JC" + str(i)]
                / k9
                * (1 - k8 * df["JC" + str(i)] / k9)
                * df["Nc"]
                / k9
                * math.sin(k13 / k14 * math.pi)
            )
            df["cte" + str(i)] = np.where(
                df["cte" + str(i)] > k9, df["cte" + str(i)], np.nan
            )
            df["Ro_ap" + str(i)] = df[potencia] * k15 * k16 / df["cte" + str(i)]
            df["cte1" + str(i)] = (
                (
                    df["JC" + str(i)] / k9 * df["Ro_ap" + str(i)]
                    - df["Ro_pulpa"] * (df["JC" + str(i)] / k9 * k17) * jfill
                )
                / k18
                / (k11 - k10)
            )
            df["cte2" + str(i)] = k10 * df["JC" + str(i)] / k9 / (k11 - k10)
            df["JB_" + str(i)] = np.where(
                df["cte" + str(i)] > k9,
                df["cte1" + str(i)] - df["cte2" + str(i)],
                np.nan,
            )
            # JA - TODO revisar prioridad de cortes con Diane
            df["JB_" + str(i)] = np.where(
                (df["JB_" + str(i)] > min_jb), df["JB_" + str(i)], np.nan
            )
            df.loc[
                ~(
                    (df[cp] >= min_sol)
                    & (df[cp] <= max_sol)
                    & (df[potencia] >= min_pot)
                    & (df[velocidad] >= min_vel)
                    & (df[presion] <= max_pres)
                    & (df[presion] >= min_pres)
                ),
                "JB_" + str(i),
            ] = np.nan

            df = df.fillna(method="pad")
            # TODO JA - entender los filtros, constantes adicionales y por ultimo, la imputacion
            tag_name_jb = tag_name + "_m_jc_" + str(m_jc[i]).replace(".", "_") + "_jb"
            tag_name_jc = tag_name + "_m_jc_" + str(m_jc[i]).replace(".", "_") + "_jc"
            df[tag_name_jb] = df["JB_" + str(i)]
            df[tag_name_jc] = df["JC" + str(i)]
            new_var_names.append(tag_name_jb)
            new_var_names.append(tag_name_jc)

    return df[[timestamp_col_name] + new_var_names]


def add_sag_features_by_shift(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Add calculated features to the aggregated-by-shift master data table.

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing source data.

    Returns:
        data_feat: Master table with additional calculated features.

    """

    # Create features
    list_features = [data]
    cuf_features = create_cuf_feature(parameters, data, model_name="fsag")
    list_features.append(cuf_features)

    # Merge data and created features
    data_feat = merge_tables_on_timestamp(parameters=parameters, df_list=list_features)

    return data_feat
