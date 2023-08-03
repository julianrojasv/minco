
import pandas as pd
import numpy as np
from functools import partial, reduce, update_wrapper
from typing import Any, Callable, Dict, List, Optional


#normalizando columnas (optimus x)
def parse_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms the column names into standarized names

    Args:
        df (pd.DataFrame): raw data

    Returns:
        pd.DataFrame: data with standarized names
    """
    # Tag dictionary
    df.columns = (
        df.columns.str.lower()
        .str.replace(":", "_")
        .str.replace(".", "_")
        .str.replace("-", "_")
        .str.replace("'", "")
        .str.replace(",", "_")
        .str.replace(" ", "_")
        .str.strip()
    )
    df = df.apply(pd.to_numeric, args=('coerce',))
    df = df.loc[:, ~df.columns.duplicated()]
    df = df[~df.index.duplicated(keep='first')]
    return df




#set de NaN a outliers fuera de percentil 0-99%
def remove_outliers(dataset):
   # import ipdb;ipdb.set_trace();
    upper_thresh = 99
    lower_thresh = 0
    upper = dataset.apply(lambda l: np.nanpercentile(l, upper_thresh))
    lower = dataset.apply(lambda l: np.nanpercentile(l, lower_thresh))
    datasetfilter = dataset.apply(lambda l: l.clip(upper=upper[l.name]))
    datasetfilter = datasetfilter.apply(lambda l: l.clip(lower=lower[l.name]))
    mean = datasetfilter.apply(lambda l: np.nanpercentile(l, 50))
    df95 = datasetfilter.apply(lambda l: np.nanpercentile(l, 95))
    std = datasetfilter[datasetfilter < df95].std()
    mean_std = mean + std * 10
    for x in range(len(datasetfilter.columns)):
        datasetfilter.loc[datasetfilter.iloc[:, x] > mean_std[datasetfilter.columns[x]], datasetfilter.columns[x]] = mean_std[
            datasetfilter.columns[x]]
    return datasetfilter




#Eliminar ventanas de resistros con más de consec_nan_max NaN consecutivos en la target
#Se asume Fecha como index
def consecutive_nan_removal(dataset,target):

    dataset = pd.concat([dataset, (dataset[target].isnull().astype(int)
                               .groupby(dataset[target].notnull().astype(int).cumsum())
                               .cumsum().to_frame('consec_count')
                               )
                       ],
                      axis=1
                      )
    l = []
    consec_nan_max = 6 * 8  # 1 turno

    dataset['Fecha'] = dataset.index
    dataset.reset_index(drop=True, inplace=True)
    k = 0
    for i, row in dataset.iterrows():
        if row['consec_count'] >= 1:
            k = k + 1
        else:
            if k >= consec_nan_max:
                # print(i)
                # print(k)
                l.append((i - k, i))
                k = 0
            else:
                k = 0

    arr = np.ones(len(dataset)) == True
    for k in l:
        arr[k[0]:k[1]] = False
    dataset = dataset[arr]
    dataset.set_index('Fecha', inplace=True)
    return dataset


#imputación de NaN por medianas.
#imputación de negativos por 0's
def data_imputation(dataset):
    dataset = dataset.fillna(dataset.median())
    dataset[dataset < 0] = 0
    return dataset

#agrupamiento por hora
def data_group_per_hour(dataset):
   # import ipdb;ipdb.set_trace();
   # dataset.set_index('Fecha', inplace=True)
    dataset.index = pd.to_datetime(dataset.index,format = '%Y-%m-%d %H:%M:%S')
    dataset = dataset.resample('1H', offset=0).mean()
    dataset.dropna(subset=[x for x in dataset.columns if 'target' in x.lower()])
    return dataset


#eliminar columnas con más NaN% que drop_columns_threshold
def drop_columns_thres(dataset):
    drop_columns_threshold = 60  # en porcentaje
    max_number_of_nas = round(dataset.shape[0] * (drop_columns_threshold) / 100)
    dataset = dataset.loc[:, (dataset.isnull().sum(axis=0) <= max_number_of_nas)]
    return dataset



#variables derivadas por modelo y generales
def create_features(params: dict, dataset: pd.DataFrame) -> pd.DataFrame:
    hour = 6
#    if model == 'R2':
        # target
    dataset.loc[:, 'target_torque_r2'] = dataset[['r_agua_r2_torque_norte', 'r_agua_r2_torque_sur']].mean(
            axis=1)
        # Variables de cambios en el torque
    dataset.loc[:, 'delta_torque_r2'] = dataset['target_torque_r2'] - dataset['target_torque_r2'].shift(1)
    dataset.loc[:, 'delta_torque_r2_1hora'] = dataset['target_torque_r2'] - dataset['target_torque_r2'].shift(
            1 * hour)
    dataset.loc[:, 'delta_torque_r2_2hora'] = dataset['target_torque_r2'] - dataset['target_torque_r2'].shift(
            2 * hour)
    dataset.loc[:, 'delta_torque_r2_4hora'] = dataset['target_torque_r2'] - dataset['target_torque_r2'].shift(
            4 * hour)
    dataset.loc[:, 'delta_torque_r2_8hora'] = dataset['target_torque_r2'] - dataset['target_torque_r2'].shift(
            8 * hour)
        # Dosificacion floculante promedio (suavizado 2 horas)
    dataset.loc[:, 'dosificacion_acumulada_floculante_r2'] = dataset['r_agua_flocu_fi_3820'].shift(0).ewm(
            halflife=2*hour).mean().shift(8*hour)
        #promedio densimetros suavizados
    dataset.loc[:, 'promedio_densimetros_r2'] = dataset[['r_agua_r2_dit_01', 'r_agua_r2_dit_02']].mean(axis=1)
    dataset.loc[:, 'media_mov_densimetros_r2'] = dataset['promedio_densimetros_r2'].shift(2*hour).ewm(
            halflife=hour * 1).mean().shift(1*hour)

        #promedio apertura valvulas de descarga de relaves
    dataset.loc[:, 'promedio_apertura_valv_descarga_r2'] = dataset[['r_agua_r2_valv_01', 'r_agua_r2_valv_02',
                                                                    'r_agua_r2_valv_03',
                                                                    'r_agua_r2_valv_04']].mean(axis=1)
    dataset.loc[:, 'promedio_apertura_valv_descarga_l1_r2'] = dataset[['r_agua_r2_valv_01',
                                                                       'r_agua_r2_valv_04']].mean(axis=1)
    dataset.loc[:, 'promedio_apertura_valv_descarga_l2_r2'] = dataset[['r_agua_r2_valv_02',
                                                                       'r_agua_r2_valv_03']].mean(axis=1)

    dataset.loc[:, 'nivel_rastras_r2'] = dataset['r_agua_r2_nivel_rastras'].rolling(6*hour).mean()

    dataset.loc[:, 'agua_clara_media_r2'] = dataset['r_agua_r2_a330_fit_208'].rolling(2*hour).mean()

    dataset.loc[:, 'flujo_alimentacion_r2'] = dataset['r_agua_r2_a330_fit_205'].rolling(6*hour).mean()

    dataset.loc[:, 'nivel_interfaz_r2'] = dataset['r_agua_r2_a330_lit_206_pulg'].rolling(6*hour).mean()

    dataset.loc[:, 'valvula_fcv_r2'] = dataset['r_agua_r2_a330_zit_203'].rolling(6*hour).mean()

    dataset.loc[:, 'turbidez_r2'] = dataset['r_agua_r2_a330_ait_207'].rolling(2*hour).mean()

    #if model == 'R3':
        # target
    dataset.loc[:, 'target_torque_r3'] = dataset[['r_agua_r3_torque_norte', 'r_agua_r3_torque_sur']].mean(
            axis=1)
        # Variables de cambios en el torque
    dataset.loc[:, 'delta_torque_r3'] = dataset['target_torque_r3'] - dataset['target_torque_r3'].shift(1)
    dataset.loc[:, 'delta_torque_r3_1hora'] = dataset['target_torque_r3'] - dataset['target_torque_r3'].shift(
            1 * hour)
    dataset.loc[:, 'delta_torque_r3_2hora'] = dataset['target_torque_r3'] - dataset['target_torque_r3'].shift(
            2 * hour)
    dataset.loc[:, 'delta_torque_r3_4hora'] = dataset['target_torque_r3'] - dataset['target_torque_r3'].shift(
            4 * hour)
    dataset.loc[:, 'delta_torque_r3_8hora'] = dataset['target_torque_r3'] - dataset['target_torque_r3'].shift(
            8 * hour)
        # Dosificacion floculante promedio (suavizado 2 horas)
    dataset.loc[:, 'dosificacion_acumulada_floculante_r3'] = dataset['r_agua_flocu_fi_3824'].shift(0 * hour).ewm(
            halflife=2 * hour).mean().shift(8 * hour)
        # promedio densimetros suavizados
    dataset.loc[:, 'promedio_densimetros_r3'] = dataset[['r_agua_r3_dit_01', 'r_agua_r3_dit_02']].mean(axis=1)
    dataset.loc[:, 'media_mov_densimetros_r3'] = dataset['promedio_densimetros_r3'].shift(2 * hour).ewm(
            halflife=hour * 1).mean().shift(1 * hour)

        # promedio apertura valvulas de descarga de relaves
    dataset.loc[:, 'promedio_apertura_valv_descarga_r3'] = dataset[['r_agua_r3_valv_01', 'r_agua_r3_valv_02',
                                                                        'r_agua_r3_valv_03',
                                                                        'r_agua_r3_valv_04']].mean(axis=1)
    dataset.loc[:, 'promedio_apertura_valv_descarga_l1_r3'] = dataset[['r_agua_r3_valv_01',
                                                                       'r_agua_r3_valv_04']].mean(axis=1)
    dataset.loc[:, 'promedio_apertura_valv_descarga_l2_r3'] = dataset[['r_agua_r3_valv_02',
                                                                       'r_agua_r3_valv_03']].mean(axis=1)
    dataset.loc[:, 'nivel_rastras_r3'] = dataset['r_agua_r3_nivel_rastras'].rolling(6 * hour).mean()

    dataset.loc[:, 'agua_clara_media_r3'] = dataset['r_agua_r3_a330_fit_308'].rolling(2 * hour).mean()

    dataset.loc[:, 'flujo_alimentacion_r3'] = dataset['r_agua_r3_a330_fit_305'].rolling(6 * hour).mean()

    dataset.loc[:, 'nivel_interfaz_r3'] = dataset['r_agua_r3_a330_lit_306_pulg'].rolling(6 * hour).mean()

    dataset.loc[:, 'valvula_fcv_r3'] = dataset['r_agua_r3_a330_zit_303'].rolling(6 * hour).mean()

    dataset.loc[:, 'turbidez_r3'] = dataset['r_agua_r3_a330_ait_307'].rolling(2 * hour).mean()





    #if model == 'R4':
        # target
    dataset.loc[:, 'target_torque_r4'] = dataset[['r_agua_r4_torque_norte', 'r_agua_r4_torque_sur']].mean(
            axis=1)
        # Variables de cambios en el torque
    dataset.loc[:, 'delta_torque_r4'] = dataset['target_torque_r4'] - dataset['target_torque_r4'].shift(1)
    dataset.loc[:, 'delta_torque_r4_1hora'] = dataset['target_torque_r4'] - dataset['target_torque_r4'].shift(
            1 * hour)
    dataset.loc[:, 'delta_torque_r4_2hora'] = dataset['target_torque_r4'] - dataset['target_torque_r4'].shift(
            2 * hour)
    dataset.loc[:, 'delta_torque_r4_4hora'] = dataset['target_torque_r4'] - dataset['target_torque_r4'].shift(
            4 * hour)
    dataset.loc[:, 'delta_torque_r4_8hora'] = dataset['target_torque_r4'] - dataset['target_torque_r4'].shift(
            8 * hour)
        # Dosificacion floculante promedio (suavizado 2 horas)
    dataset.loc[:, 'dosificacion_acumulada_floculante_r4'] = dataset['r_agua_flocu_fi_3828'].shift(0*hour).ewm(
            halflife=2*hour).mean().shift(8*hour)
        #promedio densimetros suavizados
    dataset.loc[:, 'promedio_densimetros_r4'] = dataset[['r_agua_r4_dit_01', 'r_agua_r4_dit_02']].mean(axis=1)
    dataset.loc[:, 'media_mov_densimetros_r4'] = dataset['promedio_densimetros_r4'].shift(2*hour).ewm(
            halflife=hour * 1).mean().shift(1*hour)

        #promedio apertura valvulas de descarga de relaves
    dataset.loc[:, 'promedio_apertura_valv_descarga_r4'] = dataset[['r_agua_r4_valv_01', 'r_agua_r4_valv_02',
                                                                         'r_agua_r4_valv_03',
                                                                         'r_agua_r4_valv_04']].mean(axis=1)
    dataset.loc[:, 'promedio_apertura_valv_descarga_l1_r4'] = dataset[['r_agua_r4_valv_01',
                                                                       'r_agua_r4_valv_04']].mean(axis=1)
    dataset.loc[:, 'promedio_apertura_valv_descarga_l2_r4'] = dataset[['r_agua_r4_valv_02',
                                                                       'r_agua_r4_valv_03']].mean(axis=1)
    dataset.loc[:, 'nivel_rastras_r4'] = dataset['r_agua_r4_nivel_rastras'].rolling(6*hour).mean()

    dataset.loc[:, 'agua_clara_media_r4'] = dataset['r_agua_r4_a330_fit_408'].rolling(2*hour).mean()

    dataset.loc[:, 'flujo_alimentacion_r4'] = dataset['r_agua_r4_a330_fit_405'].rolling(6*hour).mean()

    dataset.loc[:, 'nivel_interfaz_r4'] = dataset['r_agua_r4_a330_lit_406_pulg'].rolling(6*hour).mean()

    dataset.loc[:, 'valvula_fcv_r4'] = dataset['r_agua_r4_a330_zit_403'].rolling(6*hour).mean()

    dataset.loc[:, 'turbidez_r4'] = dataset['r_agua_r4_a330_ait_407'].rolling(2*hour).mean()



    #if model == 'R5':
        # target
    dataset.loc[:, 'target_torque_r5'] = dataset[['r_agua_r5_torque_norte', 'r_agua_r5_torque_sur']].mean(
            axis=1)
        # Variables de cambios en el torque
    dataset.loc[:, 'delta_torque_r5'] = dataset['target_torque_r5'] - dataset['target_torque_r5'].shift(1)
    dataset.loc[:, 'delta_torque_r5_1hora'] = dataset['target_torque_r5'] - dataset['target_torque_r5'].shift(
            1 * hour)
    dataset.loc[:, 'delta_torque_r5_2hora'] = dataset['target_torque_r5'] - dataset['target_torque_r5'].shift(
            2 * hour)
    dataset.loc[:, 'delta_torque_r5_4hora'] = dataset['target_torque_r5'] - dataset['target_torque_r5'].shift(
            4 * hour)
    dataset.loc[:, 'delta_torque_r5_8hora'] = dataset['target_torque_r5'] - dataset['target_torque_r5'].shift(
            8 * hour)
        # Dosificacion floculante promedio (suavizado 2 horas)
    dataset.loc[:,'dosificacion_acumulada_floculante_r5'] = dataset['r_agua_flocu_fi_3832'].shift(0*hour).ewm(
            halflife=2*hour).mean().shift(8*hour)
        #promedio densimetros suavizados
    dataset.loc[:, 'promedio_densimetros_r5'] = dataset[['r_agua_r5_dit_01', 'r_agua_r5_dit_02']].mean(axis=1)
    dataset.loc[:, 'media_mov_densimetros_r5'] = dataset['promedio_densimetros_r5'].shift(2*hour).ewm(
            halflife=hour * 1).mean().shift(1*hour)

        #promedio apertura valvulas de descarga de relaves
    dataset.loc[:, 'promedio_apertura_valv_descarga_r5'] = dataset[['r_agua_r5_valv_01', 'r_agua_r5_valv_02',
                                                                         'r_agua_r5_valv_03',
                                                                         'r_agua_r5_valv_04']].mean(axis=1)
    dataset.loc[:, 'promedio_apertura_valv_descarga_l1_r5'] = dataset[['r_agua_r5_valv_01',
                                                                       'r_agua_r5_valv_04']].mean(axis=1)
    dataset.loc[:, 'promedio_apertura_valv_descarga_l2_r5'] = dataset[['r_agua_r5_valv_02',
                                                                       'r_agua_r5_valv_03']].mean(axis=1)
    dataset.loc[:, 'nivel_rastras_r5'] = dataset['r_agua_r5_nivel_rastras'].rolling(6*hour).mean()

    dataset.loc[:, 'agua_clara_media_r5'] = dataset['r_agua_r5_a330_fit_508'].rolling(2*hour).mean()

    dataset.loc[:, 'flujo_alimentacion_r5'] = dataset['r_agua_r5_a330_fit_505'].rolling(6*hour).mean()

    dataset.loc[:, 'nivel_interfaz_r5'] = dataset['r_agua_r5_a330_lit_506_pulg'].rolling(6*hour).mean()

    dataset.loc[:, 'valvula_fcv_r5'] = dataset['r_agua_r5_a330_zit_503'].rolling(6*hour).mean()

    #dataset.loc[:, 'turbidez_r5'] = dataset['r_agua_r3_a330_ait_507'].rolling(2*hour).mean()


    #if model == 'R6':
        # target
    dataset.loc[:, 'target_torque_r6'] = dataset[['r_agua_r6_torque_norte', 'r_agua_r6_torque_sur']].mean(
            axis=1)
        # Variables de cambios en el torque
    dataset.loc[:, 'delta_torque_r6'] = dataset['target_torque_r6'] - dataset['target_torque_r6'].shift(1)
    dataset.loc[:, 'delta_torque_r6_1hora'] = dataset['target_torque_r6'] - dataset['target_torque_r6'].shift(
            1 * hour)
    dataset.loc[:, 'delta_torque_r6_2hora'] = dataset['target_torque_r6'] - dataset['target_torque_r6'].shift(
            2 * hour)
    dataset.loc[:, 'delta_torque_r6_4hora'] = dataset['target_torque_r6'] - dataset['target_torque_r6'].shift(
            4 * hour)
    dataset.loc[:, 'delta_torque_r6_8hora'] = dataset['target_torque_r6'] - dataset['target_torque_r6'].shift(
            8 * hour)
        # Dosificacion floculante promedio (suavizado 2 horas)
    dataset.loc[:,'dosificacion_acumulada_floculante_r6'] = dataset['r_agua_flocu_fi_3836'].shift(0*hour).ewm(
            halflife=2*hour).mean().shift(8*hour)
        #promedio densimetros suavizados
    dataset.loc[:, 'promedio_densimetros_r6'] = dataset[['r_agua_r6_dit_01', 'r_agua_r6_dit_02']].mean(axis=1)
    dataset.loc[:, 'media_mov_densimetros_r6'] = dataset['promedio_densimetros_r6'].shift(2*hour).ewm(
            halflife=hour * 1).mean().shift(1*hour)

        #promedio apertura valvulas de descarga de relaves
    dataset.loc[:, 'promedio_apertura_valv_descarga_r6'] = dataset[['r_agua_r6_valv_01', 'r_agua_r6_valv_02',
                                                                         'r_agua_r6_valv_03',
                                                                         'r_agua_r6_valv_04']].mean(axis=1)
    dataset.loc[:, 'promedio_apertura_valv_descarga_l1_r6'] = dataset[['r_agua_r6_valv_01',
                                                                       'r_agua_r6_valv_04']].mean(axis=1)
    dataset.loc[:, 'promedio_apertura_valv_descarga_l2_r6'] = dataset[['r_agua_r6_valv_02',
                                                                       'r_agua_r6_valv_03']].mean(axis=1)
    dataset.loc[:, 'nivel_rastras_r6'] = dataset['r_agua_r6_nivel_rastras'].rolling(6*hour).mean()

    dataset.loc[:, 'agua_clara_media_r6'] = dataset['r_agua_r6_a330_fit_608'].rolling(2*hour).mean()

    dataset.loc[:, 'flujo_alimentacion_r6'] = dataset['r_agua_r6_a330_fit_605'].rolling(6*hour).mean()

    dataset.loc[:, 'nivel_interfaz_r6'] = dataset['r_agua_r6_a330_lit_606_pulg'].rolling(6*hour).mean()

    dataset.loc[:, 'valvula_fcv_r6'] = dataset['r_agua_r6_a330_zit_603'].rolling(6*hour).mean()

    dataset.loc[:, 'turbidez_r6'] = dataset['r_agua_r6_a330_ait_607'].rolling(2*hour).mean()


    #if model == 'R7':
        # target
    dataset.loc[:, 'target_torque_r7'] = dataset[['r_agua_r7_torque_norte', 'r_agua_r7_torque_sur']].mean(
            axis=1)
        # Variables de cambios en el torque
    dataset.loc[:, 'delta_torque_r7'] = dataset['target_torque_r7'] - dataset['target_torque_r7'].shift(1)
    dataset.loc[:, 'delta_torque_r7_1hora'] = dataset['target_torque_r7'] - dataset['target_torque_r7'].shift(
            1 * hour)
    dataset.loc[:, 'delta_torque_r7_2hora'] = dataset['target_torque_r7'] - dataset['target_torque_r7'].shift(
            2 * hour)
    dataset.loc[:, 'delta_torque_r7_4hora'] = dataset['target_torque_r7'] - dataset['target_torque_r7'].shift(
            4 * hour)
    dataset.loc[:, 'delta_torque_r7_8hora'] = dataset['target_torque_r7'] - dataset['target_torque_r7'].shift(
            8 * hour)
        # Dosificacion floculante promedio (suavizado 2 horas)
    dataset.loc[:,'dosificacion_acumulada_floculante_r7'] = dataset['r_agua_flocu_fi_3840'].shift(0*hour).ewm(
            halflife=2*hour).mean().shift(8*hour)
        #promedio densimetros suavizados
    dataset.loc[:, 'promedio_densimetros_r7'] = dataset[['r_agua_r7_dit_01', 'r_agua_r7_dit_02']].mean(axis=1)
    dataset.loc[:, 'media_mov_densimetros_r7'] = dataset['promedio_densimetros_r7'].shift(2*hour).ewm(
            halflife=hour * 1).mean().shift(1*hour)

        #promedio apertura valvulas de descarga de relaves
    dataset.loc[:, 'promedio_apertura_valv_descarga_r7'] = dataset[['r_agua_r7_valv_01', 'r_agua_r7_valv_02',
                                                                         'r_agua_r7_valv_03',
                                                                         'r_agua_r7_valv_04']].mean(axis=1)
    dataset.loc[:, 'promedio_apertura_valv_descarga_l1_r7'] = dataset[['r_agua_r7_valv_01',
                                                                       'r_agua_r7_valv_04']].mean(axis=1)
    dataset.loc[:, 'promedio_apertura_valv_descarga_l2_r7'] = dataset[['r_agua_r7_valv_02',
                                                                       'r_agua_r7_valv_03']].mean(axis=1)
    dataset.loc[:, 'nivel_rastras_r7'] = dataset['r_agua_r7_nivel_rastras'].rolling(6*hour).mean()

    dataset.loc[:, 'agua_clara_media_r7'] = dataset['r_agua_r7_a330_fit_708'].rolling(2*hour).mean()

    dataset.loc[:, 'flujo_alimentacion_r7'] = dataset['r_agua_r7_a330_fit_705'].rolling(6*hour).mean()

    dataset.loc[:, 'nivel_interfaz_r7'] = dataset['r_agua_r7_a330_lit_706_pulg'].rolling(6*hour).mean()

    dataset.loc[:, 'valvula_fcv_r7'] = dataset['r_agua_r7_a330_zit_703'].rolling(6*hour).mean()

    dataset.loc[:, 'turbidez_r7'] = dataset['r_agua_r7_a330_ait_707'].rolling(2*hour).mean()

    #if model == 'R9':
        # target
    dataset.loc[:, 'target_torque_r9'] = dataset[['r_agua_r9_torque_norte', 'r_agua_r9_torque_sur']].mean(
            axis=1)
        # Variables de cambios en el torque
    dataset.loc[:, 'delta_torque_r9'] = dataset['target_torque_r9'] - dataset['target_torque_r9'].shift(1)
    dataset.loc[:, 'delta_torque_r9_1hora'] = dataset['target_torque_r9'] - dataset['target_torque_r9'].shift(
            1 * hour)
    dataset.loc[:, 'delta_torque_r9_2hora'] = dataset['target_torque_r9'] - dataset['target_torque_r9'].shift(
            2 * hour)
    dataset.loc[:, 'delta_torque_r9_4hora'] = dataset['target_torque_r9'] - dataset['target_torque_r9'].shift(
            4 * hour)
    dataset.loc[:, 'delta_torque_r9_8hora'] = dataset['target_torque_r9'] - dataset['target_torque_r9'].shift(
            8 * hour)
        # Dosificacion floculante promedio (suavizado 2 horas)
    dataset.loc[:,'dosificacion_acumulada_floculante_r9'] = dataset['r_agua_flocu_fi_3844'].shift(0*hour).ewm(
            halflife=2*hour).mean().shift(8*hour)
        #promedio densimetros suavizados
    dataset.loc[:, 'promedio_densimetros_r9'] = dataset[['r_agua_r9_dit_01', 'r_agua_r9_dit_02']].mean(axis=1)
    dataset.loc[:, 'media_mov_densimetros_r9'] = dataset['promedio_densimetros_r9'].shift(2*hour).ewm(
            halflife=hour * 1).mean().shift(1*hour)

        #promedio apertura valvulas de descarga de relaves
    dataset.loc[:, 'promedio_apertura_valv_descarga_r9'] = dataset[['r_agua_r9_valv_01', 'r_agua_r9_valv_02',
                                                                         'r_agua_r9_valv_03',
                                                                         'r_agua_r9_valv_04']].mean(axis=1)
    dataset.loc[:, 'promedio_apertura_valv_descarga_l1_r9'] = dataset[['r_agua_r9_valv_01',
                                                                       'r_agua_r9_valv_04']].mean(axis=1)
    dataset.loc[:, 'promedio_apertura_valv_descarga_l2_r9'] = dataset[['r_agua_r9_valv_02',
                                                                       'r_agua_r9_valv_03']].mean(axis=1)
    dataset.loc[:, 'nivel_rastras_r9'] = dataset['r_agua_r9_nivel_rastras'].rolling(6*hour).mean()

    dataset.loc[:, 'agua_clara_media_r9'] = dataset['r_agua_r9_a330_fit_908'].rolling(2*hour).mean()

    dataset.loc[:, 'flujo_alimentacion_r9'] = dataset['r_agua_r9_a330_fit_905'].rolling(6*hour).mean()

    #dataset.loc[:, 'nivel_interfaz_r9'] = dataset['r_agua_r9_a330_lit_906_pulg'].rolling(6*hour).mean()

    dataset.loc[:, 'valvula_fcv_r9'] = dataset['r_agua_r9_a330_zit_903'].rolling(6*hour).mean()

    dataset.loc[:, 'turbidez_r9'] = dataset['r_agua_r9_a330_ait_907'].rolling(2*hour).mean()


    #if model == 'R10':
        # target
    dataset.loc[:, 'target_torque_r10'] = dataset['r_agua_r10_torque']
        # Variables de cambios en el torque
    dataset.loc[:, 'delta_torque_r10'] = dataset['target_torque_r10'] - dataset['target_torque_r10'].shift(1)
    dataset.loc[:, 'delta_torque_r10_1hora'] = dataset['target_torque_r10'] - dataset['target_torque_r10'].shift(
            1 * hour)
    dataset.loc[:, 'delta_torque_r10_2hora'] = dataset['target_torque_r10'] - dataset['target_torque_r10'].shift(
            2 * hour)
    dataset.loc[:, 'delta_torque_r10_4hora'] = dataset['target_torque_r10'] - dataset['target_torque_r10'].shift(
            4 * hour)
    dataset.loc[:, 'delta_torque_r10_8hora'] = dataset['target_torque_r10'] - dataset['target_torque_r10'].shift(
            8 * hour)
        # Dosificacion floculante promedio (suavizado 2 horas)
    dataset.loc[:, 'dosificacion_acumulada_floculante_r10'] = dataset['r_agua_r10_fit_1154'].shift(0*hour).ewm(
            halflife=2*hour).mean().shift(8*hour)
        #promedio densimetros suavizados
    dataset.loc[:, 'promedio_densimetros_r10'] = dataset[['r_agua_r10_dit1107_dit1107', 'r_agua_r10_dit1111_dit1111']].mean(axis=1)
    dataset.loc[:, 'media_mov_densimetros_r10'] = dataset['promedio_densimetros_r10'].shift(2*hour).ewm(
            halflife=hour * 1).mean().shift(1*hour)

        #promedio apertura valvulas de descarga de relaves
    dataset.loc[:, 'promedio_apertura_valv_descarga_r10'] = dataset[['r_agua_r10_fit_1106',
                                                                         'r_agua_r10_fit_1110']].mean(axis=1)
    dataset.loc[:, 'promedio_apertura_valv_descarga_l1_r10'] = dataset['r_agua_r10_fit_1106']
    dataset.loc[:, 'promedio_apertura_valv_descarga_l2_r10'] = dataset['r_agua_r10_fit_1110']

    dataset.loc[:, 'agua_clara_media_r10'] = dataset['r_agua_r10_fit_1156'].rolling(2*hour).mean()

    dataset.loc[:, 'nivel_interfaz_r10'] = dataset['r_agua_r10_alturainterface'].rolling(6*hour).mean()

    dataset.loc[:, 'turbidez_r10'] = dataset['r_agua_r10_turbidez'].rolling(2*hour).mean()

    dataset.loc[:, 'presion_cono_decimal_suav_r10'] = dataset['r_agua_r10_presioncono_decimal'].rolling(
            6).mean()
    dataset.loc[:, 'presion_cono_decimal_r10'] = dataset['r_agua_r10_presioncono_decimal']
    dataset.loc[:, 'delta_presion_cono_decimal_r10'] = dataset['presion_cono_decimal_r10'].shift(1) - \
                                                              dataset['presion_cono_decimal_r10'].shift(2)





    #VARIABLES DERIVADAS COMUNES A TODOS LOS MODELOS


    #agua fresca colon
    dataset['rh_sics3_sel_sc'] = 0
    selector = dataset['rh_sics3_sel_sc'].copy()
    selector.index = dataset.index
    selector = selector[~selector.index.duplicated(keep='first')]
    calculo1 = dataset['rh_f19_flujo'].copy()
    calculo1.loc[dataset['rh_fit077'] > 0] = 0
    calculo2 = -dataset['rh_f69_flujo'].copy()
    aux = dataset['rh_fit003'].copy()
    aux = aux[~aux.index.duplicated(keep='first')]
    aux.loc[(selector != 'SAPOS').values] = 0
    calculo2[calculo2 > 400] = 0
    calculo2[calculo2 < -400] = 0
    calculo2 = calculo2 + aux
    dataset.loc[:, 'Agua_Fresca_Colon'] = (dataset[['rh_f_70_iz', 'rh_f_71_d', 'rh_f_53', 'rh_f_17', 'rh_f_16',
                                                 'rh_ge_plc_09_fit_097',
                                                 'rh_f10_flujo', 'rh_f11_flujo', 'rh_f12_flujo', 'rh_f_75',
                                                 'rh_f18_flujo',
                                                 'rh_f25_flujo',
                                                 'rh_fit077', 'rh_fit_020', 'rh_fit083', 'rh_fit_095a', 'rh_fit14',
                                                 'rh_f022']].fillna(0).sum(axis=1) - \
                                        dataset[['rh_fit_076']].fillna(0).sum(axis=1) + calculo1.fillna(
                    0) + calculo2.fillna(0)) * 3.6

    dataset.loc[:, 'Agua_Fresca_Colon'] = dataset['Agua_Fresca_Colon'].shift(5*hour)

    procesamiento_convencional = params['procesamiento_convencional']

    tph_sag1 = 'sag_wic2101'
    tph_sag2 = 'sag2_260_wit_1835'
    dataset.loc[:, 'Procesamiento_SAG'] = (dataset[tph_sag1] + dataset[tph_sag2]).shift(8*hour)
    dataset.loc[:, 'Procesamiento_Convencional'] = (dataset[procesamiento_convencional].sum(axis=1)).shift(8*hour)
    dataset.loc[:, 'Procesamiento_Total'] = dataset['Procesamiento_SAG'] + dataset['Procesamiento_Convencional']
    dataset.loc[:, 'Procesamiento_Total'] = dataset['Procesamiento_Total'].rolling(2*hour).mean()


    solidos_sag = params['solidos_sag']


    for i in solidos_sag:
        dataset[i][dataset[i] < 0] = np.nan
        dataset[i][dataset[i] > 95] = np.nan
    dataset.loc[:, 'Solido_SAG'] = dataset[solidos_sag].mean(axis=1)
    dataset.loc[:, 'Solido_SAG'] = (dataset['Solido_SAG'].rolling(2*hour).mean()).shift(8*hour)


    solidos_convencional = params['solidos_convencional']


    for i in solidos_convencional:
        dataset[i][dataset[i] < 0] = np.nan
        dataset[i][dataset[i] > 95] = np.nan
    dataset.loc[:, 'Solido_Convencional'] = dataset[solidos_convencional].mean(axis=1)
    dataset.loc[:, 'Solido_Convencional'] = (dataset['Solido_Convencional'].rolling(2*hour).mean()).shift(8*hour)
    #agua total / cea total


    agua_convencional = params['agua_convencional']



    agua_sag = params['agua_sag']

    dataset['Agua_SAG'] = dataset[agua_sag].sum(axis=1)
    dataset['Agua_Convencional'] = dataset[agua_convencional].sum(axis=1)
    dataset.loc[:,'Agua_Total'] = dataset['Agua_SAG'] + dataset['Agua_Convencional']
    dataset['CEA_Total'] = dataset['Agua_Total'] / dataset['Procesamiento_Total']
    dataset['CEA_Total'] = dataset['CEA_Total'].rolling(2*hour).mean()
    #TPH's
    dataset.loc[:, 'tph_sag1'] = (dataset['sag_wic2101'].rolling(2*hour).mean()).shift(8*hour)
    dataset.loc[:, 'tph_sag2'] = (dataset['sag2_260_wit_1835'].rolling(2*hour).mean()).shift(8*hour)
    dataset.loc[:, 'tph_sag_mills'] = (dataset['tph_sag1']+dataset['tph_sag2'])

    # promedio descarga bombas
    dataset['promedio_descarga_bomba_15_16'] = (dataset['r_agua_pp115_fit5315'].rolling(6*hour).mean() +
                                                    dataset['r_agua_pp116_fit5316'].rolling(6*hour).mean()) / 2
    dataset['promedio_descarga_bomba_17_18'] = (dataset['r_agua_pp117_fit5317'].rolling(6*hour).mean() +
                                                    dataset['r_agua_pp118_fit5318'].rolling(6*hour).mean()) / 2
    #Flujo agua recuperada promedio
    dataset.loc[:, 'flujo_agua_recup_prom'] = dataset[['r_agua_pp115_fit5315',
                                                               'r_agua_pp116_fit5316',
                                                               'r_agua_pp117_fit5317',
                                                               'r_agua_pp118_fit5318']].mean(axis=1)
    #flujo de agua promedio recuperadores
    dataset.loc[:, 'agua_prom_descarga_recuperadores'] = dataset[
            'r_agua_prom_descarga_recuperadores'].rolling(6*hour).mean()
    #PH suavizado
    dataset.loc[:, 'ph_ca6_suavizado'] = dataset['r_agua_r2_ph_ca6'].shift(1).ewm(halflife=1*hour).mean()

    if params['opt_solido']:
        dataset[params['esp_opt_name']] = dataset[params['esp_target_name']].mean(axis=1) - \
                                          pd.concat([dataset[params['esp_target_name_solido_r2']].mean(axis=1),
                                                     dataset[params['esp_target_name_solido_r3']].mean(axis=1),
                                                     dataset[params['esp_target_name_solido_r4']].mean(axis=1),
                                                     dataset[params['esp_target_name_solido_r5']].mean(axis=1),
                                                     dataset[params['esp_target_name_solido_r6']].mean(axis=1),
                                                     dataset[params['esp_target_name_solido_r7']].mean(axis=1),
                                                     dataset[params['esp_target_name_solido_r9']].mean(axis=1),
                                                     dataset[params['esp_target_name_solido_r10']].mean(axis=1)]
                                                    , axis=1).mean(axis=1)
    else:
        dataset[params['esp_opt_name']] = dataset[params['esp_target_name']].mean(axis=1)



    return dataset


def add_on_off_features_by_hour(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Add on off features to the shifted master data table.

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing source data.

    Returns:
        data_feat: Master table with additional calculated features.

    """

    # Create features
    list_features = [data]
    on_off = create_on_off(parameters, data)
    list_features.append(on_off)
    # Merge data and created features
    data_feat = merge_tables_on_timestamp(parameters=parameters, df_list=list_features)

    return data_feat




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

    #return df[[timestamp_col_name] + new_var_names]
    return df[new_var_names]



def merge_tables_on_timestamp(
    parameters: dict, df_list: List[pd.DataFrame]
) -> pd.DataFrame:
    """Left-merge all DataFrames in the list, based on 1-1 indices for each.

    Args:
        parameters: Dictionary of parameters.
        df_list: List of DataFrames to be merged.

    Returns:
        merged: DataFrame of merged sources.

    """

    merged = reduce(
        partial(_left_merge_on, on=parameters["timestamp_col_name"]), df_list
    )

    return merged


def _left_merge_on(df_1: pd.DataFrame, df_2: pd.DataFrame, on: str) -> pd.DataFrame:
    """Left-merge two DataFrames based on specified column.

    Args:
        df_1: DataFrame 1 to be merged.
        df_2: DataFrame 2 to be merged.
        on: Column to be merged on.

    Returns:
        df: Merged DataFrame.

    """
    # Merge on indices making sure there is 1-1 correspondence
    df = pd.merge(
        df_1, df_2, how="left", left_on=on, right_on=on, validate="one_to_one"
    )

    return df