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
Nodes of the model training pipeline.
"""
import logging
from collections import Counter
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from pandas.plotting import register_matplotlib_converters
from pandas.tseries.offsets import DateOffset
from scipy.stats import ks_2samp, kurtosis
from tqdm import tqdm
from PIL import Image
import pickle as pic
from sklearn.decomposition import PCA
#from pca import pca
from scipy.stats import norm
from typing import Mapping, Any, Iterator
import datetime as dt
import itertools
import os
import warnings
#import hdbscan

#matplotlib.rcParams.update({"font.size": 4})
#register_matplotlib_converters()

warnings.filterwarnings('ignore')
pd.set_option('float_format', '{:f}'.format)


def prepare_data(params: dict, data: pd.DataFrame):
    """
    Loads a regressor object based on given parameters.
    Args:
        params: dictionary of parameters
    Returns:
        sklearn compatible model
    """
    # COLON_Cada_Hora.csv
    data = data.applymap(lambda x: x.replace(',', '.') if isinstance(x, str) else x)
    data.set_index('Fecha',inplace = True)
    data = data.apply(pd.to_numeric, args=('coerce',))
    #data = pd.read_csv('PICOLON_201909_202009_Cada_Hora_V2.csv', delimiter=';', parse_dates=True, low_memory=False,
    #                   encoding="latin1")
    #print (data['MU:280_DHU_2013'])
    #rh = pd.read_csv('PIRH_201909_202009_1_hora_v3.csv', delimiter=';', parse_dates=True, low_memory=False)
    #df['Fecha'] = pd.to_datetime(df['Time'])
    #Fecha = pd.to_datetime(rh['Fecha'], format='%d-%m-%Y %H:%M')
    #rh.drop('Fecha', axis=1, inplace=True)
    # df.drop('Fecha', axis = 1, inplace = True)
    #rh = rh.applymap(lambda x: x.replace(',', '.') if isinstance(x, str) else x)
    #selector = data['RH:SICS3.SEL_SC'].copy()
    #rh = rh.apply(pd.to_numeric, args=('coerce',))
    #rh['Fecha'] = Fecha
    #rh.set_index('Fecha', inplace=True)
    # df=df.resample('1T').mean()
    #selector.index = rh.index
    #data = pd.merge(data, rh, left_index=True, right_index=True)

    #elector = selector[~selector.index.duplicated(keep='first')]

    data = data[~data.index.duplicated(keep='first')]
    data = data.resample('4H', base=23).mean()

    #data_dict_colon = pd.read_excel('DATADICTV2.xlsx')
    #data_dict_rh = pd.read_excel('DATADICTV2.xlsx', sheet_name=1)
    #data_dict = pd.concat([data_dict_colon, data_dict_rh])
    #import ipdb; ipdb.set_trace();
    solidos_sag = ['MU:280_DHU_2013',
                   'MU:Su_Mol13',
                   'SAG2:260_DIC_1842',
                   'SAG2:260_DI_1884',
                   'SAG2:260_DI_1896',
                   'SAG:DIT2189A',
                   'SAG:DIT2189B',
                   'SAG:DIC2150Z',
                   'SAG2:%Sol_SAG2_a_Flotacion']

    for i in solidos_sag:
        data[i][data[i] < 0] = np.nan
        data[i][data[i] > 95] = np.nan

    solidos_convencional = ['MOL:DIC69A',
                            'MOL:DIC69B',
                            'MOL:DIC69C',
                            'MOL:DIC69D',
                            'MOL:DIC69E',
                            'MOL:DIC69F',
                            'MOL:DIC69G',
                            'MOL:DIC69J',
                            'MOL:DIC69K',
                            'MOL:DIC69L',
                            'MOL:DIC69M',
                            'MOL:DIC69N',
                            'MU:280_DIT_8783']

    for i in solidos_convencional:
        data[i][data[i] < 0] = np.nan
        data[i][data[i] > 95] = np.nan

    niveles = [
        'MU:120_LIT_5701',
        'MU:120_LIT_5706',
        'MOL:LIT_CABEZA',
        'R-AGUA:LIT2713A']

    for i in niveles:
        data[i][data[i] < 0] = np.nan
        data[i][data[i] > 120] = np.nan

    procesamiento_convencional = ['MOL:WIC44A',
                                  'MOL:WIC44B',
                                  'MOL:WIC44C',
                                  'MOL:WIC44D',
                                  'MOL:WIC44E',
                                  'MOL:WIC44F',
                                  'MOL:WIC44G',
                                  'MOL:WIC44J',
                                  'MOL:WIC44K',
                                  'MOL:WIC44L',
                                  'MOL:WIC44M',
                                  'MOL:WIC44N',
                                  'MU:280_WIT_8778']

    agua_sellos_convencional = ['RH:F10.FLUJO',
                                'MOL:FITS65A',
                                'MOL:FITS65B',
                                'MOL:FITS65C',
                                'MOL:FITS65D',
                                'MOL:FITS65E',
                                'MOL:FITS65F',
                                'MOL:FITS65G',
                                'MOL:FITS65J',
                                'MOL:FITS65K',
                                'MOL:FITS65L',
                                'MOL:FITS65M',
                                'MOL:FITS65N'
                                ]

    agua_molienda_convencional = ['MOL:FIC65A',
                                  'MOL:FIC65B',
                                  'MOL:FIC65C',
                                  'MOL:FIC65D',
                                  'MOL:FIC65E',
                                  'MOL:FIC65F',
                                  'MOL:FIC65G',
                                  'MOL:FIC65J',
                                  'MOL:FIC65K',
                                  'MOL:FIC65L',
                                  'MOL:FIC65M',
                                  'MOL:FIC65N',
                                  'MU:280_FIC_8785',
                                  'MOL:230_FIC_66A',
                                  'MOL:230_FIC_66B',
                                  'MOL:230_FIC_66C',
                                  'MOL:230_FIC_66D',
                                  'MOL:230_FIC_66E',
                                  'MOL:230_FIC_66F',
                                  'MOL:230_FIC_66G',
                                  'MOL:230_FIC_66J',
                                  'MOL:230_FIC_66K',
                                  'MOL:230_FIC_66L',
                                  'MOL:230_FIC_66M',
                                  'MOL:230_FIC_66N',
                                  'MU:280_FIT_8789']

    otras_aguas_convencional = ['RH:F_75', 'RH:F12.FLUJO', 'RH:F11.FLUJO']

    agua_convencional = ['MOL:FIC65A',
                         'MOL:FIC65B',
                         'MOL:FIC65C',
                         'MOL:FIC65D',
                         'MOL:FIC65E',
                         'MOL:FIC65F',
                         'MOL:FIC65G',
                         'MOL:FIC65J',
                         'MOL:FIC65K',
                         'MOL:FIC65L',
                         'MOL:FIC65M',
                         'MOL:FIC65N',
                         'MU:280_FIC_8785',
                         'MOL:230_FIC_66A',
                         'MOL:230_FIC_66B',
                         'MOL:230_FIC_66C',
                         'MOL:230_FIC_66D',
                         'MOL:230_FIC_66E',
                         'MOL:230_FIC_66F',
                         'MOL:230_FIC_66G',
                         'MOL:230_FIC_66J',
                         'MOL:230_FIC_66K',
                         'MOL:230_FIC_66L',
                         'MOL:230_FIC_66M',
                         'MOL:230_FIC_66N',
                         'MU:280_FIT_8789',
                         'MOL:FITS65A',
                         'MOL:FITS65B',
                         'MOL:FITS65C',
                         'MOL:FITS65D',
                         'MOL:FITS65E',
                         'MOL:FITS65F',
                         'MOL:FITS65G',
                         'MOL:FITS65J',
                         'MOL:FITS65K',
                         'MOL:FITS65L',
                         'MOL:FITS65M',
                         'MOL:FITS65N',
                         'RH:F_75',
                         'RH:F12.FLUJO',
                         'RH:F11.FLUJO']
    # RECORDAR x POR 3.6
    agua_sag = ['RH:F-16',
                'RH:F-17',
                'RH:GE.PLC_09.FIT-097',
                'SAG:FIC2150Z',
                'SAG:FIT2155',
                'SAG:FIC2181A',
                'SAG:FIC2188A',
                'SAG:FIC2181B',
                'SAG:FIC2188B',
                'SAG2:260_FFIC_1842',
                'SAG2:260_FIC_1848',
                'SAG2:260_FI_1890',
                'SAG2:260_FI_1872',
                'SAG2:260_FI_1876',
                'SAG2:260_FI_1892']

    potencia_convencional = ['MU:280_ML01_X',
                             'MOL:JI56N',
                             'MOL:JI56M',
                             'MOL:JI56L',
                             'MOL:JI56K',
                             'MOL:JI56J',
                             'MOL:JI56G',
                             'MOL:JI56F',
                             'MOL:JI56E',
                             'MOL:JI56D',
                             'MOL:JI56C',
                             'MOL:JI56B',
                             'MOL:JI56A']

    estados = ['SAG2:ESTADO_PROFIT',
               'SAG2:ESTADO_PROFIT511',
               'SAG2:ESTADO_PROFIT512',
               'SAG:ESTADO_PROFIT',
               'SAG:ESTADO_PROFIT411',
               'SAG:ESTADO_PROFIT412']

    #data_dict[data_dict['Tag'].isin(procesamiento_convencional)]

    tph_sag1 = 'SAG:WIC2101'
    tph_sag2 = 'sag2:260_wit_1835'
    output_tph = 'tph_SAG_MILLS'
    data[output_tph] = data[tph_sag1] + data[tph_sag2]

    ley_sag1 = 'LAB-Q:LEY_CU_CABEZA_SAG.TURNO'
    ley_sag2 = 'LAB-Q:LEY_CU_CABEZA_SAG2.TURNO'
    ley_tails = 'LAB-Q:LEY_CU_COLA_SAG.TURNO'
    ley_conc = 'LAB-Q:LEY_CU_CONC_FINAL_SAG.TURNO'
    output_rec = 'recovery_FLOTATION'
    output_ley = 'LEY_ALIMENTACION_CALCULATED'
    malla100sag1 = 'LAB-Q:+100M_CABEZA_SAG1.TURNO'
    malla100sag2 = 'LAB-Q:+100M_CABEZA_SAG2.TURNO'
    malla65sag1 = 'LAB-Q:+65M_CABEZA_SAG1.TURNO'
    malla65sag2 = 'LAB-Q:+65M_CABEZA_SAG2.TURNO'

    data['Malla100 SAG'] = (data[tph_sag1] * data[malla100sag1] + data[tph_sag2] * data[malla100sag2]) / (
                data[tph_sag1] + data[tph_sag2])
    data['Malla65 SAG'] = (data[tph_sag1] * data[malla65sag1] + data[tph_sag2] * data[malla65sag2]) / (
                data[tph_sag1] + data[tph_sag2])

    output_cuFino = 'cuFino'
    ley_feed = (data[tph_sag1] * data[ley_sag1] + data[tph_sag2] * data[ley_sag2])
    ley_feed = ley_feed / (data[tph_sag1] + data[tph_sag2])
    data[output_ley] = ley_feed
    data[output_ley][data[output_ley] > 1.3] = np.nan
    data[output_ley][data[output_ley] < .7] = np.nan
    data[output_rec] = ((ley_feed - data[ley_tails]) * data[ley_conc]) / (
            (data[ley_conc] - data[ley_tails]) * ley_feed)
    data[output_cuFino] = data[output_rec] * data[output_ley] * data[output_tph] / 100

    df_plot = data.copy()

    df_plot['Procesamiento SAG'] = df_plot[tph_sag1] + df_plot[tph_sag2]
    df_plot['Procesamiento Convencional'] = df_plot[procesamiento_convencional].sum(axis=1)
    df_plot['Procesamiento Total'] = df_plot['Procesamiento SAG'] + df_plot['Procesamiento Convencional']

    # df_plot['Procesamiento Total'][df_plot['Procesamiento Total']<100]=np.nan
    # df_plot['Procesamiento SAG'][df_plot['Procesamiento SAG']<100]=np.nan
    # df_plot['Procesamiento Convencional'][df_plot['Procesamiento Convencional']<100]=np.nan
    df_plot['Ley alimentacion'] = df_plot['LEY_ALIMENTACION_CALCULATED']
    df_plot['Recuperacion SAG'] = df_plot[output_rec]
    # df_plot['Recuperacion Convencional']=df_plot['Rec Conv'] TO DO: FALTA CALCULARLO
    # df_plot['Ley Total']=(df_plot['Ley SAG']*df_plot['Procesamiento SAG']+df_plot['Ley Convencional']*df_plot['Procesamiento Convencional'])\
    #                        /(df_plot['Procesamiento SAG']+df_plot['Procesamiento Convencional'])

    df_plot['CuF SAG'] = df_plot[output_cuFino]
    df_plot['Agua SAG'] = df_plot[agua_sag].sum(axis=1)
    df_plot['Agua Convencional'] = df_plot[agua_convencional].sum(axis=1)
    df_plot['Agua Total'] = df_plot['Agua SAG'] + df_plot['Agua Convencional']

    df_plot['Potencia SAG'] = (df_plot['SAG:XE03'] + df_plot['SAG2:260_ml_04ji'] / 1000) / 2
    # df_plot['Potencia Convencional']= # FALTA ACA
    # df_plot['Potencia Total']=df_plot['Potencia SAG']+df_plot['Potencia Convencional']

    df_plot['CEE SAG'] = df_plot['Potencia SAG'] / df_plot['Procesamiento SAG']
    # df_plot['CEE Convencional']=df_plot['Potencia Convencional']/df_plot['Procesamiento Convencional']
    # df_plot['CEE Total']=df_plot['Potencia Total']/df_plot['Procesamiento Total']

    df_plot['Solido SAG'] = df_plot[solidos_sag].mean(axis=1)
    df_plot['Solido Convencional'] = df_plot[solidos_convencional].mean(axis=1)

    df_plot['Oferta_hidrica'] = df_plot[
        ['RH:F078', 'RH:FIT079', 'RH:F-45', 'RH:FIT080', 'RH:MB.PLC_05.FIT-096', 'RH:F03.FLUJO',
         'RH:F40.FLUJO', 'RH:EPUQUIOS.NIVEL', 'RH:F43.FLUJO', 'RH:FIT-093', 'RH:F44.FLUJO', 'RH:F48.FLUJO',
         'RH:F49.FLUJO', 'RH:FIT-047', 'GMIN:FLUJO_DESDE_PIQUE-C', 'RH:F-42', 'RH:F52', 'RH:F37.FLUJO',
         'RH:FIT-098', 'RH:F-36', 'RH:F-35', 'RH:F-34', 'RH:F08', 'RH:F022', 'RH:FIT-095A']].sum(axis=1)

    df_plot['Consumo_Agua_SAG1'] = df_plot[['SAG:FIT2155', 'SAG:FIC2150Z', 'SAG:FIC2181A',
                                            'SAG:FIC2188A', 'SAG:FIC2181B', 'SAG:FIC2188B']].sum(axis=1)
    df_plot['Consumo_Agua_SAG2'] = df_plot[['SAG2:260_FFIC_1842', 'SAG2:260_FIC_1848', 'SAG2:260_FI_1890',
                                            'SAG2:260_FI_1872', 'SAG2:260_FI_1876', 'SAG2:260_FI_1892']].sum(axis=1)
    df_plot['Consumo_Agua_Mol_Conv'] = df_plot[['MOL:230_FIC_66A', 'MOL:230_FIC_66B', 'MOL:230_FIC_66C',
                                                'MOL:230_FIC_66D', 'MOL:230_FIC_66E', 'MOL:230_FIC_66F',
                                                'MOL:230_FIC_66G', 'MOL:230_FIC_66J', 'MOL:230_FIC_66K',
                                                'MOL:230_FIC_66L', 'MOL:230_FIC_66M', 'MOL:230_FIC_66N',
                                                'MOL:FIC65A', 'MOL:FIC65B', 'MOL:FIC65C', 'MOL:FIC65D',
                                                'MOL:FIC65E', 'MOL:FIC65F', 'MOL:FIC65G', 'MOL:FIC65J',
                                                'MOL:FIC65K', 'MOL:FIC65L', 'MOL:FIC65M', 'MOL:FIC65N']].sum(axis=1)
    df_plot['Consumo_Agua_MU'] = df_plot[['MU:280_FIC_8785', 'MU:280_FIT_8789']].sum(axis=1)

    df_plot['Consumo_Agua_SAG'] = df_plot['Consumo_Agua_SAG1'] + df_plot['Consumo_Agua_SAG2'] + df_plot[
        ['RH:F-16', 'RH:F-17', 'RH:GE.PLC_09.FIT-097']].fillna(0).sum(axis=1) * 3.6
    df_plot['Consumo_Agua_Conv'] = df_plot['Consumo_Agua_Mol_Conv'] + df_plot['Consumo_Agua_MU'] + df_plot[
        ['RH:F_75', 'RH:F12.FLUJO', 'RH:F11.FLUJO']].fillna(0).sum(axis=1) * 3.6

    df_plot['Agua Total'] = df_plot['Consumo_Agua_SAG'] + df_plot['Consumo_Agua_Conv']

    df_plot['CEA SAG'] = df_plot['Consumo_Agua_SAG'] / df_plot['Procesamiento SAG']
    df_plot['CEA Convencional'] = df_plot['Consumo_Agua_Conv'] / df_plot['Procesamiento Convencional']
    df_plot['CEA Total'] = df_plot['Agua Total'] / df_plot['Procesamiento Total']

    calculo1 = df_plot['RH:F19.FLUJO'].copy()
    calculo1.loc[df_plot['RH:FIT077'] > 0] = 0
    calculo2 = -df_plot['RH:F69.FLUJO'].copy()
    aux = data['RH:FIT003'].copy()
    aux = aux[~aux.index.duplicated(keep='first')]
    #aux.loc[(selector != 'SAPOS').values] = 0
    calculo2[calculo2 > 400] = 0
    calculo2[calculo2 < -400] = 0
    calculo2 = calculo2 + aux
    df_plot['Agua_Fresca_Colon'] = (df_plot[['RH:F-70_IZ', 'RH:F-71_D', 'RH:F-53', 'RH:F-17', 'RH:F-16',
                                             'RH:GE.PLC_09.FIT-097',
                                             'RH:F10.FLUJO', 'RH:F11.FLUJO', 'RH:F12.FLUJO', 'RH:F_75', 'RH:F18.FLUJO',
                                             'RH:F25.FLUJO',
                                             'RH:FIT077', 'RH:FIT-020', 'RH:FIT083', 'RH:FIT-095A', 'RH:FIT14',
                                             'RH:F022']].fillna(0).sum(axis=1) - \
                                    df_plot[['RH:FIT-076']].fillna(0).sum(axis=1) + calculo1.fillna(
                0) + calculo2.fillna(0)) * 3.6

    df_plot['Cp Barahona'] = df_plot['LAB-Q:%_SOLIDO_COLA_BARAHONA.HORA']

    # df_plot['Make up']=1/(df_plot['LAB-Q:%_SOLIDO_COLA_BARAHONA.HORA']*0.01)-1

    df_plot['Makeup'] = df_plot['Agua_Fresca_Colon'] / df_plot['Procesamiento Total']
    # PTE:6350_FI_0904.PV

    # Esperando los datos de Esteban Y CONFIRMAR CUANDO ESTE BUENO
    df_plot['Flujo Sapos y Rio Blanco'] = df_plot['RH:F05.FLUJO'] + df_plot['RH:F38.FLUJO']

    tags_rec_conv = ['MOL:LIT_CABEZA', 'MU:120_LIT_5706']
    ##tags_conv=['RH:E41.NIVEL_PORC','MU:120_LIT_5701']
    tags_rec_sag = ['SAG:LIT2713', 'R-AGUA:LIT2713A']
    ##tags_sag=['RH:E39_40.NIVEL_PORC']

    df_plot['Nivel Recuperada Convencional'] = df_plot[tags_rec_conv].mean(axis=1)
    df_plot['Nivel Recuperada SAG'] = df_plot[tags_rec_sag].mean(axis=1)

    df_plot['Niveles agua recuperada'] = df_plot[tags_rec_sag + tags_rec_conv].mean(axis=1)
    ##df_plot['Nivel Fresca Convencional']=df_plot[tags_conv].mean(axis=1)
    ##df_plot['Nivel Fresca SAG']=df_plot[tags_sag].mean(axis=1)

    df_plot['Volumen embalse sapos'] = df_plot['RH:Volumen_Embalse_2']

    # tags para evaluar clusters
    df_plot['HL 411 potencia'] = df_plot['SAG:250_PROFIT_411_CV0_High']
    df_plot['HL 411 nivel cuba'] = df_plot['SAG:250_PROFIT_411_CV1_High']
    df_plot['HL 411 presion cicl'] = df_plot['SAG:250_PROFIT_411_CV2_High']
    df_plot['HL 411 % solido'] = df_plot['SAG:250_PROFIT_411_CV3_High']
    df_plot['HL 411 granulometria'] = df_plot['SAG:250_PROFIT_411_CV4_High']
    df_plot['HL 411 agua cuba'] = df_plot['SAG:250_PROFIT_411_MV0_High']
    df_plot['HL 411 agua molino'] = df_plot['SAG:250_PROFIT_411_MV1_High']
    df_plot['HL 411 velocidad bomba'] = df_plot['SAG:250_PROFIT_411_MV2_High']
    df_plot['HL 412 potencia'] = df_plot['SAG:250_PROFIT_412_CV0_High']
    df_plot['HL 412 nivel cuba'] = df_plot['SAG:250_PROFIT_412_CV1_High']
    df_plot['HL 412 presion cicl'] = df_plot['SAG:250_PROFIT_412_CV2_High']
    df_plot['HL 412 % solido'] = df_plot['SAG:250_PROFIT_412_CV3_High']
    df_plot['HL 412 granulometria'] = df_plot['SAG:250_PROFIT_412_CV4_High']
    df_plot['HL 412 agua cuba'] = df_plot['SAG:250_PROFIT_412_MV0_High']
    df_plot['HL 412 agua molino'] = df_plot['SAG:250_PROFIT_412_MV1_High']
    df_plot['HL 412 velocidad bomba'] = df_plot['SAG:250_PROFIT_412_MV2_High']

    df_plot['Makeup'] = df_plot['Makeup'].replace([np.inf, -np.inf], np.nan)

    # Granulometría a flotación: Malla +100
    df_plot['Malla100_SAG1'] = df_plot['LAB-Q:+100M_CABEZA_SAG1.TURNO']
    df_plot['Malla100_SAG2'] = df_plot['LAB-Q:+100M_CABEZA_SAG2.TURNO']
    # Malla +100 a flotación SAG
    df_plot['Malla100_SAG'] = (df_plot['Malla100_SAG1'] * df_plot[tph_sag1] + df_plot['Malla100_SAG2'] * df_plot[
        tph_sag2]) / df_plot['Procesamiento SAG']

    # Agua fresca en rangos normales
    filtro1 = (df_plot['Agua_Fresca_Colon'] < 8000) & (df_plot['Agua_Fresca_Colon'] > 1000)

    # Exigir mínimo de procesamiento
    filtro2 = df_plot['Procesamiento Total'] > 1500
    filtro3 = df_plot['Procesamiento SAG'] > 900
    filtro4 = (df_plot['Makeup'] > 0.4) & (df_plot['Makeup'] < 1.8)

    filtros_op = (filtro1) & (filtro2) & (filtro3) & (filtro4)
    df_plot = df_plot[filtros_op]
    df_plot = df_plot.replace([np.inf, -np.inf], np.nan)

    df_plot['HL 411 % solido'].hist(bins=100)

    variables_clusters = ['Niveles agua recuperada',
                          'Oferta_hidrica',
                          'Volumen embalse sapos']
    # 'Flujo Sapos y Rio Blanco']
    # 'Nivel Recuperada Convencional',
    # 'Nivel Recuperada SAG']
    # 'Nivel Fresca Convencional',
    # 'Nivel Fresca SAG']

    var_evaluacion = ['Cp Barahona', 'Procesamiento Total', 'Malla100_SAG', 'Ley alimentacion', 'CEA SAG',
                      'CEA Convencional', 'Recuperacion SAG']
    df_pre_clusters = df_plot[variables_clusters + var_evaluacion].dropna()


    df_clusters = df_pre_clusters[variables_clusters].dropna()
    df_eval = df_pre_clusters[var_evaluacion].dropna()

    print(df_clusters.shape, df_eval.shape)


    df_clusters = df_clusters.iloc[:-5]
    df_eval = df_eval.iloc[:-5]
    df_clusters.iloc[-5:].to_csv("df_test_clusters.csv")

    df_clusters_std, obj_esc_clusters = scaler_method(df_clusters)

    df_eval_std, obj_esc_eval = scaler_method(df_eval)
    #df_clusters_pc = pca_func(df_clusters, df_clusters_std)
    #with open('pca_results.pickle', 'rb') as dat:
    #    dic_pca = pic.load(dat)
    #df_clust_2pc = dic_pca['df_pc'].copy()
    #df_eval_2pc = dic_pca['df_pc'].copy()
    #df_ev_2pc = df_eval_2pc[['PC1', 'PC2']].copy()
    #df_cl_2pc = df_clust_2pc[['PC1', 'PC2']].copy()


    cluster_size = int(df_clusters.shape[0] / 10)
    #params = {'min_clusters': 3, 'max_clusters': 6}
    # params_cluster = { 'min_samples':np.arange(50, int(df_clusters.shape[0]/4),120),'min_cluster_size':[30]}
    #params_cluster = {'min_samples': [1, 50], 'min_cluster_size': [60, 150, 600, 1000]}
    #params_cluster = {'min_samples': params['min_samples'], 'min_cluster_size': params['min_cluster_size']}

    best_model = find_best_model(df_clusters, df_eval, data, params, obj_esc_clusters, model_name='hdbscan')

    print (best_model)
    return best_model


def find_best_model(df_cluster_vars: pd.DataFrame, df_metric_vars: pd.DataFrame, data_raw: pd.DataFrame,
                    parameters: Mapping[str, Any],
                    scaler_obj, model_name=None) -> dict:
    """ Rutina para encontrar un modelo de clusterización óptimo.
df_clust,df_eva,df_2pc_clust,df_2pc_eva, parameters = {'eps':0.38, 'min_samples':15}
    Entrena, evalúa métricas,genera gráficos y retorna DataFrame con resultados.

    Args:
        df_cluster_vars: DataFrame con variables a clusterizar.
        df_metrics_vars: DataFrame con variables de evaluación.
        parameters: Diccionario o similar con parámetros de modelo clusterización.
        scaler_obj: Objeto de normalización usado para construir dataset, se usa
                    en esta rutina para generar boxplots en unidades originales.
        model_name: String indicando nombre del modelo para guardar salidas.

    Returns:
        Diccionario conteniendo modelo de clusterización óptimo y DataFrame
        con resultados de clusterización.

        {'clustering_model': object,
         'metrics': pd.DataFrame,
        }

    """
    # Inicialización de parámetros
    n = 0
    n_iters = 5
    min_cluster_size = int(parameters['init_min_cluster_size'] * len(df_cluster_vars)) #Valor inicial de min_cluster_size
    min_clusters = int(parameters['min_clusters'])
    max_clusters = int(parameters['max_clusters'])

    #Definición de flags de control
    SaveClusterizationDataframes = True

    # Escalamiento de datos de entrada
    df_scaled, df_sca_cluster_var = scaler_method(df_cluster_vars)

    # Almacenamiento de dataframes usados para genera modelo (opcional)
    if SaveClusterizationDataframes:
        df_cluster_vars.to_csv('df_raw.csv')
        df_scaled.to_csv('df_scaled.csv')
        df_sca_cluster_var.to_csv('df_norm_vars.csv')

    #Se valida previamente de que se cuenten con los parametros de sintonización acordes
    if int(min_cluster_size *0.25 / parameters['n_sintonization_trials']) == 0:
        print("Error al generar parametros para iteración, se deben ajustar parametros de entrada.")
        return {'clustering_model': None, 'metrics': None}

    # Se itera n_iters veces variando el parámetro de minb_cluster_size hasta encontrar una combinación que cumpla
    # con las restricciones impuestas
    for x in range(0, n_iters):
        #print("Min CLuster Size Init: " + str(min_cluster_size) + ", steps: " + str(int(min_cluster_size *0.25 / parameters['n_sintonization_trials'])))
        min_sample_range = np.arange(1, int(min_cluster_size *0.25), int(min_cluster_size *0.25 / parameters['n_sintonization_trials']))
        metric_table = pd.DataFrame(index=np.arange(0, len(min_sample_range)),
                                    columns=['min_sample', 'n_clusters', 'metric_var_std', 'kl_div_cluster_vars',
                                             'kl_div_cluster_metrics', 'smallest_cluster', 'largest_cluster',
                                             'noise_cluster_size', 'kl_div_total'])
        for min_sample in min_sample_range:
            # Creación de modelo de clusterización con los parametros iterando
            print("Creando modelo de clusterización con: Min Samples: " + str(min_sample) + " Min Cluster Size: " + str(min_cluster_size))
            clustering_model = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size),
                                               min_samples=int(min_sample),
                                               prediction_data=True, cluster_selection_method="leaf")

            clustering_model.fit(df_scaled)
            clustering_model.model = 'Clustering_' + model_name
            clustering_model.columns = df_cluster_vars.columns
            cluster_labels = get_model_labels(clustering_model)
            metric_table = calculate_metrics(metric_table, df_cluster_vars, df_metric_vars, cluster_labels,
                                             min_sample, n)
            create_charts(df_cluster_vars, df_metric_vars, metric_table, model_name, scaler_obj,
                          cluster_labels, n)
            n += 1


        # Filtrar por min y max numero de clusters
        metric_table_opt = metric_table.copy()
        metric_table_opt = metric_table_opt[metric_table_opt['n_clusters'] >= min_clusters]
        metric_table_opt = metric_table_opt[metric_table_opt['n_clusters'] <= max_clusters]

        if len(metric_table_opt) > 0: #Existe alguna solución para el valor de min_cluster_size definido?
            break
        else:
            min_cluster_size = min_cluster_size * 1.5

    # Si no cumple las restricciones de número de clusters, retornar None
    if len(metric_table_opt) == 0:
        print('No se cumplio restricción de min/max clusters. Cluster mínimo: {}, cluster máximo: {}.'.format(
            str(min_clusters), str(max_clusters)))
        return {'clustering_model': None, 'metrics': metric_table_opt}

    print(metric_table_opt)
    min_cluster_size_used = min_cluster_size
    min_sample_opt = metric_table_opt.sort_values('kl_div_total')['min_sample'].iloc[-1]
    aux_noise = metric_table_opt.sort_values('kl_div_total')['noise_cluster_size'].iloc[-1]

    # Cálculo de métricas
    for x in range (1,len(metric_table_opt)+1):
        if (metric_table_opt.sort_values('kl_div_total')['kl_div_total'].iloc[-x] > metric_table_opt.sort_values('kl_div_total')['kl_div_total'].iloc[-1]*0.90):
            if (metric_table_opt.sort_values('kl_div_total')['noise_cluster_size'].iloc[-x] < aux_noise):
                min_sample_opt = metric_table_opt.sort_values('kl_div_total')['min_sample'].iloc[-x]
                aux_noise = metric_table_opt.sort_values('kl_div_total')['noise_cluster_size'].iloc[-x]
    print("Min Sample Optimo: " + str(min_sample_opt) + ", Min Cluster Size: " + str(min_cluster_size_used))

    # Calcular cluster optimo
    clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size), min_samples=int(min_sample_opt),
                                prediction_data=True, cluster_selection_method="leaf")
    clusterer.fit(df_scaled)
    clusterer.model = model_name
    clusterer.columns = df_cluster_vars.columns
    clusterer.model_date = dt.datetime.now()
    df_cluster_vars['Cluster'] = clusterer.labels_ + 1
    data_clustered = pd.concat([df_cluster_vars, df_metric_vars], axis=1)
    cluster_labels = get_model_labels(clusterer)  # hacer función
    clusterer.model_date = dt.datetime.now()

    #import ipdb; ipdb.set_trace();
    #Se guarda objeto de clusterización óptimo

    path = './data/01_raw//'
    with open('clusterer.pickle', 'wb') as handle:
        pic.dump(clusterer, handle, protocol=pic.HIGHEST_PROTOCOL)

    aux = data_raw.join(df_cluster_vars, on="Fecha", how='left')

    for Cluster, Cluster_data in aux.groupby('Cluster'):
        filename = path + 'data_cluster_' + str(int(Cluster)) + '.csv'
        Cluster_data.to_csv(filename, encoding='utf-8')

    #data_raw['Cluster'] = df_cluster_vars['Cluster'].resample('10MIN').mean()
    #data_raw.to_csv("df_raw_clusters.csv")
    aux.to_csv("df_raw_clus.csv")
    print(aux['Cluster'])
    print (data_clustered)
    print (cluster_labels)

    return cluster_labels


def calculate_metrics(metric_table, df_cluster_vars, df_metric_vars, cluster_labels, min_sample, n):
    def kl_divergence_sym(p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0)) + np.sum(np.where(q != 0, q * np.log(q / p), 0))

    df_cluster_vars['Cluster'] = cluster_labels + 1
    clusters = df_cluster_vars.groupby('Cluster').groups
    df_cluster_vars.drop('Cluster', axis=1, inplace=True)
    num_clusters = len(clusters.keys())
    # import ipdb; ipdb.set_trace()
    cluster_sizes = [(len(df_cluster_vars.loc[clusters[i]]) / len(df_cluster_vars))
                     for i in range(0, num_clusters + 0)]
    metric_var_std = 0
    for metric_var in df_metric_vars.columns:
        metric_var_std += np.std([df_metric_vars.loc[clusters[i], metric_var].mean()
                                  for i in range(0, num_clusters + 0)])

    # Crear PDFs de cada cluster, por variable
    pdfs = {}
    # ipdb.set_trace()
    # Variables de clusterizacion
    for var in df_cluster_vars.columns:
        df_temp_pdf = pd.DataFrame()
        # Identificar max y min para PDF
        max_range = max(df_cluster_vars[var])
        min_range = min(df_cluster_vars[var])
        for i in clusters.keys():
            mu, std = norm.fit(df_cluster_vars.loc[clusters[i], var])
            pdf_temp = norm.pdf(np.arange(min_range, max_range + 0.1, 0.1), mu, std)
            df_temp_pdf[i] = pdf_temp.copy()
        pdfs[var] = df_temp_pdf.copy()

    # Variables de metricas
    for var in df_metric_vars.columns:
        df_temp_pdf = pd.DataFrame()
        # Identificar max y min para PDF
        max_range = max(df_metric_vars[var])
        min_range = min(df_metric_vars[var])
        # ipdb.set_trace()
        #print(n, var)
        for i in clusters.keys():
            mu, std = norm.fit(df_metric_vars.loc[clusters[i], var])
            pdf_temp = norm.pdf(np.arange(min_range, max_range + 0.1, 0.1), mu, std)
            df_temp_pdf[i] = pdf_temp.copy()
        pdfs[var] = df_temp_pdf.copy()

    kl_div_vars = 0
    for var in df_cluster_vars.columns:
        kl_div_vars_max = 0
        for cluster_comb in itertools.combinations(np.arange(1, num_clusters), 2):
            if kl_divergence_sym(pdfs[var][cluster_comb[0]], pdfs[var][cluster_comb[1]]) > kl_div_vars_max:
                kl_div_vars_max = kl_divergence_sym(pdfs[var][cluster_comb[0]], pdfs[var][cluster_comb[1]])
        kl_div_vars = kl_div_vars + kl_div_vars_max
    kl_div_vars = kl_div_vars / len(df_cluster_vars.columns)

    kl_div_metrics = 0
    aux_count = 0
    # import ipdb; ipdb.set_trace()
    for var in df_metric_vars.columns:
        for cluster_comb in itertools.combinations(np.arange(1, num_clusters), 2):
            aux_count += 1
            kl_div_metrics += kl_divergence_sym(pdfs[var][cluster_comb[0]], pdfs[var][cluster_comb[1]])
    # kl_div_metrics = kl_div_metrics / aux_count
    # import ipdb; ipdb.set_trace()
    metric_table.loc[n, 'min_sample'] = min_sample
    metric_table.loc[n, 'n_clusters'] = num_clusters
    metric_table.loc[n, 'smallest_cluster'] = min(cluster_sizes)
    metric_table.loc[n, 'largest_cluster'] = max(cluster_sizes)
    metric_table.loc[n, 'noise_cluster_size'] = len(df_cluster_vars.loc[clusters[0]]) / len(df_cluster_vars)  # solo para HDBSCAN
    metric_table.loc[n, 'metric_var_std'] = metric_var_std
    metric_table.loc[n, 'kl_div_cluster_vars'] = kl_div_vars
    metric_table.loc[n, 'kl_div_cluster_metrics'] = kl_div_metrics
    metric_table.loc[n, 'kl_div_total'] = metric_table.loc[n, 'kl_div_cluster_metrics'] * 0.5 \
                                          + metric_table.loc[n, 'kl_div_cluster_vars'] * 0.5

    return metric_table


def create_charts(df_cluster_vars, df_metric_vars, metric_table, model_name, scaler_obj,
                  cluster_labels, n):
    kl_div_vars = metric_table.loc[n, 'kl_div_cluster_vars']
    kl_div_metrics = metric_table.loc[n, 'kl_div_cluster_metrics']
    metric_var_std = metric_table.loc[n, 'metric_var_std']

    data_clustered = pd.concat([df_cluster_vars, df_metric_vars], axis=1)
    # for col in data_clustered:
    #    if col in scaler_obj.index:
    #        data_clustered[col] = data_clustered[col] * scaler_obj.loc[col, 'std'] + \
    #                              scaler_obj.loc[col, 'mean']
    data_clustered['Cluster'] = cluster_labels + 1
    clusters = data_clustered.groupby('Cluster').groups
    data_clustered.drop('Cluster', axis=1, inplace=True)
    n_clusters = len(clusters.keys())
    n_cols = int(np.ceil(len(data_clustered.columns) / 2))

    fig = plt.figure(figsize=[12, 6])
    k = 1
    for var in data_clustered.columns:
        ax = fig.add_subplot(2, n_cols, k)
        data_list = []
        for c in sorted(clusters.keys()):
            data_list.append(data_clustered.loc[clusters[c], var])
        ax.boxplot(data_list, showfliers=False)
        var_title = ''
        skip = 0
        for v in var.split(' '):
            if skip == 0:
                var_title += v + ' '
                skip = 1
            else:
                var_title += v + '\n'
                skip = 0
        ax.set_title(var_title[:-1], fontsize=8)
        ax.set_xlabel('Cluster')
        # import ipdb; ipdb.set_trace()
        ax.xaxis.set_ticklabels([
            format((len(data_clustered.loc[clusters[i]]) / len(data_clustered)) * 100, '.0f') + '%'
            for i in range(0, n_clusters)])
        k += 1
    # import ipdb; ipdb.set_trace();
    fig.suptitle('KL Vars: ' + str(round(kl_div_vars, 3)) +
                 ', KL Metrics: ' + str(round(kl_div_metrics, 3)) +
                 ', Metric Std: ' + str(round(metric_var_std, 3)), fontsize=10)

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    # Cambiar este path
    fig.savefig('./Clusters_Iteration_' + str(int(n)) + '_nClusters_'+ str(metric_table.loc[n, 'n_clusters'])+'.png', dpi=240)


def pca_func(df_no_std, df_std, threshold=None, num_variables=None):
    """
    df_no_std: Dataframe with cuantitative variables.
    df_std: Standardized df_no_std.
    threshold: Threshold for the proportion of variance explained by each of the selected components.
    num_variables: Variables to display in the double projection plot.
    output: Non-standardized dataframe with principal components.

    Also, "pca_results.pickle" file is created to save some results. For example:
    dic_pca:
        1-. df_pca (df_no_std with principal components).
        2-. df_std_pca (df_std with principal components).
        3-. df_pc (principal components dataframe).
        4-. scores (principal component scores).
        5-. exp_var (the amount of variance explained by each of the selected components).
        6-. exp_var_ratio (proportion of variance explained by each of the selected components).
    fig:
        1-. double projection plot.

    """
    import pickle as pic
    import pandas as pd
    import numpy as np
    from sklearn.decomposition import PCA
    from pca import pca

    if threshold is None:
        threshold = 0.80

    if num_variables is None:
        num_variables = 5

    df_no_std1 = pd.DataFrame(df_no_std)
    df_no = df_no_std1.copy()
    df_std1 = pd.DataFrame(df_std)
    df = df_std1.copy()

    pca1 = PCA(n_components=threshold)
    prinComp = pca1.fit_transform(df)
    scores = pca1.transform(df)
    exp_var = pca1.explained_variance_
    exp_var_ratio = pca1.explained_variance_ratio_
    name_pc = list()
    for i in range(len(exp_var_ratio)):
        name_pc.append('PC' + str(i + 1))
    principalDf = pd.DataFrame(data=prinComp, columns=name_pc)

    pca_df = principalDf.copy()
    pca_df['Fecha'] = df.index
    pca_df.set_index('Fecha', inplace=True)
    df_std_pca = pd.merge(df, pca_df, left_index=True, right_index=True)
    df_no_std_pca = pd.merge(df_no, pca_df, left_index=True, right_index=True)

    df_pca_plot = df.copy()
    df_pca_plot['new_index'] = 1
    df_pca_plot.reset_index(inplace=True)
    df_pca_plot.drop(columns='Fecha', inplace=True)
    df_pca_plot.set_index('new_index', inplace=True)

    model = pca(n_components=threshold)
    y_pca = model.fit_transform(df_pca_plot)
    fig, ax = model.biplot(n_feat=num_variables, cmap='tab20c', figsize=(8, 6))
    fig.savefig('double_projection.png')

    dic_pca = {'df_pca': df_no_std_pca, 'df_std_pca': df_std_pca, 'df_pc': principalDf, 'scores': scores,
               'exp_var': exp_var, 'exp_var_ratio': exp_var_ratio}
    with open('pca_results.pickle', 'wb') as dat:
        pic.dump(dic_pca, dat, protocol=pic.HIGHEST_PROTOCOL)
        pic.dump(fig, dat, protocol=pic.HIGHEST_PROTOCOL)

    print('Note: The \"pca_results.pickle" was created.')
    print(
        'Its components are: dic_pca["df_pca","df_std_pca","df_pc","scores","exp_var","exp_var_ratio"] and fig (double_projection).')
    print('Current output is dic_pca["df_pca"].')

    return df_no_std_pca


def scaler_method(df, method=None):
    """
    df: dataframe with cuantitative variables
    method: method for tranform variables (standarization) ['standar','robust']
    output: standardized df
    Also, "summary.pickle" file is created to save some results. For example:
    1-. Input df (no_std_df)
    2-. Output df (std_df)
    3-. Input df statistics (no_std_tags_stats)
    4-. Output df statistics (std_tags_stats)

    """
    import pickle as pic
    import pandas as pd
    import numpy as np

    if method is None:
        method = 'standar'

    df1 = df.copy()
    col = list(df1)
    ncol = list(df1)
    ed = list()
    for i in range(len(col)):
        ncol[i] = col[i] + '_std'
        summ = df1[col[i]].describe()
        # import ipdb; ipdb.set_trace()
        ed.append([col[i], summ['mean'], summ['std'], summ['min'], summ['25%'], summ['50%'], summ['75%'], summ['max'],
                   method])
        if method == 'standar':
            df1[col[i] + '_std'] = df1[col[i]].apply(lambda y: (y - summ['mean']) / (summ['std']))
        elif method == 'robust':
            df1[col[i] + '_std'] = df1[col[i]].apply(lambda y: (y - summ['50%']) / (summ['75%'] - summ['25%']))
    df1_new = df1[ncol].copy()

    df1_est = pd.DataFrame(df1_new.describe().T)
    df1_est.reset_index(inplace=True)
    df1_est.rename(columns={'index': 'tag'}, inplace=True)
    df1_est.set_index('tag', inplace=True)
    df1_est = df1_est.iloc[:, 1:].copy()
    df1_est['std_method'] = method

    df_est = pd.DataFrame(ed)
    df_est.rename(
        columns={0: 'tag', 1: 'mean', 2: 'std', 3: 'min', 4: '25%', 5: '50%', 6: '75%', 7: 'max', 8: 'std_method'},
        inplace=True)
    df_est.set_index('tag', inplace=True)

    dic_results = {'no_std_df': df, 'std_df': df1_new, 'no_std_tags_stats': df_est, 'std_tags_stats': df1_est}

    print('Note: The \"summary.pickle" dictionary was created.')
    print('Its components are dic_results["no_std_df","std_df","no_std_tags_stats","std_tags_stats"].')
    print('Current output is dic_results["std_df"].')

    return df1_new, df_est


def create_params_grid(key_list,parameters: Mapping[str, Iterator]):
    keys, values = zip(*parameters.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return permutations_dicts

def get_model_labels(model):
    return model.labels_