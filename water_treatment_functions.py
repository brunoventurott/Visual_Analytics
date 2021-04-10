import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.transforms as transforms
import numpy as np
import ipywidgets as widgets
import seaborn as sns
from scipy import stats
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import ipyvolume as ipv

def orig_cols():
    """
    Retorna a lista com os nomes das colunas
    """
    return ['Q-E','ZN-E','PH-E','DBO-E','DQO-E','SS-E','SSV-E','SED-E','COND-E','PH-P','DBO-P','SS-P','SSV-P',
                    'SED-P','COND-P','PH-D','DBO-D','DQO-D','SS-D','SSV-D','SED-D','COND-D','PH-S','DBO-S','DQO-S',
                    'SS-S','SSV-S','SED-S','COND-S','RD-DBO-P','RD-SS-P','RD-SED-P','RD-DBO-S','RD-DQO-S','RD-DBO-G',
                    'RD-DQO-G','RD-SS-G','RD-SED-G']
def cols_group(nome):
    """
    Como os gráficos não são capazes de lidar com o grande número de variáveis, vou criar grupos que correlacionem as colunas
    """
    group = {
        'input': ['Q-E','ZN-E','PH-E','DBO-E','DQO-E','SSF-E','SSV-E','SST-E','SED-E','COND-E','PH-P','DBO-P','SSF-P',
                  'SSV-P','SST-P','SED-P','COND-P','PH-D','DBO-D','DQO-D','SSF-D','SST-D','SSV-D','SED-D','COND-D'],
        'output': ['PH-S','DBO-S','DQO-S','SST-S','SED-S','COND-S'],
        'performance_old': ['RD-DBO-P','RD-SS-P','RD-SED-P','RD-DBO-S','RD-DQO-S','RD-DBO-G',
                    'RD-DQO-G','RD-SS-G','RD-SED-G'],
        'performance_new':['P_DBO-E', 'P_DBO-P', 'P_DBO-D', 'P_DBO-G',
           'P_DQO-D', 'P_DQO-G', 'P_SST-E', 'P_SST-P', 'P_SST-D', 'P_SST-G',
           'P_SED-E', 'P_SED-P', 'P_SED-D', 'P_SED-G', 'P_COND-E', 'P_COND-P',
           'P_COND-D', 'P_COND-G'],
        
        'performance_E_new':['P_DBO-E', 'P_SST-E', 'P_SED-E', 'P_COND-E'],
        'performance_P_new':['P_DBO-P', 'P_SST-P', 'P_SED-P', 'P_COND-P'],
        'performance_D_new':['P_DBO-D', 'P_DQO-D', 'P_SST-D', 'P_SED-D', 'P_COND-D'],
        'performance_G_new':['P_DBO-G', 'P_DQO-G', 'P_SST-G', 'P_SED-G', 'P_COND-G'],
        'DBO': ['DBO-E','DBO-P','DBO-D','DBO-S','RD-DBO-P', 'RD-DBO-S', 'RD-DBO-G'],
        'DQO': ['DQO-E', 'DQO-D', 'DQO-S', 'RD-DQO-S', 'RD-DQO-G'],
        'SED': ['SED-E','SED-P','SED-D','SED-S','RD-SED-P','RD-SED-G'],
        'COND': ['COND-E','COND-P','COND-D'],
        'input_plant': ['Q-E','ZN-E','PH-E','DBO-E','DQO-E','SS-E','SSV-E','SED-E','COND-E'],
        'input_1_settler': ['PH-P','DBO-P','SSF-P','SSV-P', 'SST-P','SED-P','COND-P'],
        'P_clean_variables': ['PH-P','DBO-P','SST-P','SED-P','COND-P','P_DBO-P','P_SST-P','P_SED-P','P_COND-P'],
        'input_2_settler': ['PH-D','DBO-D','DQO-D','SST-D','SED-D','COND-D'],
        'performance_var': ['DBO','DQO','SST','SED','COND'],
        'performance_values': ['DBO-E', 'DBO-P', 'DBO-D', 'DBO-S', 'DQO-D', 'DQO-S', 'SST-E', 'SST-P', 'SST-D',
          'SST-S', 'SED-E', 'SED-P', 'SED-D', 'SED-S', 'COND-E', 'COND-P', 'COND-D', 'COND-S'],
        'in_out_names': ['E','P','D','S'],
        'value':['Q-E', 'ZN-E', 'PH-E', 'DBO-E', 'DQO-E', 'SST-E', 'SED-E',
           'COND-E', 'PH-P', 'DBO-P', 'SST-P', 'SED-P', 'COND-P', 'PH-D', 'DBO-D',
           'DQO-D', 'SST-D', 'SED-D', 'COND-D', 'PH-S', 'DBO-S', 'DQO-S', 'SST-S',
           'SED-S', 'COND-S', 'RD-DBO-P', 'RD-SS-P', 'RD-SED-P', 'RD-DBO-S',
           'RD-DQO-S', 'RD-DBO-G', 'RD-DQO-G', 'RD-SS-G', 'RD-SED-G'],
        'numerics': ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'],
        'variable': ['Q-E', 'ZN-E', 'PH-E', 'DBO-E', 'DQO-E', 'SST-E', 'SED-E',
           'COND-E', 'PH-P', 'DBO-P', 'SST-P', 'SED-P', 'COND-P', 'PH-D', 'DBO-D',
           'DQO-D', 'SST-D', 'SED-D', 'COND-D', 'PH-S', 'DBO-S', 'DQO-S', 'SST-S',
           'SED-S', 'COND-S'],
        'clean_variable':  ['Q-E', 'ZN-E', 'PH-E', 'DBO-E', 'DQO-E', 'SST-E', 'SED-E',
            'PH-P', 'DBO-P','SST-P', 'SED-P', 'COND-P', 'PH-D', 'DBO-D',
           'DQO-D', 'SST-D', 'SED-D', 'PH-S', 'DBO-S', 'DQO-S', 'SST-S',
           'SED-S', 'COND-S'],
        'levelS': ['level_PH-S','level_ZN-S', 'level_SED-S', 'level_COND-S','level_SST-S','level_DBO-S','level_DQO-S'],
        'levelP': ['level_P_DBO-E', 'level_P_DBO-P', 'level_P_DBO-D', 'level_P_DBO-G',
           'level_P_DQO-D', 'level_P_DQO-G', 'level_P_SST-E', 'level_P_SST-P',
           'level_P_SST-D', 'level_P_SST-G', 'level_P_SED-E', 'level_P_SED-P',
           'level_P_SED-D', 'level_P_SED-G', 'level_P_COND-E', 'level_P_COND-P',
           'level_P_COND-D', 'level_P_COND-G']
       
    }
    return group[nome]

def dictionary(col):
    """
    Apresenta as definições de cada coluna da tabela
    """
    dic = {
    'Q-E': '(input flow to plant)',
    'ZN-E': '(input Zinc to plant)',
    'PH-E': '(input pH to plant)',
    'DBO-E': '(input Biological demand of oxygen to plant)',
    'DQO-E': '(input chemical demand of oxygen to plant)',
    'SS-E': '(input suspended solids to plant)',
    'SSV-E': '(input volatile supended solids to plant)',
    'SED-E': '(input sediments to plant)',
    'COND-E': '(input conductivity to plant)',
    'PH-P': '(input pH to primary settler)',
    'DBO-P': '(input Biological demand of oxygen to primary settler)',
    'SS-P': '(input suspended solids to primary settler)',
    'SSV-P': '(input volatile supended solids to primary settler)',
    'SED-P': '(input sediments to primary settler)',
    'COND-P': '(input conductivity to primary settler)',
    'PH-D': '(input pH to secondary settler)',
    'DBO-D': '(input Biological demand of oxygen to secondary settler)',
    'DQO-D': '(input chemical demand of oxygen to secondary settler)',
    'SS-D': '(input suspended solids to secondary settler)',
    'SSV-D': '(input volatile supended solids to secondary settler)',
    'SED-D': '(input sediments to secondary settler)',
    'COND-D': '(input conductivity to secondary settler)',
    'PH-S': '(output pH)',
    'DBO-S': '(output Biological demand of oxygen)',
    'DQO-S': '(output chemical demand of oxygen)',
    'SS-S': '(output suspended solids)',
    'SSV-S': '(output volatile supended solids)',
    'SED-S': '(output sediments)',
    'COND-S': '(output conductivity)',
    'RD-DBO-P': '(performance input Biological demand of oxygen in primary settler)',
    'RD-SS-P': '(performance input suspended solids to primary settler)',
    'RD-SED-P': '(performance input sediments to primary settler)',
    'RD-DBO-S': '(performance input Biological demand of oxygen to secondary settler)',
    'RD-DQO-S': '(performance input chemical demand of oxygen to secondary settler)',
    'RD-DBO-G': '(global performance input Biological demand of oxygen)',
    'RD-DQO-G': '(global performance input chemical demand of oxygen)',
    'RD-SS-G': '(global performance input suspended solids)',
    'RD-SED-G': '(global performance input sediments)'}
    return dic[col]

def import_wt_data():
    """
    Importa os dados, realiza o tratamento e classificação dos dados
    """
    #Importar os dados
    data = pd.read_csv(r"C:\Users\Bruno M Venturott\Documents\Poli\EAD\TCC\Python arquives\water-treatment.data",
                   sep=",", header = None, index_col=0)
    #Nomear as colunas
    data.columns = orig_cols()

    #definir data como index e formatar em data
    data.index = data.index.str.replace(r'D-', '')
    data.index = pd.to_datetime(data.index)
    #substituir os valores null
    data = data.replace('?', np.nan)
    #formatar os valores em números
    for c in data.columns:
        data[c] = pd.to_numeric(data[c])
    #ordenar cronologicamente
    data = data.sort_index()
    #criar uma coluna de mês
    data['month'] = data.index.strftime('%Y/%m')
    data = data[['month']+orig_cols()]
    
    #Renomear colunas SS para SST (toal), SSV está em porcentagem de SS
    data = data.rename(columns={'SS-E':'SST-E','SS-P':'SST-P','SS-D':'SST-D','SS-S':'SST-S'})

    #tranformar valores infinitos em nan
    data = data.replace([np.inf, -np.inf], np.nan)
    return class_data(data)

def class_data(data):
    """
    Coloca as classificações dos dados
    """  
    #Cálculo da performance
    var = cols_group('performance_var')
    for v in var:
        e = np.nan
        p = np.nan
        d = np.nan
        g = np.nan
        #filtro do DQO
        if v != 'DQO':
            #E e P
            e = (data[v+'-E']-data[v+'-P'])/data[v+'-E']
            #P e D
            p = (data[v+'-P']-data[v+'-D'])
            #Adicionar os valores nas colunas sem DQO
            data['P_'+v+'-E'] = e
            data['P_'+v+'-P'] = p
        #D e S
        d = (data[v+'-D']-data[v+'-S'])
        #Global
        g = (data[v+'-E']-data[v+'-S'])
        
        data['P_'+v+'-D'] = d
        data['P_'+v+'-G'] = g
    
    #Classificação dos dias em função do documento
    data_class = []
    data_class_simp = []
    invalid_columns = []
    Class5 = pd.to_datetime(['1990-01-28','1990-04-02','1990-04-03','1990-04-09','1990-05-07','1990-05-27',
                             '1990-05-28','1990-05-30','1990-06-26','1990-06-27','1990-07-01','1990-07-03',
                             '1990-07-20','1990-07-22','1990-07-24','1990-07-25','1990-08-16','1990-08-28',
                             '1990-08-31','1990-09-02','1990-09-03','1990-10-24','1990-10-25','1990-11-02',
                             '1990-11-05','1990-11-09','1990-11-12','1990-11-13','1990-12-07','1991-01-09',
                             '1991-03-01','1991-03-08','1991-03-17','1991-03-26','1991-03-31','1991-04-14',
                             '1991-04-22','1991-04-24','1991-04-25','1991-05-10','1991-05-16','1991-05-20',
                             '1991-05-29','1991-05-30','1991-06-05','1991-06-10','1991-06-12','1991-06-14',
                             '1991-07-05','1991-07-08','1991-07-09','1991-07-21','1991-07-26','1991-10-02',
                             '1991-10-08','1991-10-09','1991-10-11','1991-10-13','1991-10-16'

    ])
    Class9 = pd.to_datetime(['1990-08-13','1990-08-15','1990-08-19','1990-08-20','1990-08-27','1990-10-21',
                             '1990-10-23','1990-10-26','1990-10-28','1990-11-01','1990-11-04','1990-11-11',
                             '1990-11-19','1991-03-07','1991-03-24','1991-03-25','1991-04-26','1991-04-28',
                             '1991-04-29','1991-05-01','1991-05-05','1991-05-08','1991-05-09','1991-05-12',
                             '1991-05-13','1991-05-26','1991-05-27','1991-06-09','1991-06-24','1991-07-02',
                             '1991-07-14','1991-07-29','1991-08-04','1991-08-28','1991-08-30','1991-10-01',
                             '1991-10-03','1991-10-05','1991-10-12','1991-10-15'
    ])
    ClassNormal2 = pd.to_datetime(['1990-10-19','1990-11-15','1990-12-02','1990-12-04','1990-12-06','1990-12-21',
                                   '1990-12-26','1991-01-03','1991-01-04','1991-01-07','1991-02-01','1991-02-04',
                                   '1991-02-06','1991-02-07','1991-02-15','1991-02-19','1991-03-19','1991-03-21',
                                   '1991-03-22','1991-04-04','1991-05-06'
    ])

    for d in data.index:
        #Classificação em função das datas
        if d == pd.to_datetime('1990-03-13'): 
            data_class.append('Secondary settler problems-1')
            data_class_simp.append('Secondary settler problems')
            
        elif d == pd.to_datetime('1990-08-12'):
            data_class.append('Storm-2')
            data_class_simp.append('Storm')

        elif d == pd.to_datetime('1990-03-14'):
            data_class.append('Secondary settler problems-2')
            data_class_simp.append('Secondary settler problems')

        elif d == pd.to_datetime('1990-03-15') or (
            pd.to_datetime('1991-07-19') >= d >= pd.to_datetime('1991-07-17')):
            data_class.append('Secondary settler problems-3')
            data_class_simp.append('Secondary settler problems')

        elif d in Class5 or (
            pd.to_datetime('1990-07-31') >= d >= pd.to_datetime('1990-07-27')) or (
            pd.to_datetime('1990-09-27') >= d >= pd.to_datetime('1990-09-24')) or (
            pd.to_datetime('1990-06-22') >= d >= pd.to_datetime('1990-06-10')) or (
            pd.to_datetime('1990-07-11') >= d >= pd.to_datetime('1990-07-09')) or (
            pd.to_datetime('1990-08-07') >= d >= pd.to_datetime('1990-08-01')) or (
            pd.to_datetime('1990-07-18') >= d >= pd.to_datetime('1990-07-16')) or (
            pd.to_datetime('1990-09-13') >= d >= pd.to_datetime('1990-09-06')) or (
            pd.to_datetime('1990-09-21') >= d >= pd.to_datetime('1990-09-16')) or (
            pd.to_datetime('1990-05-23') >= d >= pd.to_datetime('1990-05-21')) or (
            pd.to_datetime('1990-04-26') >= d >= pd.to_datetime('1990-04-22')) or (
            pd.to_datetime('1990-10-05') >= d >= pd.to_datetime('1990-10-01')):

            data_class.append('Normal situation with performance over the mean')
            data_class_simp.append('Normal')

        elif d in pd.to_datetime(['1990-06-05','1991-05-28','1991-05-31']):
            data_class.append('Solids overload-1')
            data_class_simp.append('Solids overload')

        elif d == pd.to_datetime('1990-04-29'):
            data_class.append('Secondary settler problems-4')
            data_class_simp.append('Secondary settler problems')

        elif d == pd.to_datetime('1990-09-14'):
            data_class.append('Storm-1')
            data_class_simp.append('Storm')

        elif d in Class9 or (
            pd.to_datetime('1990-08-10') >= d >= pd.to_datetime('1990-08-08')) or (
            pd.to_datetime('1990-10-09') >= d >= pd.to_datetime('1990-10-07')) or (
            pd.to_datetime('1990-10-17') >= d >= pd.to_datetime('1990-10-12')) or (
            pd.to_datetime('1991-08-26') >= d >= pd.to_datetime('1991-08-09')):

            data_class.append('Normal situation with low influent')
            data_class_simp.append('Normal')

        

        elif d == pd.to_datetime('1990-10-22'):       
            data_class.append('Storm-3')
            data_class_simp.append('Storm')

        elif d == pd.to_datetime('1991-05-24'):      
            data_class.append('Solids overload-2')
            data_class_simp.append('Solids overload')

        elif d in ClassNormal2 or(
            pd.to_datetime('1991-03-15') >= d >= pd.to_datetime('1991-03-13')) or (
            pd.to_datetime('1991-02-28') >= d >= pd.to_datetime('1991-02-25')) or (
            pd.to_datetime('1991-01-31') >= d >= pd.to_datetime('1991-01-27')) or (
            pd.to_datetime('1990-11-26') >= d >= pd.to_datetime('1990-11-22')) or (
            pd.to_datetime('1990-11-30') >= d >= pd.to_datetime('1990-11-28')) or (
            pd.to_datetime('1991-02-13') >= d >= pd.to_datetime('1991-02-10')) or (
            pd.to_datetime('1990-12-14') >= d >= pd.to_datetime('1990-12-10')) or (
            pd.to_datetime('1991-01-24') >= d >= pd.to_datetime('1991-01-21')):
            data_class.append('Normal2')
            data_class_simp.append('Normal')

        else:
            data_class.append('Normal')
            data_class_simp.append('Normal')
        
        #CLassificação em função dos dados
        nan_values = []
        row_values = np.array(pd.isna(data.loc[d]))
        
        for i in range(len(row_values)):
            if row_values[i] == True:
                nan_values.append(i)
        invalid_columns.append(nan_values)
            
    data['class'] = data_class
    data['class_simp'] = data_class_simp
    data['invalid_columns'] = invalid_columns
    
    #Classificação da vazão
    data.loc[data['Q-E'] < 30000, 'class_vazao'] = 'Low flow' 
    data.loc[data['Q-E'].between(30000,45000), 'class_vazao'] = 'Avg flow'
    data.loc[data['Q-E'] > 45000, 'class_vazao'] = 'High flow'
    
    #Classificação das variáveis de saída
    #Acceptable Levels = True
        #PH
    data['level_PH-S'] = False
    data.loc[data['PH-S'].between(6.5,9),'level_PH-S'] = True
    data.loc[data['PH-S'].isnull(),'level_PH-S'] = np.nan
    #data.loc[data['PH-S'] < 6.5, 'class_PH-S'] = 'Acid'
    #data.loc[data['PH-S'].between(6.5,9), 'class_PH-S'] = 'Normal'
    #data.loc[data['PH-S'] > 9, 'class_PH-S'] = 'Basic'
        #ZN - não é removido durante o processo, logo ZN-E = ZN-S
    data['level_ZN-S'] = False
    data.loc[data['ZN-E'] < 50,'level_ZN-S'] = True
    data.loc[data['ZN-E'].isnull(),'level_ZN-S'] = np.nan
    #data.loc[data['ZN-E'] >= 50, 'class_ZN-S'] = 'High Level'
    #data.loc[data['ZN-E'] < 50, 'class_ZN-S'] = 'Low Level'
        #SED
    data['level_SED-S'] = False
    data.loc[data['SED-S'] <= 0.05,'level_SED-S'] = True
    data.loc[data['SED-S'].isnull(),'level_SED-S'] = np.nan
    #data.loc[data['SED-S'] > 0.05, 'class_SED-S'] = 'High Level'
    #data.loc[data['SED-S'] <= 0.05, 'class_SED-S'] = 'Low Level'
        #COND
    data['level_COND-S'] = False
    data.loc[data['COND-S']<=1500,'level_COND-S'] = True
    data.loc[data['COND-S'].isnull(),'level_COND-S'] = np.nan
    #data.loc[data['COND-S'] < 3, 'class_COND-S'] = 'Distilled Water'
    #data.loc[data['COND-S'].between(30,1500), 'class_COND-S'] = 'Tap Water'
    #data.loc[data['COND-S'].between(1500,2000,inclusive=False), 'class_COND-S'] = 'Freshwater stream'
    #data.loc[data['COND-S'] >= 2000, 'class_COND-S'] = 'Industrial Waste'
        #SST
    data['level_SST-S'] = False
    data.loc[data['SST-S'] <= 20,'level_SST-S'] = True
    data.loc[data['SST-S'].isnull(),'level_SST-S'] = np.nan
    #data.loc[data['COND-S'] <= 40, 'class_SST-S'] = 'Clear'
    #data.loc[data['COND-S'] > 40, 'class_SST-S'] = 'Cloudy'
        #SSV
        #DBO
    data['level_DBO-S'] = False
    data.loc[data['DBO-S'] <= 20,'level_DBO-S'] = True
    data.loc[data['DBO-S'].isnull(),'level_DBO-S'] = np.nan
        #DQO
    data['level_DQO-S'] = False
    data.loc[data['DQO-S'] <= 100,'level_DQO-S'] = True
    data.loc[data['DQO-S'].isnull(),'level_DQO-S'] = np.nan
    #Global, retorna true se todas os levels forem true 
    na = data[cols_group('levelS')].isna().sum(axis=1)
    data['level_G'] = False
    data.loc[data[cols_group('levelS')].sum(axis=1,skipna=True) == (len(cols_group('levelS'))-na), 'level_G'] = True
    
    #Classificação da performance
       #porcentagem de tolerância
    factor_p = .3 
    for v in cols_group('performance_new'):
        #variable name
        name = v.split('_')[-1].split('-')[0]
        #filtrar dados a serem considerados, aqueles que apresentam operação normal naquela variável
        x = data.copy()
        x = x[x['level_'+name+'-S']==True]
        #tirar valores nulos da performance
        x = x[v].dropna()
        #determinar variância
        x_std = x.std()
        #calcular a mediana por kernal density
        nparam_density = stats.kde.gaussian_kde(x)
        nparam_density = nparam_density(x)
        limit = pd.Series(nparam_density,index=x).idxmax()
        #criar classificação da performance
        data['level_'+ v] = 'Normal' 
        data.loc[data[v] <= limit-x_std/2,'level_'+ v] = 'Low'
        data.loc[data[v] >= limit+x_std/2,'level_'+ v] = 'High'
        data.loc[data[v].isnull(),'level_'+ v] = np.nan


    return(data)

def limit_values(var):
    """
    Apresenta as definições de cada coluna da tabela
    """
    dic = {
    'PH': [6.5,9],
    'ZN': [0,50],
    'SED':[0,0.05],
    'COND':[0,1500],
    'SST':[0,20],
    'DBO':[0,20],
    'DQO':[0,100]
   }
    return dic[var]

def build_graph(dados, classes, medida):
    """
    Constroi o gráfico de linhas para a utlização do widgets
    """
    for classe in classes:
        y_data = dados[dados.index.get_level_values('class_simp') == classe]
    plt.figure(figsize=(20,5))
    plt.plot(y_data.index.get_level_values(0), y_data[medida[0]], marker='o')
    plt.legend(classes)
    plt.show()
    
    
    y_data = dados[dados.index.get_level_values('class_simp') == 'Normal']
    plt.figure(figsize=(20,5))
    plt.plot(y_data.index.get_level_values(0), y_data[medida[0]], marker='o')
    plt.legend(classes)
    plt.show()
    return

def var_var_graph(dados, var1, var2,classif):
    """
    Constroi o gráfico de dispersão para comparar a relação entre as variáveis
    """
    classif = classif[0]
    var1 = var1[0]
    var2 = var2[0]
    #filtrar os dados nulos de cada variável analizada
    dados = dados[dados[var1].isnull()==False]
    dados = dados[dados[var2].isnull()==False]
    #calcular o coeficiente global dos pontos
    coef = np.polyfit(dados[var1],dados[var2],1)
    corr = np.corrcoef(dados[var1],dados[var2])
    print('Global:'+str(var2)+' = '+str(round(coef[0],3))+'.'+str(var2)+' + '+str(round(coef[1],3))+' R^2 = '+str(round(corr[0][1],3)))

    
    dados.index.name = 'index'
    dados = dados.groupby([dados.index.name,var1,classif])[var2].sum().unstack(classif)
    
    dados = dados.reset_index()
    fig, ax = plt.subplots()
    for c in dados.columns[2:]:
        print(c)
        ax = plt.scatter(x=dados[var1], y=dados[c], alpha=0.4)
        col  = dados[dados[c].isnull()==False]
        coef = np.polyfit(col[var1],col[c],1)
        corr = np.corrcoef(col[var1],col[c])
        d = round(corr[0][1],3)
        
        print(str(c)+': '+str(var2)+' = '+str(round(coef[0],3))+'.'+str(var2)+' + '+str(round(coef[1],3))+' R^2 = '+str(d))
        
    plt.ylabel(var2)
    plt.xlabel(var1)
    plt.title(var1+' vs '+var2)
    plt.legend(dados.columns[2:])
    

    
    return 

def flow_x_variable(dados):
    """
    Constri gráfico para comparar a vazão da planta com as demais variáveis
    """
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def f_tsne(data, n=50, hue=None):
    """
    Função para plotar o gráfico usando o tsne e permitir o uso dos widgets
    """
    
    tsne = data.copy()
    #Eliminar as linhas com valores inválidos
    tsne = tsne[tsne['invalid_columns'].str.len()==0]
    #determinar o learning_rate
    m = TSNE(learning_rate=n)
    #transformar em duas linhas
    tsne_features = m.fit_transform(tsne.select_dtypes(include=cols_group('numerics')))
    
    tsne['tsne-x'] = tsne_features[:,0]
    tsne['tsne-y'] = tsne_features[:,1]
    print('n = ',n)
    print('hue = ', hue)
    sns.scatterplot(x='tsne-x',y='tsne-y',data=tsne, hue=hue[0], palette='bright')

    return

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = patches.Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def df_failure_motive(data):
    index = data.index
    fail_matrix = []
    in_out_names = cols_group('in_out_names')
    #Percorrer as linhas
    for i in index:
        #Percorrer as colunas de classificação  
        for j in cols_group('levelS'):
            #Se a variável for falha
            if data.loc[i][j] == False:
                #Extrair nome da variável global
                var = j.split('_')[1].split('-')[0]
                #Percorrer as variáveis da falha
                nv = 1
                level_streams = []
                streams = []
                while nv < len(in_out_names):
                    #Definir o qual é a entrada
                    v = in_out_names[nv]
                    #pular a variável DQO-P
                    if var+'-'+v == 'DQO-P':
                        nv+=1
                        v = in_out_names[nv]
                    #Definir a variável responsável
                    resp = np.nan
                    #Verificar se a variável da corrente está num nível maior do que o limite
                    if data.loc[i][var+'-'+v] >= limit_values(var)[1]:
                        resp = var+'-'+v
                        streams.append(resp)
                        #Ver o nível de performance das correntes falhas
                            #no caso da corrente de saída, retorna a performance global
                        if v == 'S':
                            v = 'G'
                        #Verificar se há performance
                        if var == 'PH':
                            level_streams.append(np.nan)
                        else:
                            level_streams.append(data.loc[i]['level_P_'+var+'-'+v])
                    nv+=1
                #Verifica se resp está nan, nan==nan -> False
                if resp == resp:
                    row = [i,j,streams[0].split('-')[1],streams[0],streams,level_streams]
                    fail_matrix.append(row)
    fail_df = pd.DataFrame(data=fail_matrix, columns=['index','fault','first_faulty_stream','first_faulty_variable','faulty_variable','perf_streams'])
    return fail_df

