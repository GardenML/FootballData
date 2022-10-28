#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: miguelcabezon
"""
#-----------------IMPORTAMOS PAQUETES-----------------#
import pandas as pd
import json
import seaborn as sns

from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont

#----------------------FUNCIONES------------------#

def unpack(df, column, fillna=None):
    ret = None
    if fillna is None:
        ret = pd.concat([df, pd.DataFrame((d for idx, d in df[column].iteritems()))], axis=1)
        #del ret[column]
    else:
        ret = pd.concat([df, pd.DataFrame((d for idx, d in df[column].iteritems())).fillna(fillna)], axis=1)
        #del ret[column]
    return ret

#----------------LECTURA DE INFORMACION --------------------#

matches=pd.read_json("/Users/miguelcabezon/Documents/GitHub/Football_DataScientis/opendata/data/matches.json")

juntos = pd.DataFrame()

matches["id"].unique()
for i in matches["id"].unique():
    with open('/Users/miguelcabezon/Documents/GitHub/Football_DataScientis/opendata/data/matches/{}/match_data.json'.format(i), 'r') as f:
        prueba_dict = json.load(f)
    df=pd.DataFrame(prueba_dict["players"])
    df["match_id"]=i
    juntos=pd.concat([juntos,df],ignore_index=True)


posicion_campo2=pd.DataFrame()
for match in [2269, 2417]:
    data=pd.read_json("/Users/miguelcabezon/Documents/GitHub/Football_DataScientis/opendata/data/matches/{}/structured_data.json".format(match))
    for row in data.itertuples():
        a=row    
        for i in row[3]:
            print(a[5])
            prueba2=pd.DataFrame(i, index=['i',])
            prueba2["time"]=a[5]
            prueba2["period"]=a[4]
            prueba2["match_id"]=match
            posicion_campo2=pd.concat([posicion_campo2,prueba2],ignore_index=True)
        
#-----------SAVE DATA ------------#
posicion_campo2.to_csv("/Users/miguelcabezon/Documents/GitHub/Football_DataScientis/posicion_campo_period_2269_2417.csv",sep=';',index=False)
#posicion_campo2=pd.read_csv("/Users/miguelcabezon/Documents/GitHub/Football_DataScientis/posicion_campo.csv",sep=';')


#----------------LIMPIEZA Y TRATAMIENTO DE DATOS --------------------#
posicion_campo3 = posicion_campo2[posicion_campo2['time'].notna()]
posicion_campo3['x_corregida']=posicion_campo3['x']
posicion_campo3.loc[posicion_campo3['period']==2, 'x_corregida']=posicion_campo3['x']*(-1)
posicion_campo3['y_corregida']=posicion_campo3['y']
posicion_campo3.loc[posicion_campo3['period']==2, 'y_corregida']=posicion_campo3['y']*(-1)

medianas=posicion_campo3.groupby(['trackable_object','match_id'])['x_corregida','y_corregida'].agg('median')
medianas=medianas.reset_index()

pegar_nombre=medianas.merge(juntos, how='inner', left_on=['trackable_object','match_id'], right_on=['trackable_object','match_id'])
pegar_nombre=pegar_nombre.drop(columns=['id'])

pegar_nombre=pegar_nombre.reset_index(drop=True)
unpack_df=unpack(pegar_nombre, 'player_role')
unpack_df = unpack_df[unpack_df['trackable_object'].notna()]

unpack_df.loc[(unpack_df['id']==0), 'posicion']='goalkeeper'
unpack_df.loc[(unpack_df['id'].isin([1,2,3,4,5,6]) ), 'posicion']='defense'
unpack_df.loc[unpack_df['id'].isin([7,8,9,10,11]) , 'posicion']='Midfield'
unpack_df.loc[unpack_df['id'].isin([12,13,14,15,16]) , 'posicion']='Forward'

unpack_df.loc[unpack_df['team_id'].isin([76,100]), 'x_corregida']=unpack_df['x_corregida']*(-1)
unpack_df.loc[unpack_df['team_id'].isin([76,100]), 'y_corregida']=unpack_df['y_corregida']*(-1)


#---------------ARBOL DECISION--------------#


cv = KFold(n_splits=10) # Numero deseado de "folds" que haremos
accuracies = list()
max_attributes = len(list(unpack_df))
depth_range = range(1, max_attributes + 1)
 
# Testearemos la profundidad de 1 a cantidad de atributos +1
for depth in depth_range:
    fold_accuracy = []
    tree_model = tree.DecisionTreeClassifier(criterion='entropy',
                                             min_samples_split=2,
                                             min_samples_leaf=1,
                                             max_depth = depth,
                                             class_weight={'Forward':1.0, 'Midfield':1.0, 'defense':1.0, 'goalkeeper': 1.0})
    for train_fold, valid_fold in cv.split(unpack_df):
        f_train = unpack_df.loc[train_fold] 
        f_valid = unpack_df.loc[valid_fold] 
 
        model = tree_model.fit(X = f_train[['x_corregida','y_corregida']], 
                               y = f_train["posicion"]) 
        valid_acc = model.score(X = f_valid[['x_corregida','y_corregida']], 
                                y = f_valid["posicion"]) # calculamos la precision con el segmento de validacion
        fold_accuracy.append(valid_acc)
 
    avg = sum(fold_accuracy)/len(fold_accuracy)
    accuracies.append(avg)


#---------------- DIBUJAMOS EL ULTIMO MODELO ENTRENADO-------#
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(model, 
                   feature_names=list(unpack_df[['x_corregida','y_corregida']]),  
                   class_names=['Forward', 'Midfield', 'defense', 'goalkeeper'],
                   filled=True)

#---------------COMPARAMOS LOS DISTINTOS MODELOS ----------#
df_x = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})
df_x = df_x[["Max Depth", "Average Accuracy"]]
print(df_x.to_string(index=False))

#EL MEJOR MODELO ES EL DE PROFUNDIDAD 2
y_train = unpack_df[['x_corregida','y_corregida']]
x_train = unpack_df["posicion"].values
 
le=preprocessing.LabelEncoder()

le.fit(x_train)
le.classes_
le.transform(x_train)
# Crear Arbol de decision con profundidad = 4
decision_tree = tree.DecisionTreeClassifier(criterion='entropy',
                                            min_samples_split=2,
                                            min_samples_leaf=1,
                                            max_depth = 2,
                                            class_weight={'Forward':1.0, 'Midfield':1.0, 'defense':1.0, 'goalkeeper': 1.0})

decision_tree.fit(y_train, x_train)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(decision_tree, 
                   feature_names=list(unpack_df[['x_corregida','y_corregida']]),  
                   class_names=['Forward', 'Midfield', 'defense', 'goalkeeper'],
                   filled=True)



#---------------GRAFICOS Y REPRESENTACION VISUAL----------------#


ax = sns.scatterplot(
    x="x_corregida",
    y="y_corregida",
    hue="posicion",
    style="team_id",
    s=50,
    data=unpack_df.reset_index(),
)

plt.show()
ax.set_title("Web Sessions Data of Users")
ax.set_xlabel("No.Of.Users")
ax.set_ylabel("Mean Hours Users Spends on the Website")

fg = sns.FacetGrid(data=unpack_df, hue='posicion', aspect=1.61)
fg.map(plt.scatter, 'x_corregida', 'y_corregida').add_legend()

plt.axvline(x=-3.801, ymin=-30, ymax=30)
plt.axvline(x=-37.645, ymin=-30, ymax=30)
plt.axvline(x=6.27, ymin=-30, ymax=30)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
