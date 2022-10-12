import streamlit as st
import requests
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import plotly.graph_objects as go
from matplotlib.patches import Rectangle, FancyArrowPatch
import panel as pn
from IPython.display import HTML
from sklearn.neighbors import NearestNeighbors

from flask import Flask

URL_API = "https://prj7app.herokuapp.com/"

# import des modèles
dfPrediction = joblib.load('dfPrediction.joblib')
f_importances = joblib.load('feature_importances.joblib')
exp = joblib.load('exp.joblib')
X_test_init = joblib.load('X_test_init.joblib')
X_test = joblib.load('X_test.joblib')

def main():

    st.set_page_config(page_title="Projet 7 API-Dashboard", page_icon="✅", layout="wide",)

    # Affichage du titre et du sous-titre
    st.title("Projet 7 : Implémenter un modèle de scoring")
    
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Affichage d'informations dans la sidebar
    st.sidebar.subheader("Informations générales")
    
    # Chargement du logo
    logo = load_logo()
    st.sidebar.image(logo, width=200)

    # Chargement de la selectbox
    lst_id = load_selectbox()
    
    global id_client
    id_client = st.sidebar.selectbox("ID Client", lst_id)
    
    # Chargement des informations générales
    nb_credits, rev_moy, credits_moy, targets = load_infos_gen()

    # Affichage des infos dans la sidebar Nombre de crédits existants
    st.sidebar.markdown("<u>Nombre total des crédits demandés :</u>", unsafe_allow_html=True)
    st.sidebar.text(nb_credits)

    # Graphique camembert
    st.sidebar.markdown("<u>Représentation des données selon le Target</u>", unsafe_allow_html=True)

    plt.pie(targets, explode=[0, 0.1], labels=["sans risque", "à risque"], autopct='%1.1f%%', shadow=True, startangle=90)
    st.sidebar.pyplot()

    # Revenus moyens
    st.sidebar.markdown("<u>Revenu moyen des clients en $(USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(rev_moy)

    # Montant crédits moyen
    st.sidebar.markdown("<u>Montant moyen des crédits demandés en $(USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(credits_moy)
    
    # Affichage de l'ID client choisi
    st.write("Vous avez choisi le client :", id_client)

    # Affichage informations du client
    st.header("**Données du client**")
    
    infos_client = identite_client()  
    
    # Situation familiale
    if infos_client["NAME_FAMILY_STATUS_Married"][0] == 1 :
        st.write("Situation familiale : ", "Married")
    if infos_client["NAME_FAMILY_STATUS_Single / not married"][0] == 1 :
         st.write("Situation familiale : ", "Single / not married")
    if infos_client["NAME_FAMILY_STATUS_Civil marriage"][0] == 1 :
        st.write("Situation familiale : ", "Civil marriage")
    if infos_client["NAME_FAMILY_STATUS_Separated"][0] == 1 :
        st.write("Situation familiale : ", "Separated")
    if infos_client["NAME_FAMILY_STATUS_Widow"][0] == 1 :
        st.write("Situation familia le : ", "Widow"[0])
    
    st.write("Nombre d'enfant(s) à charge : ", infos_client["CNT_CHILDREN"][0])        
    st.write("Age de client : ", int(infos_client["DAYS_BIRTH"].values / -365), "ans.")
    
    # Possession voiture 
    if infos_client["FLAG_OWN_CAR"][0] == 1 :
        st.write("Possède voiture : ", "Oui")
    else :
        st.write("Possède voiture : ", "Non")
         
    data_age = load_age_population()
    # Set the style of plots
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(8, 8))
    # Plot the distribution of ages in years
    plt.hist(data_age, edgecolor = 'k', bins = 20)
    plt.axvline(int(infos_client["DAYS_BIRTH"].values / -365), color="black", linestyle=":")
    plt.title('Age du Client')
    plt.xlabel('Age (années)')
    plt.ylabel('Nombre')
    st.pyplot()

    st.subheader("*Données sur les revenus du client*")
    st.write("Total des revenus du client :", infos_client["AMT_INCOME_TOTAL"][0], "$")

    data_revenus = load_revenus_population()
        
    # Set the style of plots
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(8, 8))
       
    # Plot the distribution of revenus
    plt.hist(data_revenus, edgecolor = 'k')
    plt.axvline(infos_client["AMT_INCOME_TOTAL"][0], color="black", linestyle=":")
    plt.title('Revenus du Client')
    plt.xlabel('Revenus en USD ($) ')
    plt.ylabel('Nombre')
    st.pyplot()

    st.write("Montant du crédit demandé:", infos_client["AMT_CREDIT"][0], "$")
    st.write("Montant Annuité du crédit :", infos_client["AMT_ANNUITY"][0], "$")
    st.write("Valeur des biens du client :", infos_client["AMT_GOODS_PRICE"][0], "$")

    # Affichage de l'analyse du client
    st.header("**Analyse du dossier du client**")

    #if st.button("Afficher le scoring du client"):    
        
    st.markdown("<u>Donneés sur le scoring du client :</u>", unsafe_allow_html=True)
    data_predict = dfPrediction[dfPrediction["SK_ID_CURR"] == int(id_client)]
    prediction = round(float(data_predict["SCORE_0"]),2)  
    
    if prediction <= 0.5 :
        st.write("Score du client = ", round(prediction*100, 2), "%  ==> Client à risque")
    else :
        st.write("Score du client = ", round(prediction*100, 2), "%  ==> Client non à risque")
        
    fig = go.Figure(go.Indicator(domain = {'row': 0, 'column': 0}, #{'x': [0, 1], 'y': [0, 1]},
                                 value = round(prediction*100, 2),
                                 mode = "gauge+number",
                                 gauge = {'bar': {'color': "white"},
                                          'axis': {'range': [0, 100]},
                                          'steps' : [{'range': [0, 50], 'color': "red"},{'range': [50, 100], 'color': "green"}],
                                 }))
    st.plotly_chart(fig)
    
    st.markdown("<u>Importances des Features :</u>", unsafe_allow_html=True)
    
    # Afficher features importances globales                
    features_and_importances = np.array(f_importances)

    plt.figure(figsize=(8, 8), edgecolor='black', linewidth=4)
    plt.style.use('seaborn')
    plt.title('Features importances globales')
    plt.barh(features_and_importances[:15, 0][::-1], features_and_importances[:15, 1][::-1].astype(float))
    st.pyplot()
    
    # Afficher features importances locales (LIME)    
    exp.as_pyplot_figure()
    st.pyplot()
    plt.title('Features importances locales (LIME)')
    plt.clf()
                
    # Affichage des dossiers similaires   
    st.markdown("<u>Liste des 10 dossiers les plus proches de ce client :</u>", unsafe_allow_html=True)
    #index_client = X_test_init[X_test_init["SK_ID_CURR"] == int(id_client)].index.values    
    
    neigh = NearestNeighbors(n_neighbors=10, algorithm='auto')
    neigh.fit(X_test)
    X_ID = X_test_init[X_test_init["SK_ID_CURR"] == int(id_client)] 
    X_ID.drop(['SK_ID_CURR'], axis = 1, inplace = True)
    idx = neigh.kneighbors(X=X_ID, n_neighbors=10, return_distance=False)
    similar_id = X_test_init.iloc[idx[0], :] 
    
    resultats = similar_id[['SK_ID_CURR', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'DAYS_BIRTH',
                        'DAYS_EMPLOYED', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_EMPLOYED_PERC', 
                        'INCOME_CREDIT_PERC', 'INCOME_PER_PERSON']]    
    
    st.write(resultats)
    #st.markdown("<i>Target 1 = Client à risque</i>", unsafe_allow_html=True)
    #st.markdown("<i>Target 0 = Client sans risque</i>", unsafe_allow_html=True)

@st.cache()
def load_logo():
    # Construction de la sidebar
    # Chargement du logo
    logo = Image.open("logo.PNG") 
    
    return logo

@st.cache()
def load_selectbox():
    # Requête permettant de récupérer la liste des ID clients
    data_json = requests.get(URL_API + "load_data")
    data = data_json.json()

    # Récupération des valeurs sans les [] de la réponse
    lst_id = []
    for i in data:
        lst_id.append(i[0])

    return lst_id

@st.cache()
def load_infos_gen():

    # Requête permettant de récupérer :
    # Le nombre de lignes de crédits existants dans la base
    # Le revenus moyens des clients
    # Le montant moyen des crédits existants
    infos_gen = requests.get(URL_API + "infos_gen")
    infos_gen = infos_gen.json()

    nb_credits = infos_gen[0]
    rev_moy = infos_gen[1]
    credits_moy = infos_gen[2]

    # Requête permettant de récupérer le nombre de target dans la classe 0 et la classe 1
    targets = requests.get(URL_API + "disparite_target")    
    targets = targets.json()

    return nb_credits, rev_moy, credits_moy, targets

def identite_client():

    # Requête permettant de récupérer les informations du client sélectionné
    infos_client = requests.get(URL_API + "infos_client", params={"id_client":id_client})
   
    # On transforme la réponse en dictionnaire python
    infos_client = json.loads(infos_client.content.decode("utf-8"))
    
    # On transforme le dictionnaire en dataframe
    infos_client = pd.DataFrame.from_dict(infos_client).T

    return infos_client

@st.cache
def load_age_population():
    
    # Requête permettant de récupérer les âges de la population pour le graphique situant le client
    data_age_json = requests.get(URL_API + "load_age_population")
    data_age = data_age_json.json()

    return data_age

@st.cache
def load_revenus_population():
    
    # Requête permettant de récupérer des tranches de revenus de la population pour le graphique situant le client
    data_revenus_json = requests.get(URL_API + "load_revenus_population")
    
    data_revenus = data_revenus_json.json()

    return data_revenus

def load_prediction():
    
    # Requête permettant de récupérer la prédiction de faillite du client sélectionné
    prediction = requests.get(URL_API + "predict", params={"id_client":id_client})    
    prediction = prediction.json() 
    
    return prediction

def load_voisins():
    
    # Requête permettant de récupérer les 10 dossiers les plus proches de l'ID client choisi
    voisins = requests.get(URL_API + "load_voisins", params={"id_client":id_client})

    # On transforme la réponse en dictionnaire python
    voisins = json.loads(voisins.content.decode("utf-8"))
    
    # On transforme le dictionnaire en dataframe
    voisins = pd.DataFrame.from_dict(voisins).T

    # On déplace la colonne TARGET en premier pour plus de lisibilité
    target = voisins["TARGET"]
    voisins.drop(labels=["TARGET"], axis=1, inplace=True)
    voisins.insert(0, "TARGET", target)
    
    return voisins

def load_features_importances():

    # Requête permettant de récupérer les informations du client sélectionné
    features_importances = requests.get(URL_API + "features_importances")
   
    # On transforme la réponse en dictionnaire python
    features_importances = json.loads(features_importances.content.decode("utf-8"))
    
    # On transforme le dictionnaire en dataframe
    features_importances = pd.DataFrame.from_dict(features_importances).T

    return features_importances

if __name__ == "__main__":
    main()