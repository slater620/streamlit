import streamlit as st
import joblib

# Charger le modèle
model = joblib.load('model.joblib')

# Interface utilisateur
st.title('Prédiction du nutri_score')

# Entrée des caractéristiques
product = st.text_input('nom du produit', value="")
energy = st.number_input('Énergie pour 100g',step=1.,format="%.2f")
saturated_fat = st.number_input('Matières grasses saturées pour 100g',step=1.,format="%.2f")
sugars = st.number_input('Sucres pour 100g',step=1.,format="%.2f")
fiber = st.number_input('Fibres pour 100g',step=1.,format="%.2f")
proteins = st.number_input('Protéines pour 100g',step=1.,format="%.2f")
salt = st.number_input('Sel pour 100g',step=1.,format="%.2f")

# Prédiction
if st.button('Prédire'):
    features = [[energy, saturated_fat, sugars, fiber, proteins, salt]]
    prediction = model.predict(features)
    if (prediction == 0):
        st.write('Le nutri_score de', product, 'est A')
    if (prediction == 1):
        st.write('Le nutri_score de', product, 'est B')
    if (prediction == 2):
        st.write('Le nutri_score de', product, 'est C')
    if (prediction == 3):
        st.write('Le nutri_score de', product, 'est D')
    if (prediction == 4):
        st.write('Le nutri_score de', product, 'est E')
