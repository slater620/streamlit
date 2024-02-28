import streamlit as st
import joblib

# Charger le modèle
model = joblib.load('model_adaBoot.joblib')

# Interface utilisateur
st.title('Prédiction du nutri_score')

# Entrée des caractéristiques
product = st.text_input('nom du produit', value="")
energy = st.number_input('Énergie pour 100g',step=1.,format="%.2f")
saturated_fat = st.number_input('Matières grasses saturées pour 100g',step=1.,format="%.2f")
sugars = st.number_input('Sucres pour 100g', value=0)
fiber = st.number_input('Fibres pour 100g', value=0)
proteins = st.number_input('Protéines pour 100g', value=0)
salt = st.number_input('Sel pour 100g', value=0)

# Prédiction
if st.button('Prédire'):
    features = [[energy, saturated_fat, sugars, fiber, proteins, salt]]
    prediction = model.predict(features)
    st.write('Le nutri_score de', product, 'est', prediction[0])
