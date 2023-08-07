import streamlit as st
import joblib
import pandas as pd

# Chargez votre modèle entraîné
model = joblib.load('model_opti.pkl')

# Définissez une fonction pour prédire les faux billets
def predict_fake_bill(features):
    X = pd.DataFrame([features])
    
    prediction = model.predict(X)[0]
    return prediction

# Créez votre interface Streamlit
def main():
    st.title('Détection de faux billet')
    st.write('Veuillez entrer les caractéristiques du billet à analyser :')

    # Entrée des caractéristiques du billet
    diagonal = st.number_input('Diagonal', min_value=0.0)
    height_left = st.number_input('Hauteur gauche', min_value=0.0)
    height_right = st.number_input('Hauteur droite', min_value=0.0)
    margin_low = st.number_input('Marge inférieure', min_value=0.0)
    margin_up = st.number_input('Marge supérieure', min_value=0.0)
    length = st.number_input('Longueur', min_value=0.0)

    # Bouton pour prédire
    if st.button('Prédire'):
        features = [diagonal, height_left, height_right, margin_low, margin_up, length]
        prediction = predict_fake_bill(features)
        if prediction == 0:
            st.write('Le billet est authentique.')
        else:
            st.write('Le billet est un faux.')

if __name__ == '__main__':
    main()