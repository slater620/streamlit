import streamlit as st
import joblib
import pandas as pd

# Charger le modèle entraîné
model = joblib.load('model_opti.pkl')

# Fonction de prédiction et de probabilité
def predict_fake_bill(features):
    X = pd.DataFrame([features])
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return prediction, proba

# Définition d'un widget personnalisé pour aligner les caractéristiques horizontalement
def horizontal_input(label, min_value=0.0):
    col1, col2 = st.beta_columns(2)
    return col1.slider(label, min_value=min_value)

# Personnalisation du style avec CSS
style = """
<style>
.horizontal-input-wrapper .stSlider { width: 300px; }
.horizontal-input-wrapper .stNumberInput { width: 70px; }
</style>
"""

# Créer votre interface Streamlit
def main():
    st.title('Détection de faux billet')
    st.write('Veuillez entrer les caractéristiques du billet à analyser :')

    # Appliquer le style CSS
    st.markdown(style, unsafe_allow_html=True)

    # Entrée des caractéristiques du billet
    with st.container():
        st.write("Diagonal et Hauteur gauche")
        diagonal = horizontal_input('Diagonal')
        height_left = horizontal_input('Hauteur gauche')

    with st.container():
        st.write("Hauteur droite et Marge inférieure")
        height_right = horizontal_input('Hauteur droite')
        margin_low = horizontal_input('Marge inférieure')

    with st.container():
        st.write("Marge supérieure et Longueur")
        margin_up = horizontal_input('Marge supérieure')
        length = horizontal_input('Longueur')

    # Bouton pour prédire
    if st.button('Prédire'):
        features = [diagonal, height_left, height_right, margin_low, margin_up, length]
        prediction, proba = predict_fake_bill(features)
        if prediction == 0:
            st.write('Le billet est authentique.')
        else:
            st.write('Le billet est un faux.')
        
        # Afficher la jauge de probabilité
        st.progress(proba[1])  # Probabilité d'obtention d'un faux billet

        st.write(f'Probabilité d\'obtention d\'un vrai billet : {proba[0]:.2f}')
        st.write(f'Probabilité d\'obtention d\'un faux billet : {proba[1]:.2f}')

if __name__ == '__main__':
    main()
