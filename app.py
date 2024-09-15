import streamlit as st
import pandas as pd
import pickle
from sklearn.datasets import load_iris


# Load the saved model from the pickle file
def load_model():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


# Function to get user input
def user_input_features():
    st.sidebar.header('User Input Parameters')
    
    sepal_length = st.sidebar.slider('Sepal length (cm)', 4.0, 8.0, 5.0)
    sepal_width = st.sidebar.slider('Sepal width (cm)', 2.0, 4.5, 3.0)
    petal_length = st.sidebar.slider('Petal length (cm)', 1.0, 7.0, 4.0)
    petal_width = st.sidebar.slider('Petal width (cm)', 0.1, 2.5, 1.0)


    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }


    features = pd.DataFrame(data, index=[0])
    return features


# Main function for the app
def main():
    # Title and description of the app
    st.set_page_config(page_title="Iris Flower Classification", layout="centered")
    st.title("🌸 Iris Flower Classification App")
    st.markdown("""
    This app uses **Machine Learning** to classify the species of Iris flowers based on sepal and petal measurements.
    The prediction is made using the best trained model (Random Forest, SVM, or K-Neighbors) from the dataset.
    """)
    
    st.sidebar.subheader("Model Details")
    
    # Load the trained model
    model = load_model()


    # Input data
    input_df = user_input_features()
    
    st.header("User Input Parameters")
    st.write("Modify the parameters from the sidebar.")
    
    # Display input data without index
    st.write(input_df.to_html(index=False), unsafe_allow_html=True)


    # Iris dataset for class labels
    iris = load_iris()


    # Prediction and prediction probability
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    # Display results
    st.header("Prediction Results")
    st.success(f"Predicted Class: **{iris.target_names[prediction][0]}** 🌼")


    st.subheader("Prediction Probabilities:")
    prob_df = pd.DataFrame(prediction_proba, columns=iris.target_names)
    st.write(prob_df.to_html(index=False), unsafe_allow_html=True)


    st.sidebar.info("Created by Dev. [GitHub](https://github.com/DebaMohanty1)")


if __name__ == '__main__':
    main()
