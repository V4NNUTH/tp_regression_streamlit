import streamlit as st
import pickle
import numpy as np
import torch


# Sidebar for selecting model
model_choice = st.sidebar.selectbox("Please Select Model", ["2-Petal Width (Simple) : Scikit-learn", "2-Petal Width (Simple) : Torch",
                                                     "3-Petal Width : Scikit-learn", "3-Petal Width : Torch",
                                                     "4-Species : Scikit-learn", "4-Species : Torch",
                                                     "5-Fuel Efficiency : Scikit-learn", "5-Fuel Efficiency : Torch"])  # Add more model names as needed
st.sidebar.header("Model Details")



if model_choice == "5-Fuel Efficiency : Scikit-learn":
    with open("Model/model6_1.pkl", "rb") as file:
        model = pickle.load(file)
    with open("Model/scaler6_1.pkl", "rb") as file:
        scaler = pickle.load(file)

    # Title for the web app
    st.title("Model 5: Predicting Fuel Efficiency with Scikit-learn Logistic Regression")

    # Input features
    st.header("Enter Input Features")
    feature_1 = st.number_input("Weight", min_value=100.0, max_value=10000.0, value=100.0)
    feature_2 = st.number_input("Horsepower", min_value=0.0, max_value=300.0, value=0.0)
    feature_3 = st.number_input("Displacement", min_value=50.0, max_value=500.0, value=50.0)
    feature_4 = st.selectbox("Cylinders", options=[3.0, 4.0, 5.0, 6.0, 8.0], index =0)
    

    # Collect input features into a numpy array
    features = np.array([[feature_1, feature_2, feature_3, feature_4]])
    features_scaled = scaler.transform(features)

    # Predict button
    if st.button("Predict"):
        # Make a prediction
        prediction = model.predict(features_scaled)
        if prediction[0] == 1:
            prediction = "higher fuel efficiency than 23 miles per gallon"
        if prediction[0] == 0:
            prediction = "lower fuel efficiency than 23 miles per gallon"
        st.success(f"The car model has {prediction}")

    # Optional: Display extra details or logs
    st.sidebar.write(f"This section uses pre-trained scikit-learn Logistic Regression model to make predictions.")
    
    

if model_choice == "5-Fuel Efficiency : Torch":
    from Model.torch61 import LogisticRegression
    model = LogisticRegression(4)
    model = torch.load("Model/model6_1.pth")
    
    with open("Model/scaler6_1.pkl", "rb") as file:
        scaler = pickle.load(file)
    
    # Title for the web app
    st.title("Model 6.2: Predicting Fuel Efficiency with Torch Logistic Regression")

    # Input features
    st.header("Enter Input Features")
    feature_1 = st.number_input("Weight", min_value=100.0, max_value=10000.0, value=100.0)
    feature_2 = st.number_input("Horsepower", min_value=0.0, max_value=300.0, value=0.0)
    feature_3 = st.number_input("Displacement", min_value=50.0, max_value=500.0, value=50.0)
    feature_4 = st.selectbox("Cylinders", options=[3.0, 4.0, 5.0, 6.0, 8.0], index =0)

    # Collect input features into a numpy array
    features = np.array([[feature_1, feature_2, feature_3, feature_4]])
    features_scaled = torch.tensor(scaler.transform(features), dtype=torch.float32)

    # Predict button
    if st.button("Predict"):
        # Make a prediction
        model.eval()
        with torch.no_grad():
            y_pred = model(features_scaled)
            prediction = (y_pred >= 0.5).float()
    
        if prediction[0] == 1:
            prediction = "higher fuel efficiency than 23 miles per gallon"
        if prediction[0] == 0:
            prediction = "lower fuel efficiency than 23 miles per gallon"
        st.success(f"The car model has {prediction}")

    # Optional: Display extra details or logs
    st.sidebar.write(f"This app uses pre-trained Logistic Regression model built with Torch to make predictions.")



if model_choice == "2-Petal Width (Simple) : Scikit-learn":
    with open("Model/model2.pkl", "rb") as file:
        model = pickle.load(file)

    # Title for the web app
    st.title("Model 2: Predicting Petal Width with Scikit-learn Simple Linear Regression")

    # Input features
    st.header("Enter Input Features")
    feature_1 = st.number_input("Petal Length", min_value=1.0, max_value=20.0, value=1.0)
    feature_2 = st.number_input("Sepal_length", min_value=1.0, max_value=20.0, value=1.0)
    feature_3 = st.number_input("Species", min_value=1.0, max_value=20.0, value=1.0)

    # Collect input features into a numpy array
    features = np.array([[feature_1],[feature_2],[feature_3]])

    # Predict button
    if st.button("Predict"):
        # Make a prediction
        prediction = model.predict(features)
        st.success(f"The expected petal width is {prediction[0]}")

    # Optional: Display extra details or logs
    st.sidebar.write(f"This section uses pre-trained scikit-learn Simple Linear Regression model to make predictions.")
    
    
    
if model_choice == "2-Petal Width (Simple) : Torch":
    from Model.torch2 import SimpleLinearRegression
    model = SimpleLinearRegression()
    model = torch.load("Model/model2.pth")

    # Title for the web app
    st.title("Model 2: Predicting Petal Width with Scikit-learn Simple Linear Regression")

    # Input features
    st.header("Enter Input Features")
    feature_1 = st.number_input("Petal Length", min_value=1.0, max_value=20.0, value=1.0)
    feature_2 = st.number_input("Sepal_length", min_value=1.0, max_value=20.0, value=1.0)
    feature_3 = st.number_input("Species", min_value=1.0, max_value=20.0, value=1.0)

    # Collect input features into a numpy array
    features = np.array([[feature_1],[feature_2],[feature_3]])
    features = torch.tensor(features, dtype=torch.float32)

    # Predict button
    if st.button("Predict"):
        # Make a prediction
        model.eval()
        with torch.no_grad():
            prediction = model(features)
        st.success(f"The expected petal width is {prediction[0][0]}")

    # Optional: Display extra details or logs
    st.sidebar.write(f"This section uses pre-trained Simple Linear Regression model built with Torch to make predictions.")
        
        

if model_choice == "3-Petal Width : Scikit-learn":
    with open("Model/model3.pkl", "rb") as file:
        model = pickle.load(file)
    with open("Model/encoder3.pkl", "rb") as file:
        encoder = pickle.load(file)

    # Title for the web app
    st.title("Model 3: Predicting Petal Width with Scikit-learn Linear Regression")

    # Input features
    st.header("Enter Input Features")
    feature_1 = st.selectbox("Species", options=["setosa", "versicolor", "virginica"], index=0)
    feature_2 = st.number_input("Petal Length", min_value=1.0, max_value=20.0, value=1.0)
    feature_3 = st.number_input("Sepal Length", min_value=1.0, max_value=20.0, value=1.0)

    # Collect input features into a numpy array
    feature_1 = encoder.transform([[feature_1]]).flatten()  # Encode and flatten
    features = np.array([*feature_1, feature_2, feature_3])  # Combine arrays correctly
    features = features.reshape(1, -1)  # Reshape to 2D array

    # Predict button
    if st.button("Predict"):
        # Make a prediction
        prediction = model.predict(features)
        st.success(f"The expected petal width is {prediction[0]}")

    # Optional: Display extra details or logs
    st.sidebar.write(f"This section uses pre-trained scikit-learn Linear Regression model to make predictions.")
    
    
    
if model_choice == "3-Petal Width : Torch":
    from Model.torch3 import LinearRegressionModel
    model = torch.load("Model/model3.pth")
    with open("Model/encoder3.pkl", "rb") as file:
        encoder = pickle.load(file)

    # Title for the web app
    st.title("Model 3: Predicting Petal Width with Torch Linear Regression")

    # Input features
    st.header("Enter Input Features")
    feature_1 = st.selectbox("Species", options=["setosa", "versicolor", "virginica"], index=0)
    feature_2 = st.number_input("Petal Length", min_value=1.0, max_value=20.0, value=1.0)
    feature_3 = st.number_input("Sepal Length", min_value=1.0, max_value=20.0, value=1.0)

    # Collect input features into a numpy array
    feature_1 = encoder.transform([[feature_1]]).flatten()  # Encode and flatten
    features = np.array([*feature_1, feature_2, feature_3])  # Combine arrays correctly
    features = features.reshape(1, -1)  # Reshape to 2D array
    features = torch.tensor(features, dtype=torch.float32)

    # Predict button
    if st.button("Predict"):
        # Make a prediction
        model.eval()
        with torch.no_grad():
            prediction = model(features)
        st.success(f"The expected petal width is {prediction[0][0]}")

    # Optional: Display extra details or logs
    st.sidebar.write(f"This section uses pre-trained LinearRegression model built with Torch to make predictions.")
    
    
    
if model_choice == "4-Species : Scikit-learn":
    with open("Model/model5.pkl", "rb") as file:
        model = pickle.load(file)
    with open("Model/encoder5_island.pkl", "rb") as file:
        encoder = pickle.load(file)
    with open("Model/encoder5_species.pkl", "rb") as file:
        y_encoder = pickle.load(file)
    with open("Model/scaler5.pkl", "rb") as file:
        scaler = pickle.load(file)
        
    # Title for the web app
    st.title("Model 4: Predicting Species Type with Scikit-learn Multi-class Logistic Regression")

    # Input features
    st.header("Enter Input Features")
    feature_1 = st.selectbox("Island", options=['Torgersen', 'Biscoe', 'Dream'], index=0)
    feature_2 = st.number_input("Bill Depth (mm)", min_value=1.0, max_value=40.0, value=20.0)
    feature_3 = st.number_input("Bill Length (mm)", min_value=15.0, max_value=70.0, value=30.0)
    feature_4 = st.number_input("Flipper Length (mm)", min_value=50.0, max_value=350.0, value=100.0)

    # Collect input features into a numpy array
    feature_1 = encoder.transform([[feature_1]])[0]  # Encode and flatten
    feature2_4 = scaler.transform([[feature_2, feature_3, feature_4, 1]])[0][:3]
    features = np.array([feature_1, *feature2_4])  # Combine arrays correctly
    features = features.reshape(1, -1)  # Reshape to 2D array

    # Predict button
    if st.button("Predict"):
        # Make a prediction
        prediction = model.predict(features)
        st.success(f"The expected species type is {y_encoder.inverse_transform(prediction)[0]}")

    # Optional: Display extra details or logs
    st.sidebar.write(f"This section uses pre-trained scikit-learn Multi-class Logistic Regression model to make predictions.")
    
    
if model_choice == "4-Species : Torch":
    with open("Model/encoder5_island.pkl", "rb") as file:
        encoder = pickle.load(file)
    with open("Model/encoder5_species.pkl", "rb") as file:
        y_encoder = pickle.load(file)
    with open("Model/scaler5.pkl", "rb") as file:
        scaler = pickle.load(file)
    
    from Model.torch5 import LogisticRegressionModel
    model = LogisticRegressionModel(4, 3)    
    model = torch.load("Model/model5.pth")
        
    # Title for the web app
    st.title("Model 5: Predicting Species Type with Torch Multi-class Logistic Regression")

    # Input features
    st.header("Enter Input Features")
    feature_1 = st.selectbox("Island", options=['Torgersen', 'Biscoe', 'Dream'], index=0)
    feature_2 = st.number_input("Bill Depth (mm)", min_value=1.0, max_value=40.0, value=20.0)
    feature_3 = st.number_input("Bill Length (mm)", min_value=15.0, max_value=70.0, value=30.0)
    feature_4 = st.number_input("Flipper Length (mm)", min_value=50.0, max_value=350.0, value=100.0)

    # Collect input features into a numpy array
    feature_1 = encoder.transform([[feature_1]])[0]  # Encode and flatten
    feature2_4 = scaler.transform([[feature_2, feature_3, feature_4, 1]])[0][:3]
    features = np.array([feature_1, *feature2_4])  # Combine arrays correctly
    features = features.reshape(1, -1)  # Reshape to 2D array
    features = torch.tensor(features, dtype=torch.float32)

    # Predict button
    if st.button("Predict"):
        # Make a prediction
        model.eval()
        with torch.no_grad():
            prediction = model(features)
            prediction = torch.argmax(prediction, axis=1)
        st.success(f"The expected species type is {y_encoder.inverse_transform(prediction)[0]}")

    # Optional: Display extra details or logs
    st.sidebar.write(f"This section uses pre-trained Multi-class Logistic Regression model built with Torch to make predictions.")