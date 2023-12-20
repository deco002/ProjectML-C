import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# Function to load the model using joblib
def load_model(model_path):
    return joblib.load(model_path)

# Function to make predictions
def predict(model, input_data):
    return model.predict(input_data.reshape(1, -1))[0]

# Function to classify efficiency based on overall load
def classify_efficiency(overall_load):
    if overall_load < 29:
        return 'Low'
    elif overall_load < 64:
        return 'Average'
    else:
        return 'High'

# Function to display the input form and make predictions
def prediction_form(model, input_data, prediction_type, scaler):
    st.title(f"{prediction_type.capitalize()} Prediction App")

    # Input form
    st.header("Input Features")

    relative_compactness = st.number_input("Relative Compactness", 0.0, 1.0, 0.5)
    surface_area = st.number_input("Surface Area", 500, 1000, 750)
    wall_area = st.number_input("Wall Area", 100, 300, 200)
    roof_area = st.number_input("Roof Area", 100, 300, 200)
    overall_height = st.number_input("Overall Height", 2, 7, 4)
    orientation = st.number_input("Orientation", 0, 270, 0, step=90)
    glazing_area = st.number_input("Glazing Area", 0.0, 0.9, 0.5)
    glazing_area_distribution = st.number_input("Glazing Area Distribution", 1, 5, 1)

    # Prepare input data as a NumPy array
    input_data = np.array([relative_compactness, surface_area, wall_area, roof_area,
                           overall_height, orientation, glazing_area, glazing_area_distribution])

    # Scale the input features using MinMaxScaler
    scaled_input_data = scaler.transform(input_data.reshape(1, -1))

    # Make prediction
    result = predict(model, scaled_input_data)

    # Display the result
    st.subheader(f"{prediction_type.capitalize()} Prediction Result:")
    st.write(result)

    if prediction_type == 'efficiency':
        efficiency_class = classify_efficiency(result)
        st.subheader("Efficiency Classification:")
        st.write(efficiency_class)

# Run the app
if __name__ == "__main__":
    st.sidebar.title("Choose a Prediction Type")
    prediction_types = ["heating", "cooling", "efficiency"]
    chosen_prediction_type = st.sidebar.radio("Select Prediction Type", prediction_types)

    st.sidebar.title("Choose a Model Set")
    model_set_names = ["Heating Model Set", "Cooling Model Set"]
    chosen_model_set = st.sidebar.selectbox("Select a Model Set", model_set_names)

    st.sidebar.title("Choose a Model")
    model_names = [" Linear regression", "KNN regressor", "Ridge", "Lasso", "Linear SVR", "SVM", "Decision Tree", "Ensemble Gradient Boosting"]
    chosen_model = st.sidebar.selectbox("Select a Model", model_names)

    # Dummy paths for the 9 heating models, 9 cooling models, and scaler (replace these with actual paths)
    heating_model_paths = [
        "E:\BIMA\KULIAH\SEM 7\Pembelajaran Mesin\Proyek akhir\Machine-Learning-Project\model\Heating\linearRegression_model.sav",
        "E:\BIMA\KULIAH\SEM 7\Pembelajaran Mesin\Proyek akhir\Machine-Learning-Project\model\Heating\KNNregressor_model.sav",
        "E:\BIMA\KULIAH\SEM 7\Pembelajaran Mesin\Proyek akhir\Machine-Learning-Project\model\Heating\Ridge_model.sav",
        "E:\BIMA\KULIAH\SEM 7\Pembelajaran Mesin\Proyek akhir\Machine-Learning-Project\model\Heating\Lasso_model.sav",
        "E:\BIMA\KULIAH\SEM 7\Pembelajaran Mesin\Proyek akhir\Machine-Learning-Project\model\Heating\LinearSVR.sav",
        "E:\BIMA\KULIAH\SEM 7\Pembelajaran Mesin\Proyek akhir\Machine-Learning-Project\model\Heating\SVMregressor.sav",
        "E:\BIMA\KULIAH\SEM 7\Pembelajaran Mesin\Proyek akhir\Machine-Learning-Project\model\Heating\DecisionTree.sav",
        "E:\BIMA\KULIAH\SEM 7\Pembelajaran Mesin\Proyek akhir\Machine-Learning-Project\model\Heating\EnsembleGradientBoosting.sav"

        
    ]

    cooling_model_paths = [
        "E:\BIMA\KULIAH\SEM 7\Pembelajaran Mesin\Proyek akhir\Machine-Learning-Project\model\Cooling\LinearRegressionCool.sav",
        "E:\BIMA\KULIAH\SEM 7\Pembelajaran Mesin\Proyek akhir\Machine-Learning-Project\model\Cooling\KNNregressorCool_model.sav",
        "E:\BIMA\KULIAH\SEM 7\Pembelajaran Mesin\Proyek akhir\Machine-Learning-Project\model\Cooling\Ridge_modelCool.sav",
        "E:\BIMA\KULIAH\SEM 7\Pembelajaran Mesin\Proyek akhir\Machine-Learning-Project\model\Cooling\lassoCool.sav",
        "E:\BIMA\KULIAH\SEM 7\Pembelajaran Mesin\Proyek akhir\Machine-Learning-Project\model\Cooling\LinearSVRCOOL.sav",
        "E:\BIMA\KULIAH\SEM 7\Pembelajaran Mesin\Proyek akhir\Machine-Learning-Project\model\Cooling\SVMregressorCOOL.sav",
        "E:\BIMA\KULIAH\SEM 7\Pembelajaran Mesin\Proyek akhir\Machine-Learning-Project\model\Cooling\DecisionTreeCOOL.sav",
        "E:\BIMA\KULIAH\SEM 7\Pembelajaran Mesin\Proyek akhir\Machine-Learning-Project\model\Cooling\EnsembleGradientBoostingCOOL.sav"


    ]

    # Load the selected model using joblib
    if chosen_model_set == "Heating Model Set":
        model_path = heating_model_paths[model_names.index(chosen_model)]
    elif chosen_model_set == "Cooling Model Set":
        model_path = cooling_model_paths[model_names.index(chosen_model)]
    
    loaded_model = load_model(model_path)

    # Load the MinMaxScaler used in your project code
    scaler_path = "E:\BIMA\KULIAH\SEM 7\Pembelajaran Mesin\Proyek akhir\Machine-Learning-Project\model\scaler.pkl"  # Replace with the actual path
    scaler = joblib.load(scaler_path)

    # Run the app with the loaded model, scaler, and chosen prediction type
    prediction_form(loaded_model, None, chosen_prediction_type, scaler)
