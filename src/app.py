import os
from flask import Flask, request, render_template
from joblib import load
import numpy as np 

app = Flask(__name__)

basedir = os.path.abspath(os.path.dirname(__file__))

wine_model_path = os.path.join(basedir, "..", "models", "k_nearest_neighbor_default_42.sav")
wine_scaler_path = os.path.join(basedir, "..", "models", "k_nearest_neighbor_default_42.sav")

# Cargar el modelo KNN y el scaler
try:
    wine_model = load(open(wine_model_path, "rb"))
    wine_scaler = load(open(wine_scaler_path, "rb"))
    print("Modelo KNN de vino y Scaler cargados exitosamente.")
except FileNotFoundError:
    print(f"Error: No se pudieron cargar el modelo de vino o el scaler.")
    print(f"Asegúrate de que los archivos existan en estas rutas:")
    print(f"Modelo: {wine_model_path}")
    print(f"Scaler: {wine_scaler_path}")
    exit(1) 

quality_class_dict = {
    0: "Baja Calidad (No tan bueno)",
    1: "Buena Calidad (Recomendado)"
}

@app.route("/", methods = ["GET", "POST"])
def index():
    pred_quality = None
    if request.method == "POST":
        
        try:
           
            val1 = float(request.form["fixed_acidity"])
            val2 = float(request.form["volatile_acidity"])
            val3 = float(request.form["citric_acid"])
            val4 = float(request.form["residual_sugar"])
            val5 = float(request.form["chlorides"])
            val6 = float(request.form["free_sulfur_dioxide"])
            val7 = float(request.form["total_sulfur_dioxide"])
            val8 = float(request.form["density"])
            val9 = float(request.form["pH"])
            val10 = float(request.form["sulphates"])
            val11 = float(request.form["alcohol"])

            
            input_data = np.array([[val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11]])

            scaled_input_data = wine_scaler.transform(input_data)

            prediction_numeric = wine_model.predict(scaled_input_data)[0]

            pred_quality = quality_class_dict.get(prediction_numeric, "Desconocido")

        except Exception as e:
            pred_quality = f"Error al procesar la entrada: {e}"
            print(f"Error en POST request: {e}")

    return render_template("index.html", prediction = pred_quality)

if __name__ == '__main__':
    app.run(debug=True)