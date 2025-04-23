import gradio as gr
import numpy as np

# Dummy prediction function (replace with your model logic)
def predict_outcome(age, anaemia, creatinine_phosphokinase, diabetes,
                    ejection_fraction, high_blood_pressure, platelets,
                    serum_creatinine, serum_sodium, sex, smoking, time):
    # Example: simple linear model just for demonstration
    features = np.array([
        age, anaemia, creatinine_phosphokinase, diabetes,
        ejection_fraction, high_blood_pressure, platelets,
        serum_creatinine, serum_sodium, sex, smoking, time
    ], dtype=float)
    
    # Dummy output, replace with model.predict or similar
    prediction = np.dot(features, np.random.rand(len(features)))
    return f"Predicted risk score: {prediction:.2f}"

# Interface
iface = gr.Interface(
    fn=predict_outcome,
    inputs=[
        gr.Number(label="Age"),
        gr.Radio([0, 1], label="Anaemia (0 = No, 1 = Yes)"),
        gr.Number(label="Creatinine Phosphokinase"),
        gr.Radio([0, 1], label="Diabetes (0 = No, 1 = Yes)"),
        gr.Number(label="Ejection Fraction"),
        gr.Radio([0, 1], label="High Blood Pressure (0 = No, 1 = Yes)"),
        gr.Number(label="Platelets"),
        gr.Number(label="Serum Creatinine"),
        gr.Number(label="Serum Sodium"),
        gr.Radio([0, 1], label="Sex (0 = Female, 1 = Male)"),
        gr.Radio([0, 1], label="Smoking (0 = No, 1 = Yes)"),
        gr.Number(label="Time (days)"),
    ],
    outputs="text",
    title="Heart Failure Risk Predictor",
    description="Enter patient details to predict a risk score."
)

iface.launch(server_port=8000)
