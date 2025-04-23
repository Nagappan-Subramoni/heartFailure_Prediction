import gradio as gr
import numpy as np
import joblib

save_file_name = 'xgboost-model.pkl'  # Path to the saved model
model = joblib.load(save_file_name)

# Dummy prediction function (replace with your model logic)

def predict_death_event(age, anaemia, creatinine_phosphokinase, diabetes,
       ejection_fraction, high_blood_pressure, platelets,
       serum_creatinine, serum_sodium, sex, smoking, time):
    input_array = np.array([age, anaemia, creatinine_phosphokinase, diabetes,
       ejection_fraction, high_blood_pressure, platelets,
       serum_creatinine, serum_sodium, sex, smoking, time]).reshape(1, -1)
    prediction = model.predict(input_array)
    if prediction[0] == 1:
        return "Patient died during the follow-up period"
    else:
        return "Patient survived during the follow-up period"



# Interface
iface = gr.Interface(
    fn=predict_death_event,
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

# iface.launch(server_port=8000)
# iface.launch(server_name="0.0.0.0", server_port=8000)
iface.launch(server_name="0.0.0.0", server_port=8000, share=False, inbrowser=False)


