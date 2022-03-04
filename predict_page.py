
import streamlit as sl
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl','rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

    
loaded_model = data["model"]
loaded_le_gender = data["le_gender"]
loaded_le_size = data["le_size"]

def show_prediction_page():
    sl.title("Techcare Innovation - Glove Size Selection")
    sl.write("""### Hi, kindly fillup informations needed below""")

    genders = (
    "Male",
    "Female"
    )

    gender = sl.selectbox("Gender",genders)

    height = sl.slider("Height of Person in cm",min_value=130,max_value=220,step=1)
    weight = sl.slider("Weigh of Person in Kg",min_value=40,max_value=120,step=1)
    handWidth = sl.slider("Hand Width of Person in cm",min_value=5.0,max_value=25.0,step=0.5,value=10.0)
    handLength = sl.slider("Hand-Length of Person in cm",min_value=5.0,max_value=25.0,step=0.5,value=10.0)

    ok = sl.button("Predict Glove Size!")

    if ok:
        g_input = np.array([[  gender,   weight,  height,   handLength,    handWidth]])
        g_input[:,0] = loaded_le_gender.transform(g_input[:,0])
        g_input = g_input.astype(float)

        predictedSize = loaded_model.predict(g_input)
        size = loaded_le_size.inverse_transform(predictedSize)
        sl.subheader("The estimated size is " + str(size[0]))
       ## g_input = g_input.astype(float)



##tutorial

## https://www.youtube.com/watch?v=xl0N7tHiwlw 
## https://www.youtube.com/watch?v=nJHrSvYxzjE&t=321s

