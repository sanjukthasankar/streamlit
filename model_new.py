import streamlit as st
import numpy as np
from prediction import predict
sl=st.slider("Enter SepalLength",1.0,10.0,0.5)
sw=st.slider("Enter Sepalwidth",1.0,10.0,0.5)
pl=st.slider("Enter PetalLength",0.0,10.0,0.5)
pw=st.slider("Enter Petalwidth",0.0,10.0,0.5)
if st.button("Predict the Iris-Species"):
    save=predict(np.array([[sw,sl,pw,pl]]))
    st.success(save[0])
    
    
