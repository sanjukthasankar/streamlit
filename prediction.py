import joblib

def predict(vals):

    cl=joblib.load("dt_iris")
    return cl.predict(vals)