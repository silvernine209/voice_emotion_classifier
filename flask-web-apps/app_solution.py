# minimal example from:
# http://flask.pocoo.org/docs/quickstart/

import pickle
import numpy as np
import flask
from flask import render_template, request, Flask

app = Flask(__name__)  # create instance of Flask class

repeat_doge = 30

with open("lr.pkl", "rb") as f:
    lr_model = pickle.load(f)

@app.route('/')  # the site to route to, index/main in this case
def magical_image() -> str:
    print('You Mad Bro?')
    return '<html>' + repeat_doge * '<img src="/static/emo_birds.jpg">' + '</html>'


@app.route("/classify", methods=["POST", "GET"])
def predict():

    x_input = []
    for i in range(len(lr_model.feature_names)):
        f_value = float(
            request.args.get(lr_model.feature_names[i], "0")
            )
        x_input.append(f_value)

    pred_probs = lr_model.predict_proba([x_input]).flat

    return flask.render_template('predict_final.html',
    feature_names=lr_model.feature_names,
    x_input=x_input,
    prediction=list(np.argsort(pred_probs)[::-1])
    )                         

if __name__ == '__main__':
    app.run()
