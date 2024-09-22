from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import sklearn
print(sklearn.__version__)

app = Flask(__name__)

# Load the trained model

def load_model():
    """

    :return:
    """
    model_file=None
    # Load the trained model
    with open('model_linear_regression.pkl', 'rb') as model_file:
        print('Hi')
        model_file = pickle.load(model_file)
    return model_file


@app.route('/')
def hello_geek():
    return render_template('index.html')
#return '<h1>Hello from Flask & Docker</h2>'


@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)
    model=load_model()
    prediction = model.predict(final_features)
    # return render_template('index.html', prediction_text='Predicted House Price: ${:.2f}'.format(prediction[0]))
    return render_template('index.html')



if __name__ == "__main__":
    model_f=load_model()
    print(model_f)
    app.run(debug=True)
