import flask
import pickle
import pandas as pd

# Use pickle to load in the pre-trained model.

model_instance = pickle.load(open("Diabetes.pkl", "rb"))

app = flask.Flask(__name__, template_folder='webpage')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))
    if flask.request.method == 'POST':
        Pregnancies = flask.request.form['Pregnancies']
        Glucose = flask.request.form['Glucose']
        BloodPressure = flask.request.form['BloodPressure']
        SkinThickness = flask.request.form['SkinThickness']
        Insulin = flask.request.form['Insulin']
        BMI = flask.request.form['BMI']
        DiabetesPedigreeFunction = flask.request.form['DiabetesPedigreeFunction']
        Age = flask.request.form['Age']
        
        input_variables = pd.DataFrame([pd.Series([Pregnancies,Glucose,BloodPressure, SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])])
        prediction=model_instance.predict_proba(input_variables)
        output='{0:.{1}f}'.format(prediction[0][1], 2)
        output = str(float(output)*100)+'%'
        return flask.render_template('index.html',original_input={'Pregnancies': Pregnancies,
                                                     'Glucose':Glucose,
                                                     'BloodPressure':BloodPressure,
                                                     'SkinThickness':SkinThickness,
                                                     'Insulin':Insulin,
                                                     'BMI':BMI,
                                                     'DiabetesPedigreeFunction':DiabetesPedigreeFunction,
                                                     'Age':Age},
                                         pred=f'Probability of having Diabetes is {output}')


if __name__ == '__main__':
    app.run()