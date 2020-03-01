import flask
import pickle
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import sys

model=pickle.load(open('model/clf.pickle','rb'))

app = flask.Flask(__name__, template_folder='templates')

@app.route('/')
def hello_world():
    return flask.render_template("index.html")

@app.route('/clas',methods=['POST','GET'])
def clas():
	if flask.request.method=='GET':
    		return(flask.render_template('index.html'))
	if flask.request.method=='POST':
		text=[x for x in flask.request.form.values()]
		#text="മോശം "
		tok=word_tokenize(text[0])
		
		result=model.classify(dict([token,True] for token in tok))
		print(result)
		#return redirect('/')
		return flask.render_template('index.html',pred=result)
if __name__ == '__main__':
    app.debug=True

app.run()
