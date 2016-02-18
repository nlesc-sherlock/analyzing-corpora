#!/usr/bin/python
from prepare_for_visualisation import prepare_for_visualisation

utomat = 'data/enron_lda_15_norm.csv'
dictionary = 'data/enron_dic.csv'
delim = ','
maxword = 100
wpdict = prepare_for_visualisation(utomat,dictionary,delim,maxword)

print wpdict['Topic 1']

from flask import Flask, jsonify
from flask.ext.cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/getdata')
def getData():
    return jsonify(
            words=wpdict
            # utoorder, utonorm, winninguto, Nwordsto
        )

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0')
