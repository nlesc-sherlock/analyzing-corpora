#!/usr/bin/python
from prepare_for_visualisation import prepare_for_visualisation

utomat = 'data/enron_lda_15_norm.csv'
dictionary = 'data/enron_dic.csv'
delim = ','
allWords, utoorder, utonorm, winninguto, Nwordsto = prepare_for_visualisation(utomat,dictionary,delim)

from flask import Flask, jsonify
from flask.ext.cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/getdata')
def getData():
    # words = allWords[:10,:].tolist()
    words = [
        [ ('hello', 0.9), ('world', 0.05), ('!', 0.05)],  # T1
        [ ('hola', 0.9), ('mundo', 0.05), ('!', 0.05)],  # T2
    ]
    return jsonify(
            words=words
            # utoorder, utonorm, winninguto, Nwordsto
        )

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0')
