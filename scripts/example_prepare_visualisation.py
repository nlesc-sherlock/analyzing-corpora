#!/usr/bin/env python
#
# Copyright 2015 Netherlands eScience Center <info@esciencecenter.nl>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from prepare_for_visualisation import prepare_for_visualisation

utomat = 'data/enron_lda_15_norm.csv'
dictionary = 'data/enron_dic.csv'
delim = ','
maxword = 100
wpdict = prepare_for_visualisation(utomat, dictionary, delim, maxword)

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
