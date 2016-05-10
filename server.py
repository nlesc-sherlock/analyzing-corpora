from flask import Flask, jsonify
from flask_restful import reqparse
from flask.ext.cors import CORS

app = Flask(__name__)
CORS(app)

def getTopic(n):
    topic = {
        'name'    : 'nameOfTopic',
        'children': 'xxx'
    }
    return topic

@app.route('/ronald')
def ronaldData():
    topics = [ getTopic(n) for n in range(10) ]
    data = {
        'name'    : 'ENRON',
        'children': topics
    }
    return jsonify(data=data)

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0',threaded=True)
