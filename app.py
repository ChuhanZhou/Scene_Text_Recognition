from flask import Flask, render_template, request, url_for, make_response, jsonify

from werkzeug.utils import secure_filename
import os
import cv2
from gevent import pywsgi
import time
from datetime import timedelta
import numpy as np
from algorithm_interface import recognize_text
import datetime

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    result = "[Wrong request]"
    if request.method == 'POST':
        f = request.files.get('file').filename
        print("f in python is "+ f)

        if not (f and allowed_file(f)):
            msg = "check the format of picture, only for png、PNG、jpg、JPG、bmp"
            return msg

        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'static\images', 'origin')
        request.files.get('file').save(upload_path)

        print("path is " + str(upload_path))
        #becuase the file path contain Chinses characters so that it should be decoded
        #img = cv2.imread(upload_path)
        #cv2.imwrite(os.path.join(basepath, 'static/images', 'test.jpg'), img)

        img = cv2.imdecode(np.fromfile(upload_path,dtype=np.uint8),cv2.IMREAD_COLOR)
        start = datetime.datetime.now()
        result = dict()
        result['text'] = recognize_text(img)
        end = datetime.datetime.now()
        print(end-start)
        out_path = os.path.join(basepath, 'out','r.png')
        result['outPath'] = str(out_path)


        print("img is " + str(img.size))
        #result['shape'] = str(img.shape)
    return result


@app.route('/', methods=['POST', 'GET'])
def website():  # put application's code her
    return render_template('converter.html')


if __name__ == '__main__':
    print(app.url_map)
    app.run(debug=True)
