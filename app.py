import base64

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
import  base64

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPEG', 'PNG', 'bmp'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    result = "[Wrong request]"
    if request.method == 'POST':
        f = request.files.get('file').filename
        print("f in python is "+ f)
        result = dict()

        if not (f and allowed_file(f)):
            msg = "check the format of picture, only for png、PNG、jpg、JPEG、bmp"
            result['text'] = msg
            return result

        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'static\images', 'origin')
        request.files.get('file').save(upload_path)

        print("path is " + str(upload_path))
        #becuase the file path contain Chinses characters so that it should be decoded
        #img = cv2.imread(upload_path)
        #cv2.imwrite(os.path.join(basepath, 'static/images', 'test.jpg'), img)

        img = cv2.imdecode(np.fromfile(upload_path,dtype=np.uint8),cv2.IMREAD_COLOR)
        start = datetime.datetime.now()

        result['text'] = recognize_text(img)
        end = datetime.datetime.now()
        print(end-start)
        out_path = os.path.join(basepath, 'out', 'r.png')
        # b64encode，b64decode
        img = cv2.imdecode(np.fromfile(out_path,dtype=np.uint8),cv2.IMREAD_COLOR)
        img_data = cv2.imencode('.png', img)[1].tobytes()
        # The image is encoded into stream data, placed in the memory cache, and then converted to string format
        base64_data = base64.b64encode(img_data)
        img_base64 = str(base64_data, encoding='utf-8')
        # base64.b64decode(base64data)
        result['outPath'] = img_base64
    return result


@app.route('/', methods=['POST', 'GET'])
def website():  # put application's code her
    return render_template('converter.html')


if __name__ == '__main__':
    print(app.url_map)
    app.run(debug=True)
