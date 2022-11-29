from flask import Flask, render_template, request, url_for, make_response, jsonify

from werkzeug.utils import secure_filename
import os
import cv2
from gevent import pywsgi
import time
from datetime import timedelta
import numpy as np

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)

result = {}

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files.get('file').filename
        print("f in python is "+ f)

        if not (f and allowed_file(f)):
            return jsonify({"error": 1001, "msg": "check the format of picture, only for png、PNG、jpg、JPG、bmp"})

        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'static\images', f)
        request.files.get('file').save(upload_path)

        print("path is " + str(upload_path))
        #becuase the file path contain Chinses characters so that it should be decoded
        #img = cv2.imread(upload_path)
        #cv2.imwrite(os.path.join(basepath, 'static/images', 'test.jpg'), img)

        img = cv2.imdecode(np.fromfile(upload_path,dtype=np.uint8),cv2.IMREAD_COLOR)

        print("img is " + str(img.size))
        result['shape'] = str(img.shape)
    return "image shape is "+result['shape']


@app.route('/', methods=['POST', 'GET'])
def website():  # put application's code her
    return render_template('converter.html')


if __name__ == '__main__':
    print(app.url_map)
    app.run(debug=True)
