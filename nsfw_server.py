#!/usr/bin/env python

# Use Python 3
# Install dependencies:     pip install tensorflow flask
# Run server:               python nsfw_server.py
# Open in browser:          http://localhost:8082/classify?image_path=/home/user/image.jpg

import sys
import argparse
import tensorflow as tf
from model import OpenNsfwModel, InputType
from image_utils import create_yahoo_image_loader
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/classify', methods=['GET'])
def classify():
    filename = request.args["image_path"]
    image = create_yahoo_image_loader()(filename)
    predictions = sess.run(model.predictions, feed_dict={model.input: image})
    # print("\tSFW score:\t{}\n\tNSFW score:\t{}".format(*predictions[0]))

    predictions = predictions[0].tolist()
    return jsonify(dict(sfw=predictions[0], nsfw=predictions[1]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", default=8082, help="server http port")
    args = parser.parse_args()

    model = OpenNsfwModel()

    with tf.compat.v1.Session() as sess:
        model.build(weights_path="data/open_nsfw-weights.npy", input_type=InputType["TENSOR"])
        sess.run(tf.compat.v1.global_variables_initializer())

        app.run(port=args.port)
