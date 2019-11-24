# -*- coding: utf-8 -*-
"""
    :author: Grey Li <withlihui@gmail.com>
    :copyright: (c) 2017 by Grey Li.
    :license: MIT, see LICENSE for more details.
"""
import os

from flask import Flask, render_template, request, send_from_directory
from flask_dropzone import Dropzone

from main import connect_db, main

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)

app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'uploads'),
    DROPZONE_MAX_FILE_SIZE=1024,
    DROPZONE_MAX_FILES=1,
    DROPZONE_REDIRECT_VIEW='completed'  # set redirect view
)

dropzone = Dropzone(app)


@app.route('/', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files.get('file')
        filename = os.path.join(app.config['UPLOADED_PATH'], 'test.mp4')
        f.save(filename)
    return render_template('index.html')


@app.route('/completed')
def completed():
    filename = os.path.join(app.config['UPLOADED_PATH'], 'test.mp4')
    main(filename, True)
    return render_template('completed.html')


@app.route('/result')
def result():
    conn = connect_db()
    cur = conn.cursor()
    sql_str = "SELECT * FROM t_persons"
    cur.execute(sql_str)
    content = cur.fetchall()

    sql_str = "SHOW FIELDS FROM t_persons"
    cur.execute(sql_str)
    labels = cur.fetchall()
    labels = [l[0] for l in labels]

    return render_template('result.html', labels=labels, content=content)


@app.route("/download/<filename>", methods=['GET'])
def download_file(filename):
    directory = os.getcwd()
    return send_from_directory(directory, filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
