from flask import Flask, request, redirect, url_for, render_template
import os
from web import to_predict

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/test',methods = ['POST','GET'])
def test():
    if request.method == 'POST':
        f = request.files['file']
        basepath =os.path.dirname(__file__)
        upload_path = os.path.join(basepath,r'static\to_pridict',"to_pridect.png")
        f.save(upload_path)
        return redirect(url_for('ans'))
    return render_template('test.html')

@app.route('/ans')
def ans():
    pct =  to_predict.to_predict()
    return pct

if __name__ == '__main__':
    app.run()