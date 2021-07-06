from models.experimental import attempt_load
from utils.torch_utils import select_device
from werkzeug.utils import secure_filename
from PIL import Image
import base64
import os
import io
from flask import Flask, render_template, Response, request, session, redirect, url_for, send_from_directory, jsonify, flash
import json
import numpy as np
from backend.predict import predict
from pathlib import Path
from backend.camera import Camera
#导入数据库模块
import pymysql
import traceback  
#传递根目录
SECRET_KEY = 'This is my key'
# 传入__name__实例化Flask
app = Flask(__name__)

app.secret_key = SECRET_KEY

#获取注册请求及处理
@app.route('/registuser')
def getRigistRequest():

    # db = pymysql.connect(host=“localhost”, user=“root”, password=“root”, database=“testdb”)
    db = pymysql.connect(host="localhost",user="root",password="Mm123456",database="maskdetection" )
    # 使用cursor()方法获取操作游标 
    cursor = db.cursor()
    # SQL 插入语句
    sql = "INSERT INTO user(name, password, phonenumber) VALUES ('"+request.args.get('name')+"', '"+request.args.get('password')+"','"+request.args.get('phonenumber')+"')"
    try:
        # 执行sql语句
        cursor.execute(sql)
        # 提交到数据库执行
        db.commit()
         #注册成功之后跳转到登录页面
        return render_template('login.html') 
    except:
        #抛出错误信息
        traceback.print_exc()
        # 如果发生错误则回滚
        db.rollback()
        return '注册失败'
    # 关闭数据库连接
    db.close()

#获取登录参数及处理
@app.route('/loginuser')
def getLoginRequest():
    db = pymysql.connect(host="localhost",user="root",password="Mm123456",database="maskdetection" )
    cursor = db.cursor()
    sql = "select * from user where name='"+request.args.get('name')+"' and password='"+request.args.get('password')+"'"
    try:
        # 执行sql语句
        cursor.execute(sql)
        results = cursor.fetchall()
        print(len(results))
        if len(results)==1:
            session['name'] = request.args.get('name')
            return render_template('photo.html', name = session.get('name')) 
        else:
            return render_template('login.html', error = '用户名或密码错误')
        # 提交到数据库执行
        db.commit()
    except:
        # 如果发生错误则回滚
        traceback.print_exc()
        db.rollback()
    # 关闭数据库连接
    db.close()

@app.route('/logout')
def logout():
    session.pop('name', None)
    return redirect(url_for('index'))

# 读取flask配置
with open('./backend/flask_config.json','r',encoding='utf8')as fp:
    opt = json.load(fp)
    print('Flask Config : ', opt)

# 选择设备
device = select_device(opt['device'])
# 加载模型
model = attempt_load(opt['weights'], map_location=device)  

@app.route('/predict/', methods=['POST'])
# 响应POST消息的预测函数
def get_prediction():
    response = request.get_json()
    data_str = response['image']
    point = data_str.find(',')
    base64_str = data_str[point:]  # remove unused part like this: "data:image/jpeg;base64,"
    image = base64.b64decode(base64_str) # base64图像解码
    img = Image.open(io.BytesIO(image)) # 打开文件
    if (img.mode != 'RGB'):
        img = img.convert("RGB")
    save_path = str(Path(opt['source']) / Path("img4predict.jpg")) # 保存路径
    img.save(save_path) # 保存文件
    # img.save("./frontend/static/images/img4predict.jpg")  

    # convert to numpy array.
    img_arr = np.array(img)
    # print('img_arr shape = %s \n' % str(img_arr.shape))

    results = predict(opt, model, img_arr) # 预测图像

    return jsonify(results)

@app.route('/index')
def index():
    """Video streaming home page."""
    print(session.get('name'))
    return render_template('index.html', name = session.get('name'))

@app.route('/present')
def present():
    """Video streaming home page."""
    return render_template('present.html', name = session.get('name'))

def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/login')
def login():
    return render_template('login.html', name = session.get('name'))

@app.route('/register')
def register():
    return render_template('register.html', name = session.get('name'))

@app.route('/photo')
def photo():
    return render_template('photo.html', name = session.get('name'))

@app.route('/video')
#@app.route('/video/<filename>', methods=['GET'])
def video():
    return render_template('video.html', path = 'static/static/real/20210505214950.mp4')


@app.before_request
def before_action():
    repath= request.path.split('/')
    print(request.path.split('/'))
    if repath[1] != 'index' and repath[1] != 'login' and repath[1] != 'loginuser':
        if session.get('name') == None:
            return render_template('index.html', warning = '请登录后再进行操作')

# @app.route('/video', methods=['POST'])
# def video():
#     return render_template('video.html', src = "static/static/real/20210505212554.mp4")

@app.after_request
def add_headers(response):
    # 允许跨域
    response.headers.add('Access-Control-Allow-Origin', '*') 
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1')
    #app.run(debug=False, host='127.0.0.1')



