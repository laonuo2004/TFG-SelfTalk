from flask import Flask, render_template, request

app = Flask(__name__)

# 首页
@app.route('/')
def index():
    return render_template('index.html')

# 模型训练页面 & 表单提交
@app.route('/model_training', methods=['GET', 'POST'])
def model_training():
    video_path = None
    if request.method == 'POST':
        # TODO: 后端同学在这里接“训练模型”的逻辑
        data = {
            'model_choice': request.form.get('model_choice'),
            'dataset': request.form.get('dataset'),
            'train_subjects': request.form.get('train_subjects'),
            'val_subjects': request.form.get('val_subjects'),
            'epochs': request.form.get('epochs'),
            'gpu_choice': request.form.get('gpu_choice'),
        }
        
        from backend.model_trainer import train_model
        result = train_model(data)

        return render_template('model_training.html', result=result)
    
        # 前端现在只需要有个占位视频路径，方便调试界面
        video_path = '/static/videos/sample.mp4'
    return render_template('model_training.html', video_path=video_path)

# 视频生成页面 & 表单提交
@app.route('/video_generation', methods=['GET', 'POST'])
def video_generation():
    video_path = None
    if request.method == 'POST':
        # TODO: 后端同学在这里接“生成视频”的逻辑
        video_path = '/static/videos/sample.mp4'
    return render_template('video_generation.html', video_path=video_path)

# 人机对话页面 & 表单提交
@app.route('/chat_system', methods=['GET', 'POST'])
def chat_system():
    video_path = None
    if request.method == 'POST':
        # TODO: 后端同学在这里接“实时对话/语音克隆”的逻辑
        video_path = '/static/videos/sample.mp4'
    return render_template('chat_system.html', video_path=video_path)

# 录音上传接口（前端会用 fetch 调用）
@app.route('/save_audio', methods=['POST'])
def save_audio():
    """
    占位接口：
    - 现在先不真正保存录音，只返回 'ok'
    - 将来后端同学在这里读取 request.files[...]，按需保存即可
    """
    return 'ok'

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    # 监听所有网卡，方便 VS Code 端口转发
    app.run(host='0.0.0.0', port=5001, debug=True)
