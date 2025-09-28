import os
import shutil  # Nuevo módulo para eliminar carpetas
import subprocess
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image

app = Flask(__name__)

BASE_DIR = os.path.expanduser('~/SatVision2/')
BASE_OUTPUT_DIR = os.path.join(BASE_DIR, 'ckpt','test','test')
PROCESED_IMAGES_DIR = os.path.join(BASE_OUTPUT_DIR, 'compose')
RESULT_DIR = os.path.join(BASE_DIR, 'SatVision', 'data', 'iSAID', 'test', 'images')
os.makedirs(RESULT_DIR, exist_ok=True)
Image.MAX_IMAGE_PIXELS = None


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    allowed_exts = ('.png', '.tif', '.tiff')
    filename_lower = file.filename.lower()
    if not filename_lower.endswith(allowed_exts):
        return jsonify({'error': 'Formato no permitido', 'details': 'Solo se permiten imágenes PNG o TIFF.'}), 400

    try:
        image = Image.open(file.stream)
        width, height = image.size
        if width * height > 5000000:
            return jsonify({
              'error': 'Imagen demasiado grande',
              'details': f'La imagen subida tiene {width * height} píxeles, la cual supera el limite de 5,000,000 píxeles.'
            }), 400
    except Exception as e:
        return jsonify({'error': 'Error procesando la imagen', 'details': str(e)}), 400
    file.seek(0)
    
    for directory in [RESULT_DIR, BASE_OUTPUT_DIR]:
        if os.path.exists(directory):
            for entry in os.listdir(directory):
                path = os.path.join(directory, entry)
                try:
                    if os.path.isfile(path) or os.path.islink(path):
                        os.unlink(path)
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
                except Exception as e:
                    print(f'No se pudo eliminar {path}. Razón: {e}')

    image_path = os.path.join(RESULT_DIR, file.filename)
    file.save(image_path)
    
    script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cmd = [
        "sh", "./scripts/pointflow/test/test_iSAID_pfnet_R50.sh",
        BASE_DIR+"snapshot/pfnet_r50_iSAID.pth",
        BASE_DIR+"ckpt/"
    ]
    process = subprocess.run(cmd, cwd=script_dir, capture_output=True, text=True)
    process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode != 0:
        return jsonify({'error': 'Processing failed', 'details': process.stderr}), 500

    stat_file_path = os.path.join(BASE_OUTPUT_DIR, "class_statistics.txt")
    if os.path.exists(stat_file_path):
        with open(stat_file_path, "r") as stat_file:
            statistics = stat_file.read()
    else:
        statistics = "No statistics available"

    return jsonify({
        'message': 'Imagen procesada correctamente',
        'result_directory': BASE_OUTPUT_DIR,
        'uploaded_image': file.filename,
        'statistics': statistics
    })

@app.route('/results/<filename>', methods=['GET'])
def get_result_file(filename):
    base, ext = os.path.splitext(filename)
    processed_filename = base + '_compose' + '.png'
    return send_from_directory(PROCESED_IMAGES_DIR, processed_filename)

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(RESULT_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)