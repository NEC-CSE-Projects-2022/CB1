from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pandas as pd
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}  # Allowed file extensions
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    """Check if file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about/about.html')

@app.route('/flowchart')
def flowchart():
    return render_template('flowchart/flowchart.html')

@app.route('/metrics')
def metrics():
    return render_template('metrics/metrics.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded!')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected!')
            return redirect(request.url)
        
        if not allowed_file(file.filename):
            flash('Invalid file type! Please upload a CSV or Excel file.')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Check if the file is empty
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
            
            if df.empty:
                os.remove(filepath)  # Remove the empty file
                flash('Empty file! Please upload a file with data.')
                return redirect(request.url)
        except Exception as e:
            os.remove(filepath)  # Remove file if there's an issue reading it
            flash(f'Error reading file: {str(e)}')
            return redirect(request.url)

        flash('File uploaded successfully!')
        return redirect(url_for('train', filename=filename))

    return render_template('prediction/base.html')

@app.route('/train/<filename>', methods=['GET'])
def train(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        from model.train_model import train_model
        results = train_model(filepath, target_column='Label')
        return render_template('prediction/results.html', results=results)
    except Exception as e:
        return f"An error occurred during training: {str(e)}"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        return "Prediction logic not implemented yet."
    return render_template('prediction/base.html')

if __name__ == '__main__':
    app.run(debug=True)