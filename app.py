import numpy as np 
import os

from flask import Flask, render_template, session, redirect, url_for, session, request
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from wtforms import TextField, SubmitField


base_dir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'
app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(base_dir, 'uploads')

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB

# Load the model:
cnn_model = load_model('<write-here-the-name-of-the-file>.h5')
CLASS_INDICES = {0: 'cat', 1: 'dog'}


# Form where image will be uploaded:
class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, 'Image Only!'), FileRequired('Choose a file to upload!')])
    submit = SubmitField('Get Prediction')


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    # If the form is valid on submission:
    if form.validate_on_submit():
        # Saving file to folder
        file = request.files['photo']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for("prediction", filename=filename))
    else: 
        return render_template('home.html', form=UploadForm())


@app.route('/prediction/<filename>')
def prediction(filename):
    results = return_prediction(model=cnn_model, filename=filename)
    return render_template('prediction.html', results=results)


def return_prediction(model, filename):
    # Note: in some computers such as Macs, the line below may throw an error. If that's your case, add 'uploads/' before the filename.
    # e.g.: _image_process('uploads/' + filename)
    input_image_matrix = _image_process(filename)
    score = cnn_model.predict(input_image_matrix)
    class_index = cnn_model.predict_classes(input_image_matrix, batch_size=1)
    
    return CLASS_INDICES[class_index[0]], score


def _image_process(filename):
    img = image.load_img(filename, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    input_matrix = np.vstack([x])
    input_matrix /= 255.
    return input_matrix


if __name__ == '__main__':
    app.run(debug=True)
