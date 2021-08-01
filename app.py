from os import listdir, remove
from os.path import isfile, join, splitext
import os 
from PIL import Image, ImageDraw
import numpy as np
from datetime import datetime
import face_recognition  as fr
from flask import Flask, jsonify, request, render_template, flash, redirect, url_for
from flask_cors import CORS
from werkzeug.exceptions import BadRequest
from flask_sqlalchemy import SQLAlchemy

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global storage for images
faces_dict = {}
persistent_faces = "/root/faces"

# Create flask app
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'faces/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
#app.secret_key = "secret key"

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///clients.sqlite3'
app.config['SECRET_KEY'] = "random string"

db = SQLAlchemy(app)

class clients(db.Model):
    id = db.Column('id', db.Integer, primary_key = True)
    name = db.Column(db.String(100))
    age = db.Column(db.Integer)
    attendance = db.Column(db.String(200))
    violence = db.Column(db.String(200)) 
    over_consum = db.Column(db.String(200))
    black_listed = db.Column(db.String(200))

    def __init__(self, name, age, attendance, violence, over_consum, black_listed):
        self.name = name
        self.age = age
        self.attendance = attendance
        self.violence = violence
        self.over_consum = over_consum
        self.black_listed = black_listed

# Black list
black_list = []
"""
black_names = clients.query.filter_by(black_listed='yes')
for name in black_names.name:
    black_list.append(name)
black_list = ['jayz', 'ross', 'monica']"""

def is_picture(filename):
    image_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in image_extensions

def get_all_picture_files(path):
    files_in_dir = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    return [f for f in files_in_dir if is_picture(f)]

def remove_file_ext(filename):
    return splitext(filename.rsplit('/', 1)[-1])[0]

def calc_face_encoding(image):
    # Currently only use first face found on picture
    loaded_image = fr.load_image_file(image)
    faces = fr.face_encodings(loaded_image)

    # If more than one face on the given image was found -> error
    if len(faces) > 1:
        raise Exception(
            "Found more than one face in the given training image.")

    # If none face on the given image was found -> error
    if not faces:
        raise Exception("Could not find any face in the given training image.")

    return faces[0]

def get_faces_dict(path):
    image_files = get_all_picture_files(path)
    return dict([(remove_file_ext(image), calc_face_encoding(image))
        for image in image_files])

def detect_faces_in_image(file_stream):
    # Load the uploaded image file
    img = fr.load_image_file(file_stream)

    # Get face encodings for any faces in the uploaded image
    uploaded_faces = fr.face_encodings(img)

    # Defaults for the result object
    faces_found = len(uploaded_faces)
    faces = []

    if faces_found:
        face_encodings = list(faces_dict.values())
        for uploaded_face in uploaded_faces:
            match_results = fr.compare_faces(
                face_encodings, uploaded_face)
            for idx, match in enumerate(match_results):
                if match:
                    match = list(faces_dict.keys())[idx]
                    match_encoding = face_encodings[idx]
                    dist = fr.face_distance([match_encoding],
                            uploaded_face)[0]
                    faces.append({
                        "id": match,
                        "dist": dist
                    })

    return {
        "count": faces_found,
        "faces": faces
    }

def draw_faces_in_image(file_stream):
    # Load an image with an unknown face
    unknown_image = fr.load_image_file(file_stream)
    
    # Find all the faces and face encodings in the unknown image
    face_locations = fr.face_locations(unknown_image)
    face_encodings = fr.face_encodings(unknown_image, face_locations)
    
    # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
    pil_image = Image.fromarray(unknown_image)
    # Create a Pillow ImageDraw Draw instance to draw with
    draw = ImageDraw.Draw(pil_image)

    # face encogins for knowing image
    known_face_encodings = list(faces_dict.values())
    # Detected names 
    names = []
    # Loop through each face found in the unknown image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = fr.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = fr.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = list(faces_dict.keys())[best_match_index]
            names.append(name)

        """
        if name in black_list:
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 0))
            # Draw a label with a name below the face
            text_width, text_height = draw.textsize(name)
            draw.rectangle(((left, bottom), (right, bottom + text_height + 10)), fill=(0, 0, 0), outline=(0, 0, 0))
            draw.text((left + 10, bottom + 5), name + "  BL", fill=(255, 255, 255, 255))
        else :   
        """
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom), (right, bottom + text_height + 10)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 10, bottom + 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw
    image_name = "img_rect-{}.jpg".format(datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p"))
    pil_image.save("static/image_rect/{}".format(image_name))
    # Display the resulting image
    image_path = os.path.join("static/image_rect/", image_name)
    return image_path, names

@app.route('/')
def upload_form():
	return render_template('home.html')
    
@app.route('/', methods=['POST'])
def web_recognize():
    file = extract_image(request)

    if file and is_picture(file.filename):
        # The image file seems valid! Detect faces and return the result.
        image_path, namess = draw_faces_in_image(file)
        return render_template("show_image.html", user_image = image_path, names = namess)
        #return jsonify(detect_faces_in_image(file))
    else:
        raise BadRequest("Given file is invalid!")

@app.route('/upload')
def upload_form2():
	return render_template('upload_train.html')

@app.route('/faces', methods=['GET', 'POST', 'DELETE'])
def web_faces():
    # GET
    if request.method == 'GET':
        return jsonify(list(faces_dict.keys()))

    # POST/DELETE
    file = extract_image(request)
    #if 'id' not in request.args:
    #    raise BadRequest("Identifier for the face was not given!")
    id_name = file.filename.rsplit('.', 1)[0]
    if request.method == 'POST':
        app.logger.info('%s loaded', file.filename)
        # HINT jpg included just for the image check -> this is faster then passing boolean var through few methods        
        # TODO add method for extension persistence - do not forget abut the deletion
        file.save("{0}/{1}.jpg".format(persistent_faces, id_name))
        try:
            new_encoding = calc_face_encoding(file)
            faces_dict.update({id_name: new_encoding})
        except Exception as exception:
            raise BadRequest(exception)

    elif request.method == 'DELETE':
        faces_dict.pop(request.args.get('id'))
        remove("{0}/{1}.jpg".format(persistent_faces, request.args.get('id')))

    return jsonify(list(faces_dict.keys()))

@app.route('/clients')
def show_all():
    return render_template('show_all.html', clients = clients.query.all() )

@app.route('/clients/new', methods = ['GET', 'POST'])
def new():
    if request.method == 'POST':
        if not request.form['name'] or not request.form['violence'] or not request.form['black_listed']:
            flash('Please enter all the fields', 'error')
        else:
            client = clients(request.form['name'], request.form['age'],
               request.form['attendance'], request.form['violence'], request.form['over_consum'], request.form['black_listed'])

            db.session.add(client)
            db.session.commit()
            flash('Record was successfully added')
            return redirect(url_for('show_all'))
    return render_template('new.html')

@app.route('/clients/<name>')
def info_perso(name):
    client_info = clients.query.filter_by(name=name).first()
    if client_info :
        return render_template('info_perso.html', client_info = client_info)
    return "Client with name = {} doesn't exist".format(name)

def extract_image(request):
    # Check if a valid image file was uploaded
    if 'file' not in request.files:
        raise BadRequest("Missing file parameter!")

    file = request.files['file']
    if file.filename == '':
        raise BadRequest("Given file is invalid")

    return file
# </Controller>


if __name__ == "__main__":
    print("Starting by generating encodings for found images...")
    # Calculate known faces
    faces_dict = get_faces_dict(persistent_faces)
    #print(faces_dict)
    db.create_all()
    # Start app
    print("Starting WebServer...")
    app.run(host='0.0.0.0', port=8080, debug=False)