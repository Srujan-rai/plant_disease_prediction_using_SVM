from flask import Flask,request,jsonify,render_template,url_for
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import joblib
from werkzeug.utils import secure_filename
import os

vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model_filename = 'model/svm_model.joblib'
svm_model = joblib.load(model_filename)

app=Flask(__name__)

class_indices={
    'healthy':0,
    'powdery':1,
    'rust':2
    
}

IMAGE_SIZE=(224,224)
@app.route('/')
def index():
    return render_template('index.html')


def predict_image(image_path, model, class_indices):
    img = load_img(image_path, target_size=IMAGE_SIZE)
    
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = vgg16_model.predict(img_array)
    features = features.reshape((features.shape[0], -1))

    prediction = model.predict(features)
    prediction_prob = model.predict_proba(features)

    predicted_class = list(class_indices.keys())
    predicted_label = predicted_class[prediction[0]]
    confidence = prediction_prob[0][prediction[0]]

    return predicted_label, confidence

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method=="POST":
        
        image=request.files["file"]
        
        uploads_folder="uploads"
        os.makedirs(uploads_folder,exist_ok=True)
        file_path=os.path.join(uploads_folder,secure_filename(image.filename))
        image.save(file_path)
        
        predicted_label, confidence = predict_image(file_path, svm_model, class_indices)
        os.remove(file_path)
        
        
        print(predicted_label)
        

        return jsonify({'disease':predicted_label})
    
    else:
        return url_for(index)    
        



if __name__=="__main__":
    app.run(debug=True,host='0.0.0.0')