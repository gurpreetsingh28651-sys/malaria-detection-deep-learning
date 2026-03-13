import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model('model/malaria_model.h5')

# Path of test image
img_path = 'test.jpg'   # change this to your image

# Load image
img = image.load_img(img_path, target_size=(64,64))
img_array = image.img_to_array(img)

# Normalize
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Prediction
prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("🦠 Parasitized (Malaria Infected)")
else:
    print("✅ Uninfected (Healthy Cell)")