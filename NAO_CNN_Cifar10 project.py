
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from naoqi import ALProxy


# NAO connection details
NAO_IP = "172.18.16.54"  # replace with your robot's IP
NAO_PORT = 9559

# CIFAR-10 class labels
cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', ' horse', 'ship', 'truck']

# 1. Connect to NAO camera
video = ALProxy("ALVideoDevice", "172.18.16.54" , 9559)
resolution = 2    # 640x480
color_space = 11  # RGB
fps = 5

# Subscribe to the video feed
name_id = video.subscribeCamera("python_client",0, resolution, color_space, fps)

# 2. Get image from NAO
nao_image = video.getImageRemote(name_id)
video.unsubscribe(name_id)

width = nao_image[0]
height = nao_image[1]
array = nao_image[6]

# Convert image to np array
image = np.frombuffer(array, dtype=np.uint8).reshape((height, width, 3))

# Optional: Save or display
cv2.imwrite("nao_captured.jpg", image)
cv2.imshow("Captured", image)
cv2.waitKey(0)

# 3. Preprocess image to 32x32 as in CIFAR-10
resized_img = cv2.resize(image, (32, 32))  # Resizing the image
normalized_img = resized_img.astype('float32') / 255.0  # Normalizing the image
input_img = normalized_img.reshape(1, 32, 32, 3)  # Reshaping for the model

# 4. Build the same CNN model used for CIFAR-10
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. Load trained weights (ensure you have the right weights)
model.load_weights('my_model_weights1.h5')

# 6. Predict the class
pred = model.predict(input_img)
predicted_class = np.argmax(pred)
predicted_label = cifar10_labels[predicted_class]

print("Predicted class:", predicted_label)

# 7. Make NAO speak the result
tts = ALProxy("ALTextToSpeech", "172.18.16.54" , 9559)
tts.say(" this is a " + predicted_label)
file = open("result.txt",'w')
file.write(predicted_label)
file.close()