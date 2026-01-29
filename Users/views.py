from django.shortcuts import render, redirect
from .models import UserRegistrationModel
from django.contrib import messages
from django.conf import settings
import os
import numpy as np
import tensorflow as tf
import cv2
import math
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ===================== BASIC VIEWS =====================
def UserBase(request):
    return render(request, 'users/UserBase.html')

def UserHome(request):
    return render(request, 'users/UserHome.html')


# ===================== USER REGISTRATION =====================
from .forms import UserRegistrationForm
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'You have been successfully registered')
            return render(request, 'UserRegistration.html', {'form': UserRegistrationForm()})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
    return render(request, 'UserRegistration.html', {'form': UserRegistrationForm()})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            if check.status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                return render(request, 'users/UserHome.html')
            else:
                messages.success(request, 'Your Account Not Activated')
        except:
            messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html')


# ===================== TRAINING ==================

from django.shortcuts import render, redirect
from .models import UserRegistrationModel
from django.contrib import messages
from django.conf import settings
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import cv2
import math




# ================= TRAINING =================
def Training(request):

    data_dir = os.path.join(settings.BASE_DIR, 'media/archive (2)/dataset')
    img_height, img_width = 128, 128
    batch_size = 32
    epochs = 3

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs
    )

    model.save(os.path.join(settings.BASE_DIR, 'kidney_stone_model.keras'))

    return render(request, 'users/Training.html', {
        'train_accuracy': round(history.history['accuracy'][-1] * 100, 2),
        'val_accuracy': round(history.history['val_accuracy'][-1] * 100, 2),
    })


# ================= ANALYSIS =================
def analyze_stone(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = cv2.equalizeHist(img)

    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    stone_pixels = cv2.countNonZero(thresh)

    PIXEL_TO_MM = 0.264
    stone_size_mm = round(math.sqrt(stone_pixels) * PIXEL_TO_MM, 2)

    if stone_size_mm < 5:
        severity = "Mild (≤ 5 mm)"
    elif stone_size_mm <= 10:
        severity = "Moderate (5–10 mm)"
    else:
        severity = "Severe (> 10 mm)"

    region = "Unknown"
    if stone_pixels > 0:
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            if cx < 42:
                region = "Left Kidney Region"
            elif cx > 84:
                region = "Right Kidney Region"
            else:
                region = "Central Kidney Region"

    return stone_size_mm, severity, region





def treatment_recommendation(stone_size_mm):
    if stone_size_mm < 5:
        return "Increase water intake and monitor regularly."
    elif stone_size_mm <= 10:
        return "Medication and medical supervision recommended."
    else:
        return "Surgical intervention may be required. Consult a urologist."

# ================= PREDICTION =================
loaded_model = None

def prediction(request):
    global loaded_model

    if request.method == 'POST':

        if 'image' not in request.FILES:
            return render(request, 'users/prediction.html', {
                'result': 'Please upload an image'
            })

        # Load model once
        if loaded_model is None:
            loaded_model = tf.keras.models.load_model(
                os.path.join(settings.BASE_DIR, 'kidney_stone_model.keras'),
                compile=False
            )

        # Save uploaded image
        img_file = request.FILES['image']
        temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp_images')
        os.makedirs(temp_dir, exist_ok=True)
        img_path = os.path.join(temp_dir, img_file.name)

        with open(img_path, 'wb+') as f:
            for chunk in img_file.chunks():
                f.write(chunk)

        # Preprocess
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = np.expand_dims(
            image.img_to_array(img) / 255.0, axis=0
        )

        # Predict
        pred = loaded_model.predict(img_array)[0][0]
        confidence = round(pred * 100, 2)

        if pred > 0.4:
            stone_size, severity, region = analyze_stone(img_path)
            treatment = treatment_recommendation(stone_size)
            result = "Kidney Stone Detected"
        else:
            stone_size = 0
            severity = "None"
            region = "N/A"
            treatment = "No treatment required."
            result = "No Kidney Stone Detected"

        return render(request, 'users/prediction.html', {
            'result': result,
            'confidence': confidence,
            'stone_size': stone_size,
            'severity': severity,
            'region': region,
            'treatment': treatment,
            'image_path': settings.MEDIA_URL + 'temp_images/' + os.path.basename(img_path)
        })

    return render(request, 'users/prediction.html')
