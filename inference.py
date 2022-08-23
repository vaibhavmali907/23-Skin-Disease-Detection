import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt



unique_labels = ['Acne and Rosacea Photos','Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions','Atopic Dermatitis Photos','Bullous Disease Photos',
'Cellulitis Impetigo and other Bacterial Infections','Eczema Photos','Exanthems and Drug Eruptions','Hair Loss Photos Alopecia and other Hair Diseases',
'Herpes HPV and other STDs Photos','Light Diseases and Disorders of Pigmentation','Lupus and other Connective Tissue diseases','Melanoma Skin Cancer Nevi and Moles',
'Nail Fungus and other Nail Disease','Poison Ivy Photos and other Contact Dermatitis','Psoriasis pictures Lichen Planus and related diseases','Scabies Lyme Disease and other Infestations and Bites',
'Seborrheic Keratoses and other Benign Tumors','Systemic Disease','Tinea Ringworm Candidiasis and other Fungal Infections','Urticaria Hives','Vascular Tumors','Vasculitis Photos','Warts Molluscum and other Viral Infections']
model = tf.keras.models.load_model("model.h5")


def predict(img_path):  # mandatory: function name should be predict and it accepts a string which is image location
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(48,48,1))
    plt.imshow(img)
    img = np.expand_dims(img, axis=0)
    img = img.reshape(3, 48, 48)
    yhat = model.predict(img)
    yhat = np.array(yhat)
    indices = np.argmax(yhat, axis=1)
    scores = yhat[np.arange(len(yhat)), indices]
    predicted_categories = [unique_labels[i] for i in indices]
    category = predicted_categories[0]
    confidence = round(scores[0] * 100, 2)
    output = category + " (Confidence: " + str(confidence) + "%)"
    return output