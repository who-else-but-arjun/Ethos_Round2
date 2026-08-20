import os
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def load_images(dataset_dir, img_height, img_width):
    images = []
    labels = []
    class_names = os.listdir(dataset_dir)  
    
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)
        
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (img_width, img_height))
                    images.append(img)
                    labels.append(label)
    images = np.array(images, dtype='float32') / 255.0
    labels = np.array(labels)
    
    return images, labels, class_names
    
def batch_generator(X, Y, batch_size):
    n_samples = X.shape[0]
    while True:
        for offset in range(0, n_samples, batch_size):
            X_batch = X[offset:offset+batch_size]
            Y_batch = Y[offset:offset+batch_size]
            yield X_batch, Y_batch