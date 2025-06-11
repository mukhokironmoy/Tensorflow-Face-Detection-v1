# 8. Building the Face Detection Neural Network

In this step, we design and build the neural network model that performs both **image classification** and **bounding box regression** for face detection. We use a pre-trained **VGG16** model as the backbone and then add our own custom layers.

---

## 8.1 Importing Required Components

We begin by importing the necessary classes from `tensorflow.keras`:

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16
```

### Explanation:

- `Model`: The class used to define custom models in TensorFlow.
    
- `Input`: Defines the shape and type of input to the model.
    
- `Conv2D`: Used in CNNs to extract spatial features (already part of VGG16).
    
- `Dense`: Fully connected layers for classification and regression.
    
- `GlobalMaxPooling2D`: Reduces tensor dimensions by taking the maximum of each feature map.
    
- `VGG16`: A well-known deep CNN architecture pre-trained on ImageNet.
    

---

## 8.2 Initializing the Base Model (VGG16)

```python
vgg = VGG16(include_top= False)
vgg.summary()
```

### Explanation:

- `include_top=False`: Removes the original fully connected layers (used for 1000-class classification).
    
- We remove the top layers because we are building a **custom head** for our specific face detection task.
    
- `vgg.summary()` shows the architecture and number of parameters (~14.7M).
    

This pretrained model acts as a feature extractor.

---

## 8.3 Building the Model with Functional API

We define a function `build_model()` to create the face detection model:

```python
def build_model():
    input_layer = Input(shape = (120,120,3))

    vgg = VGG16(include_top= False)(input_layer)

    # Classification Head (for determining if a face is present)
    f1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048, activation='relu')(f1)
    class2 = Dense(1, activation='sigmoid')(class1)

    # Regression Head (for predicting bounding box coordinates)
    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(2048, activation='relu')(f2)
    regress2 = Dense(4, activation='sigmoid')(regress1)

    facetracker = Model(inputs = input_layer, outputs = [class2, regress2])
    return facetracker
```

### Detailed Breakdown:

- **Input Layer**: Accepts images of shape `(120, 120, 3)`.
    
- **VGG16 Backbone**: Acts as the core feature extractor.
    

### Classification Head:

- **GlobalMaxPooling2D**: Collapses spatial dimensions to 1D.
    
- **Dense(2048, relu)**: Fully connected hidden layer.
    
- **Dense(1, sigmoid)**: Outputs probability of face presence (between 0 and 1).
    

### Regression Head:

- **GlobalMaxPooling2D**: Same pooling applied.
    
- **Dense(2048, relu)**: Fully connected hidden layer.
    
- **Dense(4, sigmoid)**: Outputs normalized bounding box coordinates `(x_center, y_center, width, height)`.
    

---

## 8.4 Instantiating the Model and Viewing Summary

```python
facetracker = build_model()
facetracker.summary()
```

### Key Observations:

- **Dual Outputs**: One for classification and one for regression.
    
- **Total Parameters**: ~16.8 million (mostly from the VGG16 backbone).
    
- Sigmoid is used for outputs so we get normalized values in [0, 1] range.
    

---

## 8.5 Testing with Sample Batch

We can test the model with a sample batch from the training dataset:

```python
X, y = train.as_numpy_iterator().next()
X.shape

classes, coords = facetracker.predict(X)
classes, coords
```

### What Happens:

- `X` contains a batch of input images.
    
- `classes`: Predicted probabilities of whether each image contains a face.
    
- `coords`: Predicted bounding boxes for faces.
    

Note: The model is not yet trained, so predictions will be random, but the pipeline works!

---

## ✅ Summary of Completed Step 8

- ✅ Imported necessary TensorFlow/Keras layers and VGG16 model.
    
- ✅ Initialized VGG16 without the top classification layer.
    
- ✅ Built a custom model with two output heads: one for classification and one for regression.
    
- ✅ Instantiated and summarized the model.
    
- ✅ Ran a test batch to verify the model outputs correctly shaped predictions.
    

Next step: **Training the model** with two loss functions — one for classification (binary crossentropy) and one for bounding box regression (mean squared error).