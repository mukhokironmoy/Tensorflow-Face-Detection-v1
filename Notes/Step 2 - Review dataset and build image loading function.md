# 2. Reviewing Dataset and Building the Image Loading Pipeline

## 2.1 Overview of Steps

To prepare our dataset for training a face detection model, we will:

1. **Review and load images into TensorFlow.**
    
2. **Partition unaugmented data.**
    
3. **Apply image augmentation using `albumentations`.**
    
4. **Build and run the augmentation pipeline.**
    
5. **Prepare labels and combine everything.**
    
6. **Final result**: A full data pipeline (train, test, validation splits) with augmented, reshaped, and pre-processed images ready for model input.
    

---

## 2.2 Importing Required Libraries

We begin by importing the libraries necessary for data loading and preprocessing:

- `tensorflow as tf`: For building the data pipeline and model.
    
- `json`: To load annotation data stored in JSON format.
    
- `numpy as np`: For numerical operations.
    
- `matplotlib.pyplot as plt`: To visualize the images.
    

```python
import tensorflow as tf
import json
import numpy as np
import matplotlib.pyplot as plt
```

Note: TensorFlow is used here early on to configure GPU memory usage.

---

## 2.3 GPU Memory Growth Limitation

TensorFlow, by default, can consume all available GPU memory, which can lead to out-of-memory errors. To avoid this:

```python
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

You can verify GPU availability with:

```python
tf.config.list_physical_devices('GPU')
```

This helps ensure your code can train models faster using GPU.

---

## 2.4 Loading Images into TensorFlow Data Pipeline

We load image file paths using:

```python
images = tf.data.Dataset.list_files('data/images/*.jpg', shuffle=False)
```

- `data/images/` is the folder where all `.jpg` images are stored.
    
- The `*.jpg` wildcard picks all image files.
    
- `shuffle=False` ensures the data is loaded in a fixed order.
    

You can inspect the dataset:

```python
images.as_numpy_iterator().next()
```

This should return the full file path of an image. If not, the path is incorrect and should be fixed.

---

## 2.5 Defining the Image Loading Function

To load and decode images into tensors, we define:

```python
def load_image(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    return img
```

- `tf.io.read_file`: Reads the file as raw bytes.
    
- `tf.io.decode_jpeg`: Converts raw bytes into an image tensor.
    

We map this function to our dataset using:

```python
images = images.map(load_image)
```

This maps every file path in `images` to the actual image data.

---

## 2.6 TensorFlow Data Pipeline Confirmation

You can check if `images` is a valid TensorFlow data pipeline:

```python
type(images)
```

It should return:

```
tensorflow.python.data.ops.dataset_ops.MapDataset
```

---

## 2.7 Visualizing Images with Matplotlib

To view batches of images:

### a. Batch the images

```python
image_generator = images.batch(4).as_numpy_iterator()
```

- Creates batches of 4 images.
    

### b. Fetch a batch and visualize

```python
plot_images = image_generator.next()
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(plot_images):
    ax[idx].imshow(img)
    ax[idx].axis('off')
plt.show()
```

Running this multiple times will show different batches. You can re-enable shuffling to get randomized outputs by setting:

```python
images = tf.data.Dataset.list_files('data/images/*.jpg', shuffle=True)
```

---
## 2.8 Moving the Labels with a Script

To move the corresponding label files to the right folders:

- A Python script is used to:
    
    1. Loop through the folder names: `train`, `test`, and `val`
        
    2. Look up matching image filenames
        
    3. Move the corresponding label JSON files from `data/labels/` into `data/<split>/labels/`
        

After running the script:

- `data/train/labels/` has 63 label files
    
- `data/test/labels/` has 14 label files
    
- `data/val/labels/` has 13 label files


## ✅ Summary of Completed Steps

- ✅ **Imported all necessary dependencies** (`tensorflow`, `json`, `numpy`, `matplotlib`).
    
- ✅ **Limited GPU memory growth** to prevent memory overflow.
    
- ✅ **Loaded file paths using TensorFlow’s `list_files()`**.
    
- ✅ **Created an image loading function** using TensorFlow I/O utilities.
    
- ✅ **Mapped image loading to all image paths** to create a TensorFlow dataset.
    
- ✅ **Batched and visualized image data** using `matplotlib`.
    

You now have a working TensorFlow data pipeline with image loading and visualization. Next up: **label loading, augmentation, and dataset partitioning.**