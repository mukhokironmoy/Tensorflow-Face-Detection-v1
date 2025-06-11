Here are the detailed and beginner-friendly structured notes for **Step 9: Defining Losses and Optimizer**, based on your provided code.

---

# 9. Defining Losses and Optimizer

To effectively train our custom face detection model, we need to define:

1. **An Optimizer** – to apply gradients and update weights.
    
2. **A Learning Rate Decay Schedule** – to reduce the learning rate over time.
    
3. **A Classification Loss Function** – for the presence/absence of a face.
    
4. **A Regression Loss Function** – for bounding box coordinate prediction.
    

---

## 9.1 Learning Rate Decay

We want the learning rate to **decay after each epoch**, so training slows down over time, helping the model converge better and avoid overfitting or gradient explosion.

### Code:

```python
batches_per_epoc = len(train)
lr_decay = (1./0.75 - 1)/batches_per_epoc
```

### Explanation:

- `batches_per_epoc`: This is the number of batches in one epoch. We calculate it using the length of the training dataset.
    
- `lr_decay`: This formula is derived from a common strategy where the learning rate is **reduced to 75% of its original value after each epoch**. The formula calculates the per-batch decay needed to achieve this.
    

---

## 9.2 Defining the Optimizer

We use the **Adam optimizer**, which is a widely used adaptive optimizer combining the benefits of RMSProp and SGD with momentum.

### Code:

```python
opt = tf.keras.optimizers.Adam(learning_rate = 0.0001, decay = lr_decay)
```

### Explanation:

- `learning_rate`: Set to `0.0001` initially.
    
- `decay`: Applies the decay value calculated above to reduce the learning rate over time.
    
- `opt`: This variable holds the configured optimizer that will be passed to our model later during compilation.
    

---

## 9.3 Localization Loss (Bounding Box Loss)

This is our **custom regression loss function** to evaluate how close the predicted bounding box is to the actual one.

### Code:

```python
def localization_loss(y_true, yhat):
    y_true = tf.squeeze(y_true, axis=1)

    delta_coord = tf.reduce_sum(tf.square(y_true[:,:2] - yhat[:,:2]))
                  
    h_true = y_true[:,3] - y_true[:,1] 
    w_true = y_true[:,2] - y_true[:,0] 

    h_pred = yhat[:,3] - yhat[:,1] 
    w_pred = yhat[:,2] - yhat[:,0] 
    
    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))
    
    return delta_coord + delta_size
```

### Explanation:

- `y_true`: The actual bounding box, shape `[batch, 1, 4]`, squeezed to `[batch, 4]`.
    
- `yhat`: The predicted bounding box, shape `[batch, 4]`.
    
- `delta_coord`: Squared error between predicted and true center points (x and y).
    
- `h_true`, `w_true`: True height and width.
    
- `h_pred`, `w_pred`: Predicted height and width.
    
- `delta_size`: Squared error between predicted and true width/height.
    
- The final loss is the **sum of coordinate loss and size loss**, giving a total localization loss.
    

This function is used to train the **bounding box regression head** of the model.

---

## 9.4 Classification Loss

We use **Binary Cross-Entropy Loss** for the classification head since the task is binary (face vs no face).

### Code:

```python
classloss = tf.keras.losses.BinaryCrossentropy()
```

### Explanation:

- Binary cross-entropy computes the log loss between true class (`0` or `1`) and the predicted probability (between 0 and 1 from sigmoid output).
    
- Suitable for single-label binary classification problems.
    

---

## 9.5 Assign Loss Functions to Model

Once both loss functions are defined, we can assign them:

```python
regressloss = localization_loss
```

Now we can test them on sample predictions:

---

## 9.6 Testing Loss Functions

We test the losses using sample predictions:

```python
localization_loss(y[1], coords)
# Output: <tf.Tensor: shape=(), dtype=float16, numpy=3.5390625>

classloss(y[0], classes)
# Output: <tf.Tensor: shape=(), dtype=float32, numpy=0.7886611223220825>

regressloss(y[1], coords)
# Output: <tf.Tensor: shape=(), dtype=float16, numpy=3.5390625>
```

### Explanation:

- The regression loss (localization) is quite large at the start (≈ 3.54) because the model hasn't learned anything yet.
    
- The classification loss (≈ 0.78) also reflects early stage predictions being far from accurate.
    
- These values will **decrease as training progresses**.
    

---

## ✅ Summary of Step 9

- ✅ Defined a **custom learning rate decay schedule**.
    
- ✅ Initialized the **Adam optimizer** with decay.
    
- ✅ Created a **custom bounding box regression loss** using coordinate and size differences.
    
- ✅ Assigned **Binary Cross-Entropy** for classification.
    
- ✅ Verified both loss functions using initial outputs.
    

Next up: compiling and training the model using these configurations!