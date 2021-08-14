# Neural Style Transfer


```python
import tensorflow as tf 

from tensorflow.python.keras.applications.vgg19 import VGG19
```

# Importing Model


```python
model = VGG19(
    include_top = False,
    weights = 'imagenet'
)

model.trainable = False
model.summary()
```


```python
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

from tensorflow.python.keras.models import Model

from tensorflow.python.keras.applications.vgg19 import preprocess_input
```


```python
import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
```

# Image Functions


```python
def load_and_process_image(image_path):
    img = load_img(image_path)
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis = 0)
    return img


def deprocess(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.799
    x[:, :, 2] += 123.456
    
    x = x[:, :,::-1]
    
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def display_image(image):
    if len(image.shape)==4:
        img = np.squeeze(image, axis = 0)
        
    img = deprocess(img)
    
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    
    return

```


```python
display_image(load_and_process_image('Style.jpg')) #Style Image
```


```python
display_image(load_and_process_image('Content.jpg')) #Content Image
```


```python
content_layer = 'block5_conv2'

style_layers = [
    'block1_conv1',
    'block3_conv1',
    'block5_conv1'
]

content_model = Model(
    inputs = model.input,
    outputs = model.get_layer(content_layer).output
)

style_models = [Model(inputs = model.input,
                    outputs = model.get_layer(layer).output) for layer in style_layers]
```

# Content Cost


```python
def content_cost(content, generated):
    a_C = content_model(content)
    a_G = content_model(generated)
    cost = tf.reduce_mean(tf.square(a_C - a_G))
    return cost 
```

# Gram Matrix


```python
def gram_matrix(A):
    n_C = int(A.shape[-1])
    a = tf.reshape(A, [-1, n_C])
    n = tf.shape(a)[0]
    G = tf.matmul(a, a, transpose_a = True)
    return G/tf.cast(n, tf.float32)
```


```python
lam = 1./len(style_models)

def style_cost(style, generated):
    J_style = 0
    
    for style_model in style_models:
        a_S = style_model(style)
        a_G = style_model(generated)
        GS = gram_matrix(a_S)
        GG = gram_matrix(a_G)
        
        current_cost = tf.reduce_mean(tf.square(GS - GG))
        J_style += current_cost * lam
    
    return J_style
```


```python
import time
generated_images = [] 

def training_loop(content_path, style_path, iterations = 50,
                 alpha = 10., beta = 20.):
    
    content = load_and_process_image(content_path)
    
    style = load_and_process_image(style_path)
    
    generated = tf.Variable(content, dtype = tf.float32)
    
    opt = tf.optimizers.Adam(learning_rate = 5.)
    
    best_cost = 1e12 + 0.1
    best_image = None
    
    start_time = time.time()
    
    for i in range(iterations):
        with tf.GradientTape() as tape:
            J_content = content_cost(content, generated)
            J_style = style_cost(style, generated)
            J_total = (alpha * J_content) + (beta * J_style)
            
        grads = tape.gradient(J_total, generated)
        opt.apply_gradients([(grads, generated)])
        
        if J_total < best_cost:
            best_cost = J_total
            best_image = generated.numpy()
            
        print('Cost at {}: {}. Time elapsed: {}'.format(i, J_total, time.time() - start_time))
        generated_images.append(generated.numpy())
        
    return best_image
```


```python
best_image = training_loop('Content.jpg','Style.jpg')
```


```python
display_image(best_image) # Generated Image
```


```python
plt.figure(figsize = (10, 10))

for i in range(50):
    plt.subplot(10, 5, i+1)
    display_image(generated_images[i])
    
plt.show()
```
