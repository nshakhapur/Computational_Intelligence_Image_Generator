
module_path = 'https://tfhub.dev/deepmind/biggan-deep-128/1'  


import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

import os
import io
import IPython.display
import numpy as np
import PIL.Image
from scipy.stats import truncnorm
import tensorflow_hub as hub


# In[3]:


tf.compat.v1.reset_default_graph()
print('Loading BigGAN module from:', module_path)
module = hub.Module(module_path)
inputs = {k: tf.compat.v1.placeholder(v.dtype, v.get_shape().as_list(), k)
          for k, v in module.get_input_info_dict().items()}
output = module(inputs)

print()
print('Inputs:\n', '\n'.join(
    '  {}: {}'.format(*kv) for kv in inputs.items()))
print()
print('Output:', output)


# In[4]:


input_z = inputs['z']
input_y = inputs['y']
input_trunc = inputs['truncation']

dim_z = input_z.shape.as_list()[1]
vocab_size = input_y.shape.as_list()[1]

def truncated_z_sample(batch_size, truncation=1., seed=None):
  state = None if seed is None else np.random.RandomState(seed)
  values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state)
  return truncation * values

def one_hot(index, vocab_size=vocab_size):
  index = np.asarray(index)
  if len(index.shape) == 0:
    index = np.asarray([index])
  assert len(index.shape) == 1
  num = index.shape[0]
  output = np.zeros((num, vocab_size), dtype=np.float32)
  output[np.arange(num), index] = 1
  return output

def one_hot_if_needed(label, vocab_size=vocab_size):
  label = np.asarray(label)
  if len(label.shape) <= 1:
    label = one_hot(label, vocab_size)
  assert len(label.shape) == 2
  return label

def sample(sess, noise, label, truncation=1., batch_size=8,
           vocab_size=vocab_size):
  noise = np.asarray(noise)
  label = np.asarray(label)
  num = noise.shape[0]
  if len(label.shape) == 0:
    label = np.asarray([label] * num)
  if label.shape[0] != num:
    raise ValueError('Got # noise samples ({}) != # label samples ({})'
                     .format(noise.shape[0], label.shape[0]))
  label = one_hot_if_needed(label, vocab_size)
  ims = []
  for batch_start in range(0, num, batch_size):
    s = slice(batch_start, min(num, batch_start + batch_size))
    feed_dict = {input_z: noise[s], input_y: label[s], input_trunc: truncation}
    ims.append(sess.run(output, feed_dict=feed_dict))
  ims = np.concatenate(ims, axis=0)
  assert ims.shape[0] == num
  ims = np.clip(((ims + 1) / 2.0) * 256, 0, 255)
  ims = np.uint8(ims)
  return ims

def interpolate(A, B, num_interps):
  if A.shape != B.shape:
    raise ValueError('A and B must have the same shape to interpolate.')
  alphas = np.linspace(0, 1, num_interps)
  return np.array([(1-a)*A + a*B for a in alphas])

def imgrid(imarray, cols=5, pad=1):
  if imarray.dtype != np.uint8:
    raise ValueError('imgrid input imarray must be uint8')
  pad = int(pad)
  assert pad >= 0
  cols = int(cols)
  assert cols >= 1
  N, H, W, C = imarray.shape
  rows = N // cols + int(N % cols != 0)
  batch_pad = rows * cols - N
  assert batch_pad >= 0
  post_pad = [batch_pad, pad, pad, 0]
  pad_arg = [[0, p] for p in post_pad]
  imarray = np.pad(imarray, pad_arg, 'constant', constant_values=255)
  H += pad
  W += pad
  grid = (imarray
          .reshape(rows, cols, H, W, C)
          .transpose(0, 2, 1, 3, 4)
          .reshape(rows*H, cols*W, C))
  if pad:
    grid = grid[:-pad, :-pad]
  return grid

def imshow(a, format='png', jpeg_fallback=True):
  a = np.asarray(a, dtype=np.uint8)
  data = io.BytesIO()
  PIL.Image.fromarray(a).save(data, format)
  im_data = data.getvalue()
  try:
    disp = IPython.display.display(IPython.display.Image(im_data))
  except IOError:
    if jpeg_fallback and format != 'jpeg':
      print(('Warning: image was too large to display in format "{}"; '
             'trying jpeg instead.').format(format))
      return imshow(a, format='jpeg')
    else:
      raise
  return disp




import streamlit as st
import random

# Define subcategories and their associated animals with category numbers
bird_subcategories = {
    "Flightless Birds": [("ostrich", 9)],
    "Songbirds": [("brambling", 10), ("goldfinch", 11), ("house_finch", 12), ("junco", 13), ("indigo_bunting", 14), ("robin", 15), ("bulbul", 16)],
    "Corvids": [("jay", 17), ("magpie", 18)],
    "Chickadees": [("chickadee", 19)],
    "Water Birds": [("water_ouzel", 20), ("kite", 21), ("bald_eagle", 22), ("vulture", 23), ("white_stork", 128), ("black_stork", 129),
                    ("spoonbill", 130), ("flamingo", 131), ("little_blue_heron", 132), ("American_egret", 133), ("bittern", 134),
                    ("crane", 135), ("limpkin", 136), ("European_gallinule", 137), ("American_coot", 138), ("bustard", 139),
                    ("ruddy_turnstone", 140), ("red-backed_sandpiper", 141), ("redshank", 142), ("dowitcher", 143),
                    ("oystercatcher", 144), ("pelican", 145), ("king_penguin", 146), ("albatross", 147)],
    "Exotic Birds": [("bee_eater", 148), ("hornbill", 149), ("hummingbird", 150), ("jacamar", 151), ("toucan", 152), ("red-breasted_merganser", 153)]
}

mammal_subcategories = {
    "Aquatic Mammals": [("dugong", 149), ("sea_lion", 150)],
    "Marsupials": [("wallaby", 104), ("koala", 105), ("wombat", 106)],
    "Marine Mammals": [("grey_whale", 147), ("killer_whale", 148)],
    "Canines": [("dingo", 273), ("dhole", 274), ("African_hunting_dog", 275)],
    "Felids": [("tiger", 291), ("lion", 292), ("cheetah", 293), ("cougar", 286), ("lynx", 287), ("leopard", 288), ("snow_leopard", 289), ("jaguar", 290)],
    "Ursids": [("brown_bear", 294), ("American_black_bear", 295), ("ice_bear", 296)],
    "Big Cats": [("tiger", 291), ("lion", 292), ("cheetah", 293), ("cougar", 286), ("lynx", 287), ("leopard", 288), ("snow_leopard", 289), ("jaguar", 290)],
    "Canids": [("dingo", 273), ("dhole", 274), ("African_hunting_dog", 275)],
    "Foxes": [("red_fox", 277), ("kit_fox", 278), ("Arctic_fox", 279), ("grey_fox", 280)]
}

# Streamlit app title
st.title("AnimalGen")
st.subheader("A Generative AI Project to generate images of the animals")

# Choose between Birds and Mammals
animal_type = st.selectbox("Choose Animal Type:", ["Birds", "Mammals"])

# Choose subcategory based on selected animal type
if animal_type == "Birds":
    subcategory = st.selectbox("Choose Bird Subcategory:", list(bird_subcategories.keys()))
    animals = bird_subcategories[subcategory]
else:
    subcategory = st.selectbox("Choose Mammal Subcategory:", list(mammal_subcategories.keys()))
    animals = mammal_subcategories[subcategory]

# Create an empty variable to store the category number
selected_category_number = None

# Choose an animal from the subcategory
selected_animal = st.radio("Select an Animal:", [animal[0] for animal in animals])
selected_category_number = [animal[1] for animal in animals if animal[0] == selected_animal][0]






initializer = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session()
sess.run(initializer)


# In[6]:

if st.button("Generate Image", key="generate_image_button"):
 num_samples = 10
 truncation = 0.4
 noise_seed = 0
 category = str(selected_category_number)

 z = truncated_z_sample(num_samples, truncation, noise_seed)
 y = int(category.split(')')[0])

 ims = sample(sess, z, y, truncation=truncation)

 for i, img in enumerate(ims):
        st.image(img, caption=f"Generated Image {i}", use_column_width=True); break
