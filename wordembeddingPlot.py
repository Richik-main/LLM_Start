#%%
import os
import requests
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#%%

# Define the URL and local paths for the GloVe model
url = 'http://nlp.stanford.edu/data/glove.6B.zip'
local_zip_path = 'glove.6B.zip'
glove_dir = 'glove.6B'

# Download the GloVe model
if not os.path.exists(local_zip_path):
    print("Downloading the GloVe model...")
    response = requests.get(url, stream=True)
    with open(local_zip_path, 'wb') as f:
        f.write(response.content)
    print("Download completed.")

# Extract the GloVe model
if not os.path.exists(glove_dir):
    print("Extracting the GloVe model...")
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(glove_dir)
    print("Extraction completed.")

# Load the GloVe model
def load_glove_model(glove_file):
    print("Loading the GloVe model...")
    model = {}
    with open(glove_file, 'r', encoding="utf-8") as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            model[word] = embedding
    print("Model loaded.")
    return model

# Specify the GloVe file path (50-dimensional version)
glove_file_path = os.path.join(glove_dir, 'glove.6B.50d.txt')
glove_model = load_glove_model(glove_file_path)
# %%
# Select words to plot
words = ['king', 'queen', 'man', 'woman', 'apple', 'orange', 'car', 'bicycle']
word_vectors = [glove_model[word] for word in words]

# Reduce dimensions with PCA
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(word_vectors)

# Plot the embeddings
plt.figure(figsize=(10, 7))
for word, (x, y) in zip(words, reduced_vectors):
    plt.scatter(x, y)
    plt.text(x+0.01, y+0.01, word)

plt.title('Word Embeddings Visualization using PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()
# %%
from sklearn.manifold import TSNE
words = ['king', 'queen', 'man', 'woman', 'apple', 'orange', 'car', 'bicycle']
word_vectors = [glove_model[word] for word in words]
word_vectors = np.array(word_vectors)
# Reduce dimensions with t-SNE
tsne = TSNE(n_components=2, random_state=0, perplexity=3)
reduced_vectors = tsne.fit_transform(word_vectors)

# Plot the embeddings
plt.figure(figsize=(10, 7))
for word, (x, y) in zip(words, reduced_vectors):
    plt.scatter(x, y)
    plt.text(x+0.01, y+0.01, word)

plt.title('Word Embeddings Visualization using t-SNE')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()
# %%
from mpl_toolkits.mplot3d import Axes3D


words = ['king', 'queen', 'man', 'woman', 'apple', 'orange', 'car', 'bicycle']
word_vectors = [glove_model[word] for word in words]

word_vectors = np.array(word_vectors)

# Reduce dimensions with t-SNE
tsne = TSNE(n_components=3, random_state=0, perplexity=3)  
reduced_vectors = tsne.fit_transform(word_vectors)

# Plot the embeddings in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

for word, (x, y, z) in zip(words, reduced_vectors):
    ax.scatter(x, y, z)
    ax.text(x, y, z, word)

ax.set_title('Word Embeddings Visualization using t-SNE (3D)')
ax.set_xlabel('t-SNE Component 1')
ax.set_ylabel('t-SNE Component 2')
ax.set_zlabel('t-SNE Component 3')
plt.show()
# %%
