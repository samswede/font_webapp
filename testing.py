#%%

import matplotlib.pyplot as plt
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd # this module is useful to work with tabular data
import random # this module will be used to select random samples from a collection
import os # this module will be used just to create directories in the local filesystem

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from OLD_variational_autoencoder import *
from manager import *
from vector_database import *
from dimensionality_reduction import *

#%%
model_path='./models/big_vae_L9_E700.pt'
embeddings_path='./data/embeddings/cleaned_big_L9_E700.csv'

vae = VAEModel(model_path= model_path, 
               embeddings_path= embeddings_path, 
               latent_dims=9)

#%%
font_1_index = 300
font_2_index = 310
interpolation_fraction = 0.5

vae.generate_interpolated_image(font_1_index=font_1_index, font_2_index=font_2_index, interpolation_fraction=interpolation_fraction)

vae.create_interpolation_gif(font_1_index=font_1_index, font_2_index=font_2_index, gif_path=f'interpolation_font_{font_1_index}_to_{font_2_index}.gif')

# %%

def get_similar_fonts(chosen_font_label, distance_metric='euclidean'):
    
    # Translate indication name to index in indication diffusion profiles, to retrieve diffusion profile
    #chosen_font_label = graph_manager.mapping_indication_name_to_label[chosen_indication_name]
    chosen_font_index = dict_font_labels_to_indices[chosen_font_label]
    chosen_font_diffusion_profile = font_embeddings_array[chosen_font_index]

    #====================================
    # Querying Vector Database to return drug candidates
    #====================================
    num_recommendations = 10

    query = chosen_font_diffusion_profile

    font_candidates_indices = font_vector_db.nearest_neighbors(query, distance_metric, num_recommendations)

    font_candidates_labels = [dict_font_indices_to_labels[index] for index in font_candidates_indices]
    #drug_candidates_names = [graph_manager.mapping_drug_label_to_name[i] for i in font_candidates_labels]

    return font_candidates_labels # List
#%%

# load font embeddings as pandas df
df = pd.read_csv(embeddings_path)

# drop index column
df = df.drop(df.columns[0], axis=1)

# Convert the DataFrame to a numpy array
font_embeddings_array = df.values


#%%

#dict_font_labels_to_indices = load_data_dict(f'{data_path}dict_font_labels_to_indices')
dict_font_labels_to_indices = {i: i for i in range(font_embeddings_array.shape[0])}
dict_font_indices_to_labels = {v: k for k, v in dict_font_labels_to_indices.items()}

metrics = ['angular', 'euclidean', 'manhattan', 'hamming', 'dot']

font_vector_db = MultiMetricDatabase(dimensions=font_embeddings_array.shape[1], metrics= metrics, n_trees=30)

# Add all fonts to vector database
font_vector_db.add_vectors(font_embeddings_array, dict_font_labels_to_indices)

# %%

font_candidates = get_similar_fonts(chosen_font_label=0, distance_metric='euclidean')
list_of_font_candidates = [
    {"value": label, "name": label}
    for label in font_candidates
]

print(list_of_font_candidates)
# %%
