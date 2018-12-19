
# coding: utf-8

# # HMM implementation for genre sequence prediction

# ## Table of contents
# 
# ### 1. [Introduction](#introduction)
# ### 2. [Dataset](#dataset)
# ### 3. [Dataset analysis](#dataset_analysis)
# ### 4. [Feature extraction](#feature_extraction)
# ### 5. [Model details](#model_details)
# ### 6. [Model implementation](#model_implementation)
# ### 7. [Model training](#model_training)
# ### 8. [Model testing](#model_testing)
# ### 9. [Results](#results)

# -----
# -----

# # 1. Introduction<a id='introduction'></a>

# Hidden Markov Model (HMM) is a widely used method for sequence prediction. In this project, we present a method for generating track predictions for users of music applications where the listening history of the users in readily available.

# The dataset is collected using LastFM'S Python API : pylast

# The listening history of a given user is collected using the API. Previous 10000 listened tracks are used to train and test the model.

# ----
# ----

# # 2. Dataset<a id='dataset'></a>

# ### Setup filesystem paths and read in the dataset from the directory

# In[9]:


# Setup the filesystem path to the data file

# System library to be used for path management
from os import path

# Print the home directory path
user_home_dir = str(path.expanduser('~'))
print('Home directory for the current user : ', user_home_dir)


# In[10]:


# Setup the path to the data file for the user
user_file_path = path.join(user_home_dir,
                                  'Documents\\GitHub\\comp_4710\\data\\user_1.csv')

print('Path to the data file currently being used : ', user_file_path)


# In[11]:


# Extract the data from the csv file and construct the dataframe
import pandas as pd

user_data_df = pd.read_csv(user_file_path, engine='python', encoding='utf-8 sig')

print('Printing the first 5 columns of the input dataset :')
print(user_data_df.head(5))


# In[12]:


# Extract the feature relevant for model implementation
user_sequence = user_data_df['genre_sequence'].values
print('Printing the sequence of the top tag for each of the songs listened to by the user :')
print(user_sequence)


# ----
# ----

# # 3. Dataset analysis<a id='dataset_analysis'></a>

# ## Extract all of the unique genres listened to by the user in the last 10000 tracks

# In[13]:


# Extract the unique genres in the sequence with their corresponding frequencies
import numpy as np

user_unique_tags, user_unique_counts = np.unique(user_sequence, return_counts=True)

print('Printing the list of all the unique tags for songs listened to by the user in the last valid 10000 tracks.')
print(user_unique_tags)

print('Printing the count of all the unique tags for songs listened to by the user in the last valid 10000 tracks.')
print(user_unique_counts)


# ## Determine a minimum suppost (minsup) to prune tags which do not occur frequently in the sequence

# We have to test the model performance for various minsup on the tag frequency in the sequence.

# Set the cap to be 10000/100 = 100 and prune all the genres which have a frequency lower than the minsup.

# In[34]:


# Prune the genres which are not frequent according to the threshold
minsup = 100

# Construct the array of genres frequent in the sequence
user_frequent_tags = []
user_frequent_counts = []

for i in range(len(user_unique_tags)):
    if(user_unique_counts[i] >= minsup):
        user_frequent_tags.append(user_unique_tags[i])
        user_frequent_counts.append(user_unique_counts[i])
        
# Print the pruned set of genres and their corresponding frequency
print('Printing the set of frequent tags.')
print(user_frequent_tags)

print('Printing the corresponding count of the tags in the set of frequent tags.')
print(user_frequent_counts)


# ## Create a mapping for the tags in the frequent set

# In[35]:


# Construct the map to be applied to the whole sequence
tag_map = {}
tag_inverse_map = {}

# Iterate over the frequent tags and construct a map
for i in range(len(user_frequent_tags)):
    tag_map[user_frequent_tags[i]] = i
    tag_inverse_map[i] = user_frequent_tags[i]

print('Printing the map to be applied to the user sequence :')
print(tag_map)


# ## Remove the infrequent tags from the user sequence

# In[36]:


# Prune the infrquent tags from the user sequence
user_sequence_pruned = []

for tag in user_sequence:
    if tag in user_frequent_tags:
        user_sequence_pruned.append(tag)
        
print('Printing the pruned sequence of tags for songs listened to by the user.')
print(user_sequence_pruned)


# ## Plot the frequencies for the frequent tags in the user sequence

# In[66]:


# Initialize the sequence array to hold the mapped values
user_sequence_mapped = []
mapper = lambda tag: tag_map[tag]

for tag in user_sequence_pruned:
    user_sequence_mapped.append(mapper(tag))
    
print('Printing the user sequence mapped to a set of integers.')
print(user_sequence_mapped)


# In[67]:


# Plot the frequencies of the frequent tags
import matplotlib.pyplot as plt

# Initialize the plot features
fig = plt.gcf()
fig.set_size_inches(28, 10.5, forward=True)
plt.style.use('seaborn-darkgrid')

# Plot the bar graph of the histogram with the frequencies per tag
plt.bar(user_frequent_tags, user_frequent_counts)
plt.title('Frequencies of the frequent tags with minsup=100 for a given user')
plt.xlabel('Frequent Tags')
plt.ylabel('Frequency in the sequence')
plt.savefig('tag_frequencies.png')
plt.show()


# ----
# ----

# # 4. Feature Extraction<a id='feature_extraction'></a>

# We have already extracted the desired features i.e. the tags from the track info where the track is in the set of last 10000 tracks listened to by the user. 

# The sequence has also been pruned to remove the infrequent tags from the original sequence.

# ---
# ---

# # 5. Model details<a id='model_details'></a>

# We will implement a Multinomial Hidden Markov Model (Multinomial HMM) for fitting the training dataset and generating
# predictions.

# ---
# ---

# # 6. Model implementation<a id='model_implementation'></a>

# ## Implement the Multinomial HMM model using :
# ## hmmlearn - https://hmmlearn.readthedocs.io/en/stable/

# In[68]:


# Import the install module for initializing and implementing the Multinomial HMM
from hmmlearn import hmm


# In[69]:


# Initialize the model
multinomial_hmm = hmm.MultinomialHMM(n_components=10)


# In[70]:


# Prepare the input dataset to be made suitable for training the model
user_sequence_mapped = np.array(user_sequence_mapped)

print('Printing the mapped sequence array.')
print(user_sequence_mapped)

print('Printing the shape of the mapped sequence array.')
print(user_sequence_mapped.shape)


# In[71]:


# Reshape the input data sequence
data_shape = (len(user_sequence_mapped), 1)

# Make the dataset divisible by the desired sequence lengths
user_sequence_mapped = user_sequence_mapped.reshape(data_shape)

# Print the reshaped dataset
print('Printing the reshaped input sequence.')
print(user_sequence_mapped)

print('Printing the shape of the reshaped input sequence.')
print(user_sequence_mapped.shape)


# In[72]:


multinomial_hmm.fit(user_sequence_mapped)

