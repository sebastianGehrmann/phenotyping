
# Patient Phenotyping

This repository contains the code and data for the paper 
"Comparing deep learning and concept extraction based methods for patient phenotyping".


## Data 

In data/annotations.csv, you can find our annotations as well 
as unique identifiers for patient visits in MIMIC-III, namely 
the hospital admission ID, subject ID, and chart time. 
Due to HIPAA requirements, we cannot provide the text of the patients' 
discharge summary in this repository. With the information 
named above and access to MIMIC-III, it is easy to extract the text 
from the identifiers. 
We are in the process of submitting the annotations as a direct
add-on to MIMIC-III to physionet which will make the linking-step 
obsolete. If you experience difficulties with the data, please 
contact us and we are happy to help! 

In the following sections, we assume that annotations.csv is extended
by an additional column named "text" that contains the discharge summary.

## Code

Here is how to run the code for baselines and deep learning components. 

### Preprocessing

To run all the code on the same training and test splits, we provide preprocessing code in
preprocessing.py. We assume that you ran word2vec on the extracted texts first and saved 
the resulting vectors in a file named "w2v.txt". If you need assistance or want to use
our vectors, please contact us (as the file size is too large for this repository).

run it with the following command (with python 2.7): 

```
python preprocess.py data/annotations.csv w2v.txt 
```

This will create one file data.h5 and one file data-nobatch.h5 in your main directory. 
Use the batched file for a speedup in the lua code, and the non-batched file for the baselines. 




### Baselines

The code for baselines can be found in basic_models.py. It is compatible with both 
python 2 and 3. To run it, simply enter

```
python basic_models.py --data data-nobatch.h5 --ngram 5
```


### Convolutional Neural Net

We recommend that you have a GPU with a cuda installation. Otherwise, training might take 
a very long time! This code is based off of the following repository:
https://github.com/harvardnlp/sent-conv-torch 
You can find more information there!



#### Install Torch

From http://torch.ch/docs/getting-started.html :

```
git clone https://github.com/torch/distro.git
cd distro; bash install-deps;
./install.sh
```

#### Install Torch packages

    luarocks install hdf5


#### Running the code

Please run the code with the following command:

```
th main.lua [OPTIONS]
```

You can find all options documented in the file itself. 
Important ones are "-gpuid" to set your GPU as well as "-label_index" to define
which phenotype you want to detect. Here is the list that explains the indices:

1: cohort (is the patient frequent flier)
2: Obesity
3: Non Adherence
4: Developmental Delay Retardation
5: Advanced Heart Disease
6: Advanced Lung Disease
7: Schizophrenia and other Psychiatric Disorders
8: Alcohol Abuse
9: Other Substance Abuse
10: Chronic Pain Fibromyalgia
11: Chronic Neurological Dystrophies
12: Advanced Cancer
13: Depression
14: Dementia
