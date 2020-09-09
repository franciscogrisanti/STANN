"""
   ___                  _                
  / _/______ ____  ____(_)__ _______     
 / _/ __/ _ `/ _ \/ __/ (_-</ __/ _ \    
/_//_/  \_,_/_//_/\__/_/___/\__/\___/    
  ___ _____(_)__ ___ ____  / /_(_)       
 / _ `/ __/ (_-</ _ `/ _ \/ __/ /        
 \_, /_/ /_/___/\_,_/_//_/\__/_/         
/___/

Samee Lab @ Baylor College Of Medicine
francisco.grisanticanozo@bcm.edu
Date: 12/2020

"""

# USAGE
# python train.py --model STANN --data seqfish --output AE_seqfish_1


from numpy.random import seed
seed(123)

#from tensorflow import set_random_seed
#set_random_seed(234)

# import the necessary packages
import matplotlib.pyplot as plt
import logging
import argparse
import os
import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import scanpy as sc
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# local functions
from STANN.models import STANN
from autoencoder_models.utils import cross_validate, plot_confusion_matrix


#logging.getLogger("tensorflow").setLevel(logging.CRITICAL)


################construct the argument parser and parse the arguments###################
ap = argparse.ArgumentParser()

ap.add_argument(
    "-m",
    "--model",
    type=str,
    default="autoencoder",
    choices=["autoencoder", "class"],
    help="type of model architecture",
)

ap.add_argument(
    "-d",
    "--data",
    type=str,
    required=True,
    choices=["scrna", "seqfish","other"],
    help="type of data scrna or seqfish"
)

ap.add_argument(
    "-dp",
    "--data_path",
    type=str,
    required=False,
    help="path of dataset"
)

ap.add_argument(
    "-o",
    "--output",
    type=str, 
    default="AE_temp",
    help="model filename"
)

ap.add_argument(
    "-cv",
    "--cross_validate",
    type=str,
    default=False,
    choices=[True, False],
    help="cross-validation"
)


args = vars(ap.parse_args())

################create np array vectors###################
def create_data(adata,cell_type_id=None,model=None,pc=False):  
    """
    create your data
    returns -> X,Y
    """
    if pc is False:
        if model == "autoencoder":
        # split into input (X) and output (y) variables
            X = adata.to_df().values.astype(float)
            return X     
        else:
            X = adata.to_df().values.astype(float)
            Y = pd.DataFrame(adata.obs[cell_type_id].copy()).values
            return X,Y
    else:
        if model == "autoencoder":
        # split into input (X) and output (y) variables
            X = adata_all.obsm['X_pca'].astype(float)
            return X     
        else:
            X = adata_all.obsm['X_pca'].astype(float)
            Y = pd.DataFrame(adata.obs[cell_type_id].copy()).values
            return X,Y

################encoding of target variable###################
def encode_data(Y):
    """
    encode target vector
    returns -> encoded_Y
    """
    #encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = keras.utils.to_categorical(encoded_Y)
    return dummy_y , encoder 

################create model###################
def create_model():  
    """
    initiates and creates keras sequential model
    returns -> model
    """
    model = autoencoder_v2(n_input=X.shape[1],
                        inner_neurons=30,
                        outer_neurons=75,
                        activation_choice = 'relu'
                      )

    return model


################LOAD DATA###################
print("[INFO] loading dataset...")

# check to see which data
if args["data"] == "scrna":
    print("[INFO] using scRNA data...")
    adata = sc.read_h5ad("data/scRNA.h5ad")

elif args["data"] == "seqfish":
    print("[INFO] using seqfish data...")
    adata = sc.read_h5ad("data/seqfish.h5ad")

elif args["data"] == "other":

    print(f"[INFO] using "+ args["data_path"] +"...")
    adata = sc.read_h5ad(args["data_path"])

################PROCESSING AND DATA CREATION###################
X = create_data(adata,model=args['model'])

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

################BUILD MODEL###################

#check to see if we are using a Keras Sequential model
if args["model"] == "autoencoder":
    # instantiate a Keras Sequential model
    print("[INFO] using " + str(args["model"]) + " model...")
    model = create_model()


################ EARLY STOPPING###################
es = EarlyStopping(monitor="loss", mode="min", verbose=1,patience=5)


################ COMPILE & OPTIMIZER ###################
print("[INFO] compile network...")

opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)

model.compile(
  loss="mean_squared_error", 
  optimizer='adam',
	metrics=["accuracy"])


################ TRAINING NETWORK###################
print("[INFO] training network...")

## CROSS VALIDATION ##

if args["cross_validate"] == True:
    # instantiate a Keras Sequential model
    print("[INFO] using sequential model...")
    cv_scores, model_history = cross_validate(n_fols=10,model=model,X=X,Y=X)
    # save to json:
    print("[INFO] saving history .json ...")
    hist_df = pd.DataFrame(cv_scores) 
    hist_json_file = 'history.json'
    with open(hist_json_file, mode='w') as f:
      hist_df.to_json("../autoencoder/output/"+args["output"]+"_"+hist_json_file)

## NORMAL ##

else:
    # instantiate a Keras Sequential model
    X_train, X_test, x_train, x_test = train_test_split(X, X, train_size=0.80, random_state=10)
    history = model.fit(X_train, X_train, validation_data=(X_test,X_test), shuffle = True,batch_size=300, epochs=30, callbacks=[es])
    
    # save to json:
    print("[INFO] saving history .json ...")
    hist_df = pd.DataFrame(history.history)
    hist_json_file = 'history.json'
    with open(hist_json_file, mode='w') as f:
      hist_df.to_json("../autoencoder/output/"+args["output"]+"_"+hist_json_file)


################ SAVE MODEL ###################

print("[INFO] saving .h5 model ...")
model.save("../STANN/output/"+args["output"]+".h5")
    


################ PLOT MODEL ###################
# plot training history
#print("\n" + "Training Loss: ", history.history['loss'][-1])
plt.style.use("ggplot")
plt.figure(figsize=(10,10))
plt.plot(history.history['loss'], label='train')
#plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig("../autoencoder/output/"+args["output"]+"_plot.png")

