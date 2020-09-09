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

#import the necessary packages
import logging
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import sys
import pandas as pd
import warnings
import sklearn
import seaborn as sb
import venn
import sklearn
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score
import pickle
import tqdm
import scanpy as sc
import anndata
import tensorflow as tf
import tensorflow.keras as keras
import scanpy as sc
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# local functions
from STANN.models import STANN
import STANN.utils as utils

logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

#Reproducibility
seed = 10
np.random.seed(seed)
tf.random.set_seed(seed)

################construct the argument parser and parse the arguments###################
ap = argparse.ArgumentParser()

ap.add_argument(
    "-m",
    "--model",
    type=str,
    default="STANN",
    choices=["autoencoder", "class"],
    help="type of model architecture",
)

ap.add_argument(
    "-dp",
    "--data_train",
    type=str,
    required=False,
    help="Path of training dataset"
)

ap.add_argument(
    "-dp",
    "--data_predict",
    type=str,
    required=False,
    help="Path of prediction dataset"
)

ap.add_argument(
    "-o",
    "--output",
    type=str, 
    default="./",
    help="Path to output"
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


################LOAD DATA###################

# check to see which data
print("[INFO] loading training data...")
adata_train = sc.read_h5ad("../data/scrna.h5ad")

print("[INFO] loading predict data...")
adata_predict = sc.read_h5ad("../data/seqfish.h5ad")



model = STANN(act_fun='tanh',
              first_dense=160,
              second_dense=145.0,
              learning_rate=0.01,input_dim=adata_train.X.shape[1],
              output_dim=len(adata_train.obs.celltype.unique()))

model.summary()


X_train, Y_train, X_predict = utils.organize_data(adata_train=adata_train,
                                            adata_predict=adata_predict)


X_train_scaled , scaler_train = utils.min_max(X=X_train)

X_predict_scaled , scaler_predict = utils.min_max(X=X_predict)

Y_train_dummy,Y_train_ohe,encoder = utils.label_encoder(Y_train=Y_train)

x_train, x_test, y_train, y_test = utils.train_test_split(X_train_scaled,
                                                    Y_train_ohe,
                                                    test_size=0.10, 
                                                    random_state=40)

class_weights = utils.get_class_weights(Y_train_ohe=y_train)

es = tf.keras.callbacks.EarlyStopping(monitor='accuracy', 
                                      mode='min', 
                                      verbose=1,
                                      patience=30)

history = model.fit(x_train, 
                    y_train, 
                    validation_data=(x_test, y_test),
                    epochs=30,
                    class_weight=class_weight,
                    callbacks=[es],verbose=0)

 utils.print_metrics(model=model,
                  x_train=x_train,
                  y_train=y_train,
                  x_test=x_test,
                  y_test=y_test)

predictions = utils.make_predictions(model=model,
                     X_predict=X_predict_scaled,
                     encoder=encoder,
                     adata_predict=adata_predict,
                     probabilities=False,
                     save=True
                    )
                    
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
    
################ SAVE METADATA RESULTS ###################

print("[INFO] saving .h5 model ...")
model.save("../STANN/output/"+args["output"]+".h5")
    