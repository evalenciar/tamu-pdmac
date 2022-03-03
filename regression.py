# -*- coding: utf-8 -*-
"""
Data Input
02/10/2022
Updates:
    - Convolutional Neural Network (CNN)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# All scikit-learn related modules
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm
# All TensorFlow related modules
import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow.keras import layers, models
# import pathlib
from openpyxl import load_workbook
import datetime
import random
import time
import os
import sys

# Will need to have Input Data as 3-dimensional ILI layers
# - var1: number of ILI data files
# - var2: dimension in the longitudinal (along pipe axis)
# - var3: dimension in the lateral (along pipe circumference)
# To simplify the preprocessing, will use square resolution, meaning that the
# unit of length in var2 should be equal to that of var3. This way the length
# is not warped in any direction (more resolution in one direction). However,
# this does not mean that values var2 and var3 need to be equal. The resultant
# image should be like a rectangle.

# =============================================================================
# FUNCTIONS
# =============================================================================

def load_training_data(training_data_path):
    # Load Training Data
    image_data_path = []
    label_data_path = []

    for subdir, dirs, files in os.walk(training_data_path):
        for file in files:
            if 'radius' in file:
                image_data_path.append(os.path.join(subdir, file))
            if 'SCF' in file:
                label_data_path.append(os.path.join(subdir, file))
                
    # Load ALL of the data .npy files
    if len(image_data_path) != len(label_data_path):
        sys.exit()

    for i in range(len(image_data_path)):
        if i == 0: 
            image_data = np.load(image_data_path[i])
            label_data = np.load(label_data_path[i])
            continue
        
        image_data = np.concatenate((image_data, np.load(image_data_path[i])), axis=0)
        label_data = np.concatenate((label_data, np.load(label_data_path[i])), axis=0)
        
    return image_data, label_data

def regressor_model(shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(shape[0], shape[1], 1)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(256, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    
    # Add dense layers on top
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1))
    
    # Compile and train model
    model.compile(optimizer='RMSprop', loss='mae', metrics=['mse'])
    
    return model

def plot_image_prediction(j, image, label_true, label_predicted):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, cmap=plt.cm.RdYlGn)
    error_perc = (label_true - label_predicted)/label_true * 100
    plt.xlabel("#%d | %.3f (%.3f) | %.0f%%" % (j, label_predicted, label_true, error_perc))

def metrics(y_true, y_pred):
    # R^2 / Adjusted R^2
    # Measures how much variability in the dependent variable can be explained by the model.
    # It is the square of the Correlation Coefficient, R.
    # Relative measure of how well the model fits dependent variables.
    r2 = skm.r2_score(y_true, y_pred)
    
    # Mean Square Error (MSE) / Root Mean Square Error (RMSE)
    # Absolute measure of the goodness for the fit. Sum of square of error.
    # Gives larger penalization to big prediction error by squaring it.
    mse = skm.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error (MAE)
    # Sum of the absolute value of error. More direct representation of sum of error terms. 
    # Treats all errors the same. 
    mae = skm.mean_absolute_error(y_true, y_pred)
    
    # Max Error
    max_err = skm.max_error(y_true, y_pred)
    
    return r2, mse, rmse, mae, max_err

def metrics_data_print():
    # Metrics Values
    # Metrics Values in list organization
    metrics_data = [str(datetime.datetime.now()),   # Time stamp
                    wbMs_row - 1,                   # Run number
                    str(Conv2D_shapes),
                    str(Conv2D_activations),
                    str(Dense_shapes),
                    str(Dense_activations),
                    compile_optimizer,
                    compile_loss,
                    compile_metrics,
                    label_data.shape[0],
                    labels_train.shape[0],
                    training_time_stop - training_time_start,   # Training time
                    labels_test.shape[0],
                    test_val_loss,
                    test_val_metric,
                    r2,
                    mse,
                    rmse,
                    mae,
                    max_error,
                    max_error_perc,
                    time.time() - time_iteration]       # Total time
    return metrics_data

# =============================================================================
# PARAMETERS
# =============================================================================

time_start = time.time()

# Input Data dimensions
axial_len = 200
circ_len = 200
shape = [axial_len, circ_len]

metrics_path = 'metrics/'
metrics_name = 'metrics_regression.xlsx'
training_data_path = 'training_data'

# =============================================================================
# SCRIPT ITERATIONS
# =============================================================================

run_inf = True
while run_inf:
    
    # Parameter Options
    Conv2D_shapes = [32,64,128,256]
    Conv2D_activations = ['relu','relu','relu','relu']
    Dense_shapes = [100,10,1]
    Dense_activations = ['relu','relu']
    compile_optimizer = 'RMSprop'
    compile_loss = 'mae'
    compile_metrics = 'mse'
    
    # Current Time
    test_iteration = 1
    time_iteration = time.time()
    time_current = time.time() - time_start
    time_m, time_s = divmod(time_current, 60)
    time_h, time_m = divmod(time_m, 60)
    print('========== START ==========')
    print('%.0f:%.0f:%.0f | Program began execution.' % (time_h, time_m, time_s))
    
    # Load All Data
    image_data, label_data = load_training_data(training_data_path)
    
    # Split the data into Training and Test sets
    images_train, images_test, labels_train, labels_test = train_test_split(image_data, label_data, test_size=0.2)
    
    # Test 9 random images: Generate a random set of test images and labels
    rand_list_test = random.sample(range(images_train.shape[0]), 9)
    plt.figure(figsize=(10,10))
    for i, j in enumerate(rand_list_test):
        plt.subplot(3,3,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images_train[j], cmap=plt.cm.RdYlGn)
        plt.xlabel(round(labels_train[j],3))
        plt.colorbar()
    plt.show()
    
    model = regressor_model(shape)  # Create the regression ML model
    model.summary()                 # Print the current model information
    
    training_time_start = time.time()
    history = model.fit(images_train, labels_train, epochs=50,
                        validation_data=(images_test, labels_test))
    training_time_stop = time.time()
    
    # Evaluate the model: Metrics
    plt.figure(figsize=(8,6))
    plt.subplot(2,1,1)
    plt.plot(history.history['mse'], label='MSE')
    plt.plot(history.history['val_mse'], label='Val. MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    
    # Evaluate the model: Losses
    plt.subplot(2,1,2)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Val. Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    
    test_val_loss, test_val_metric = model.evaluate(images_test, labels_test, verbose=2)
    print('%.0f:%.0f:%.0f | Run %03d | Train Val Loss = %.3f | Train Val Metric = %.3f' % (time_h, time_m, time_s, test_iteration, test_val_loss, test_val_metric))
    
    # =============================================================================
    # VALIDATION
    # =============================================================================
    
    # Make predictions
    predictions = model.predict(images_test)
    
    # Plot some predictions
    num_rows = 3
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(8,8))
    # Generate a random set of test images and labels
    rand_list = random.sample(range(predictions.shape[0]), num_images)
    for i, j in enumerate(rand_list):
        plt.subplot(num_rows, num_cols, i + 1)
        plot_image_prediction(j, images_test[j], labels_test[j], predictions[j][0])
    plt.tight_layout()
    plt.show()
    
    # Unity Plot for SCF Values
    unity_line = np.linspace(0,15,100)
    plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    plt.plot(unity_line, unity_line, '-k')
    plt.plot(labels_test, np.ravel(predictions), 'or')
    plt.xlabel('Truth Data (SCF)')
    plt.ylabel('Predicted Data (SCF)')
    plt.xlim([0,14])
    plt.ylim([0,14])
    # Unity Plot for Difference
    plt.subplot(1,2,2)
    plt.plot(unity_line, np.linspace(0,0,100), '-k')
    plt.plot(labels_test, np.ravel(predictions) - labels_test, 'or')
    plt.xlabel('Truth Data (SCF)')
    plt.ylabel('Residual Error (SCF)')
    plt.xlim([0,14])
    plt.ylim([-8,8])
    
    # Metrics
    r2, mse, rmse, mae, max_err = metrics(labels_test, predictions)
    max_error = skm.max_error(labels_test, predictions)
    max_error_perc = max((np.ravel(predictions) - labels_test)/labels_test)
        
    # =============================================================================
    # METRICS
    # =============================================================================
    
    # y_test = labels_test.copy()
    # y_pred = [np.argmax(i) for i in predictions]
    # y_pred_prob = predictions.copy()
    
    # Print the results to view
    # df_data = [metrics_roc_auc, metrics_kappa, metrics_mcc, metrics_log_loss]
    # df_labels = ['ROC AUC', 'Kappa', 'MCC', 'Log Loss']
    # df = pd.DataFrame(data=df_data, index=df_labels, columns=['Value'])
    # print(df)
    
    # Save the SCF value to an Excel sheet
    # Save the Metrics to the metrics.xlsx Workbook
    
    metrics_file = metrics_path + metrics_name
    wbM = load_workbook(metrics_file)
    wbM_sn = wbM.sheetnames
    wbMs = wbM[wbM_sn[0]]
    wbMs_row = wbMs.max_row
    
    metrics_labels = wbMs[2]
    metrics_labels = [cell.value for cell in metrics_labels]
    
    metrics_data = metrics_data_print()
    
    # Write the metric values to the Excel workbook
    for i, value in enumerate(metrics_data):
        wbMs.cell(row=wbMs_row+1, column=i+1, value=value)
    wbM.save(metrics_file)
    wbM.close()     # Close the Results Workbook
    
    # Total time
    time_current = time.time() - time_start
    time_m, time_s = divmod(time_current, 60)
    time_h, time_m = divmod(time_m, 60)
    print('%.0f:%.0f:%.0f | Overall total time.' % (time_h, time_m, time_s))
    print('=========== END ===========')