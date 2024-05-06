# © 2024. Triad National Security, LLC. All rights reserved.
"""
Use convolutional neural network (CNN) methods to detect
signals in infrasound beamforming results.

"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import warnings

from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout, MultiHeadAttention
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from riddl.utils import data_io

def define_callbacks() -> list:
    return [EarlyStopping(monitor='loss', patience=500, mode='min', min_delta=0.0),
            ModelCheckpoint('/tmp/checkpoint', monitor='accuracy', save_best_only=True, mode='max')]

def define_CNN(input_shape: tuple, label_cnt: int) -> Model:
    """
    Define the ConvNet architecture and compile the model.

    Args:
        input_shape (tuple):
        label_cnt (int):

    Returns:
        model: The compiled ML model
    """
    ##############
    # Build CNN
    # https://stackoverflow.com/questions/50701913/how-to-split-the-input-into-different-channels-in-keras
    # https://www.machinecurve.com/index.php/2020/11/24/one-hot-encoding-for-machine-learning-with-tensorflow-and-keras/
    #############

    inputs = Input(input_shape)

    # Block 1
    x = Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape)(inputs)
    x = BatchNormalization()(x)
    x = Conv1D(64, kernel_size=5, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(128, kernel_size=5, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.25)(x)

    x = MultiHeadAttention(num_heads=8, key_dim=128, dropout=0.25)(x, x)
    x = MultiHeadAttention(num_heads=8, key_dim=128, dropout=0.25)(x, x)

    # Block 2
    x = Conv1D(256, kernel_size=5, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.25)(x)

    x = MultiHeadAttention(num_heads=8, key_dim=256, dropout=0.25)(x, x)

    # Block 3
    x = Conv1D(256, kernel_size=5, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.25)(x)

    x = MultiHeadAttention(num_heads=8, key_dim=256, dropout=0.25)(x, x)

    # Decode
    x = Conv1D(256, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.25)(x)
    x = Conv1D(256, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.25)(x)
    x = Conv1D(512, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.25)(x)

    # Prediction
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.25)(x)
    outputs = Dense(label_cnt, activation='softmax')(x)
    model = Model(inputs, outputs)
    # model.summary()
    model.compile(optimizer=Adam(learning_rate=3e-6), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train_ML_model(X_train, Y_train, epoch_cnt: int = 5000, batch_size: int = 32, verbosity: bool = 1) -> Model:
    """
    Train the ConvNet and return the model.

    Args:
        X_train (array): Training data
        Y_train (array): Training labels (one-hot encoded)
        epoch_cnt (int): Number of training epochs
        batch_size (int): batch size
        verbosity (bool): verbose output (1) or not (0)
    
    Returns:
        (tensorflow.keras.Model):
        model : The trained ML model
    """
    # Define the architecture + callbacks
    model = define_CNN(X_train.shape[1:], Y_train.shape[1])
    keras_callbacks = define_callbacks()

    # Train the CNN
    model.fit(X_train, Y_train, epochs=epoch_cnt, batch_size=batch_size, verbose=verbosity, callbacks=keras_callbacks)

    return model


def evaluate_model(X_test, Y_test, model: Model, batch_size: int = 32, verbosity: bool = 1) -> tuple:
    """
    Evaluate the ML model on test data.

    Args:
        X_test: Test data
        Y_test: Test labels (one-hot encoded)
        model: A trailed ML model
        batch_size: batch size
        verbosity: verbose output (1) or not (0)
    
    Returns:
        score: 
        M: A confusion matrix
    """

    score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=verbosity)
    classes = np.argmax(model.predict(X_test), axis=1)
    M = confusion_matrix(classes, np.argmax(Y_test, axis=1))

    return score, M


def run_kfold(X, Y, n_splits: int = 5, epoch_cnt: int = 5000, batch_size: int = 32, model_out: str = None, verbosity: bool = 1) -> dict:
    """
    Run a k-fold analysis and save the best fitting model.

    Args:
        X (array): Test data
        Y (array): Test labels (one-hot encoded)
        n_splits (int): number of folds
        epoch_cnt (int): Number of training epochs
        batch_size (int): batch size
        model_out (str): model name
        verbosity (bool): verbose output (1) or not (0)
    
    Returns:
        results:
    """

    # Define per-fold score containers
    kfold_results = {'acc': [], 'loss': [], 'test': [], 'train': [],
               'M': [], 'precision': [], 'recall': [], 'F': []}
    
    # Collapse targets for Stratified K-fold
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
    # Note: https://stackoverflow.com/questions/48508036/sklearn-stratifiedkfold-valueerror-supported-target-types-are-binary-mul/48512157
    y_target = np.zeros((np.shape(Y)[0],))
    for jj in range(0, np.shape(Y)[0]):
        y_target[jj] = np.argmax(Y[jj])

    # Model save
    if model_out is not None:
        acc_max = 0.0

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    fold_no = 1
    for train, test in kfold.split(X, y_target):
        print('--------------------------------------------------------')
        print(f'Training for fold {fold_no}/{n_splits} ...')

        kfold_results['test'].append(np.array(test))
        kfold_results['train'].append(np.array(train))

        model = train_ML_model(X[train], Y[train], epoch_cnt=epoch_cnt, batch_size=batch_size, verbosity=verbosity)
        score, M = evaluate_model(X[test], Y[test], model=model, batch_size=batch_size, verbosity=verbosity)

        # Save most accurate model
        if model_out is not None:
            if (score[1]) > acc_max:
                model.save(model_out)
                acc_max = score[1]

        # Calculate and save training metrics
        print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
        kfold_results['acc'].append(score[1] * 100)
        kfold_results['loss'].append(score[0])

        predictions = model.predict(X[test])
        classes = np.argmax(predictions, axis=1)
        y_true = np.argmax(Y[test], axis=1)
        M = confusion_matrix(classes, y_true)
        kfold_results['M'].append(M)

        precision = np.array([M[n, n] / np.sum(M[:, n]) for n in range(M.shape[0])])
        recall = np.array([M[n, n] / np.sum(M[n, :]) for n in range(M.shape[1])])
        fscore = 2 * (precision * recall) / (precision + recall)

        kfold_results['precision'].append(precision)
        kfold_results['recall'].append(recall)
        kfold_results['F'].append(fscore)

        fold_no += 1

    return kfold_results


def summarize_kfold(kfold_results: dict, drop_minmax: bool = False) -> None:
    """
    Summarize k-fold results with optional extreme value trimming.

    Args:
        kfold_results (dict): Dictionary of k_fold results (accuracy, precision, recall, F, etc...)
        drop_minmax (bool):

    Returns:
        Prints to screen
    """
    mask = np.ones_like(kfold_results['acc'], dtype=bool)
    if drop_minmax:
        mask[np.argmin(kfold_results['acc'])] = False
        mask[np.argmax(kfold_results['acc'])] = False

    M_avg = np.average(np.array(kfold_results['M'])[mask], axis=0)
    for n in range(M_avg.shape[1]):
        M_avg[:, n] = M_avg[:, n] / np.sum(M_avg[:, n]) * 100.0
    
    precision = np.array([M_avg[n, n] / np.sum(M_avg[:, n]) for n in range(M_avg.shape[0])])
    recall = np.array([M_avg[n, n] / np.sum(M_avg[n, :]) for n in range(M_avg.shape[1])])
    fscore = 2 * (precision * recall) / (precision + recall)

    print('\n' + "k-fold results summary:" + '\n' + '-' * 50)
    print("Fold count: ", len(kfold_results['acc']))
    print("Accuracy: ", np.mean(np.array(kfold_results['acc'])[mask]), "+/-", np.std(np.array(kfold_results['acc'])[mask]))
    print("Loss: ", np.mean(np.array(kfold_results['loss'])[mask]), "+/-", np.std(np.array(kfold_results['loss'])[mask]))
    print('\n' + "Confusion Matrix (true labels along horizontals) [%]: ", '\n', M_avg.T)

    print("\nPrecision = " + str(precision))
    print("\nRecall = " + str(recall))
    print("\nF-score = " + str(fscore))


def run(fk_label: str, model_path: str) -> tuple:
    """
    """
    # Load in fk analysis results
    if fk_label[-15:] != ".fk_results.dat":
        print(fk_label)
        fk_label += ".fk_results.dat"

    print("Loading fk results from " + fk_label + "...")
    try:
        fk_results = np.loadtxt(fk_label).T
    except OSError:
        raise Exception("fk results file '" + fk_label + "' not found. Check that the file exists and that the (relative?) path is correct.") from None
    
    # Load CNN model for detection
    print("Loading ConvNet from " + model_path + "...")
    try:
        model = load_model(model_path)
    except OSError:
        raise Exception("model file '" + model_path + "' not found. Check that the model folder exists and that the (relative?) path is correct.") from None

    # Compute the window count from the model input shape and length of fk results
    window_cnt = np.floor(fk_results.shape[1] / (model.input_shape[1] / 2)).astype(int) - 1

    # Separate fk results into model input shape and record label times
    X = np.zeros((window_cnt, model.input_shape[1], model.input_shape[2]))
    det_times = []
    for n in range(window_cnt):
        j0 = int(model.input_shape[1] * n / 2)

        det_times = det_times + [fk_results[0][j0 + int(model.input_shape[1] / 2)]]
        X[n] = data_io.prep_fk_results(fk_results[:, j0:j0 + model.input_shape[1]])

    # An extra normalization was left in the model construction, so it
    # needs to be here until we re-build the model
    for jj in range(window_cnt):
        for m in range(0, 3):
            X[jj, :, m] /= np.max(np.abs(X[jj, :, m]))

    det_times = np.array(det_times)
    predictions = np.argmax(model.predict(X), axis=1)

    return det_times, predictions


def plot(fk_label: str, det_times: list, predictions) -> None:
    """
    Plot results.
    """

    # Load in fk analysis results
    if ".fk_results.dat" not in fk_label:
        fk_label += ".fk_results.dat"
    
    print("Loading fk results from " + fk_label + "...")
    try:
        fk_results = np.loadtxt(fk_label).T
    except:
        warnings.warn("fk_results file not found")
        return (None, None)

    # Plot results
    _, axis = plt.subplots(3, figsize=(10, 6), sharex=True)

    axis[0].set_ylabel("F-value")
    axis[0].set_ylim(0, 7.0)
    axis[1].set_ylabel("Trace Velocity \n[m/s]")
    axis[1].set_ylim(300, 600)
    axis[2].set_ylabel("Back Azimuth \n[deg.]")
    axis[2].set_ylim(-180, 180)

    axis[0].plot(fk_results[0], fk_results[3], 'k.')
    axis[1].plot(fk_results[0], fk_results[2], 'k.')
    axis[2].plot(fk_results[0], fk_results[1], 'k.')

    for n in range(0, len(det_times)):
        if predictions[n] == 1:
            for j in range(0, 3):
                axis[j].axvspan(det_times[n] - 9.9 * 60, det_times[n] + 9.9 * 60, color='b', alpha=0.4)
        elif predictions[n] == 2:
            for j in range(0, 3):
                axis[j].axvspan(det_times[n] - 9.9 * 60, det_times[n] + 9.9 * 60, color='g', alpha=0.4)

    plt.show()
