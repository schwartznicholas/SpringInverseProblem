import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from math import *
from scipy.integrate import odeint
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import time
def odeintway_damp(x0, v0, k, m, b, dt, tf):

    def DifEq(U, t, k, b, m): #defines our Differential Equation
        x = U[0]
        v = U[1] #= dr/dy

        a = (-b*v - k*x)/m #= d^2x/dt^2

        return [v, a]

    U0 = [x0, v0]

    ts = np.linspace(0, tf, ceil(tf/dt))
    Us = odeint(DifEq, U0, ts, args=(k, b, m))
    xs = Us[:,0]
    vs = Us[:,1]

    return xs, vs, ts

def odeintway_nodamp(x0, v0, k, m, dt, tf):

    def DifEq(U, t, k, m): #defines our Differential Equation
        x = U[0]
        v = U[1] #= dr/dy

        a = (-k*x)/m #= d^2x/dt^2

        return [v, a]

    U0 = [x0, v0]

    ts = np.linspace(0, tf, ceil(tf/dt))
    Us = odeint(DifEq, U0, ts, args=(k, m))
    xs = Us[:,0]
    vs = Us[:,1]

    return xs, vs, ts

def graph_random_data(data): # taken from matplotlib documentation and modified
    fig, axs = plt.subplots(2,2)
    x_data = np.arange(len(data[0][0]))/10
    y_data = np.random.choice(len(data),4)
    axs[0, 0].plot(x_data, data[y_data[0],0]) # graphs a 2x2 grid of the position behavior
    axs[0, 0].set_title('Example 1')
    axs[0, 1].plot(x_data, data[y_data[1],0])
    axs[0, 1].set_title('Example 2')
    axs[1, 0].plot(x_data, data[y_data[2],0])
    axs[1, 0].set_title('Example 3')
    axs[1, 1].plot(x_data, data[y_data[3],0])
    axs[1, 1].set_title('Example 4')

    for ax in axs.flat:
        ax.set(xlabel='Time (sec)', ylabel='X Position (m)')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

def build_and_compile_model(norm,optim,lr=0.001):
    model = keras.Sequential([
        norm,
        layers.Dense(600, activation='relu'), # sequential NN with descending layers sizes
        layers.Dense(500, activation='relu'),
        layers.Dense(400, activation='relu'),
        layers.Dense(300, activation='relu'),
        layers.Dense(200, activation='relu'),
        layers.Dense(100, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(loss=tf.keras.metrics.mean_squared_error,
                  metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')], # using RMSE b/c for values <1 MSE may
                  optimizer=optim(lr))                    # make the error look smaller than it actually is
    return model


def build_and_compile_model_fc(norm,optim,lr=0.001):
    model = keras.Sequential([
        norm,
        layers.Dense(600, activation='relu'), # sequential NN with all fully connected layers
        layers.Dense(600, activation='relu'),
        layers.Dense(600, activation='relu'),
        layers.Dense(600, activation='relu'),
        layers.Dense(600, activation='relu'),
        layers.Dense(600, activation='relu'),
        layers.Dense(600, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(loss=tf.keras.metrics.mean_squared_error,
                  metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')],
                  optimizer=optim(lr))
    return model


def plot_loss(history):
    '''
    plots the loss function for the training and validation data
    '''
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)

def calc_test_RMSE(pred,real):
    '''
    pred: predicted data points
    real: real data points

    returns the root mean squared error
    '''
    RMSE = 0
    for i in range(len(pred)):
        for j in range(2):
            RMSE += (pred[i][j] - real[i][j])**2
    return (RMSE[0]/(2*len(pred)))**0.5

def run_kfold(model_func,data,targets,k_fold_num,data_normalizer,optim,optim_name,epochs=100,shuffle=False,random_state=None):
    '''

    arguments:

    model_func: pass keras tensorflow model compiler
    data: pass data must be same size as targets
    targets: pass targets must be same size as data
    k_fold_num: k-fold cross validation number; suggested sizes: 5, 10
    data_normalizer: pass the normalizer that is used to normalize the data
    epochs: number of epochs to train the model on
    shuffle: shuffle your data and targets; default = False
    random_state: set a seed for shuffling; default = None

    returns: dictionary of cross validation number and the test RMSE
    '''
    kf_scores = {}
    kf = KFold(n_splits=k_fold_num,shuffle=shuffle,random_state=random_state)
    count = 1
    for train_index, test_index in kf.split(data):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = targets[train_index], targets[test_index]
        model = model_func(data_normalizer,optim)
        model.fit(X_train,y_train,validation_split=0.2,verbose=0,epochs=epochs)
        test_predictions = model.predict(X_test)
        kf_scores[str(count)] = calc_test_RMSE(test_predictions,y_test)
        count += 1
    return(kf_scores)

def run_epoch_comparison(model,epoch_list,data,targets,data_normalizer,optim,cv,optim_name):
    start = time.time()
    for i in range(len(epoch_list)):
        start_iter = time.time()
        kf_scores = run_kfold(model, data, targets, cv, data_normalizer,optim,optim_name, epochs=epoch_list[i])
        plt.style.use('ggplot')
        plt.bar(kf_scores.keys(), kf_scores.values())
        plt.ylabel('RMSE')
        plt.xlabel('K-fold')
        plt.title('RMSE across K-fold CV with '+str(epoch_list[i])+' Epochs and '+optim_name+' optimizer')
        plt.show()
        end = time.time()
        print('Task Runtime: ', np.round((end - start_iter) / 60, 2), ' minutes')
        print('Total Runtime: ', np.round((end - start) / 60, 2), ' minutes')