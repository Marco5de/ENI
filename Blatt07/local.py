import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import keras
import itertools as it

'''
Die Lernregel für die quadratische Fehlerfunktion enhält die Ableitung der Transferfunktion. Wird als 
Transferfunktion die Heaviside-Funktion gewählt, so existiert in 0 die Ableitung nicht --> keine gute Wahl
'''
def binary_lin_comb(n):
    comb_list = []
    #2**n possible combs, will all be represented in binary
    for i in range(2**n):
        x = bin(i)[2:]
        x = "0" * (n-len(x)) + x
        comb = []
        for j in range(n):
            if x[j]=="1":
                comb.append(1)
            else:
                comb.append(0)
        comb_list.append(comb)
        
    return np.array(comb_list)

def lin_comb(n):
    x = binary_lin_comb(n)
    y = binary_lin_comb(n)
    return np.array(list(it.product(x,y)))

def teaching_signals(combs):
    T = []
    for i in range(len(combs)):
        T.append(np.dot(combs[i][0],combs[i][1])%2)
    return np.array(T)   
        

def init_data(n):
    combs = lin_comb(n)
    T = teaching_signals(combs)
    combs[combs == 0] = -1
    T[T == 0] = -1
    
    print(combs.shape)
    
    #todo flatten data for input to array
    combs = combs.reshape(len(combs),combs.shape[1]*combs.shape[2])

    return combs,T
    
    
def train_network(n_hidden,data,T):
    '''
    Trains a neural network and returns the lowest error.
    :param n_hidden: Number of hidden neurons to use per
    layer (as vector to indicate when multiple hidden
    layers should be used). For example, [2] uses one
    hidden layer with two neurons and [2, 2] uses two
    hidden layers each with two neurons.
    :return: The lowest error (MSE) occurred over all
    training epochs.
    ''' 
    # Start fresh and at least try to get reproducible
    tf.reset_default_graph()
    K.clear_session()
    tf.set_random_seed(42)
    np.random.seed(42)
    
    session = tf.Session(config=None)
    keras.backend.set_session(session)
    
    n_input = sum(data[0].shape)
    
    model = keras.models.Sequential()
    
   
    
    if(isinstance(n_hidden, int) or len(n_hidden)==1):
        #only hidden layer in net
        model.add(keras.layers.Dense(output_dim = n_hidden,
                                    input_dim=n_input,
                                    activation='tanh',
                                    kernel_initializer = keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=42)))
        #output layer
        model.add(keras.layers.Dense(output_dim=1,
                                    activation='tanh',
            ))
    else:
        #first hidden layer
        model.add(keras.layers.Dense(output_dim = n_hidden[1],
                                input_dim=n_input,
                                activation='tanh',
                                kernel_initializer = keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=42)))
        #middle hidden layers
        for i in range(1,n_hidden[0]):
            model.add(keras.layers.Dense(output_dim=n_hidden[1],
                                        activation='tanh',
                                        kernel_initializer = keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=42))
                     )

        #output layer
        model.add(keras.layers.Dense(output_dim=1,
                                    activation='tanh',
                 ))
      
    keras.utils.plot_model(model,"test.png",show_shapes = True);

    with session.as_default():
        with session.graph.as_default():
            model.compile(loss='mse',optimizer = keras.optimizers.SGD(lr=0.2, decay=0.0001, momentum=0.9,nesterov=True))
            history = model.fit(data,T,epochs=300,batch_size=16)
            return(min(history.history['loss']), len(model.layers))

    
n=3
combs,T = init_data(n)

min_loss_flat = []
min_loss_deep = []

#net should have nx[0]+1 layers
nx = [7,7];
#nx = 8;
_,length = train_network(nx,combs,T)
print(length)
'''
for i in range(2,2**n+4):
    min_loss_flat.append(train_network(i,combs,T))
    
for i in range(1,n+4):
    min_loss_deep.append(train_network([i,i],combs,T))
print("Min Loss Flat: " + str(min_loss_flat))
print("Min Loss Deep " + str(min_loss_deep))
'''
