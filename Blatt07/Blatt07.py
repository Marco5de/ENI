import numpy as np;
import itertools as it
import tensorflow as tf;
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

def binaryCombinations(n):
    combinations = [];
    for i in range(2**n): #2^n possibilities to cover
        binNumber = format(i, 'b').zfill(n)
        vector = [];
        for j in range(n):
            if(binNumber[j] == '1'):
                vector.append(1);
            else:
                vector.append(0)
                
        combinations.append(vector)
                
    return np.array(combinations)

def generateInputCombinations(n): 
    x = binaryCombinations(n)
    y = binaryCombinations(n)
    return np.array(list(it.product(x,y)))

def generateTeachingSignal(n):
    teachingSignal = []
    inputSignal = generateInputCombinations(n);
    
    for i in range(len(inputSignal)):
        setpoint = np.dot(inputSignal[i][0],inputSignal[i][1]) % 2;
        teachingSignal.append(setpoint);
    
    inputSignal[inputSignal == 0] = -1;
    teachingSignal[teachingSignal == 0] = -1;
    
    return inputSignal, np.array(teachingSignal)  

def train_network(n_hidden,data,T):
    '''
    Trains a neural network and returns the lowest error.
    :param n_hidden: Number of hidden neurons to use per layer (as vector to indicate when multiple hidden
        layers should be used). For example, [2] uses one hidden layer with two neurons and [2, 2] uses two 
        hidden layers each with two neurons.
    :return: The lowest error (MSE) occurred over all training epochs.
    ''' 
    
    usedNeurons = 0;
    
    mySeed = 42;
    
    # Start fresh and at least try to get reproducible
    tf.reset_default_graph()
    K.clear_session()
    tf.set_random_seed(mySeed)
    np.random.seed(mySeed)
    
    session = tf.Session(config=None)
    tf.keras.backend.set_session(session)
    
    model = tf.keras.models.Sequential()
    
    batch, dim, length = data.shape;
    
    inputNeurons = dim*length;
    
    data = data.reshape(batch, inputNeurons);
    
    init = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=mySeed)
    acti = 'tanh'  
    
    if(isinstance(n_hidden, int)):
        numberOfLayers = 1;
        n_firstHidden = n_hidden;
    else:
        numberOfLayers = np.shape(n_hidden)[0]
        n_firstHidden = n_hidden[0];
        
    # Add first layer
    model.add(tf.keras.layers.Dense(units=n_firstHidden, input_dim = inputNeurons, activation=acti, kernel_initializer = init)) 
    usedNeurons += n_firstHidden;
    
    for layerNum in range(1, numberOfLayers): 
        model.add(tf.keras.layers.Dense(units = n_hidden[layerNum], activation=acti, kernel_initializer = init)) 
        usedNeurons += n_hidden[layerNum]
    
    # Add output layer
    model.add(tf.keras.layers.Dense(units = 1, activation=acti)) 
    usedNeurons += 1;
    
    #Plot model for debugging
    path = "models/" + str(n_hidden) + ".png"
    tf.keras.utils.plot_model(model,path,show_shapes = True);
    
    sgd = tf.keras.optimizers.SGD(lr=0.02, decay=0.0001, momentum=0.9,nesterov=True)

    with session.as_default():
        with session.graph.as_default():
            model.compile(loss='mse',optimizer =sgd)
            history = model.fit(data,T,epochs=300)
            
            #Return min error and number of neurons
            return (min(history.history['loss']), usedNeurons)

for n in range(3,5):
	min_loss_flat = []
	neurons_flat = []
	min_loss_deep = []
	neurons_deep = []

	data, T = generateTeachingSignal(n);

	for i in range(1,2**(n+1)+4):
		loss, neurons = train_network(i,data,T);
		min_loss_flat.append(loss);
		neurons_flat.append(neurons);

	for i in range(1,(n+1)+4):
		loss, neurons = train_network([i,i],data,T);
		min_loss_deep.append(loss)
		neurons_deep.append(neurons)

	fig = plt.figure()
	plt.title("n = " + str(n))
	plt.xlabel('Number of neurons')
	plt.ylabel('MSE')
	plt.plot(neurons_flat, min_loss_flat, label='Flat hierarchy', marker='o')
	plt.plot(neurons_deep, min_loss_deep, label='Deep hierarchy', marker='o')
	plt.tight_layout()
	plt.legend()
	plt.grid(color='lightgrey', linestyle='-', linewidth=1)

	figureName = "n=" + str(n) + ".png";
	plt.savefig(figureName)