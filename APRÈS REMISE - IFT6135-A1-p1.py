
# coding: utf-8

# # IFT6135 - Assignment 1 - Problem 1
# By Orlando Marquez (ID 1062617) & Nicholas Vachon (ID 64249)

# In[1]:


import numpy as np
import gzip,pickle
import torch
from torch.nn import Parameter
from torch.autograd import Variable
import matplotlib.pyplot as plt


# In[2]:


f=gzip.open('mnist.pkl.gz')
data=pickle.load(f)

traindata_mnist = np.concatenate((data[0][0], np.array([data[0][1]]).T), axis=1)
validdata_mnist = np.concatenate((data[1][0], np.array([data[1][1]]).T), axis=1)
testdata_mnist = np.concatenate((data[2][0], np.array([data[2][1]]).T), axis=1)


# In[101]:


class SimpleMLP:
    def __init__(self, batch_size=100, d_in=784, d_h1=500, d_h2=500, d_out=10, learning_rate=0.001):
        self.batch_size = batch_size
        self.d_in = d_in
        self.d_h1 = d_h1
        self.d_h2 = d_h2
        self.d_out = d_out
        self.learning_rate = learning_rate
        self.layer_h1 = torch.nn.Linear(self.d_in, self.d_h1)
        self.layer_h2 = torch.nn.Linear(self.d_h1, self.d_h2)
        self.layer_out = torch.nn.Linear(self.d_h2, self.d_out)
        self.layer_h1.bias = Parameter(torch.zeros(self.d_h1))        
        self.layer_h2.bias = Parameter(torch.zeros(self.d_h2))
        self.layer_out.bias = Parameter(torch.zeros(self.d_out))  
        
    def num_params(self):
        return self.d_h1*(self.d_in+1) + self.d_h2*(self.d_h1+1) + self.d_out*(self.d_h2+1)
        
    def init_layers(self, type_init='normal', std=1):
        if type_init == 'zero':
            self.layer_h1.weight = Parameter(torch.zeros(self.d_h1, self.d_in))
            self.layer_h2.weight = Parameter(torch.zeros(self.d_h2, self.d_h1))
            self.layer_out.weight = Parameter(torch.zeros(self.d_out, self.d_h2))
        elif type_init == 'glorot':            
            dl_h1 = np.sqrt(6.0/(self.d_in+self.d_h1))
            dl_h2 = np.sqrt(6.0/(self.d_h1+self.d_h2))
            dl_out = np.sqrt(6.0/(self.d_h2+self.d_out))
            self.layer_h1.weight = Parameter(torch.FloatTensor(np.random.uniform(-dl_h1, dl_h1, size=(self.d_h1, self.d_in))))
            self.layer_h2.weight = Parameter(torch.FloatTensor(np.random.uniform(-dl_h2, dl_h2, size=(self.d_h2, self.d_h1))))
            self.layer_out.weight = Parameter(torch.FloatTensor(np.random.uniform(-dl_out, dl_out, size=(self.d_out, self.d_h2))))
        else:
            self.layer_h1.weight = Parameter(torch.FloatTensor(np.random.normal(0, std, size=(self.d_h1, self.d_in))))
            self.layer_h2.weight = Parameter(torch.FloatTensor(np.random.normal(0, std, size=(self.d_h2, self.d_h1))))
            self.layer_out.weight = Parameter(torch.FloatTensor(np.random.normal(0, std, size=(self.d_out, self.d_h2))))            
    
    def compute_accuracy(self, data):
        x = Variable(torch.FloatTensor(data[:, :-1]))
        y = Variable(torch.LongTensor(data[:, -1]))
        _, indices = torch.max(self.model(x), 1)
        return np.mean((indices == y).data)        

    def train(self, traindata, validationdata=None, compute_accuracy=False, num_epochs=10, testdata=None, verbose=False):
        num_samples = traindata.shape[0]

        # Create Tensors to hold inputs and outputs, and wrap them in Variables.
        x = Variable(torch.FloatTensor(traindata[:, :-1]))
        y = Variable(torch.LongTensor(traindata[:, -1]), requires_grad=False)

        self.model = torch.nn.Sequential(
            self.layer_h1,
            torch.nn.ReLU(),
            self.layer_h2,
            torch.nn.ReLU(),   
            self.layer_out,
            torch.nn.Softmax(dim=1)
        )
        loss_fn = torch.nn.NLLLoss()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        batch_ends = np.linspace(self.batch_size, num_samples, num_samples/self.batch_size).astype(int)
        
        avg_losses = np.zeros(num_epochs)
        accuracy_training = np.zeros(num_epochs, dtype=np.float16)
        accuracy_validation = np.zeros(num_epochs, dtype=np.float16)       
        accuracy_test = np.zeros(num_epochs, dtype=np.float16)

        #epochs
        for e in range(num_epochs):
            total_loss = 0

            for batch_idx, batch_end in enumerate(batch_ends):
                if batch_idx == 0:
                    batch_start = 0
                else:
                    batch_start = batch_ends[batch_idx - 1]

                # Forward pass: compute predicted y by passing x to the model.
                y_pred = self.model(x[batch_start: batch_end])

                loss = loss_fn(y_pred, y[batch_start: batch_end])
                
                total_loss+=loss

                optimizer.zero_grad()

                # Backward pass: compute gradient of the loss with respect to model
                # parameters
                loss.backward()

                # Calling the step function on an Optimizer makes an update to its
                # parameters
                optimizer.step()

            avg_losses[e] = total_loss.data[0]/num_samples
            #print avg_losses[e]
            
            if compute_accuracy == True:            
                
                _, indices = torch.max(self.model(x), 1)        
                accuracy_training[e] = np.mean((indices == y).data)
                
                accuracy_validation[e] = self.compute_accuracy(validationdata)
                
                if testdata is not None:
                    accuracy_test[e] = self.compute_accuracy(testdata)
                    
            if verbose:
                print "Epoch {}: Avg L {}, Acc train {} (if app), Acc valid {} (if app)".format(e+1, avg_losses[e], accuracy_training[e], accuracy_validation[e])
                    
        return avg_losses, accuracy_training, accuracy_validation, accuracy_test


# Architecture of model used below unless otherwise indicated:
# <span style="color:blue">
# * <span style="color:blue"> Number of units in 1st hidden layer = 500 <span>
# * <span style="color:blue"> Number of units in 2nd hidden layer = 500 <span>
# * <span style="color:blue"> Total number of parameters = 648,010 <span>
# * <span style="color:blue"> Nonlinearity activation function = RELU <span>
# * <span style="color:blue"> Learning rate = 0.6 <span>
# * <span style="color:blue"> Mini-batch size = 100 <span>

# ### Initialization

# Testing that weights stay at 0 if they are initialized at 0

# In[102]:


num_neurons=500
lr = 0.6
mlp = SimpleMLP(d_h1=num_neurons, d_h2=num_neurons, learning_rate=lr)
print "# neurons per layer: {}, # params: {}, RELU activations, \
learning rate={}, mini-batch size=100".format(num_neurons, mlp.num_params(), lr)
print 'Init weights at zero'
mlp.init_layers('zero')
print mlp.layer_h1.weight
print mlp.layer_h2.weight
print mlp.layer_out.weight
_, _, _, _ = mlp.train(traindata_mnist, 10, verbose=True)
print mlp.layer_h1.weight
print mlp.layer_h2.weight
print mlp.layer_out.weight


# In[103]:


num_neurons=500
lr = 0.01
mlp = SimpleMLP(d_h1=num_neurons, d_h2=num_neurons, learning_rate=lr)
print "# neurons per hidden layer: {}, # params: {}, RELU activations, \
learning rate={}, mini-batch size=100".format(num_neurons, mlp.num_params(), lr)
print 'Init weights at zero'
mlp.init_layers('zero')
avg_losses_zero_init, _, _, _ = mlp.train(traindata_mnist, 10, verbose=True)

print 'Init weights from normal distribution'
mlp = SimpleMLP(d_h1=num_neurons, d_h2=num_neurons, learning_rate=lr)
mlp.init_layers('normal')
avg_losses_normal_init, _, _, _ = mlp.train(traindata_mnist, 10, verbose=True)

print 'Glorot initialization'
mlp = SimpleMLP(d_h1=num_neurons, d_h2=num_neurons, learning_rate=lr)
mlp.init_layers('glorot')
avg_losses_glorot_init, _, _, _ = mlp.train(traindata_mnist, 10, verbose=True)

plt.figure(figsize=(10,8))
plt.plot(range(1,11), avg_losses_zero_init, label="Zero initialization")
plt.plot(range(1,11), avg_losses_normal_init, label="N(0,1) initialization")
plt.plot(range(1,11), avg_losses_glorot_init, label="Glorot initialization")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average loss')
plt.xlim(1,10)
plt.show()


# <p style='color:blue'>
# <b>We used the same architecture as specified above, except with a learning rate of 0.01 (which is low) to better compare the average losses</b>
# </p>
# <ul style='color:blue'>
# <li>When the weight parameters are initialized to zero, they remain zero throughout training and thus the loss barely decreases. Only the biases can be updated through gradient descent. An MLP that does not make use of its weights to learn a predictor function is useless.</li>
# <li>When the weight parameters are initialized from sampling N(0,1), the loss decreases faster than if they started at 0. However, there is no mechanism to control the variance of the weights throughout the network. We can empirically see that this retards learning.</li>
# <li>Glorot initialization performs the best as it decreases the loss the fastest. Controlling the variance of the weights through the layers is empirically demonstrated to be a good idea.</li>
# </ul>

# ### Learning Curves

# In[104]:


num_neurons=500
lr=0.6
mlp = SimpleMLP(d_h1=num_neurons, d_h2=num_neurons, learning_rate=lr)
mlp.init_layers('glorot')
print "# neurons per hidden layer: {}, # params: {}, RELU activations, \
learning rate={}, mini-batch size=100".format(num_neurons, mlp.num_params(), lr)

_, accuracy_train_500, accuracy_valid_500, _ = mlp.train(traindata_mnist, validdata_mnist, True, 100, verbose=True)


# In[123]:


print 'Train and valid error after 5 epochs: {}, {}'.format(accuracy_train_500[4]*100, accuracy_valid_500[4]*100)
print 'Train and valid error after 10 epochs: {}, {}'.format(accuracy_train_500[9]*100, accuracy_valid_500[9]*100)
print 'Train and valid error after 20 epochs: {}, {}'.format(accuracy_train_500[19]*100, accuracy_valid_500[19]*100)
print 'Train and valid error after 100 epochs: {}, {}'.format(accuracy_train_500[99]*100, accuracy_valid_500[99]*100)
print 'Max validation accuracy (basis for early stopping): {}'.format(accuracy_valid_500.max()*100)

plt.figure(figsize=(10,5))
plt.plot(range(1,101), accuracy_train_500*100, label="Training")
plt.plot(range(1,101), accuracy_valid_500*100, label="Validation")
plt.legend()
plt.title("Training and validation accuracy for MLP that achives >98% validation accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.yticks(np.arange(80, 100, 2))
plt.xticks(np.arange(0, 100, 5))
plt.xlim(1,100)
plt.show()


# #### Doubling the capacity

# Architecture of model with double capacity:
# <span style="color:blue">
# * <span style="color:blue"> Number of units in 1st hidden layer = 808 <span>
# * <span style="color:blue"> Number of units in 2nd hidden layer = 808 <span>
# * <span style="color:blue"> Total number of parameters = 1,296,042 <span>
# * <span style="color:blue"> Nonlinearity activation function = RELU <span>
# * <span style="color:blue"> Learning rate = 0.6 <span>
# * <span style="color:blue"> Mini-batch size = 100 <span>

# In[90]:


num_neurons=808
lr=0.6
mlp = SimpleMLP(d_h1=num_neurons, d_h2=num_neurons, learning_rate=lr)
mlp.init_layers('glorot')
print "# neurons per hidden layer: {}, # params: {}, RELU activations, \
learning rate={}, mini-batch size=100".format(num_neurons, mlp.num_params(), lr)

_, accuracy_train_808, accuracy_valid_808, _ = mlp.train(traindata_mnist, validdata_mnist, True, 100, verbose=True)


# In[124]:


print 'Train and valid error after 5 epochs: {}, {}'.format(accuracy_train_808[4]*100, accuracy_valid_808[4]*100)
print 'Train and valid error after 10 epochs: {}, {}'.format(accuracy_train_808[9]*100, accuracy_valid_808[9]*100)
print 'Train and valid error after 20 epochs: {}, {}'.format(accuracy_train_808[19]*100, accuracy_valid_808[19]*100)
print 'Train and valid error after 100 epochs: {}, {}'.format(accuracy_train_808[99]*100, accuracy_valid_808[99]*100)
print 'Max validation accuracy (basis for early stopping): {}'.format(accuracy_valid_808.max()*100)

f = plt.figure(figsize=(10,5))
plt.plot(range(1,101), accuracy_train_808*100, label="Training")
plt.plot(range(1,101), accuracy_valid_808*100, label="Validation")
plt.legend()
plt.title("Training and validation accuracy for MLP with double capacity")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.yticks(np.arange(80, 100, 2))
plt.xticks(np.arange(0, 100, 5))
plt.xlim(1,100)
plt.show()


# In[115]:


f = plt.figure(figsize=(10,5))
plt.plot(range(1,101), accuracy_valid_500*100, label="Validation accuracy (648010 parameters)")
plt.plot(range(1,101), accuracy_valid_808*100, label="Validation accuracy (1296042 parameters)")
plt.title("Comparing validation accuracy with a model with double capacity")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.yticks(np.arange(80, 100, 2))
plt.xticks(np.arange(0, 100, 5))
plt.xlim(1,100)
plt.show()


# <p style="color:blue">
# The accuracy is about the same even if we double the numbers of parameters (thus increasing the capacity of the model). It didn't have impact because the model we started with was already pretty accurate (more than 98%), so it was capturing a function pretty close to the optimal one. Trying to reduce the bias (by increasing the capacity) had no real effect because there no large bias in the first place.</p>
# 
# <p style="color:blue">This means that for this kind of task, a model with high capacity is not needed. A model with about 650 000 parameters finds a function that is close enough to the true solution. Using a higher capacity model does not produce a better function and might have higher variance. Let's run a little test to verify that. Bellow, we trained 2 models on 30,000 examples chose randomly from the 50,000 MNIST example 5 times (5 different subsets). The only difference between the 2 models is the number of parameters, from simple to double. And below that, we ran a model with three times the capacity.</p>
# 
# <p style='color:blue'>
# Our results are interesting: Both variances are very low and tripling the capacity gives us exactly the same results. We can conclude that the amount of data we have is large enough to support a higher capacity model. That is, we can attempt to decrease the bias without danger of increasing variance. However, a model with double or triple capacity did not end up reducing the bias.
# </p>

# In[126]:


Variance = np.zeros((5,2))

for ite in range(5):
    rand_ind = np.arange(50000)
    np.random.shuffle(rand_ind)
    rand_ind = rand_ind[:30000]
    
    mlp = SimpleMLP(d_h1=500, d_h2=500, learning_rate=0.6)
    mlp.init_layers('glorot')
    _, _, accuracy_valid_simple, _ = mlp.train(traindata_mnist[rand_ind], validdata_mnist, True, num_epochs=50)
    print "Max validation with normal capacity: ", np.max(accuracy_valid_simple)
    
    mlp = SimpleMLP(d_h1=808, d_h2=808, learning_rate=0.6)
    mlp.init_layers('glorot')
    _, _, accuracy_valid_double, _ = mlp.train(traindata_mnist[rand_ind], validdata_mnist, True, num_epochs=50)
    print "Max validation with double capacity: ", np.max(accuracy_valid_double)
    
    Variance[ite, 0] = np.max(accuracy_valid_simple)
    Variance[ite, 1] = np.max(accuracy_valid_double)


# In[134]:


print(Variance)
print(np.mean(Variance, axis=0))
print(np.var(Variance, axis=0))


# <b>Tripling the capacity</b>

# In[128]:


num_neurons=1000
lr=0.6
mlp = SimpleMLP(d_h1=num_neurons, d_h2=num_neurons, learning_rate=lr)
mlp.init_layers('glorot')
print "# neurons per hidden layer: {}, # params: {}, RELU activations, \
learning rate={}, mini-batch size=100".format(num_neurons, mlp.num_params(), lr)

_, accuracy_train_1000, accuracy_valid_1000, _ = mlp.train(traindata_mnist, validdata_mnist, True, 100, verbose=True)


# ### Training Set Size, Generalization Gap and Standard Error

# In[92]:


training_set_size = traindata_mnist.shape[0]
a = [0.01, 0.02, 0.05, 0.1, 1.0]
num_neurons=500
lr=0.6
gaps = np.zeros((5,5))

for trials_idx in range(5):
    for data_idx, data_fraction in enumerate(a):
        indices = np.random.choice(training_set_size, int(data_fraction*training_set_size), replace=False)
        traindata = traindata_mnist[indices]

        print "Number of samples: ", traindata.shape[0]    
        mlp = SimpleMLP(d_h1=num_neurons, d_h2=num_neurons, learning_rate=lr)
        mlp.init_layers('glorot')
        print "# neurons per hidden layer: {}, # params: {}, RELUs, learning rate={}, mini-batch size=100".format(num_neurons, mlp.num_params(), lr)

        _, acc_train, acc_valid, acc_test = mlp.train(traindata, validdata_mnist, True, 100, testdata_mnist)
        print acc_train
        print acc_valid
        print acc_test
        print "Gap: ", acc_train[acc_valid.argmax()] - acc_test[acc_valid.argmax()]
        gaps[trials_idx][data_idx] = acc_train[acc_valid.argmax()] - acc_test[acc_valid.argmax()]
    


# In[93]:


from scipy import stats

print "Generalization gaps(0.01, 0.02, 0.05, 0.1, 1.0):"
print gaps*100
print "Average generalization gap"
print (gaps*100).mean(axis=0)
print "Average generalization standard error"
print (gaps*100).std(axis=0)


# <ul style='color:blue'>
# <li>More training data allows for better generalization as the generalization error decreases as data increases.</li>
# <li>More training data means a smaller difference between the training and generalization error (10.5% decreased to 1.41%). This means that with more data, we are more confident that driving down the training error will yield the lowest generalization error. </li>
# <li>The standard error is decreasing as the number of sample increases. This means that the more we have data, the more the variance is reduced. More data also means that the generalization error produced by our model is more representative of how that model will perform with unseen data. When training with only 1% of the data, we would report a generalization gap that would be inaccurate by 0.5%. Whereas when training with 10% of the data, the reported gap would be inaccurate by 0.37% and with the entire training data set, it would be inaccurate by only 0.06%.</li> 
# </ul>
