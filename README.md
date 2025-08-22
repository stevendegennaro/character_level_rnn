# Character-level Recurrent Neural Networks

This generative AI uses a character-level recurrent neural network to generate strings of letters. When trained on lists of words or names, the network will create novel words or names one character at a time.

The original network was built entirely from scratch, using only native Python and NumPy.

The network uses a training list to build a one-hot-encoded vocabulary of individual letters. The model generally consists of two simple RNN layers with 32 hidden neurons each, followed by a linear layer (though this is customizable). The Simple RNN layers use a hidden state that is updated from one character to the next for each word in the training data (and subsequently for the generation of new words). Additional functions track the speed and accuracy of both training and generation. The model is then trained on the input data for enough epochs to generate plausible-sounding novel names without recreating names already in the list.

The original version of the code was created to train on a list of the first and last names of baseball players. A more generalized version can accept any list of strings. I used a list of company names from the New York Stock Exchange and NASDAQ in order to create fake company names for use in the feature film Remote.

I later rewrote the networks using Keras and TensorFlow, partly as an exercise in teaching myself those packages, and partly for rather considerable gains in training speed.

A more complete description of the project is documented in a series of blog posts:
[Building a Character-level Recurrent Neural Network to Generate Fake Baseball Player Names, Part 1](https://medium.com/@datasciencefilmmaker/whos-on-first-1-394dda0db523)<br>
[Building a Character-level Recurrent Neural Network to Generate Fake Baseball Player Names, Part 2](https://medium.com/@datasciencefilmmaker/whos-on-first-2-8a857f887124)<br>
[Using SimpleRNN in Keras to Generate Fake Baseball Player Names](https://medium.com/@datasciencefilmmaker/whos-on-first-3-569da9d2b4f3)<br>
[Testing Keras Against a Recurrent-Neural Network Built from Scratch](https://medium.com/@datasciencefilmmaker/whos-on-first-4-2d6417be8db0)<br>
[Further Testing of Character-level Recurrent-Neural Networks](https://medium.com/@datasciencefilmmaker/whos-on-first-5-6-further-testing-of-character-level-recurrent-neural-networks-304784dff445)<br>
[Using a Keras Recurrent Neural Network Trained on Stock Exchange Data to Generate Fake Company Names](https://medium.com/@datasciencefilmmaker/whos-on-first-6-acb64a0d8a44)<br>

The repository consists of two folders, each containing two versions of the networks.

## Baseball

This is the original version, created to generate fake names for baseball players based on a list
of every MLB player in history. Both networks use player.py, which defines an object class to store information about baseball players, and scrape_baseball_names.py, which gets the names from the MLB website and outputs them to a file with one name per line.

The network is trained separately on first names and last names.

The scratch version of the network utilizes dsfs_deep.py and dsfs_vocab.py, which are modified versions of the deep learning algorithms in [Data Science From Scratch](https://www.amazon.com/Data-Science-Scratch-Principles-Python/dp/1492041130/ref=sr_1_1?dib=eyJ2IjoiMSJ9.8bSg9CnzT6ILYn7fN65f9GXxN6LDuQMHjkfZpjA27wP5vzgnYENSVsds_W1E3VyBuTGBf8LwA0TfkU2-RizNv8SdirGX1xO0D1COqOlqq0BsAwroRUiHs3FeMB1VjxL7RjuZLO3DTZyp0rNSLLzpdifu6jBep8zuPRAVZGRkLidNjALKzJTU-p3rQyidnAlZ_ro3tFv8WhV8zFIRnxVJe5So7M0uHeWI2Q_8jhJsnzo.lmQ9ZgsysOJUbv4KU3sIDe1jyp7wKXbci0XFaiAmugM&dib_tag=se&hvadid=694174772857&hvdev=c&hvexpln=67&hvlocphy=9028322&hvnetw=g&hvocijid=4535312885576838464--&hvqmt=e&hvrand=4535312885576838464&hvtargid=kwd-299673867783&hydadcr=16405_13457201&keywords=data+science+from+scratch&mcid=13cabc72f4ab36cfbc6431b4b647ef70&qid=1755877709&sr=8-1) by Joel Grus. I used his code as a launching point, but replaced native Python with numpy for speed.

### dsfs_vocab.py

Defines a vocabulary class that stores every letter seen by the network as one-hot-encoded lists. Also containes functions for importing and exporting the vocabulary to file.

### dsfs_deep.py

Defines the classes and functions necessary to create the deep-learning neural network. 

The Layer class is a general purpose class that defines functions for returning the current values of the weights, biases, and gradients, as well as functions for feeding the layer forward and for back propagation.

Linear defines simple linear layers with weights, biases, and gradients for each.

SimpleRnn layers add a hidden unit with its own weights, biases, and gradients. This hidden unit retains its state from one letter to the next to give the network a "memory".

The module also defines a generic Loss class and the specific loss function SoftMaxCrossEntropy, a generic Optimizer class with a simple GradientDescent version and one utilizing Momentum.

And finall, there is a Model class, based on the Layer class, that contains the layers of the network, the loss function, and the optimizer.

Additional functions generate random initialization condions, save and load weights, and to sample from the weights randomly.

### name_network_scratch.py

This is the main scratch network.

**import_names()**

Imports the list of names and puts them into Player objects.

**get_vocab()**

Uses the training list to create a vocabulary object that includes every letter in every name. Currently not used in the module because the vocab does not need to be recreated every time unless the list of names changes.

**calculate_accuracy()**

Calculates the accuracy of a model once it has been trained (or during training). There are two different methods possible to calculate accuracy. The 'argmax' method always chooses the letter with the highest probability of being next. The 'sample_from' method chooses a letter at random based on its probability of being next.

**plot_history()**

Plots the accuracy of the network vs time during training, to judge convergence.

**train()**

Trains the network. It takes a vocabulary object and two lists of names -- one for training, one for validation. User can choose batch size, number of epochs, as well as files to write or plot the accuracy over time during training. It keeps track of the accuracy over time and continually saves the weights to file every epoch where the accuracy improves over its previous maximum.

**generate()**

Creates new names entirely from scratch. It starts with the START charater, predicts the first letter of the name, then the second based on the first letter, the third based on the first two letters, etc, until it predicts the STOP character or reaches a maximum length.

**create_model()**

creates a model that consists of two SimpleRnn hidden layers (with dimensions specified by the user) and a linear layer, a softmax cross entropy loss, and a momentum optimizer.

**run_network()**

The main function of the module. Creates the model, imports names, trains, and generates new names.

**generate_players()**

Generates a final batch of first and last names based on final weights from the training sessions.

**training_speed_test()**

Tests the time it takes to train the network.

**generation_test()**

Tests how long it takes to generate n_players names, then calculates how many of those names are already in the names list. Repeats for last names. Returns a dictionary with lists of the duplicates.

**manual_accuracy_test()**, **predict_accuracy()**, **get_most_likely()**, **calculate_max_accuracy()**, adn **calculate_max_likelihood()**

These methods are used to calculate the actual theoretical maximum accuracy of the network. Because names have a lot of redundancy, it's impossible for _any_ network to obtain 100% prediction accuracy. For example, the input "B" can be followed by "o" for "Bob" or "i" for "Bill" or "a" for "Barry". In the accuracy calculations that the network does as it trains, if the input is "B" and the target is "o", then the network is "wrong" if it predicts "i" for that particular target. These functions represent two different ways to figure out how accurate the network could possibly be in the best case scenario, if it chose among the possible letters. (So, for instance, if "B" can be follwed by "o" or "i" or "a" with equal frequency, then the network can only be correct at most a third of the time for that input. The calculation changes slightly if the probabilities are not equal, but the idea remains the same.) These functions take every single possible input string, figure out every possible target for that input, and then calculates the probability of hitting *a particular* one of those targets.

## name_network_keras.py

Recreates the functions in name_network_scratch.py more or less 1-to-1, but using Keras/TensorFlow. Utilizes a converstion from Keras to tfLite for reasons that I can no longer remember.