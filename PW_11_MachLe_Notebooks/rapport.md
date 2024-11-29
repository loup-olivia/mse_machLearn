# Practical work 11 -  Convolutional Neural Networks 
## Students
- Liechti Matthieu
- Loup Olivia
## Summary of work to include in the report
1. What is the learning algorithm being used to train the neural networks?
What are the parameters (arguments) being used by that algorithm?
What cost function is being used?
Please, answer the equation(s) and describe (e.g., please include your code for
this part) how did you create the training, validation and test datasets.

2. For each experiment except the last one (shallow network learning from raw data,
shallow network learning from features and CNN):
a select a neural network topology and describe the inputs, indicate how many
are they, and how many outputs.
b Compute the number of weights of each model (e.g., how many weights
between the input and the hidden layer, how many weights between each
pair of layers, biases, etc..) and explain how you get to the total number of
weights.
c Test every notebook for at least three different meaningful cases (e.g., for the
MLP exploiting raw data, test different models varying the number of hidden
neurons, for the feature-based model, test pix_p_cell 4 and 7, and number of
orientations or number of hidden neurons, for the CNN, try different number
of neurons in the feed-forward part) describe the model and present the
performance of the system (e.g., plot of the evolution of the error, final
evaluation scores and confusion matrices). Comment the differences in
results. Are there particular digits that are frequently confused?

3. Do the deep neural networks have much more “capacity” (i.e., do they have more
weights?) than the shallow ones? explain with one example
4. Train a CNN for the chest x-ray pneumonia recognition. In order to do so, complete
the code to reproduce the architecture plotted in the notebook. Present the
confusion matrix, accuracy and F1-score of the validation and test datasets and
discuss your results.