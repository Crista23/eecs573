This folder includes all the codes, intermediate files and final results of the linear mapping.
The main algorithm is implemented with python and MATLAB.

The subfolder includes the codes for generating word2vec features we use here.

The sim_results include one test results using our model.

new_linear_map.m is the main MATLAB src code for train the linear mapping model. After training, the param.mat is generated, saving the parameters used for testing. new_test.m is used to test our linear model, generating the related document id. Use id2doc to convert those ids into text.
