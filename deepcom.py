import numpy as np
import torch
import sys
import json
import os
from itertools import product

__version__ = '0.1.0'


class SurpAlgorithm:
    def __init__(self, input_normalized, laplacian_lambda, beta):
        self.input_normalized = input_normalized
        self.encoder_lambda = laplacian_lambda
        self.decoder_lambda = laplacian_lambda
        self.beta = beta
        self.num_variable = input_normalized.size
        self.output_normalized = np.zeros(self.num_variable)

    def encoder(self):
        larger_element_loc = np.where(
            self.input_normalized > 1 / self.encoder_lambda * np.log(self.num_variable / self.beta))
        if larger_element_loc[0].size == 0:
            self.encoder_lambda = 1 / np.mean(self.input_normalized)
            self.decoder_lambda = self.encoder_lambda
            loc_to_decoder = None
            print(
                f'Refreshment with lambda {self.encoder_lambda}, larger beta value is required.')
        else:
            loc_to_decoder = np.random.choice(larger_element_loc[0])
            self.input_normalized[loc_to_decoder] -= 1 / \
                self.encoder_lambda * np.log(self.num_variable / self.beta)

        self.encoder_lambda = (self.num_variable / (
            self.num_variable - np.log(self.num_variable / self.beta))) * self.encoder_lambda

        return loc_to_decoder

    def decoder(self, loc_to_decoder):
        if loc_to_decoder is not None:
            self.output_normalized[loc_to_decoder] += 1 / \
                self.decoder_lambda * np.log(self.num_variable / self.beta)

        self.decoder_lambda = (self.num_variable / (
            self.num_variable - np.log(self.num_variable / self.beta))) * self.decoder_lambda


def surp_algorithm(input, sparsity_target, beta_scale=0, iteration_target=0):
    """
    A sparse compression algorithm for Laplacian sequence.

    Example: 
    laplace_sequence = np.random.laplace(0, 0.1, 10000)
    laplace_sequence_decoded = dlcom.surp_algorithm(laplace_sequence, sparsity_target=0.7,beta_scale=100)

    Algorithms stops when sparsity target is satisfied or iteration number reaches the specified constraint.
    Defaut value of beta_scale = np.log(float(input.size)).
    """

    input = input.flatten()
    num_variable = float(input.size)
    input_normalized = np.abs(input) / np.linalg.norm(input, ord=1)
    lambda_laplacian = 1 / np.mean(input_normalized)

    if beta_scale == 0:
        beta = np.log(num_variable)
    else:
        beta = beta_scale * np.log(num_variable)

    algorithm_iteration = SurpAlgorithm(
        input_normalized, lambda_laplacian, beta)

    print('SuRP algorithm processing...')
    if iteration_target != 0:
        stop_condition = 'iteration_num < iteration_target'
    else:
        stop_condition = 'np.count_nonzero(algorithm_iteration.output_normalized) / num_variable < 1 - sparsity_target'

    iteration_num = 0
    while eval(stop_condition):
        loc_to_decoder = algorithm_iteration.encoder()
        algorithm_iteration.decoder(loc_to_decoder)
        if algorithm_iteration.encoder_lambda == np.inf or algorithm_iteration.encoder_lambda == np.inf:
            break
        iteration_num += 1

    print(
        f'beta value is {beta}, number of parameters is {input.size}, achieving sparsity {1 - np.count_nonzero(algorithm_iteration.output_normalized) / num_variable} with {iteration_num} iterations')

    return algorithm_iteration.output_normalized * np.linalg.norm(input, ord=1) * np.sign(input)


def model2params(model):
    """
    Convert model parameters as a flattened numpy array on CPU.

    Example: model2params(model)
    """
    parameters_collection = torch.tensor(
        []).to(next(model.parameters()).device)
    for parameters in model.parameters():
        parameters_collection = torch.cat(
            (parameters_collection, parameters.detach().reshape(-1)), dim=0)

    return parameters_collection.cpu().numpy()


def params2model(model, parameters):
    """
    Load a numpy array as model parameters.

    Example: params2model(model,parameters)
    """
    for key, value in model.state_dict().items():
        layer_parameters_num = value.numel()
        layer_parameters_pruned = parameters[0:layer_parameters_num]
        model.state_dict()[key].copy_(torch.tensor(layer_parameters_pruned.reshape(
            value.shape), requires_grad=True).to(next(model.parameters()).device))
        parameters = np.delete(parameters, range(layer_parameters_num))

    return model


def moving_average(sequence, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(sequence, window, 'same')


def mutual_info(x_sequence, y_sequence):
    """
    Calculate the mutual information of two random variables.
    Each variable is represented by a numpy sequence.

    Example: 
    x_sequence = np.random.normal(0,1-np.sqrt(0.5),sequence_lenth)
    y_sequence = x_sequence+np.random.normal(0,np.sqrt(0.5),sequence_lenth)
    mutual_info(x_sequence,y_sequence)
    [which should equal to 0.5*np.log2(1/np.sqrt(0.5))=0.25]
    """
    x_sequence, y_sequence = x_sequence.flatten(), y_sequence.flatten()
    x_sequence_len, y_sequence_len = x_sequence.size, y_sequence.size
    assert x_sequence_len == y_sequence_len, 'Length of two sequences must be the same'

    x_intervals = np.min([50, int(10/np.std(x_sequence))]) + \
        50  # a param to tradeoff pdf accuracy
    y_intervals = np.min([50, int(10/np.std(y_sequence))])+50

    # for one dimensional distribution Px and Py
    x_pdf = np.histogram(x_sequence, bins=x_intervals)[0]/float(x_sequence_len)
    y_pdf = np.histogram(y_sequence, bins=y_intervals)[0]/float(y_sequence_len)
    x_pdf = moving_average(x_pdf, 2)  # smoothing pdf curve
    y_pdf = moving_average(y_pdf, 2)

    # for two dimensional distribution Pxy
    xy_pdf = np.histogram2d(x_sequence, y_sequence, bins=[
                            x_intervals, y_intervals])[0]/float(x_sequence_len)
    xy_pdf = moving_average(xy_pdf.flatten(), 2).reshape(xy_pdf.shape)

    # Calculate mutual information
    # I(X;Y) = \sum_{x,y}p(x,y)log(p(x,y))-\sum_{x,y}p(x,y)log(p(x)p(y))
    minuend = np.sum(xy_pdf*np.log2(xy_pdf+1e-16))
    x_pdf_2d = np.repeat(x_pdf.reshape(x_intervals, 1),
                         y_intervals, axis=1)  # repeat each x collumn
    y_pdf_2d = np.repeat(y_pdf.reshape(1, y_intervals),
                         x_intervals, axis=0)  # repeat each y row
    subtrahend = np.sum(xy_pdf*np.log2(x_pdf_2d*y_pdf_2d+1e-16))
    return minuend-subtrahend


if __name__ == '__main__':
    """
    Batch processing with enumerating argument values, based on argparse module.
    Learn more with configuration json file https://github.com/mh-lan/deepcom/blob/main/config.json
    """
    config_file_name = sys.argv[1]
    with open(config_file_name, 'r') as config_file:
        jsondata = ''.join(
            line for line in config_file if not line.startswith('//'))
        config_data = json.loads(jsondata)

    script_file_name = config_data["script"]
    config_data.pop("script")  # remain parameters only

    batch_processing_info = f'Batch processing script {script_file_name} with parameters file {config_file_name}'
    print(batch_processing_info)
    print('-'*len(batch_processing_info))

    params_key = []
    params_possible_values = []  # for params with valus, e.g., --params_key params_value
    params_action = []  # for params without value, e.g., --action
    for key in config_data:
        if config_data[key] == []:
            params_action.append(key)
        else:
            params_key.append(key)
            # collect each value for enumeration
            params_possible_values.append(config_data[key])

    command = f'python {script_file_name}'
    for action in params_action:
        command += f' --{action}'

    if params_key == []:  # with only action parameters
        os_command = command
        print('-'*6+os_command+'-'*6)
        os.system(os_command)
    else:
        # enumerate values for each parameter
        for params_value in product(*params_possible_values):
            assert len(params_key) == len(
                params_value), 'Length of parameters and their values must be the same'
            os_command = command
            for index in range(len(params_key)):
                os_command += f' --{params_key[index]} {params_value[index]}'
            print('-'*6+os_command+'-'*6)
            os.system(os_command)
