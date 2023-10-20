import numpy as np
import casadi
import os, sys

def load_forces_path():

    print_paths = ["PYTHONPATH"]
    # Is forces in the python path?
    try:
        import forcespro.nlp
        print('Forces found in PYTHONPATH')

        return
    except:
        pass

    paths = [os.path.join(os.path.expanduser("~"), "forces_pro_client"),
             os.path.join(os.getcwd(), "forces"),
             os.path.join(os.getcwd(), "../forces"),
             os.path.join(os.getcwd(), "forces/forces_pro_client"),
             os.path.join(os.getcwd(), "../forces/forces_pro_client")]
    for path in paths:
        if check_forces_path(path):
            return
        print_paths.append(path)

    print('Forces could not be imported, paths tried:\n')
    for path in print_paths:
        print('{}'.format(path))
    print("\n")


def check_forces_path(forces_path):
    # Otherwise is it in a folder forces_path?
    try:
        if os.path.exists(forces_path) and os.path.isdir(forces_path):
            sys.path.append(forces_path)
        else:
            raise IOError("Forces path not found")

        import forcespro.nlp
        print('Forces found in: {}'.format(forces_path))

        return True
    except:
        return False


class ParameterStructure:

    def __init__(self):
        self.parameters = dict()
        self.organization = dict() # Lists parameter grouping and indices
        self.param_idx = 0

    def add_parameter(self, name):
        self.organization[self.param_idx] = 1
        self.parameters[self.param_idx] = name
        setattr(self, name+ "_index", self.param_idx)
        self.param_idx += 1

    def add_multiple_parameters(self, name, amount):
        self.organization[self.param_idx] = amount
        for i in range(amount):
            self.parameters[self.param_idx] = name + "_" + str(i)
            setattr(self, name + "_" + str(i) + "_index", self.param_idx)
            self.param_idx += 1


    def n_par(self): # Does not need + 1, because it is always increased
        return self.param_idx

    def __str__(self):
        result = "Parameter Structure:\n"
        for idx, amount in self.organization.items():
            if amount == 1:
                result += "{}\t:\t{}\n".format(idx, self.parameters[idx])
            else:
                result += "{}\t:\t{} x{}\n".format(idx, '_'.join(self.parameters[idx].split('_')[:-1]), amount)



        result += "--------------------\n"
        return result

    # When operating, retrieve the weights from param
    def load_params(self, params):
        for key, name in self.parameters.items(): # This is a parameter name
            setattr(self, name, params[getattr(self, name+ "_index")]) # this is an index

class WeightStructure:

    # When defining the structure we define a structure with the weights as variables
    def __init__(self, parameters, weight_list):
        self.weight_list = weight_list

        for idx, weight in enumerate(weight_list):
            setattr(self, weight + "_index", parameters.param_idx)
            parameters.add_parameter(weight + "_weight")

        self.npar = len(weight_list)
        # parameters

    # When operating, retrieve the weights from param
    def set_weights(self, param):
        for weight in self.weight_list:
            # print(weight + ": " + str(getattr(self, weight+"_index")))
            setattr(self, weight , param[getattr(self, weight+"_index")])


def rotation_matrix(angle):
    return np.array([[casadi.cos(angle), -casadi.sin(angle)],
                      [casadi.sin(angle), casadi.cos(angle)]])

class SplineParameters:

    def __init__(self, param, current_spline, spline_nr):
        spline_index = spline_nr * 8
        self.current_spline = current_spline

        self.x_a = param[spline_index]
        self.x_b = param[spline_index + 1]
        self.x_c = param[spline_index + 2]
        self.x_d = param[spline_index + 3]

        self.y_a = param[spline_index + 4]
        self.y_b = param[spline_index + 5]
        self.y_c = param[spline_index + 6]
        self.y_d = param[spline_index + 7]

    def compute_path(self, spline_index):
        self.path_x = self.x_a * (spline_index - self.current_spline) ** 3 + \
                      self.x_b * (spline_index - self.current_spline) ** 2 + \
                      self.x_c * (spline_index - self.current_spline) + \
                      self.x_d

        self.path_y = self.y_a * (spline_index - self.current_spline) ** 3 + \
                      self.y_b * (spline_index - self.current_spline) ** 2 + \
                      self.y_c * (spline_index - self.current_spline) + \
                      self.y_d

        self.path_dx = 3 * self.x_a * (spline_index - self.current_spline) ** 2 + \
                       2 * self.x_b * (spline_index - self.current_spline) + \
                       self.x_c

        self.path_dy = 3 * self.y_a * (spline_index - self.current_spline) ** 2 + \
                       2 * self.y_b * (spline_index - self.current_spline) + \
                       self.y_c