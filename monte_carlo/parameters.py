"""
Parameters for Monte-Carlo 
"""
from monte_carlo.moves import get_all_move_types 
import json

class Parameter(object):
    def __init__(self, name, default, param_type, description, allowed_values=None):
        self._name = name
        self._param_type = param_type
        self._description = description
        self._allowed_values = allowed_values
        self.value = default

    def __str__(self):
        strng_rep = [
            " Name: %s" % self.name,
            " Description: %s" % self.description,
            " Value: %s" % str(self.value),
        ]
        if self._allowed_values is not None:
            nvalues = len(self._allowed_values)
            allowed = " Allowed values: " + ("%s "*nvalues) % tuple(self._allowed_values)
            strng_rep.append(allowed)

        return "\n".join(strng_rep)
            
    @property
    def value(self):
        """ value of parameter """
        return self._value

    @value.setter
    def value(self, value):
        if not isinstance(value, self._param_type):
            raise TypeError("Parameter %s must be a %s" % (self._name, str(self._type)))
        if self._allowed_values is not None and value not in self._allowed_values:
            raise ValueError("unallowed value for parameter %s" % self.name) 

        self._value = value

    @property
    def description(self):
        """ description of property """
        return self._description

    @property
    def name(self):
        """ name of parameter """
        return self._name

    @property
    def type(self):
        """ type of parameter value """
        return self._param_type

    @property
    def allowed_values(self):
         """ allowed values if set """
         return self._allowed_values

class MonteCarloParams(object):
    """
    Parameters for Monte-Carlo job 
    """
    def __init__(self):
        
        parameters = [
            Parameter(
                "max_steps",
                50000,
                int,
                "number of MC steps to perform",
            ),
            Parameter(
                "batch_size",
                50,
                int,
                "number of points to collect before training again"
            ),
            Parameter(
                "variance_thresh",
                0.5,
                float,
                "all geometries with atomic variance greater than this are written"
            ),
            Parameter(
                "beta",
                #1.0/20.0,
                1.6,
                float,
                "beta = 1/kT (kcal/mol) parameter for control of sampling"
            ),
            Parameter(
                "Pmin",
                1.0e-5, 
                float,
                "Used to set alpha = -Log(Pmin)/sigma_max"
            ),
            Parameter(
                "sigma_max",
                5.0,
                float,
                "cap on variance from committee"
            ),
            Parameter(
                "disp_max",
                1.0,
                float,
                "maximum displacement step (A)"
            ),
            Parameter(
                "stretch_max",
                0.1,
                float,
                "maximum bond stretch step (A)"
            ),
            Parameter(
                "bend_max",
                5.0,
                float,
                "maximum angle step (degrees)"
            ),
            Parameter(
                "torsion_max",
                20.0,
                float,
                "maximum torsion step (degrees)"
            ),
            Parameter(
                "trans_max",
                0.1,
                float,
                "maximum rigid translation step (A)"
            ),
            Parameter(
                "rot_max",
                20.0,
                float,
                "maximum rigid rotation step (degrees)"
            ),
            Parameter(
                "move_type",
                get_all_move_types(),
                list,
                "MC Move types",
            )
        ]

        # store as attributes
        self._parameters = {}
        for param in parameters:
            self._parameters[param.name] = param

        param_descriptions = map(str, self._parameters.values())
        self.__doc__ += "\n Available parameters via calls to getValue/setValue\n\n" + "\n\n".join(param_descriptions)

    def getValue(self, name):
        """
        get value of parameter
        See help on an instance of class to see parameter list
        """
        return self._parameters[name].value

    def setValue(self, name, value):
        """
        set value of parameter
        See help on an instance of class to see parameter list
        """
        self._parameters[name].value = value

    def setFromConfig(self, filename):
        """
        Set parameters from a config file
        """
        assert filename.endswith(".json")

        with open(filename, "r") as fin:
            config_file = json.load(fin)
        for name in config_file:
            if name not in self._parameters:
                raise RuntimeError("Unrecognized parameter in config file %s" % name)
            prm = self._parameters[name]
            self.setValue(name, prm.type(config_file[name]))

    def saveConfig(self, filename):
        """
        Write all parameters to a config file
        """
        assert filename.endswith(".json")

        params = {param.name: param.value for param in self._parameters.values()}
        config_file = json.dumps(params, indent=4)
        with open(filename, "w") as fout:
            fout.write(config_file)

            
if __name__ == "__main__":
    # print the default parameters
    params = MonteCarloParams() 
    print(params.__doc__)
