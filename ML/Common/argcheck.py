import collections

import math


def throw_on_none(argument, parameter_name):
    if argument is None:
        raise TypeError("The argument {0} was None.".format(parameter_name))


def throw_on_negative(argument, parameter_name):
    if argument < 0:
        raise TypeError("The argument {0} was negative.".format(parameter_name))


def throw_on_non_positive(argument, parameter_name):
    if argument <= 0:
        raise TypeError("The argument {0} was not positive.".format(parameter_name))


def throw_on_positive(argument, parameter_name):
    if argument > 0:
        raise TypeError("The argument {0} was positive.".format(parameter_name))


def throw_on_non_negative(argument, parameter_name):
    if argument >= 0:
        raise TypeError("The argument {0} was not negative.".format(parameter_name))


def throw_if_not_in_range(argument, min_inclusive, max_inclusive, parameter_name):
    if argument < min_inclusive or argument > max_inclusive:
        raise TypeError(
            "The argument {0} was not between {1} and {2}.".format(parameter_name, min_inclusive, max_inclusive))


def throw_on_greater_than(argument, value, parameter_name):
    if argument > value:
        raise TypeError("The argument {0} was larger than {1}.".format(parameter_name, value))


def throw_on_lower_than(argument, value, parameter_name):
    if argument < value:
        raise TypeError("The argument {0} was lower than {1}.".format(parameter_name, value))


def throw_on_nan(argument, parameter_name):
    if math.isnan(argument):
        raise TypeError("The argument {0} was nan.".format(parameter_name))


def throw_on_inf(argument, parameter_name):
    if math.isinf(argument):
        raise TypeError("The argument {0} was inf.".format(parameter_name))

def throw_on_nan_or_inf(argument, parameter_name):
    if math.isnan(argument) or math.isinf(argument):
        raise TypeError("The argument {0} was nan or inf.".format(parameter_name))


def throw_on_equal(argument, value, parameter_name):
    if argument == value:
        raise TypeError("The argument {0} was equal to {1}".format(parameter_name, value))


def throw_on_close(argument, value, parameter_name):
    if math.isclose(argument, value):
        raise TypeError("The argument {0} was close to {1}".format(parameter_name, value))


def throw_on_not_close(argument, value, parameter_name):
    if not math.isclose(argument, value):
        raise TypeError("The argument {0} was not close to {1}".format(parameter_name, value))


def throw_on_non_iterable(argument, parameter_name):
    if not isinstance(argument, collections.Iterable):
        raise TypeError("The argument {0} was not iterable.".format(parameter_name))


def throw_on_empty(argument, parameter_name):
    if len(argument) == 0:
        raise TypeError("The argument {0} was empty.".format(parameter_name))


def throw_on_none_or_empty(argument, name_of_argument):
    if argument is None or len(argument) == 0:
        raise TypeError(name_of_argument + " was None or empty.")


def throw_if_false(condition, message):
    if not condition:
        raise TypeError(message)


def throw_if_true(condition, message):
    if condition:
        raise TypeError(message)
