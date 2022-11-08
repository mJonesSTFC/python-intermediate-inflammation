"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains
inflammation data for a single patient taken over a number of days
and each column represents a single day across all patients.
"""

import numpy as np


def load_csv(filename):
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    """
    return np.loadtxt(fname=filename, delimiter=',')


def daily_mean(data):
    """Calculate the daily mean of a 2D inflammation data array.
    
    :param data: Data array that will be processed
    """
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2D inflammation data array.
    
    :param data: Data array that will be processed
    """
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2D inflammation data array.
    
    :param data: Data array that will be processed
    """
    return np.min(data, axis=0)

def patient_normalise(data):
    """Normalise patient data from the 2D data array
    
    :param data: Data array that will be processed
    """
    if np.any(data < 0):
        raise ValueError('Inflammation values should not be negative')
    maxes = np.nanmax(data, axis = 1)
    with np.errstate(invalid = 'ignore', divide = 'ignore'):
        normalised = data / maxes[:, np.newaxis]
    normalised[np.isnan(normalised)] = 0
    normalised[normalised < 0]= 0
    return normalised

class Observation:
    def __init__(self, day, value):
        self.day = day
        self.value = value

    def __str__(self):
        return str(self.value)


class Person:
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name

class Patient(Person):
    """Patient with a name and observations.
    """
    def __init__(self, name: str):
        """_summary_
        """
        super().__init__(name)
        self.observations = []

    @property
    def last_observation(self):
        return self.observations[-1]

    def add_observations(self, value, day=None):
        """_summary_

        Args:
            value (_type_): _description_
            day (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if day is None:
            if self.observations:
                day = self.observations[-1].day + 1
            else:
                day = 0

        new_observation = Observation(day, value)
        self.observations.append(new_observation)
        return new_observation
class Doctor(Person):
    def __init__(self, name: str):
        super().__init__(name)
        self.patients = []
    def add_patient(self, patient_name):
        self.patients.append(patient_name)