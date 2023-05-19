import yaml
import json
import numpy as np
from loguru import logger as printer

def readYaml(path):
        with open(path, 'r') as stream:
            try:
                parsed_yaml=yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        return parsed_yaml

def readJson(path):
    with open(path, 'r') as stream:
        try:
            search_space = json.load(stream)
        except Exception as e:
            print(e)

    return search_space

def writeJson(data, path ):
        try:
            with open(path, "w") as outfile:
                json.dump(data, outfile)
        except Exception as e:
            print(e)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            printer.warning("Early stopping counter: {}/{}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                return True
        return False