import yaml
import json

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