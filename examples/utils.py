import yaml

def parse_setup(setup_file: str):
    with open(setup_file, "r") as setup_stream:
        setup = yaml.safe_load(setup_stream)
    return setup