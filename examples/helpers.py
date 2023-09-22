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