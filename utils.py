import os


def clear_stdout():
    os.system("clear")


class dotdict(dict):
    """ To access attributes of a dict via my_dict.attribute"""

    def __getattr__(self, name):
        return self[name]
