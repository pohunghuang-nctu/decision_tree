#!/usr/bin/python3
import abc

class DecisionTree(abc.ABC):
    @abc.abstractmethod
    def split(self):
        return NotImplemented

    @abc.abstractmethod
    def feature_collect(self):
        return NotImplemented

    @abc.abstractmethod
    def train(self):
        return NotImplemented

    @abc.abstractmethod
    def test(self):
        return NotImplemented

    @abc.abstractmethod
    def evaluate(self):
        return NotImplemented                        

    @abc.abstractmethod
    def round_summary(self):
        return NotImplemented 


def init_data(data_path):
    return MovieLensDC(data_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder", help='the folder which contains dataset.')


def main():
    args = get_args()
    # initial a dataset by giving specific data set path
    data = init_data(args.data_folder)
    # separate data as training set and testing set
    # this is one-shot, and need to persist. 
    # if split flag detected, not doing it again. 
    data.split()
    # join descriptive and target features, 
    # if source data not changed, and ready flag on, 
    # then not need to do it again. 
    data.feature_collect()
    # training stage
    data.train()
    # testing stage
    data.test()
    # evaluation stage
    data.evaluate()
    # report stage
    data.report()

