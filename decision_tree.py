#!/usr/bin/python3
import utils
import argparse
import os
from DT import DecisionTree

class MovieLensDC(DecisionTree):
    def __init__(self, datapath):
        self.datapath = datapath
        self.movies = utils.toDf(os.path.join(self.datapath, 'movies.csv'))
        self.ratings = utils.toDf(os.path.join(self.datapath, 'ratings.csv'))
        self.tags = utils.toDf(os.path.join(self.datapath, 'tags.csv'))
        self.abt = self.movies[['movieId', 'title']].copy()

    def split(self):
        pass

    def feature_collect(self):
        self.collect_target()
        self.collect_descriptives()

    def collect_target(self):        
        self.rate_global_mean = self.ratings['rating'].mean()
        average_rating = self.ratings[['movieId','rating']].copy().groupby(by='movieId').mean()
        average_rating['good_or_bad'] = average_rating['rating'] > self.rate_global_mean
        average_rating['good_or_bad'] = average_rating['good_or_bad'].astype('int')     
        self.abt = self.abt.merge(average_rating, left_on='movieId', right_on='movieId')
        print(self.abt.sample(n=10))

    def collect_descriptives(self):
        self.collect_tags()

    def collect_tags(self):        
        movie_tag_counts = self.tags[['movieId', 'tag']].copy().groupby(by='movieId').count()
        self.abt = self.abt.merge(movie_tag_counts, left_on='movieId', right_on='movieId')
        print(self.abt.head(n=10))

    def train(self):
        pass

    def test(self):
        pass

    def evaluate(self):
        pass                        

    def report(self):
        pass


def init_data(data_path):
    return MovieLensDC(data_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder", help='the folder which contains dataset.')
    return parser.parse_args()


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


if __name__ == '__main__':
    main()
  