#!/usr/bin/python3
import utils
import argparse
import os
import pandas as pd
from DT import DecisionTree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate
import pydotplus
from IPython.display import Image
import numpy as np
import sys
from sklearn.metrics import recall_score


class MovieLensDC(DecisionTree):
    def __init__(self, datapath):
        self.datapath = datapath
        self.movies = utils.toDf(os.path.join(self.datapath, 'movies.csv'))
        self.ratings = utils.toDf(os.path.join(self.datapath, 'ratings.csv'))
        self.tags = utils.toDf(os.path.join(self.datapath, 'tags.csv'))
        self.abt = self.movies[['movieId', 'title']].copy()

    def split(self):
        self.training_set, self.test_set = train_test_split(self.abt, test_size=0.3)
        self.validate_data(self.training_set)
        self.validate_data(self.test_set)
        # print(self.training_set.head())
        # print(self.test_set.head())

    def validate_data(self, dataset):
        print(dataset.isnull().sum())
        if (dataset.isnull().any().any()):
            print('NAN inside.\n')
            dataset.fillna(value=0, inplace=True)
        print('Any Nan inside after fixing? %s\n' % dataset.isnull().any().any())
    
    def feature_collect(self):
        self.collect_target()
        self.collect_descriptives()
        self.abt.reset_index(inplace=True)
        self.validate_data(self.abt)

    def collect_target(self):        
        self.rate_global_mean = self.ratings['rating'].mean()
        average_rating = self.ratings[['movieId','rating']].copy().groupby(by='movieId').mean()
        average_rating['good_or_bad'] = average_rating['rating'] > self.rate_global_mean
        average_rating['good_or_bad'] = average_rating['good_or_bad'].astype('int')     
        self.abt = self.abt.merge(average_rating, left_on='movieId', right_on='movieId')
        print(self.abt.head(n=10))

    def collect_descriptives(self):
        self.collect_tags()
        self.collect_interactions()

    def collect_interactions(self):
        # concate ratings and tags
        interaction_1 = self.ratings[['movieId', 'timestamp']]
        interaction_2 = self.tags[['movieId', 'timestamp']]
        interactions = pd.concat([interaction_1, interaction_2], ignore_index=True)
        movie_total_interact = interactions.groupby('movieId').size().reset_index().rename(columns={0:'rates_and_tags'})
        interactions['date'] = pd.to_datetime(interactions['timestamp'], unit='s')
        interactions['date'] = interactions['date'].apply(lambda x: x.strftime('%Y_%m_%d'))
        interactions.drop(columns=['timestamp'], inplace=True)
        movie_date_interact = interactions.groupby(['movieId', 'date']).size().reset_index().rename(columns={0:'interact_count'}).drop(columns=['date'])
        movie_max_interact = movie_date_interact.groupby('movieId').max().reset_index().rename(columns={'interact_count': 'max'})
        movie_mean_interact = movie_date_interact.groupby('movieId').mean().reset_index().rename(columns={'interact_count': 'mean'})
        movie_var_interact = movie_date_interact.groupby('movieId').var().reset_index().rename(columns={'interact_count': 'var'})
        interact_stat = movie_max_interact.merge(movie_mean_interact, left_on='movieId', right_on='movieId')
        interact_stat = interact_stat.merge(movie_var_interact, left_on='movieId', right_on='movieId')
        interact_stat = interact_stat.merge(movie_total_interact, left_on='movieId', right_on='movieId')
        # print(movie_total_interact.head(n=50))
        self.abt = self.abt.merge(interact_stat, left_on='movieId', right_on='movieId')
        print(self.abt.head(n=50))        

    def aggregate_tags(self):
        self.tags['tag'] = self.tags['tag'].apply(lambda x: x.replace(' ', '_').upper() if isinstance(x, str) else str(x))
        movieIds = []
        aggr_tags = []
        for id, group in self.tags.groupby('movieId'):
            movieIds.append(id)
            aggr_tags.append(group['tag'].str.cat(sep=' '))
        v = CountVectorizer()
        movie_tags_sparse = pd.DataFrame.sparse.from_spmatrix(v.fit_transform(aggr_tags), columns=v.get_feature_names())
        movie_tags_sparse['movieId'] = movieIds
        movie_tags_sparse['aggr_tags'] = aggr_tags
        return movie_tags_sparse
    
    def collect_tags(self):        
        movie_tags = self.aggregate_tags()
        # print(movie_tags.head(n=10))
        print('total %d different tags.' % len(self.tags['tag'].unique().tolist()))     
        movie_tag_counts = self.tags[['movieId', 'tag']].copy().groupby(by='movieId').count()
        self.abt = self.abt.merge(movie_tag_counts, left_on='movieId', right_on='movieId').rename(columns={'tag': 'tag_count'})
        self.abt = self.abt.merge(movie_tags.drop(columns=['aggr_tags']), left_on='movieId', right_on='movieId')
        print(self.abt.head(n=10))

    def XY(self, dataset):
        Y = dataset['good_or_bad']
        X = dataset.drop(columns=['movieId', 'title', 'rating', 'good_or_bad'])
        X = X.clip(-1e11,1e11)
        Y = Y.clip(-1e11,1e11)
        print('Validating X....\n')
        self.validate_data(X)
        return X, Y

    def train(self):
        X, Y = self.XY(self.training_set)
        # print(X.head(n=20))
        dc = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_leaf=3)
        scoring_types = ['accuracy', 'precision', 'recall']
        scores = cross_validate(dc, X, Y, cv=5, return_estimator=True, scoring=scoring_types, return_train_score=True, n_jobs=5)
        print(scores.keys())
        print('==== Accuracy ==== \n')
        print(scores['test_accuracy'])        
        print('==== Recal ==== \n')
        print(scores['test_recall'])
        print('==== Precision ==== \n')
        print(scores['test_precision'])
        print('==== Train Accuracy ==== \n')
        print(scores['train_accuracy'])        
        print('==== Train Recall ==== \n')
        print(scores['train_recall'])
        print('==== Train Precision ==== \n')
        print(scores['train_precision'])        
        # self.scores = cross_val_score(dc, X, Y, cv=5)
        # print(self.scores)
        # print(type(scores['estimator']))
        estimators = scores['estimator']
        for dctree in estimators:
            dot_data = tree.export_graphviz(dctree, feature_names=X.columns, filled=True, out_file=None)
            graph = pydotplus.graph_from_dot_data(dot_data)
            graph.write_pdf("dc_%d.pdf" % estimators.index(dctree))

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
    # join descriptive and target features, 
    # if source data not changed, and ready flag on, 
    # then not need to do it again. 
    data.feature_collect()
    data.split()
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
  