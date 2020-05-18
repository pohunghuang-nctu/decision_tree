#!/usr/bin/python3
import utils
import argparse
import os
import pandas as pd
from DT import DecisionTree
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import recall_score, precision_score, accuracy_score
import pydotplus
import numpy as np
import sys
import json
import matplotlib.pyplot as plt


class MovieLensDC(DecisionTree):
    def __init__(self, args, round_no):
        self.datapath = args.data_folder
        self.hashTags = args.hash_tags
        self.withTags = args.with_tags
        self.output = args.output_folder
        self.enableKbeans = args.kbeans
        self.round = round_no
        self.depth = args.depth
        self.min_leaf_samples = args.min_leaf_samples
        self.tag_nums = args.tag_nums
        self.movies = utils.toDf(os.path.join(self.datapath, 'movies.csv'))
        print("total movies to be predict, %d" % int(len(self.movies)))
        self.ratings = utils.toDf(os.path.join(self.datapath, 'ratings.csv'))
        self.tags = utils.toDf(os.path.join(self.datapath, 'tags.csv'))
        self.abt = self.movies[['movieId', 'title', 'genres']].copy()

    def split(self):
        print('final movies to be evaluated: %d' % int(len(self.abt)))
        self.training_set, self.test_set = train_test_split(self.abt, test_size=0.3)
        self.validate_data(self.training_set)
        self.validate_data(self.test_set)
        # print(self.training_set.head())
        # print(self.test_set.head())

    def validate_data(self, dataset):
        # print(dataset.isnull().sum())
        if (dataset.isnull().any().any()):
            # print('NAN inside.\n')
            dataset.fillna(value=0, inplace=True)
        # print('Any Nan inside after fixing? %s\n' % dataset.isnull().any().any())
    
    def feature_collect(self):
        self.collect_target()
        self.collect_descriptives()
        self.abt.reset_index(drop=True, inplace=True)
        self.validate_data(self.abt)
        print('=== self.abt memory usage ===')
        print('df count: %d \n' % self.abt.memory_usage(index=True, deep=True).sum())
        # print(sys.getsizeof(self.abt))

    def collect_target(self):        
        average_rating = self.ratings[['movieId','rating']].copy().groupby(by='movieId').mean()
        self.rate_global_mean = average_rating['rating'].mean()
        average_rating['good_or_bad'] = average_rating['rating'] > self.rate_global_mean
        average_rating['good_or_bad'] = average_rating['good_or_bad'].astype('int') 
        # print(self.abt)    
        self.abt = self.abt.merge(average_rating, left_on='movieId', right_on='movieId')
        # print(self.abt)   
        # print(self.abt.head(n=10))

    def collect_descriptives(self):
        if (self.withTags):
            self.collect_tags()
        self.vectorize_genres()
        self.collect_interactions()

    def collect_interactions(self):
        # concate ratings and tags
        interaction_1 = self.ratings[['movieId', 'timestamp']]
        interaction_2 = self.tags[['movieId', 'timestamp']]
        interactions = pd.concat([interaction_1, interaction_2], ignore_index=True)
        movie_total_interact = interactions.groupby('movieId').count().reset_index().rename(columns={'timestamp':'total_interacts'})
        interactions['date'] = pd.to_datetime(interactions['timestamp'], unit='s')
        interactions['date'] = interactions['date'].apply(lambda x: x.strftime('%Y_%m_%d'))
        interactions.drop(columns=['timestamp'], inplace=True)
        # movie_date_interact = interactions.groupby(['movieId', 'date']).size().reset_index().rename(columns={'date':'interact_count'})
        movie_date_interact = interactions.groupby(['movieId', 'date']).size().reset_index().rename(columns={0:'interact_count'}).drop(columns=['date'])
        movie_max_interact = movie_date_interact.groupby('movieId').max().reset_index().rename(columns={'interact_count': 'max_day_interacts'})
        movie_mean_interact = movie_date_interact.groupby('movieId').mean().reset_index().rename(columns={'interact_count': 'mean_day_interacts'})
        movie_var_interact = movie_date_interact.groupby('movieId').var().reset_index().rename(columns={'interact_count': 'var_day_interacts'})
        interact_stat = movie_max_interact.merge(movie_mean_interact, how='left', left_on='movieId', right_on='movieId')
        interact_stat = interact_stat.merge(movie_var_interact, how='left', left_on='movieId', right_on='movieId')
        interact_stat = interact_stat.merge(movie_total_interact, how='left', left_on='movieId', right_on='movieId')
        self.validate_data(interact_stat)
        # print(interact_stat)
        if self.enableKbeans:
            interact_stat = self.kbean_transform(interact_stat)
            print(interact_stat)
        self.abt = self.abt.merge(interact_stat, how='left', left_on='movieId', right_on='movieId')
        # print(self.abt)        

    def kbean_transform(self, interact_stat):
        self.validate_data(interact_stat)
        # print('== before kbean ==')
        # print(interact_stat.head(n=20))
        quant_est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
        uni_est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
        # est2 = KBinsDiscretizer(n_bins=2, encode='ordinal')
        interact_stat['total_interacts'] = quant_est.fit_transform(pd.DataFrame(interact_stat['total_interacts']))
        interact_stat['max_day_interacts'] = uni_est.fit_transform(pd.DataFrame(interact_stat['max_day_interacts']))
        interact_stat['mean_day_interacts'] = uni_est.fit_transform(pd.DataFrame(interact_stat['mean_day_interacts']))
        interact_stat['var_day_interacts'] = uni_est.fit_transform(pd.DataFrame(interact_stat['var_day_interacts']))
        interact_stat.drop(columns=['mean_day_interacts', 'var_day_interacts'], inplace=True)
        return interact_stat

    def vectorize_genres(self):
        v = CountVectorizer()
        self.abt['genres'] = self.abt['genres'].apply(lambda x: x.replace(' ', '_').replace('-', '_').upper() if isinstance(x, str) else str(x))
        # print(self.abt)
        genres_sm = pd.DataFrame.sparse.from_spmatrix(v.fit_transform(self.abt['genres']), columns=v.get_feature_names())
        self.abt = pd.concat([self.abt, genres_sm], axis=1, sort=False)
        # print(self.abt)
        # print(self.abt.head(n=30))

    def prune_tags(self):
        self.tags['tag'] = self.tags['tag'].apply(lambda x: "__tag__" + (x.replace(' ', '_').replace('-', '_').upper() if isinstance(x, str) else str(x)))
        total_tags = len(self.tags)
        print('rows of tags: %d.\n' % total_tags)
        sort_by_count = self.tags.groupby('tag').count().reset_index().drop(columns=['userId', 'timestamp']).rename(columns={'movieId': 'count'}).sort_values('count', ascending=False).reset_index(drop=True)       
        sort_by_count['cumsum'] = sort_by_count['count'].cumsum()
        sort_by_count['percent'] = sort_by_count['cumsum'] * 100 /total_tags
        pruned_tags = sort_by_count[:self.tag_nums]
        print(pruned_tags)
        self.tags = self.tags[self.tags['tag'].isin(pruned_tags['tag'])]
        print('rows of pruned_tags: %d\n' % len(pruned_tags))   
        print(self.tags)     
        # sys.exit(0)
    
    def aggregate_tags(self):
        self.prune_tags()
        self.tags['tag'] = self.tags['tag'].apply(lambda x: "__tag__" + (x.replace(' ', '_').replace('-', '_').upper() if isinstance(x, str) else str(x)))
        movieIds = []
        aggr_tags = []
        for id, group in self.tags.groupby('movieId'):
            movieIds.append(id)
            aggr_tags.append(group['tag'].str.cat(sep=' '))
        if self.hashTags:
            v = HashingVectorizer(norm='l1', n_features=2**10)
            movie_tags_sparse = pd.DataFrame.sparse.from_spmatrix(v.fit_transform(aggr_tags))
        else:
            v = CountVectorizer()
            movie_tags_sparse = pd.DataFrame.sparse.from_spmatrix(v.fit_transform(aggr_tags))
            # movie_tags_sparse = pd.DataFrame.sparse.from_spmatrix(v.fit_transform(aggr_tags), columns=v.get_feature_names())
        movie_tags_sparse['movieId'] = movieIds
        # movie_tags_sparse['aggr_tags'] = aggr_tags
        # movie_tags_sparse[movie_tags_sparse['movieId'] == 7153].to_csv('7153.csv')
        return movie_tags_sparse
    
    def collect_tags(self):        
        movie_tags = self.aggregate_tags()
        # print(movie_tags.head(n=10))
        print('total %d different tags.' % len(self.tags['tag'].unique().tolist())) 
        print("before merge tag, ", len(self.abt))    
        movie_tag_counts = self.tags[['movieId', 'tag']].copy().groupby(by='movieId').count()
        self.abt = self.abt.merge(movie_tag_counts, how='left', left_on='movieId', right_on='movieId').rename(columns={'tag': 'tag_count'})
        self.abt = self.abt.merge(movie_tags, how='left',left_on='movieId', right_on='movieId')
        print("after merge tag, ", len(self.abt))
        # sys.exit(0)  
        # print(self.abt.head(n=10))

    def XY(self, dataset):
        Y = dataset['good_or_bad']
        X = dataset.drop(columns=['movieId', 'title', 'rating', 'good_or_bad', 'genres'])
        self.features = list(X.columns)
        # print(X[X.index.duplicated()])
        X = X.clip(-1e11,1e11)
        Y = Y.clip(-1e11,1e11)        
        if 'index' in X:
            # print('why index in X?')
            X.drop(columns=['index'], inplace=True)

        print('Validating X....\n')
        self.validate_data(X)
        return X, Y

    def display_scores(self, scores):
        self.validation_acc =  scores['test_accuracy']
        self.validation_recall =  scores['test_recall']
        self.validation_precision =  scores['test_precision']
        self.train_acc =  scores['train_accuracy']
        self.train_recall =  scores['test_recall']
        self.train_precision =  scores['train_precision']
        self.estimators = scores['estimator']
        best_acc = 0.0
        best_acc_index = 0
        for acc in self.validation_acc:
            if acc > best_acc:
                best_acc = acc
                best_acc_index = np.where(self.validation_acc == best_acc)  
        self.best_model_index = best_acc_index[0][0] 
        self.feature_importance = self.estimators[self.best_model_index].feature_importances_.tolist()
        self.feature_importance_info()
        # print(self.features)
        # print(self.feature_importance)        
        print(scores.keys())
        print('==== Accuracy #%d ==== \n' % self.round)
        print(scores['test_accuracy'])
        print('==== Recal #%d ==== \n' % self.round)
        print(scores['test_recall'])
        print('==== Precision #%d ==== \n' % self.round)
        print(scores['test_precision'])
        # print('==== NMSE #%d ==== \n' % self.round)
        # print(scores['test_neg_mean_squared_error'])        
        print('==== Train Accuracy #%d ==== \n' % self.round)
        print(scores['train_accuracy'])   
        print('==== Train Recall #%d ==== \n' % self.round)
        print(scores['train_recall'])
        print('==== Train Precision #%d ==== \n' % self.round)
        print(scores['train_precision']) 
        # print('==== Train NMSE #%d ==== \n' % self.round)
        # print(scores['train_neg_mean_squared_error'])                
        print("\nBest model index: %d \n" % self.best_model_index)              
    
    def train(self):
        X, Y = self.XY(self.training_set)
        # print(X.head(n=20))
        dc = DecisionTreeClassifier(criterion='gini', max_depth=self.depth, min_samples_leaf=self.min_leaf_samples)
        scoring_types = ['accuracy', 'precision', 'recall', 'neg_mean_squared_error']
        scores = cross_validate(dc, X, Y, cv=5, return_estimator=True, scoring=scoring_types, return_train_score=True, n_jobs=5)
        self.display_scores(scores)
        for dctree in self.estimators:
            dot_data = tree.export_graphviz(dctree, feature_names=X.columns, filled=True, out_file=None)
            graph = pydotplus.graph_from_dot_data(dot_data)
            graph.write_pdf(os.path.join(self.output, "dc_%d_%d.pdf" % (self.round, self.estimators.index(dctree))))

    def test(self):
        X, Y = self.XY(self.test_set)
        y_predict = self.estimators[self.best_model_index].predict(X)
        self.test_precision = precision_score(Y, y_predict)
        self.test_recall = recall_score(Y, y_predict)
        self.test_acc = accuracy_score(Y, y_predict)
        print(' Best Model Accuracy: %f' % self.test_acc)
        # for dc in self.estimators:
        #    acc = dc.score(X, Y)
        #    print('=== %d model ===\n' % self.estimators.index(dc))
        #    print(' Test Set Accuracy: %f\n' % acc)

    def evaluate(self):
        pass                        

    def feature_importance_info(self):
        fii = {}
        idx = 0
        for feature in self.features:
            if self.feature_importance[idx] > 0.0:
                fii[feature] = self.feature_importance[idx]
            idx+=1
        self.important_features = fii

    def round_summary(self):
        round_summary = {
            "best_model_index": int(self.best_model_index),
            "validation_acc": self.validation_acc.tolist(),
            "validation_recall": self.validation_recall.tolist(),
            "validation_precision": self.validation_precision.tolist(),
            "train_acc": self.train_acc.tolist(),
            "train_recall": self.train_recall.tolist(),
            "train_precision": self.train_precision.tolist(),
            "test_acc": self.test_acc,
            "test_recall": self.test_recall,
            "test_precision": self.test_precision,
            "feature_importance": self.important_features
        }
        print(round_summary)
        summary_path = os.path.join(self.output, "summary.json")
        summary_list = []
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as rfile:
                summary_list = json.load(rfile)
        summary_list.append(round_summary)
        with open(summary_path, 'w') as ofile:
            ofile.write(json.dumps(summary_list, indent=4))

def init_data(args, run_no):
    return MovieLensDC(args, run_no)


def report(args):
    output_folder = args.output_folder
    with open(os.path.join(output_folder, "summary.json"), 'r') as rfile:
        data_list = json.load(rfile)
    test_acc_list = []
    test_recall_list = []
    test_precision_list = []
    validation_acc_list = []
    validation_recall_list = []
    validation_precision_list = []
    train_acc_list = []
    train_recall_list = []
    train_precision_list = []
    for rs in data_list:
        test_acc_list.append(rs['test_acc']) 
        test_recall_list.append(rs['test_recall'])
        test_precision_list.append(rs['test_precision'])
        validation_acc_list.append(rs['validation_acc'][rs['best_model_index']])
        validation_recall_list.append(rs['validation_recall'][rs['best_model_index']])
        validation_precision_list.append(rs['validation_precision'][rs['best_model_index']]) 
        train_acc_list.append(rs['train_acc'][rs['best_model_index']])
        train_recall_list.append(rs['train_recall'][rs['best_model_index']])
        train_precision_list.append(rs['train_precision'][rs['best_model_index']])
    report_data = pd.DataFrame({
        "test accuracy": test_acc_list,
        "test recall": test_recall_list,
        "test precision": test_precision_list,
        "validation accuracy": validation_acc_list,
        "validation recall": validation_recall_list,
        "validation precision": validation_precision_list,
        "train accuracy": train_acc_list,
        "train recall": train_recall_list,
        "train precision": train_precision_list
    }, index = [
        'R1', 'R2',
        'R3', 'R4',
        'R5', 'R6',
        'R7', 'R8',
        'R9', 'R10'])
    print(report_data.head(n=10))
    print("test mean accuracy: %f" % report_data['test accuracy'].mean())
    report_data.boxplot(column=['test accuracy', 'test recall', 'test precision'])
    plt.savefig(os.path.join(output_folder, "test_boxplot.png"))
    barchart = report_data[['train accuracy', 'validation accuracy', 'test accuracy']].copy().plot.bar(rot=0)
    barchart.set_ylim(0.4, 1.0)
    plt.savefig(os.path.join(output_folder, "acc_bar.png"))
    with open(os.path.join(output_folder, "arguments.json"), 'w') as argfile:
        argfile.write(json.dumps(vars(args), indent=4))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder", help='the folder which contains dataset.')
    parser.add_argument("--output_folder", required=True, help='the folder which used as output base folder.')
    parser.add_argument("--runs", type=int, default=10, help='total runs of cross-validation')
    parser.add_argument("--depth", type=int, default=3, help='depth of decision tree.')
    parser.add_argument("--min_leaf_samples", type=int, default=3, help='the minimum number of samples of leaf node.')
    parser.add_argument("--with_tags", "-wt", dest='with_tags', default=False, action='store_true', help="include tags as descriptive features.")
    parser.add_argument("--tag_nums", "-tn", dest='tag_nums', default=10, type=int, help="Top N tags would be selected as features.")
    parser.add_argument("--hash_tags", "-ht", dest='hash_tags', default=False, action='store_true', help='enable hashvectorizer for tags or not.')
    parser.add_argument("--kbeans", "-kb", dest='kbeans', default=False, action='store_true', help='enable kbean for continuous values.')

    return parser.parse_args()


def main():
    # pd.set_option('display.max_columns', None)
    args = get_args()
    spath = os.path.join(args.output_folder, "summary.json")
    if os.path.exists(spath):
        os.remove(spath)
    for i in range(0, args.runs):
    # initial a dataset by giving specific data set path
        data = init_data(args, i)
        data.feature_collect()
        data.split()
        # training stage
        data.train()
        # testing stage
        data.test()
        # evaluation stage
        data.round_summary()
    report(args)


if __name__ == '__main__':
    main()
  