# -*- coding: utf-8 -*-
"""
Author: Irem Ecem Yelkanat
ID: 21702624
"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib widget

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Import necessary packages

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from skmultiflow.data.sea_generator import SEAGenerator
from skmultiflow.meta import BatchIncrementalClassifier

from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.lazy import KNNClassifier
from sklearn.neural_network import MLPClassifier

from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data.file_stream import FileStream

from sklearn.ensemble import VotingClassifier

import numpy as np
import pandas as pd

np.random.seed(1)

"""## Dataset Generation"""

sea_dataset_generated = SEAGenerator(random_state=42)
sea_dataset = sea_dataset_generated.next_sample(20000)
sea_dataset = np.append(sea_dataset[0], np.reshape(sea_dataset[1], (20000,1)), axis=1)
sea_dataset_df = pd.DataFrame(sea_dataset)
sea_dataset_df = sea_dataset_df.astype({3: int})
sea_dataset_df.to_csv("SEA Dataset.csv", index=False)

sea_dataset_generated_10 = SEAGenerator(random_state=42, noise_percentage=0.10)
sea_dataset_10 = sea_dataset_generated_10.next_sample(20000)
sea_dataset_10 = np.append(sea_dataset_10[0], np.reshape(sea_dataset_10[1], (20000,1)), axis=1)
sea_dataset_df_10 = pd.DataFrame(sea_dataset_10)
sea_dataset_df_10 = sea_dataset_df_10.astype({3: int})
sea_dataset_df_10.to_csv("SEA Dataset 10.csv", index=False)

sea_dataset_generated_70 = SEAGenerator(random_state=42, noise_percentage=0.70)
sea_dataset_70 = sea_dataset_generated_70.next_sample(20000)
sea_dataset_70 = np.append(sea_dataset_70[0], np.reshape(sea_dataset_70[1], (20000,1)), axis=1)
sea_dataset_df_70 = pd.DataFrame(sea_dataset_70)
sea_dataset_df_70 = sea_dataset_df_70.astype({3: int})
sea_dataset_df_70.to_csv("SEA Dataset 70.csv", index=False)

"""# Online Learning

## Read Data from Files as Streams
"""

sea_dataset_read = FileStream("SEA Dataset.csv")
sea_dataset_read_10 = FileStream("SEA Dataset 10.csv")
sea_dataset_read_70 = FileStream("SEA Dataset 70.csv")

"""## Data Stream Classification with Three Separate Online Single Classifiers: HT, KNN, MLP"""

HT_online_classifier = HoeffdingTreeClassifier()
KNN_online_classifier = KNNClassifier()
MLP_online_classifier = MLPClassifier(hidden_layer_sizes=(200,4))

evaluator = EvaluatePrequential(show_plot=True, metrics=["accuracy"])

evaluator.evaluate(stream=sea_dataset_read, model=[HT_online_classifier, KNN_online_classifier, MLP_online_classifier], model_names=["HT", "KNN", "MLP"])

evaluator.evaluate(stream=sea_dataset_read_10, model=[HT_online_classifier, KNN_online_classifier, MLP_online_classifier], model_names=["HT", "KNN", "MLP"])

evaluator.evaluate(stream=sea_dataset_read_70, model=[HT_online_classifier, KNN_online_classifier, MLP_online_classifier], model_names=["HT", "KNN", "MLP"])

"""### Try different batch sizes"""

batch_sizes = [5, 10, 25, 50, 100, 200, 500, 1000]

for batch_size in batch_sizes:
    HT_online_classifier = HoeffdingTreeClassifier()
    KNN_online_classifier = KNNClassifier()
    MLP_online_classifier = MLPClassifier(hidden_layer_sizes=(200,4))

    print("Evaluating for batch size", batch_size, "and dataset SEA Dataset")
    evaluator = EvaluatePrequential(show_plot=True, metrics=["accuracy"], batch_size=batch_size)
    evaluator.evaluate(stream=sea_dataset_read, model=[HT_online_classifier, KNN_online_classifier, MLP_online_classifier], model_names=["HT", "KNN", "MLP"])

batch_sizes = [5, 10, 25, 50, 100, 200, 500, 1000]

for batch_size in batch_sizes:
    HT_online_classifier = HoeffdingTreeClassifier()
    KNN_online_classifier = KNNClassifier()
    MLP_online_classifier = MLPClassifier(hidden_layer_sizes=(200,4))

    print("Evaluating for batch size", batch_size, "and dataset SEA Dataset10")
    evaluator = EvaluatePrequential(show_plot=True, metrics=["accuracy"], batch_size=batch_size)
    evaluator.evaluate(stream=sea_dataset_read_10, model=[HT_online_classifier, KNN_online_classifier, MLP_online_classifier], model_names=["HT", "KNN", "MLP"])

batch_sizes = [5, 10, 25, 50, 100, 200, 500, 1000]

for batch_size in batch_sizes:
    HT_online_classifier = HoeffdingTreeClassifier()
    KNN_online_classifier = KNNClassifier()
    MLP_online_classifier = MLPClassifier(hidden_layer_sizes=(200,4))

    print("Evaluating for batch size", batch_size, "and dataset SEA Dataset10")
    evaluator = EvaluatePrequential(show_plot=True, metrics=["accuracy"], batch_size=batch_size)
    evaluator.evaluate(stream=sea_dataset_read_70, model=[HT_online_classifier, KNN_online_classifier, MLP_online_classifier], model_names=["HT", "KNN", "MLP"])

"""## Data Stream Classification with Two Online Ensemble Classifiers: MV, WMV"""

HT_online_classifier = HoeffdingTreeClassifier()
KNN_online_classifier = KNNClassifier()
MLP_online_classifier = MLPClassifier(hidden_layer_sizes=(200,4))

MV_online = VotingClassifier(estimators=[("HT", HT_online_classifier), ("KNN", KNN_online_classifier), ("MLP", MLP_online_classifier)], voting='hard', weights=[1,1,1])
MV_online = BatchIncrementalClassifier(base_estimator=MV_online, n_estimators=3)

WMV_online = VotingClassifier(estimators=[("HT", HT_online_classifier), ("KNN", KNN_online_classifier), ("MLP", MLP_online_classifier)], voting='hard')
WMV_online = BatchIncrementalClassifier(base_estimator=WMV_online, n_estimators=3)

evaluator = EvaluatePrequential(show_plot=True, metrics=["accuracy"])

evaluator.evaluate(stream=sea_dataset_read, model=[MV_online, WMV_online], model_names=["MV", "WMV"])

evaluator.evaluate(stream=sea_dataset_read_10, model=[MV_online, WMV_online], model_names=["MV", "WMV"])

evaluator.evaluate(stream=sea_dataset_read_70, model=[MV_online, WMV_online], model_names=["MV", "WMV"])

"""### Try different batch sizes"""

batch_sizes = [5, 10, 25, 50, 100, 200, 500, 1000]

for batch_size in batch_sizes:
    HT_online_classifier = HoeffdingTreeClassifier()
    KNN_online_classifier = KNNClassifier()
    MLP_online_classifier = MLPClassifier(hidden_layer_sizes=(200,4))

    MV_online = VotingClassifier(estimators=[("HT", HT_online_classifier), ("KNN", KNN_online_classifier), ("MLP", MLP_online_classifier)], voting='hard', weights=[1,1,1])
    MV_online = BatchIncrementalClassifier(base_estimator=MV_online, n_estimators=3)

    WMV_online = VotingClassifier(estimators=[("HT", HT_online_classifier), ("KNN", KNN_online_classifier), ("MLP", MLP_online_classifier)], voting='hard')
    WMV_online = BatchIncrementalClassifier(base_estimator=WMV_online, n_estimators=3)

    print("Evaluating for batch size", batch_size)
    evaluator = EvaluatePrequential(show_plot=True, metrics=["accuracy"], batch_size=batch_size)
    evaluator.evaluate(stream=sea_dataset_read, model=[MV_online, WMV_online], model_names=["MV", "WMV"])

batch_sizes = [5, 10, 25, 50, 100, 200, 500, 1000]

for batch_size in batch_sizes:
    HT_online_classifier = HoeffdingTreeClassifier()
    KNN_online_classifier = KNNClassifier()
    MLP_online_classifier = MLPClassifier(hidden_layer_sizes=(200,4))

    MV_online = VotingClassifier(estimators=[("HT", HT_online_classifier), ("KNN", KNN_online_classifier), ("MLP", MLP_online_classifier)], voting='hard', weights=[1,1,1])
    MV_online = BatchIncrementalClassifier(base_estimator=MV_online, n_estimators=3)

    WMV_online = VotingClassifier(estimators=[("HT", HT_online_classifier), ("KNN", KNN_online_classifier), ("MLP", MLP_online_classifier)], voting='hard')
    WMV_online = BatchIncrementalClassifier(base_estimator=WMV_online, n_estimators=3)

    print("Evaluating for batch size", batch_size)
    evaluator = EvaluatePrequential(show_plot=True, metrics=["accuracy"], batch_size=batch_size)
    evaluator.evaluate(stream=sea_dataset_read_10, model=[MV_online, WMV_online], model_names=["MV", "WMV"])

batch_sizes = [5, 10, 25, 50, 100, 200, 500, 1000]

for batch_size in batch_sizes:
    HT_online_classifier = HoeffdingTreeClassifier()
    KNN_online_classifier = KNNClassifier()
    MLP_online_classifier = MLPClassifier(hidden_layer_sizes=(200,4))

    MV_online = VotingClassifier(estimators=[("HT", HT_online_classifier), ("KNN", KNN_online_classifier), ("MLP", MLP_online_classifier)], voting='hard', weights=[1,1,1])
    MV_online = BatchIncrementalClassifier(base_estimator=MV_online, n_estimators=3)

    WMV_online = VotingClassifier(estimators=[("HT", HT_online_classifier), ("KNN", KNN_online_classifier), ("MLP", MLP_online_classifier)], voting='hard')
    WMV_online = BatchIncrementalClassifier(base_estimator=WMV_online, n_estimators=3)

    print("Evaluating for batch size", batch_size)
    evaluator = EvaluatePrequential(show_plot=True, metrics=["accuracy"], batch_size=batch_size)
    evaluator.evaluate(stream=sea_dataset_read_70, model=[MV_online, WMV_online], model_names=["MV", "WMV"])

"""# Batch Learning

## Read data to np array
"""

sea_dataset = pd.read_csv("SEA Dataset.csv").values
sea_dataset_features = sea_dataset[:,:3]
sea_dataset_labels = np.array(sea_dataset[:,3], dtype=int)
train_features, test_features, train_labels, test_labels = train_test_split(sea_dataset_features, sea_dataset_labels, test_size=0.2, random_state=42)

sea_dataset_10 = pd.read_csv("SEA Dataset 10.csv").values
sea_dataset_features_10 = sea_dataset_10[:,:3]
sea_dataset_labels_10 = np.array(sea_dataset_10[:,3], dtype=int)
train_features_10, test_features_10, train_labels_10, test_labels_10 = train_test_split(sea_dataset_features_10, sea_dataset_labels_10, test_size=0.2, random_state=42)

sea_dataset_70 = pd.read_csv("SEA Dataset 70.csv").values
sea_dataset_features_70 = sea_dataset_70[:,:3]
sea_dataset_labels_70 = np.array(sea_dataset_70[:,3], dtype=int)
train_features_70, test_features_70, train_labels_70, test_labels_70 = train_test_split(sea_dataset_features_70, sea_dataset_labels_70, test_size=0.2, random_state=42)

"""## Batch Classification with Three Separate Batch Single Classifiers: HT, NB, MLP"""

HT_batch_classifier = HoeffdingTreeClassifier()
KNN_batch_classifier = KNNClassifier()
MLP_batch_classifier = MLPClassifier(hidden_layer_sizes=(200,4))

HT_batch_classifier.fit(train_features, train_labels)
pred_HT = HT_batch_classifier.predict(test_features)
acc_HT = accuracy_score(test_labels, pred_HT)

KNN_batch_classifier.fit(train_features, train_labels)
pred_KNN = KNN_batch_classifier.predict(test_features)
acc_KNN = accuracy_score(test_labels, pred_KNN)

MLP_batch_classifier.fit(train_features, train_labels)
pred_MLP = MLP_batch_classifier.predict(test_features)
acc_MLP = accuracy_score(test_labels, pred_MLP)

print("Accuracy of Batch HT on SEA Dataset is:", acc_HT)
print("Accuracy of Batch KNN on SEA Dataset is:", acc_KNN)
print("Accuracy of Batch MLP on SEA Dataset is:", acc_MLP)

HT_batch_classifier.fit(train_features_10, train_labels_10)
pred_HT = HT_batch_classifier.predict(test_features_10)
acc_HT = accuracy_score(test_labels_10, pred_HT)

KNN_batch_classifier.fit(train_features_10, train_labels_10)
pred_KNN = KNN_batch_classifier.predict(test_features_10)
acc_KNN = accuracy_score(test_labels_10, pred_KNN)

MLP_batch_classifier.fit(train_features_10, train_labels_10)
pred_MLP = MLP_batch_classifier.predict(test_features_10)
acc_MLP = accuracy_score(test_labels_10, pred_MLP)

print("Accuracy of Batch HT on SEA Dataset10 is:", acc_HT)
print("Accuracy of Batch KNN on SEA Dataset10 is:", acc_KNN)
print("Accuracy of Batch MLP on SEA Dataset10 is:", acc_MLP)

HT_batch_classifier.fit(train_features_70, train_labels_70)
pred_HT = HT_batch_classifier.predict(test_features_70)
acc_HT = accuracy_score(test_labels_70, pred_HT)

KNN_batch_classifier.fit(train_features_70, train_labels_70)
pred_KNN = KNN_batch_classifier.predict(test_features_70)
acc_KNN = accuracy_score(test_labels_70, pred_KNN)

MLP_batch_classifier.fit(train_features_70, train_labels_70)
pred_MLP = MLP_batch_classifier.predict(test_features_70)
acc_MLP = accuracy_score(test_labels_70, pred_MLP)

print("Accuracy of Batch HT on SEA Dataset70 is:", acc_HT)
print("Accuracy of Batch KNN on SEA Dataset70 is:", acc_KNN)
print("Accuracy of Batch MLP on SEA Dataset70 is:", acc_MLP)

"""## Batch Classification with Two Batch Ensemble Classifiers: MV, WMV"""

HT_batch_classifier = HoeffdingTreeClassifier()
KNN_batch_classifier = KNNClassifier()
MLP_batch_classifier = MLPClassifier(hidden_layer_sizes=(200,4))

MV_batch = VotingClassifier(estimators=[("HT", HT_batch_classifier), ("KNN", KNN_batch_classifier), ("MLP", MLP_batch_classifier)], voting='hard', weights=[1,1,1])
MV_batch.fit(train_features, train_labels)
pred_MV = MV_batch.predict(test_features)
acc_MV = accuracy_score(test_labels, pred_MV)

WMV_batch = VotingClassifier(estimators=[("HT", HT_batch_classifier), ("KNN", KNN_batch_classifier), ("MLP", MLP_batch_classifier)], voting='hard')
WMV_batch.fit(train_features, train_labels)
pred_WMV = WMV_batch.predict(test_features)
acc_WMV = accuracy_score(test_labels, pred_WMV)

print("Accuracy of MV Batch Ensemble on SEA Dataset is:", acc_MV)
print("Accuracy of WMV Batch Ensemble on SEA Dataset is:", acc_WMV)

MV_batch = VotingClassifier(estimators=[("HT", HT_batch_classifier), ("KNN", KNN_batch_classifier), ("MLP", MLP_batch_classifier)], voting='hard', weights=[1,1,1])
MV_batch.fit(train_features_10, train_labels_10)
pred_MV = MV_batch.predict(test_features_10)
acc_MV = accuracy_score(test_labels_10, pred_MV)

WMV_batch = VotingClassifier(estimators=[("HT", HT_batch_classifier), ("KNN", KNN_batch_classifier), ("MLP", MLP_batch_classifier)], voting='hard')
WMV_batch.fit(train_features_10, train_labels_10)
pred_WMV = WMV_batch.predict(test_features_10)
acc_WMV = accuracy_score(test_labels_10, pred_WMV)

print("Accuracy of MV Batch Ensemble on SEA Dataset10 is:", acc_MV)
print("Accuracy of WMV Batch Ensemble on SEA Dataset10 is:", acc_WMV)

MV_batch = VotingClassifier(estimators=[("HT", HT_batch_classifier), ("KNN", KNN_batch_classifier), ("MLP", MLP_batch_classifier)], voting='hard', weights=[1,1,1])
MV_batch.fit(train_features_70, train_labels_70)
pred_MV = MV_batch.predict(test_features_70)
acc_MV = accuracy_score(test_labels_70, pred_MV)

WMV_batch = VotingClassifier(estimators=[("HT", HT_batch_classifier), ("KNN", KNN_batch_classifier), ("MLP", MLP_batch_classifier)], voting='hard')
WMV_batch.fit(train_features_70, train_labels_70)
pred_WMV = WMV_batch.predict(test_features_70)
acc_WMV = accuracy_score(test_labels_70, pred_WMV)

print("Accuracy of MV Batch Ensemble on SEA Dataset70 is:", acc_MV)
print("Accuracy of WMV Batch Ensemble on SEA Dataset70 is:", acc_WMV)


## Accuracy Improvement

from skmultiflow.meta import AdaptiveRandomForestClassifier

random_forest_classifier = AdaptiveRandomForestClassifier()
KNN_online_classifier = KNNClassifier()

evaluator = EvaluatePrequential(show_plot=True, metrics=["accuracy"])

evaluator.evaluate(stream=sea_dataset_read_70, model=[KNN_online_classifier, random_forest_classifier], model_names=["KNN - Online", "Random Forest"])

from skmultiflow.meta import DynamicWeightedMajorityClassifier

knn = KNNClassifier()
dwm = DynamicWeightedMajorityClassifier(knn)

KNN_online_classifier = KNNClassifier()

evaluator = EvaluatePrequential(show_plot=True, metrics=["accuracy"])

evaluator.evaluate(stream=sea_dataset_read_70, model=[KNN_online_classifier, dwm], model_names=["KNN", "DWM"])