import time

import numpy as np
import pandas as pd

import threading

index = 0
number_of_finished = 0
lock = None
def custom_map(function, input_list):
    global index, number_of_finished, lock
    index = 0
    number_of_finished = 0
    lock = threading.Lock()
    output_list = [None for _ in range(len(input_list))]
    def worker():
        current_index = 0
        global index, number_of_finished
        while current_index < len(input_list):
            lock.acquire()
            current_index = index
            index += 1
            lock.release()
            if current_index >= len(input_list):
                break
            output_list[current_index] = function(input_list[current_index])

            lock.acquire()
            number_of_finished += 1
            lock.release()
        pass

    for _ in range(16):
        t = threading.Thread(target=worker, args=[])
        t.start()
    while number_of_finished < len(input_list):
        time.sleep(0.1)
    return output_list

class Configuration:
    def __init__(self):
        # Graph construction part:
        # preparation of the entire graph
        # if set to False the graph will contain ground truth relations
        self.use_realistic_graph = False
        # if set to True the graph will also contain important relations from other documents
        self.use_relations_from_other_documents = False
        # after the graph is built, add another set of inverse relations to the graph
        self.add_inverse_relations_to_graph = False
        # after the graph is built, add another set of transitive relations to the graph
        self.add_transitive_relations_to_graph = False

        # set how many relations to destroy to simulate realistic graph (do not use with actual realistic graphs)
        self.simulated_realistic_graph = {
            "use_simulated_realistic_graph": False,
            "portion_of_relations_to_keep": 0.75,
        }
        self.realistic_graph = {
            # when building the graph, only keep edges that were predicted correctly
            "remove_wrong_edges": False,
            "use_threshold_confidence": False,
            "threshold_confidence": 0.8,
        }

        # Knowledge graph builder:
        # Each relation gets the entire graph from current document (alternatively the graph contains only "important" relations)
        self.use_entire_graph = True

        # include relations from all documents
        self.no_document_filtering = False
        # WARNING only set this to False if you are using graphs from training set for testing
        self.remove_target_relation = True

        # hyper parameters for training
        self.training = {
            "text": {
                "learning_rate": 0.001,
                "weight_decay": 0.0005,
                "optimizer": "adamw",
            },
            "graph": {
                "learning_rate": 0.0001,
                "weight_decay": 0.1,
                "optimizer": "adamw",
            },
            "bimodal": {
                "learning_rate": 0.01,
                "weight_decay": 0.05,
                "optimizer": "sgd",
            },
            "bimodal2": {
                "learning_rate": 0.01,
                "weight_decay": 0.01,
                "optimizer": "sgd",
            }
        }


    def __str__(self):
        param_strings = []
        for attr, value in self.__dict__.items():
            param_strings.append(f"{attr}={repr(value)}")
        return ", ".join(param_strings)

def expand_sentence(text, start, end):
    start -= 1
    end += 1
    while text[start] != ".":
        start -= 1
        if start < 0:
            break
    while end < len(text) and text[end] != ".":
        end += 1
    return start + 1, end


def split_data(df, oversample=False, label_name='class', train_size=0.6, val_size=0.2, split_by_documents=True):
    groups = df.groupby(label_name)
    number_of_groups = 3
    number_of_relations = len(groups)
    final_dataframes = [None for _ in range(number_of_groups)]
    # train_df = None
    # val_df = None
    # test_df = None
    if split_by_documents:
        df = df.sort_values(by=['document_id'])
        border1 = int(train_size * len(df))
        border2 = int((train_size + val_size) * len(df))
        while df['document_id'][border1] == df['document_id'][border1 - 1]:
            border1 -= 1
        while df['document_id'][border2] == df['document_id'][border2 - 1]:
            border2 -= 1
        final_dataframes = np.split(df, [border1, border2])
        pass
    else:
        for group in groups.groups:
            group_df = groups.get_group(group)
            group_split = np.split(group_df.sample(frac=1, random_state=42),
                                     [int(train_size * len(group_df)), int((train_size + val_size) * len(group_df))])
            for i in range(number_of_groups):
                final_dataframes[i] = pd.concat([final_dataframes[i], group_split[i]])


    # oversampling
    if oversample:
        for i in range(number_of_groups):
            max_class_count = final_dataframes[i][label_name].value_counts().max()
            final_dataframes[i] = final_dataframes[i].groupby(label_name).apply(lambda x: x.sample(max_class_count, replace=True)).reset_index(
                drop=True)

    # shuffle rows
    for i in range(number_of_groups):
        final_dataframes[i] = final_dataframes[i].sample(frac=1, random_state=42).reset_index(drop=True)

    return final_dataframes

def oversample(df, oversample=True, label_name='class'):
    if oversample:
        max_class_count = df[label_name].value_counts().max()
        df = df.groupby(label_name).apply(lambda x: x.sample(max_class_count, replace=True)).reset_index(drop=True)
        # shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

def compute_transitive_relation(relation1, relation2):
    if relation1 == "BEFORE" and relation2 == "BEFORE":
        return "BEFORE"
    if relation1 == "BEFORE" and relation2 == "AFTER":
        return "OVERLAP"
    if relation1 == "BEFORE" and relation2 == "OVERLAP":
        return "BEFORE"
    if relation1 == "BEFORE" and relation2 == "BEGINS-ON":
        return "BEFORE"
    if relation1 == "BEFORE" and relation2 == "CONTAINED-BY":
        return "BEFORE"
    if relation1 == "BEFORE" and relation2 == "CONTAINS":
        return "BEFORE"
    if relation1 == "BEFORE" and relation2 == "ENDS-ON":
        return "BEFORE"
    if relation1 == "BEFORE" and relation2 == "CONTINUES":
        return "BEFORE"
    if relation1 == "BEFORE" and relation2 == "TERMINATES":
        return "BEFORE"
    if relation1 == "BEFORE" and relation2 == "INITIATES":
        return "BEFORE"
    if relation1 == "BEFORE" and relation2 == "REINITIATES":
        return "BEFORE"
    if relation1 == "AFTER" and relation2 == "BEFORE":
        return "OVERLAP"
    if relation1 == "AFTER" and relation2 == "AFTER":
        return "AFTER"
    if relation1 == "AFTER" and relation2 == "OVERLAP":
        return "AFTER"
    if relation1 == "AFTER" and relation2 == "BEGINS-ON":
        return "AFTER"
    if relation1 == "AFTER" and relation2 == "CONTAINED-BY":
        return "AFTER"
    if relation1 == "AFTER" and relation2 == "CONTAINS":
        return "AFTER"
    if relation1 == "AFTER" and relation2 == "ENDS-ON":
        return "AFTER"
    if relation1 == "AFTER" and relation2 == "CONTINUES":
        return "AFTER"
    if relation1 == "AFTER" and relation2 == "TERMINATES":
        return "AFTER"
    if relation1 == "AFTER" and relation2 == "INITIATES":
        return "AFTER"
    if relation1 == "AFTER" and relation2 == "REINITIATES":
        return "AFTER"
    if relation1 == "OVERLAP" and relation2 == "BEFORE":
        return "BEFORE"
    if relation1 == "OVERLAP" and relation2 == "AFTER":
        return "AFTER"
    if relation1 == "OVERLAP" and relation2 == "OVERLAP":
        return "OVERLAP"
    if relation1 == "OVERLAP" and relation2 == "BEGINS-ON":
        return "BEGINS-ON"
    if relation1 == "OVERLAP" and relation2 == "CONTAINED-BY":
        return "CONTAINED-BY"
    if relation1 == "OVERLAP" and relation2 == "CONTAINS":
        return "CONTAINS"
    if relation1 == "OVERLAP" and relation2 == "ENDS-ON":
        return "ENDS-ON"
    if relation1 == "OVERLAP" and relation2 == "CONTINUES":
        return "CONTINUES"
    if relation1 == "OVERLAP" and relation2 == "TERMINATES":
        return "TERMINATES"
    if relation1 == "OVERLAP" and relation2 == "INITIATES":
        return "INITIATES"
    if relation1 == "OVERLAP" and relation2 == "REINITIATES":
        return "REINITIATES"
    if relation1 == "BEGINS-ON" and relation2 == "BEFORE":
        return "OVERLAP"
    if relation1 == "BEGINS-ON" and relation2 == "AFTER":
        return "AFTER"
    if relation1 == "BEGINS-ON" and relation2 == "OVERLAP":
        return "BEGINS-ON"
    if relation1 == "BEGINS-ON" and relation2 == "BEGINS-ON":
        return "BEGINS-ON"
    if relation1 == "BEGINS-ON" and relation2 == "CONTAINED-BY":
        return "BEGINS-ON"
    if relation1 == "BEGINS-ON" and relation2 == "CONTAINS":
        return "BEGINS-ON"
    if relation1 == "BEGINS-ON" and relation2 == "ENDS-ON":
        return "BEGINS-ON"
    if relation1 == "BEGINS-ON" and relation2 == "CONTINUES":
        return "OVERLAP"
    if relation1 == "BEGINS-ON" and relation2 == "TERMINATES":
        return "OVERLAP"
    if relation1 == "BEGINS-ON" and relation2 == "INITIATES":
        return "OVERLAP"
    if relation1 == "BEGINS-ON" and relation2 == "REINITIATES":
        return "OVERLAP"
    if relation1 == "CONTAINED-BY" and relation2 == "BEFORE":
        return "BEFORE"
    if relation1 == "CONTAINED-BY" and relation2 == "AFTER":
        return "AFTER"
    if relation1 == "CONTAINED-BY" and relation2 == "OVERLAP":
        return "CONTAINED-BY"
    if relation1 == "CONTAINED-BY" and relation2 == "BEGINS-ON":
        return "BEGINS-ON"
    if relation1 == "CONTAINED-BY" and relation2 == "CONTAINED-BY":
        return "CONTAINED-BY"
    if relation1 == "CONTAINED-BY" and relation2 == "CONTAINS":
        return "OVERLAP"
    if relation1 == "CONTAINED-BY" and relation2 == "ENDS-ON":
        return "OVERLAP"
    if relation1 == "CONTAINED-BY" and relation2 == "CONTINUES":
        return "OVERLAP"
    if relation1 == "CONTAINED-BY" and relation2 == "TERMINATES":
        return "TERMINATES"
    if relation1 == "CONTAINED-BY" and relation2 == "INITIATES":
        return "INITIATES"
    if relation1 == "CONTAINED-BY" and relation2 == "REINITIATES":
        return "REINITIATES"
    if relation1 == "CONTAINS" and relation2 == "BEFORE":
        return "BEFORE"
    if relation1 == "CONTAINS" and relation2 == "AFTER":
        return "AFTER"
    if relation1 == "CONTAINS" and relation2 == "OVERLAP":
        return "CONTAINS"
    if relation1 == "CONTAINS" and relation2 == "BEGINS-ON":
        return "BEGINS-ON"
    if relation1 == "CONTAINS" and relation2 == "CONTAINED-BY":
        return "OVERLAP"
    if relation1 == "CONTAINS" and relation2 == "CONTAINS":
        return "CONTAINS"
    if relation1 == "CONTAINS" and relation2 == "ENDS-ON":
        return "CONTAINS"
    if relation1 == "CONTAINS" and relation2 == "CONTINUES":
        return "CONTAINS"
    if relation1 == "CONTAINS" and relation2 == "TERMINATES":
        return "OVERLAP"
    if relation1 == "CONTAINS" and relation2 == "INITIATES":
        return "OVERLAP"
    if relation1 == "CONTAINS" and relation2 == "REINITIATES":
        return "OVERLAP"
    if relation1 == "ENDS-ON" and relation2 == "BEFORE":
        return "BEFORE"
    if relation1 == "ENDS-ON" and relation2 == "AFTER":
        return "AFTER"
    if relation1 == "ENDS-ON" and relation2 == "OVERLAP":
        return "ENDS-ON"
    if relation1 == "ENDS-ON" and relation2 == "BEGINS-ON":
        return "ENDS-ON"
    if relation1 == "ENDS-ON" and relation2 == "CONTAINED-BY":
        return "ENDS-ON"
    if relation1 == "ENDS-ON" and relation2 == "CONTAINS":
        return "ENDS-ON"
    if relation1 == "ENDS-ON" and relation2 == "ENDS-ON":
        return "ENDS-ON"
    if relation1 == "ENDS-ON" and relation2 == "CONTINUES":
        return "ENDS-ON"
    if relation1 == "ENDS-ON" and relation2 == "TERMINATES":
        return "OVERLAP"
    if relation1 == "ENDS-ON" and relation2 == "INITIATES":
        return "ENDS-ON"
    if relation1 == "ENDS-ON" and relation2 == "REINITIATES":
        return "ENDS-ON"
    if relation1 == "CONTINUES" and relation2 == "BEFORE":
        return "OVERLAP"
    if relation1 == "CONTINUES" and relation2 == "AFTER":
        return "AFTER"
    if relation1 == "CONTINUES" and relation2 == "OVERLAP":
        return "CONTINUES"
    if relation1 == "CONTINUES" and relation2 == "BEGINS-ON":
        return "BEGINS-ON"
    if relation1 == "CONTINUES" and relation2 == "CONTAINED-BY":
        return "BEGINS-ON"
    if relation1 == "CONTINUES" and relation2 == "CONTAINS":
        return "BEGINS-ON"
    if relation1 == "CONTINUES" and relation2 == "ENDS-ON":
        return "BEGINS-ON"
    if relation1 == "CONTINUES" and relation2 == "CONTINUES":
        return "OVERLAP"
    if relation1 == "CONTINUES" and relation2 == "TERMINATES":
        return "OVERLAP"
    if relation1 == "CONTINUES" and relation2 == "INITIATES":
        return "OVERLAP"
    if relation1 == "CONTINUES" and relation2 == "REINITIATES":
        return "OVERLAP"
    if relation1 == "TERMINATES" and relation2 == "BEFORE":
        return "BEFORE"
    if relation1 == "TERMINATES" and relation2 == "AFTER":
        return "AFTER"
    if relation1 == "TERMINATES" and relation2 == "OVERLAP":
        return "TERMINATES"
    if relation1 == "TERMINATES" and relation2 == "BEGINS-ON":
        return "AFTER"
    if relation1 == "TERMINATES" and relation2 == "CONTAINED-BY":
        return "OVERLAP"
    if relation1 == "TERMINATES" and relation2 == "CONTAINS":
        return "OVERLAP"
    if relation1 == "TERMINATES" and relation2 == "ENDS-ON":
        return "OVERLAP"
    if relation1 == "TERMINATES" and relation2 == "CONTINUES":
        return "OVERLAP"
    if relation1 == "TERMINATES" and relation2 == "TERMINATES":
        return "TERMINATES"
    if relation1 == "TERMINATES" and relation2 == "INITIATES":
        return "OVERLAP"
    if relation1 == "TERMINATES" and relation2 == "REINITIATES":
        return "OVERLAP"
    if relation1 == "INITIATES" and relation2 == "BEFORE":
        return "BEFORE"
    if relation1 == "INITIATES" and relation2 == "AFTER":
        return "AFTER"
    if relation1 == "INITIATES" and relation2 == "OVERLAP":
        return "INITIATES"
    if relation1 == "INITIATES" and relation2 == "BEGINS-ON":
        return "OVERLAP"
    if relation1 == "INITIATES" and relation2 == "CONTAINED-BY":
        return "INITIATES"
    if relation1 == "INITIATES" and relation2 == "CONTAINS":
        return "OVERLAP"
    if relation1 == "INITIATES" and relation2 == "ENDS-ON":
        return "INITIATES"
    if relation1 == "INITIATES" and relation2 == "CONTINUES":
        return "OVERLAP"
    if relation1 == "INITIATES" and relation2 == "TERMINATES":
        return "OVERLAP"
    if relation1 == "INITIATES" and relation2 == "INITIATES":
        return "INITIATES"
    if relation1 == "INITIATES" and relation2 == "REINITIATES":
        return "REINITIATES"
    if relation1 == "REINITIATES" and relation2 == "BEFORE":
        return "OVERLAP"
    if relation1 == "REINITIATES" and relation2 == "AFTER":
        return "AFTER"
    if relation1 == "REINITIATES" and relation2 == "OVERLAP":
        return "REINITIATES"
    if relation1 == "REINITIATES" and relation2 == "BEGINS-ON":
        return "BEGINS-ON"
    if relation1 == "REINITIATES" and relation2 == "CONTAINED-BY":
        return "BEGINS-ON"
    if relation1 == "REINITIATES" and relation2 == "CONTAINS":
        return "BEGINS-ON"
    if relation1 == "REINITIATES" and relation2 == "ENDS-ON":
        return "BEGINS-ON"
    if relation1 == "REINITIATES" and relation2 == "CONTINUES":
        return "OVERLAP"
    if relation1 == "REINITIATES" and relation2 == "TERMINATES":
        return "OVERLAP"
    if relation1 == "REINITIATES" and relation2 == "INITIATES":
        return "OVERLAP"
    if relation1 == "REINITIATES" and relation2 == "REINITIATES":
        return "OVERLAP"