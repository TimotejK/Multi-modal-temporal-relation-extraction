import os
import sys
from datetime import datetime
from os import path

import pandas
import torch.nn.functional as F

import torch
import torch_geometric
from sklearn.metrics import accuracy_score, f1_score
from tensorboardX import SummaryWriter
from torch import nn
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

import common
from common import split_data, Configuration
from dataLoaders import combining_data, knowledge_graph_builder
from dataLoaders.knowledge_graph_builder import in_memory_kg, KnowledgeGraphDataset, convert_df_row
from models.helpers.DataframeDataset import DFDataset
from models.EntityBERT.Model import EntityBERTRelationExtraction, GNNRelationPrediction, MultiModalPrediction, \
    BaselineBERTrelationExtraction

tokenizer = AutoTokenizer.from_pretrained("./pretrained models/PubmedBERTbase-MimicBig-EntityBERT", )

# labels = {'BEFORE': 0,
#           'AFTER': 1,
#           'OVERLAP': 2,
#           'BEGINS-ON': 3,
#           'CONTAINED-BY': 4,
#           'CONTAINS': 5,
#           'ENDS-ON': 6
#           }

# 'AFTER', 'BEFORE', 'OVERLAP', 'BEGINS-ON', 'CONTAINED-BY', 'CONTAINS', 'ENDS-ON', 'CONTINUES', 'TERMINATES', 'INITIATES', 'REINITIATES',

labels = {'BEFORE': 0,
          'AFTER': 1,
          'OVERLAP': 2,
          'BEGINS-ON': 3,
          'CONTAINED-BY': 4,
          'CONTAINS': 5,
          'ENDS-ON': 6,
          'CONTINUES': 7,
          'TERMINATES': 8,
          'INITIATES': 9,
          'REINITIATES': 10,
          }
index_to_label = ['BEFORE', 'AFTER', 'OVERLAP', 'BEGINS-ON', 'CONTAINED-BY', 'CONTAINS', 'ENDS-ON', 'CONTINUES', 'TERMINATES', 'INITIATES', 'REINITIATES']
# labels = {'BEFORE': 0,
#           'AFTER': 1,
#           'OVERLAP': 2
#           }

def train_in_parts(model, dataLoader, dataLoader_val, dataLoader_test, dataLoader_balanced, dataLoader_val_balanced):
    report = {}
    top_performance, top_epoch, top_scores, log_name = train(model.graph_model, dataLoader_balanced, dataLoader_val_balanced, epochs=30, pretraining=True, weight_decay=0.01)
    top_performance, top_epoch, top_scores, log_name = train(model.graph_model, dataLoader, dataLoader_val, epochs=30, pretraining=False)
    report["graph_only"] = top_performance
    print(log_name)
    # top_performance, top_epoch, top_scores, log_name = train(model.text_model, dataLoader_balanced, dataLoader_val_balanced, epochs=30, pretraining=True)
    # top_performance, top_epoch, top_scores, log_name = train(model.text_model, dataLoader, dataLoader_val, epochs=30, pretraining=False)
    # report["text_only"] = top_performance
    # top_performance, top_epoch, top_scores, log_name = train(model, dataLoader_val, dataLoader_test, epochs=30)
    # report["combined"] = top_performance
    print(report)

def train(model, dataLoader, dataLoader_val, epochs=100, learning_rate=5e-2, weight_decay=0.0001, optimizer="sgd",
          batch_size=2, pretraining=False, save_path=None, log_name=None):
    description = \
        "model" + ": " + str(model.__class__.__name__) + ", " + \
        "dataset" + ": " + dataset_description + ", " + \
        "number_of_relations" + ": " + str(len(labels)) + ", " + \
        "epochs" + ": " + str(epochs) + ", " + \
        "batch_size" + ": " + str(batch_size) + ", " + \
        "learning_rate" + ": " + str(learning_rate) + ", " + \
        "weight_decay" + ": " + str(weight_decay) + ", " + \
        "pool_mode" + ": " + str(model.pooling_strategy) + ", " + \
        "pretraining" + ": " + str(pretraining) + ", " + \
        "optimizer" + ": " + str(optimizer)
    print(description)
    if log_name is None:
        log_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open("log_descriptions.txt", "a") as file_object:
        file_object.write(log_name + ": \t" + description + "\n")
    writer = SummaryWriter("./log/" + log_name)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    if optimizer == "sgd":
        optimizer = SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "adamw":
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    top_performance = 0
    top_scores = None
    top_epoch = 0

    for epoch_num in range(epochs):
        model.train()
        total_acc_train = 0
        n = 0
        for batch in tqdm(dataLoader):
            graph, _ = batch
            y = graph.y
            graph.to(device)
            y = y.to(device)
            pred = model(graph)
            # if len(y) > 1:
            #     y = y.squeeze()
            pred = F.softmax(pred, dim=1)
            loss = criterion(pred, y)
            acc = (pred.argmax(dim=1) == y).sum().item()
            total_acc_train += acc
            n += y.size()[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pass
        total_acc_val, f1 = run_test(model, dataLoader_val, device)
        if f1 > top_performance:
            top_performance = f1
            top_epoch = epoch_num
            top_scores = {"f1": f1, "acc": total_acc_val}
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                torch.save(model, path.join(save_path, "model_best.pt"))

        if writer is not None:
            writer.add_scalar("train_acc", total_acc_train / n, epoch_num)
            writer.add_scalar("test_acc", total_acc_val, epoch_num)
            writer.add_scalar("test_f1", f1, epoch_num)
        print("train_acc:", total_acc_train / n,
              "test_acc:", total_acc_val,
              "test_f1:", f1)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        torch.save(model, path.join(save_path, "model_final.pt"))

    return top_performance, top_epoch, top_scores, log_name


def run_test(model, test_data_loader, device):
    model.eval()
    model.training = False
    labels = []
    units = []
    predictions = []
    predictions_unit = []
    raw_predictions_export=[]
    for batch in test_data_loader:
        graph, _ = batch
        y = graph.y
        graph.to(device)
        y = y.to(device)
        res = model(graph)
        res = torch.softmax(res, dim=1)
        pred = res.argmax(dim=1)

        labels += y.tolist()
        predictions += pred.tolist()
        for i in range(len(pred)):
            raw_predictions_export.append({
                "document_id": graph['document_id'][i],
                "text": graph['text'][i],
                "event1_start": graph['event1_start'][i],
                "event2_start": graph['event2_start'][i],
                "event1_end": graph['event1_end'][i],
                "event2_end": graph['event2_end'][i],
                "rule_based_prediction": graph['rule_based_prediction'][i],
                "model_prediction": int(pred[i]),
                "correct_label": int(labels[i]),
            })
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='micro')
    pandas.DataFrame(raw_predictions_export).to_csv('predictions_' + dataset_description + '_' + model_description + '.csv')
    return accuracy, f1


def hyperparameters_test(pool_mode, pretrain, train_df, val_df, deeper_network=False, pretrain_epochs=20, epochs=100,
                         learning_rate=5e-2, weight_decay=0.0001, optimizer="sgd", batch_size=64):
    global dataset_description
    model = EntityBERTRelationExtraction(3, pooling_strategy=pool_mode, deeper_network=deeper_network)

    # Load dataset
    dataset = DFDataset(train_df, convert_df_row)
    dataLoader = DataLoader(dataset, batch_size=batch_size)
    dataset_val = DFDataset(val_df, convert_df_row)
    dataLoader_val = DataLoader(dataset_val, batch_size=batch_size)

    if pretrain:
        # oversample
        max_class_count = train_df['class'].value_counts().max()
        train_balanced = train_df.groupby('class').apply(
            lambda x: x.sample(max_class_count, replace=True)).reset_index(
            drop=True)
        train_balanced = train_balanced.sample(frac=1, random_state=42)
        dataset_balanced = DFDataset(train_balanced, convert_df_row)
        dataLoader_balanced = DataLoader(dataset_balanced, batch_size=batch_size)
        dataset_description = "i2b2_window_60_oversampled"
        train(model, dataLoader=dataLoader_balanced, dataLoader_val=dataLoader_val,
              epochs=pretrain_epochs, learning_rate=learning_rate, weight_decay=weight_decay, optimizer=optimizer,
              batch_size=batch_size, pretraining=True)

    dataset_description = "i2b2_window_60_unbalanced"
    log_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    top_performance, top_epoch, top_scores, log_name = train(model, dataLoader, dataLoader_val,
                                                             epochs=epochs, learning_rate=learning_rate,
                                                             weight_decay=weight_decay, optimizer=optimizer,
                                                             batch_size=batch_size,
                                                             save_path=path.join('checkpoints', log_name),
                                                             log_name=log_name)

    results = log_name + \
              "\t" + str(top_scores['f1']) + \
              "\t" + str(top_scores['acc']) + \
              "\t" + str(top_epoch) + \
              "\t" + str(pool_mode) + \
              "\t" + str(pretrain) + \
              "\t" + str(deeper_network) + \
              "\t" + str(epochs) + \
              "\t" + str(learning_rate) + \
              "\t" + str(weight_decay) + \
              "\t" + str(optimizer) + \
              "\t" + str(batch_size) + \
              "\t" + str(pool_mode) + "\n"

    header = "log_name" + \
             "\t" + str("f1") + \
             "\t" + str("accuracy") + \
             "\t" + str("top_epoch") + \
             "\t" + str("pool_mode") + \
             "\t" + str("pretrain") + \
             "\t" + str("deeper_network") + \
             "\t" + str("epochs") + \
             "\t" + str("learning_rate") + \
             "\t" + str("weight_decay") + \
             "\t" + str("optimizer") + \
             "\t" + str("batch_size") + \
             "\t" + str("pool_mode") + "\n"

    print(header)
    print(results)
    with open("hyper_parameter_test_results.txt", "a") as file_object:
        file_object.write(header)
        file_object.write(results)
    return top_performance


def generate_hyper_parameters(df, batch_size=64):
    pool_modes = ['both_events', 'cls', 'pool']
    pretraining = [True, False]
    learning_rate = [1e-1, 5e-2, 1e-2]
    best_lr = 0
    best_pool_mode = 0
    best_pretraining = 0

    train_df, val_df, test_df = split_data(df, oversample=False)
    best_score = 0
    for i, lr in enumerate(learning_rate):
        performance = hyperparameters_test(pool_modes[best_pool_mode], pretrain=pretraining[best_pretraining],
                                           train_df=train_df, val_df=val_df, learning_rate=lr, batch_size=batch_size)
        if performance > best_score:
            best_score = performance
            best_lr = i

    best_score = 0
    for i, pm in enumerate(pool_modes):
        performance = hyperparameters_test(pm, pretrain=pretraining[best_pretraining],
                                           train_df=train_df, val_df=val_df, learning_rate=learning_rate[best_lr],
                                           batch_size=batch_size)
        if performance > best_score:
            best_score = performance
            best_pool_mode = i

    best_score = 0
    for i, pre in enumerate(pretraining):
        performance = hyperparameters_test(pool_modes[best_pool_mode], pretrain=pre,
                                           train_df=train_df, val_df=val_df, learning_rate=learning_rate[best_lr],
                                           batch_size=batch_size)
        if performance > best_score:
            best_score = performance
            best_pretraining = i


def graph_model_testing():
    configuration = Configuration()
    configuration.add_inverse_relations_to_graph = True
    configuration.add_transitive_relations_to_graph = True
    configuration.use_entire_graph = False

    model = GNNRelationPrediction(use_edge_features=False)
    model.simulate_mislabeled_relations = False
    df = combining_data.window_for_entity_bert(combining_data.read_i2b2(full_text=True))
    train_df, val_df, test_df = split_data(df, oversample=True, split_by_documents=True)
    train_graph = in_memory_kg(train_df)
    val_graph = in_memory_kg(val_df)
    dataset = KnowledgeGraphDataset(train_graph, train_df, configuration=configuration)
    dataLoader = torch_geometric.loader.DataLoader(dataset, batch_size=64)
    dataset_val = KnowledgeGraphDataset(val_graph, val_df, configuration=configuration)
    dataLoader_val = torch_geometric.loader.DataLoader(dataset_val, batch_size=64)
    train(model, dataLoader=dataLoader, dataLoader_val=dataLoader_val,
          epochs=20, learning_rate=0.1, weight_decay=0.01, optimizer="sgd",
          batch_size=64, pretraining=True)

def graph_model_testing_new():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    configuration = Configuration()
    configuration.use_realistic_graph = False
    configuration.add_inverse_relations_to_graph = False
    configuration.add_transitive_relations_to_graph = False

    configuration.use_relations_from_other_documents = False
    configuration.realistic_graph['remove_wrong_edges'] = False

    i2b2_test_split = combining_data.read_i2b2(full_text=True, use_test_files=True)
    true_test_graph = construct_graph_from_text_only(i2b2_test_split, configuration, dataset_type="test")
    test_dataset = KnowledgeGraphDataset(true_test_graph, combining_data.window_for_entity_bert(i2b2_test_split),
                                         configuration=configuration)
    dataLoader_test = torch_geometric.loader.DataLoader(test_dataset, batch_size=batch_size)
    torch.save(dataLoader_test, "data_checkpoints/dataLoader_test.pt")

    dataLoader_train_balanced = torch.load("data_checkpoints/dataLoader_train_balanced.pt")
    dataLoader_val_balanced = torch.load("data_checkpoints/dataLoader_val_balanced.pt")
    dataLoader_train = torch.load("data_checkpoints/dataLoader_train.pt")
    dataLoader_val = torch.load("data_checkpoints/dataLoader_val.pt")
    dataLoader_test = torch.load("data_checkpoints/dataLoader_test.pt")

    model = GNNRelationPrediction(use_edge_features=False)
    model.simulate_mislabeled_relations = False

    train(model, dataLoader=dataLoader_train_balanced, dataLoader_val=dataLoader_val_balanced,
          epochs=20, learning_rate=0.1, weight_decay=0.01, optimizer="sgd",
          batch_size=64, pretraining=False)

    train(model, dataLoader=dataLoader_train, dataLoader_val=dataLoader_val,
          epochs=20, learning_rate=0.01, weight_decay=0.001, optimizer="sgd",
          batch_size=64, pretraining=False)
    print(run_test(model, dataLoader_test, device=device))

def text_model_testing(main_df=None, test_df=None):
    possible_plm = ['allenai/scibert_scivocab_cased', 'allenai/scibert_scivocab_uncased', 'bert-base-uncased', 'bert-base-cased']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = BaselineBERTrelationExtraction()
    model = EntityBERTRelationExtraction(dropout=0, deeper_network=False, pooling_strategy='both_events')
    if main_df is None:
        main_df = combining_data.read_i2b2(full_text=True)
    if test_df is None:
        test_df = combining_data.read_i2b2(full_text=True, use_test_files=True)
    df = combining_data.window_for_entity_bert(main_df)

    i2b2_test_split = combining_data.window_for_entity_bert(test_df)
    dataset_test = KnowledgeGraphDataset([], i2b2_test_split)
    dataLoader_test = torch_geometric.loader.DataLoader(dataset_test, batch_size=64)

    for oversample, epochs, save_path in zip([True, False], [20, 100], ['checkpoints/text_model_thyme_balanced_6-7', 'checkpoints/text_model_thyme_6-7']):
        train_df, val_df, test_df = split_data(df, oversample=oversample, split_by_documents=True)
        # train_graph = in_memory_kg(train_df)
        # val_graph = in_memory_kg(val_df)
        dataset = KnowledgeGraphDataset([], train_df)
        dataLoader = torch_geometric.loader.DataLoader(dataset, batch_size=64)
        dataset_val = KnowledgeGraphDataset([], val_df)
        dataLoader_val = torch_geometric.loader.DataLoader(dataset_val, batch_size=64)

        # train(model, dataLoader=dataLoader, dataLoader_val=dataLoader_val,
        #       epochs=20, pretraining=True)

        train(model, dataLoader, dataLoader_val, epochs=epochs, learning_rate=0.001, weight_decay=0.0005,
              optimizer="adamw",
              batch_size=64, pretraining=False, save_path=save_path)

        accuracy, f1 = run_test(model, dataLoader_test, device)
        print("accuracy", accuracy)
        print("f1", f1)
    # top_performance, top_epoch, top_scores, log_name = train(model, dataLoader_val, dataLoader_test, epochs=30)

def text_model_testing_thyme(do_train=True):
    possible_plm = ['allenai/scibert_scivocab_cased', 'allenai/scibert_scivocab_uncased', 'bert-base-uncased', 'bert-base-cased']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = BaselineBERTrelationExtraction()
    model = EntityBERTRelationExtraction(dropout=0, deeper_network=False, pooling_strategy='both_events', number_of_relations=11)
    df = torch.load("thyme_df.pt")
    df = combining_data.window_for_entity_bert(df)

    if not do_train:
        model = torch.load('checkpoints/text_model_thyme/model_best.pt')

    for oversample, epochs, save_path in zip([True, False], [5, 100], [None, 'checkpoints/text_model_thyme']):
        train_df, val_df, test_df = split_data(df, oversample=oversample, split_by_documents=True)
        # train_graph = in_memory_kg(train_df)
        # val_graph = in_memory_kg(val_df)
        dataset = KnowledgeGraphDataset([], train_df)
        dataLoader = torch_geometric.loader.DataLoader(dataset, batch_size=64)
        dataset_val = KnowledgeGraphDataset([], val_df)
        dataLoader_val = torch_geometric.loader.DataLoader(dataset_val, batch_size=64)
        dataset_test = KnowledgeGraphDataset([], test_df)
        dataLoader_test = torch_geometric.loader.DataLoader(dataset_test, batch_size=64)

        if do_train:
            train(model, dataLoader, dataLoader_val, epochs=epochs, learning_rate=0.001, weight_decay=0.0005,
                  optimizer="adamw",
                  batch_size=64, pretraining=False, save_path=save_path)

        accuracy, f1 = run_test(model, dataLoader_test, device)
        print("accuracy", accuracy)
        print("f1", f1)


def training_dataset_with_transitive_relations(full_text_df):
    train_df = combining_data.add_inverse_relations(full_text_df)
    train_df = combining_data.window_for_entity_bert(combining_data.add_transitive_relations(train_df), window_size=60, normalize_event_order=True)
    train_df = train_df.drop_duplicates().reset_index(drop=True)
    return train_df

def construct_graph_from_text_only(full_text_df, configuration, dataset_type=""):
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_df = combining_data.add_inverse_relations(full_text_df)
    test_df = combining_data.add_transitive_relations(test_df)
    test_df = combining_data.window_for_entity_bert(test_df, window_size=60, normalize_event_order=True)
    test_df = test_df.drop_duplicates().reset_index()
    test_dataset = KnowledgeGraphDataset([], test_df, configuration=configuration)
    dataLoader_test = torch_geometric.loader.DataLoader(test_dataset, batch_size=batch_size)

    text_model = torch.load("checkpoints/new_best/Transitive_relation_extraction.pt", map_location=torch.device(device))
    # text_model = torch.load("checkpoints/EntityBert_relation_extraction.pt", map_location=torch.device(device))
    in_memory_graph = []

    text_model.eval()
    text_model.training = False
    labels = []
    predictions = []
    raw_predictions = None
    correct = 0
    n = 0
    n_all = 0
    for batch in dataLoader_test:
        graph, _ = batch
        y = graph.y
        graph.to(device)
        y = y.to(device)
        res = text_model(graph)
        res = torch.softmax(res, dim=1)
        if raw_predictions is None:
            raw_predictions = res
        else:
            raw_predictions = torch.cat((raw_predictions, res), 0)
        pred = res.argmax(dim=1)

        lab = y.tolist()
        labels += lab
        pred = pred.tolist()
        predictions += pred
        for i in range(len(pred)):
            n_all += 1
            if configuration.realistic_graph['remove_wrong_edges'] and pred[i] != lab[i]:
                continue
            if configuration.use_realistic_graph:
                p = pred[i]
                raw_pred = tuple(res[i].tolist())
            else:
                p = lab[i]
                raw_pred = tuple(F.one_hot(torch.tensor(lab[i]), 3).tolist())
            if configuration.realistic_graph['use_threshold_confidence'] and \
                    raw_pred[p] < configuration.realistic_graph['threshold_confidence']:
                continue
            if p == lab[i]:
                correct += 1
            n += 1
            event1 = knowledge_graph_builder.link_entity_to_umls(graph.text[i][graph.event1_start[i]:graph.event1_end[i]])
            event2 = knowledge_graph_builder.link_entity_to_umls(graph.text[i][graph.event2_start[i]:graph.event2_end[i]])
            document_id = graph.document_id[i]
            relation = index_to_label[p]
            in_memory_graph.append([event1, relation, event2, document_id, raw_pred])
    print("Accuracy:", correct / n)
    print("Number of relations:", n / n_all)
    file1 = open("construct graph from text only.log", "a")  # append mode
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    file1.write("Data type=" + dataset_type + ", " + "Date=" + str(now) + ", " + str(configuration) + ", Accuracy=" + str(correct/n) + ", Number of relations=" + str(n / n_all) + " \n")
    file1.close()
    in_memory_graph = knowledge_graph_builder.add_inverse_relations(in_memory_graph)
    return in_memory_graph

def train_text_model_on_transitive_relations(use_thyme=True):
    use_stored_documents = True
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not use_stored_documents:
        if use_thyme:
            dataset = torch.load('thyme_df.pt')
        else:
            dataset = combining_data.read_i2b2(full_text=True)
        dataset = training_dataset_with_transitive_relations(dataset)
        train_df, val_df, test_df = split_data(dataset, oversample=False, split_by_documents=True)

        dataset = KnowledgeGraphDataset([], train_df)
        dataLoader = torch_geometric.loader.DataLoader(dataset, batch_size=64)
        dataset_val = KnowledgeGraphDataset([], val_df)
        dataLoader_val = torch_geometric.loader.DataLoader(dataset_val, batch_size=64)
        dataset_test = KnowledgeGraphDataset([], test_df)
        dataLoader_test = torch_geometric.loader.DataLoader(dataset_test, batch_size=64)

        torch.save(dataLoader, "data_checkpoints/thyme_transitive_dataloader.pt")
        torch.save(dataLoader_val, "data_checkpoints/thyme_transitive_dataloader_val.pt")
        torch.save(dataLoader_test, "data_checkpoints/thyme_transitive_dataloader_test.pt")

    else:
        dataLoader = torch.load("data_checkpoints/thyme_transitive_dataloader.pt")
        dataLoader_val = torch.load("data_checkpoints/thyme_transitive_dataloader_val.pt")
        dataLoader_test = torch.load("data_checkpoints/thyme_transitive_dataloader_test.pt")


    # model = EntityBERTRelationExtraction(3, pooling_strategy='both_events', deeper_network=False)
    # model = torch.load("checkpoints/EntityBert_relation_extraction.pt")
    if use_thyme:
        model = EntityBERTRelationExtraction(dropout=0, deeper_network=False, pooling_strategy='both_events', number_of_relations=11)
    else:
        model = EntityBERTRelationExtraction(dropout=0, deeper_network=False, pooling_strategy='both_events')
    # Settings used for i2b2 transitive model
    # epochs = 100, learning_rate = 0.001, weight_decay = 0.0005, optimizer = "adamw", batch_size = 64, pretraining = False, save_path = 'checkpoints/transitive_text_model_11-5'

    if use_thyme:
        train(model, dataLoader, dataLoader_val, epochs=100, learning_rate=0.5, weight_decay=0.005,
              optimizer="adamw",
              batch_size=64, pretraining=False, save_path='checkpoints/transitive_text_model_6-6')
    else:
        train(model, dataLoader, dataLoader_val, epochs=100, learning_rate=0.001, weight_decay=0.0005, optimizer="adamw", batch_size=64, pretraining=False, save_path='checkpoints/transitive_text_model_11-5')

    accuracy, f1 = run_test(model, dataLoader_test, device)
    print("accuracy", accuracy)
    print("f1", f1)


def realistic_test_graph(configuration, use_cached_dataset=False, dataframe=None, dataset_name='i2b2'):
    print("Realistic test")
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not use_cached_dataset:
        if dataframe is None:
            if dataset_name == 'i2b2':
                dataframe = combining_data.read_i2b2(full_text=True)
            else:
                dataframe = torch.load("thyme_df.pt")

        train_df, val_df, test_df = split_data(dataframe, oversample=False, split_by_documents=True)
        train_df_balanced, val_df_balanced, test_df_balanced = split_data(dataframe, oversample=True, split_by_documents=True)

    if not use_cached_dataset:
        train_graph = construct_graph_from_text_only(train_df, configuration, dataset_type="train")
        val_graph = construct_graph_from_text_only(val_df, configuration, dataset_type="validation")
        val2_graph = construct_graph_from_text_only(test_df, configuration, dataset_type="validation2")

    # train_dataset_balanced = KnowledgeGraphDataset(train_graph, combining_data.window_for_entity_bert(train_df_balanced), configuration=configuration)
    # val_dataset_balanced = KnowledgeGraphDataset(val_graph, combining_data.window_for_entity_bert(val_df_balanced), configuration=configuration)
    # dataLoader_train_balanced = torch_geometric.loader.DataLoader(train_dataset_balanced, batch_size=batch_size)
    # dataLoader_val_balanced = torch_geometric.loader.DataLoader(val_dataset_balanced, batch_size=batch_size)
    # torch.save(dataLoader_train_balanced, "data_checkpoints/dataLoader_train_balanced.pt")
    # torch.save(dataLoader_val_balanced, "data_checkpoints/dataLoader_val_balanced.pt")
    # dataLoader_train_balanced = torch.load("data_checkpoints/dataLoader_train_balanced.pt")
    # dataLoader_val_balanced = torch.load("data_checkpoints/dataLoader_val_balanced.pt")

    if use_cached_dataset:
        dataLoader_train = torch.load("data_checkpoints/dataLoader_train.pt")
    else:
        train_dataset = KnowledgeGraphDataset(train_graph, combining_data.window_for_entity_bert(train_df), configuration=configuration)
        dataLoader_train = torch_geometric.loader.DataLoader(train_dataset, batch_size=batch_size)
        torch.save(dataLoader_train, "data_checkpoints/dataLoader_train.pt")

    if use_cached_dataset:
        dataLoader_val = torch.load("data_checkpoints/dataLoader_val.pt")
    else:
        val_dataset = KnowledgeGraphDataset(val_graph, combining_data.window_for_entity_bert(val_df), configuration=configuration)
        dataLoader_val = torch_geometric.loader.DataLoader(val_dataset, batch_size=batch_size)
        torch.save(dataLoader_val, "data_checkpoints/dataLoader_val.pt")

    if use_cached_dataset:
        dataLoader_val2 = torch.load("data_checkpoints/dataLoader_val2.pt")
    else:
        val2_dataset = KnowledgeGraphDataset(val2_graph, combining_data.window_for_entity_bert(test_df), configuration=configuration)
        dataLoader_val2 = torch_geometric.loader.DataLoader(val2_dataset, batch_size=batch_size)
        torch.save(dataLoader_val2, "data_checkpoints/dataLoader_val2.pt")

    if use_cached_dataset:
        dataLoader_test = torch.load("data_checkpoints/dataLoader_test.pt")
    else:
        i2b2_test_split = combining_data.read_i2b2(full_text=True, use_test_files=True)
        true_test_graph = construct_graph_from_text_only(i2b2_test_split, configuration, dataset_type="test")
        test_dataset = KnowledgeGraphDataset(true_test_graph, combining_data.window_for_entity_bert(i2b2_test_split), configuration=configuration)
        dataLoader_test = torch_geometric.loader.DataLoader(test_dataset, batch_size=batch_size)
        torch.save(dataLoader_test, "data_checkpoints/dataLoader_test.pt")

    # graph_model = torch.load("checkpoints/new_best/graph_model_realistic.pt", map_location=torch.device(device))
    graph_model = GNNRelationPrediction(use_edge_features=True, dropout=0.1)

    # print("pretraining")
    # train(graph_model, dataLoader_train_balanced, dataLoader_val_balanced, epochs=60, learning_rate=0.1, weight_decay=0.01,
    #       optimizer="sgd",
    #       batch_size=64, pretraining=False)
    # print("real training")

    train(graph_model, dataLoader_train, dataLoader_val, epochs=100, learning_rate=0.0001, weight_decay=0.1,
          optimizer="adamw",
          batch_size=64, pretraining=False, save_path='checkpoints/graph_relation_extraction')

    global model_description
    # # torch.save(graph_model, "checkpoints/graph_relation_extraction_3.pt")
    model_description = "graph"
    graph_model = torch.load('checkpoints/graph_relation_extraction/model_best.pt')
    graph_accuracy, graph_f1 = run_test(graph_model, dataLoader_test, device)
    print("graph F1:", graph_f1)

    model_description = "text"
    if dataset_name == 'i2b2':
        text_model = torch.load("checkpoints/new_best/EntityBert_relation_extraction.pt")
    else:
        text_model = torch.load("checkpoints/text_model_thyme/model_best.pt")
    # text_model = torch.load("checkpoints/new_best/Transitive_relation_extraction.pt")
    # # bimodal_model = torch.load("checkpoints/bimodal_relation_extraction.pt")
    text_accuracy, text_f1 = run_test(text_model, dataLoader_test, device)
    print("text F1:", text_f1)

    model_description = "bimodal"
    bimodal_model = MultiModalPrediction(combine_embeddings=False)
    # bimodal_model = torch.load('checkpoints/new_best/bimodal_on_realistic.pt')
    # bimodal_model = torch.load('checkpoints/bimodal_relation_extraction.pt')
    bimodal_model.text_model = text_model
    bimodal_model.graph_model = graph_model

    # torch.save(dataLoader_train, "checkpoints/tmp1.pt")
    # torch.save(dataLoader_val, "checkpoints/tmp2.pt")

    train(bimodal_model, dataLoader_val, dataLoader_val2, epochs=20, learning_rate=0.1, weight_decay=0.01,
          optimizer="sgd",
          batch_size=64, pretraining=False,
          save_path='checkpoints/bimodal_threshold')
    train(bimodal_model, dataLoader_val, dataLoader_val2, epochs=30, learning_rate=0.01, weight_decay=0.01,
          optimizer="sgd",
          batch_size=64, pretraining=False,
          save_path='checkpoints/bimodal_threshold')
    bimodal_model = torch.load('checkpoints/bimodal_threshold/model_best.pt')
    bimodal_model.combine_embeddings = True
    bimodal_accuracy, bimodal_f1 = run_test(bimodal_model, dataLoader_test, device)
    print("bimodal F1:", bimodal_f1)
    # torch.save(bimodal_model, 'checkpoints/bimodal_relation_extraction2.pt')
    return {"graph_accuracy": graph_accuracy,
            "graph_f1": graph_f1,
            "text_accuracy": text_accuracy,
            "text_f1": text_f1,
            "bimodal_accuracy": bimodal_accuracy,
            "bimodal_f1": bimodal_f1}

def create_global_graph(train_df, val_df, test_df):
    configuration = Configuration()
    configuration.use_entire_graph = True
    configuration.no_document_filtering = True
    configuration.remove_target_relation = False

    graph = in_memory_kg(train_df)
    train_dataset = KnowledgeGraphDataset(graph, val_df, configuration=configuration)
    val_dataset = KnowledgeGraphDataset(graph, test_df, configuration=configuration)
    return train_dataset, val_dataset

def global_knowledge_graph_test():
    i2b2 = combining_data.window_for_entity_bert(combining_data.read_i2b2(full_text=True))
    train_df, val_df, test_df = split_data(i2b2, oversample=False, split_by_documents=True)
    train_dataset, val_dataset = create_global_graph(train_df, val_df, test_df)
    batch_size = 64
    graph_model = GNNRelationPrediction()
    dataLoader = torch_geometric.loader.DataLoader(train_dataset, batch_size=batch_size)
    dataLoader_val = torch_geometric.loader.DataLoader(val_dataset, batch_size=batch_size)
    train(graph_model, dataLoader, dataLoader_val, learning_rate=1e-1, weight_decay=0.01)

dataset_description = "i2b2_window_60"
model_description = ""

def classic_test():
    batch_size = 64
    configuration = Configuration()
    configuration.use_realistic_graph = True
    configuration.use_relations_from_other_documents = False
    configuration.add_inverse_relations_to_graph = False
    configuration.add_transitive_relations_to_graph = False
    configuration.realistic_graph['remove_wrong_edges'] = False

    i2b2_test_split = combining_data.read_i2b2(full_text=True, use_test_files=True)
    true_test_graph = in_memory_kg(i2b2_test_split)
    test_dataset = KnowledgeGraphDataset(true_test_graph, combining_data.window_for_entity_bert(i2b2_test_split),
                                         configuration=configuration)
    dataLoader_test = torch_geometric.loader.DataLoader(test_dataset, batch_size=batch_size)

    text_model = torch.load("checkpoints/EntityBert_relation_extraction.pt")
    graph_model = torch.load("checkpoints/graph_relation_extraction.pt")
    bimodal_model = torch.load("checkpoints/bimodal_relation_extraction.pt")
    accuracy, f1 = run_test(text_model, dataLoader_test, device)
    print("text F1:", f1)
    accuracy, f1 = run_test(graph_model, dataLoader_test, device)
    print("graph F1:", f1)
    accuracy, f1 = run_test(bimodal_model, dataLoader_test, device)
    print("bimodal F1:", f1)

def thyme_corpus():
    model = BaselineBERTrelationExtraction()
    df = torch.load("thyme_df.pt")
    df.loc[df['class'] == 'BEGINS-ON', 'class'] = "BEFORE"
    df.loc[df['class'] == 'ENDS-ON', 'class'] = "BEFORE"
    df.loc[df['class'] == 'CONTINUES', 'class'] = "BEFORE"
    df.loc[df['class'] == 'INITIATES', 'class'] = "BEFORE"
    df.loc[df['class'] == 'CONTAINS', 'class'] = "OVERLAP"
    df.loc[df['class'] == 'REINITIATES', 'class'] = "BEFORE"
    df.loc[df['class'] == 'TERMINATES', 'class'] = "BEFORE"


    df = combining_data.window_for_entity_bert(df)
    train_df, val_df, test_df = split_data(df, oversample=True, split_by_documents=True)
    dataset = KnowledgeGraphDataset([], train_df)
    dataLoader = torch_geometric.loader.DataLoader(dataset, batch_size=64)
    dataset_val = KnowledgeGraphDataset([], val_df)
    dataLoader_val = torch_geometric.loader.DataLoader(dataset_val, batch_size=64)
    train(model, dataLoader, dataLoader_val, epochs=30, learning_rate=0.05, weight_decay=0.0001,
          optimizer="sgd",
          batch_size=64, pretraining=False)
    accuracy, f1 = run_test(model, dataLoader_val, device)
    print("accuracy", accuracy)
    print("f1", f1)

def comparing_graphs():
    df = torch.load("thyme_df.pt")
    df.loc[df['class'] == 'BEGINS-ON', 'class'] = "BEFORE"
    df.loc[df['class'] == 'ENDS-ON', 'class'] = "BEFORE"
    df.loc[df['class'] == 'CONTINUES', 'class'] = "BEFORE"
    df.loc[df['class'] == 'INITIATES', 'class'] = "BEFORE"
    df.loc[df['class'] == 'CONTAINS', 'class'] = "OVERLAP"
    df.loc[df['class'] == 'REINITIATES', 'class'] = "BEFORE"
    df.loc[df['class'] == 'TERMINATES', 'class'] = "BEFORE"
    df = df.iloc[:5]

    graph1 = in_memory_kg(df)
    graph1 = knowledge_graph_builder.add_inverse_relations(graph1)
    graph1 = knowledge_graph_builder.add_transitive_relations(graph1, only_additional=False)

    configuration = Configuration()
    configuration.use_realistic_graph = False
    configuration.add_inverse_relations_to_graph = False
    configuration.add_transitive_relations_to_graph = False
    graph2 = construct_graph_from_text_only(df, configuration, dataset_type="thyme-entire")

    knowledge_graph_builder.visualise_graph(graph1)
    knowledge_graph_builder.visualise_graph(graph2)

def generate_threshold_curve():
    number_of_steps = 10
    for threshold in range(4, number_of_steps + 1, 1):
        configuration = Configuration()
        configuration.use_realistic_graph = False
        configuration.simulated_realistic_graph['use_simulated_realistic_graph'] = True
        configuration.simulated_realistic_graph['portion_of_relations_to_keep'] = threshold/number_of_steps
        configuration.add_inverse_relations_to_graph = True
        configuration.add_transitive_relations_to_graph = False

        configuration.use_relations_from_other_documents = False
        configuration.realistic_graph['remove_wrong_edges'] = False
        configuration.realistic_graph['use_threshold_confidence'] = False
        configuration.realistic_graph['threshold_confidence'] = threshold / number_of_steps
        # train_text_model_on_transitive_relations()
        # text_model_testing()
        results = realistic_test_graph(configuration, use_cached_dataset=False)
        results['portion_of_relations'] = configuration.realistic_graph['threshold_confidence']
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        results['time'] = str(now)
        file1 = open("test_different_portions_of_graph.log", "a")  # append mode
        file1.write(str(results) + " \n")
        file1.close()

def statistical_significance(use_thyme=False):
    number_of_tests = 1
    cache = True
    for i in range(number_of_tests):
        configuration = Configuration()
        configuration.use_realistic_graph = False
        configuration.add_inverse_relations_to_graph = True
        configuration.add_transitive_relations_to_graph = False

        configuration.use_relations_from_other_documents = False
        configuration.realistic_graph['remove_wrong_edges'] = False
        configuration.realistic_graph['use_threshold_confidence'] = False
        configuration.realistic_graph['threshold_confidence'] = 1
        # train_text_model_on_transitive_relations()
        # text_model_testing()
        results = realistic_test_graph(configuration, use_cached_dataset=cache, dataset_name='thyme' if use_thyme else 'i2b2')
        cache = True
        # now = datetime.now().strftime("%Y%m%d-%H%M%S")
        # results['time'] = str(now)
        # file1 = open("construct graph from text only.log", "a")  # append mode
        # file1.write(str(results) + " \n")
        # file1.close()

def full_test_battery_for_new_dataset(configuration, df, df_test=None, settings=None):
    print(configuration)
    if settings is None:
        settings = {
            "dataset_name": 'THYME',
            "text_model_pretraining": False,
            "oversample_dataset": True,
            "train_text_model": True,
            "train_graph_model": True,
            "use_cached_datasets": False,
        }

    # Initial configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64

    # Dataset preparation
    if df_test is not None:
        df_test_windowed = combining_data.window_for_entity_bert(df_test)
        if settings['use_cached_datasets']:
            save_path = 'checkpoints/battery_' + settings['dataset_name']
            dataLoader_test = torch.load(save_path + '/dataset_cache' + '_test.pt')
        else:
            dataset_test = KnowledgeGraphDataset([], df_test_windowed)
            dataLoader_test = torch_geometric.loader.DataLoader(dataset_test, batch_size=64)
    number_of_classes = len(df_test_windowed['class'].unique())

    # Create a basic text model for dataset
    model = EntityBERTRelationExtraction(dropout=0, deeper_network=False, pooling_strategy='both_events', number_of_relations=number_of_classes)

    # train the text model
    if settings['train_text_model']:
        if settings['text_model_pretraining']:
            steps = zip([True, False], [20, 100], ['checkpoints/battery_' + settings['dataset_name'] + '_pretrain',
                                                   'checkpoints/battery_' + settings['dataset_name']])
        else:
            steps = zip([settings['oversample_dataset']], [100], ['checkpoints/battery_' + settings['dataset_name']])
        for oversample, epochs, save_path in steps:
            train_df, val_df, test_df = split_data(df, oversample=oversample, split_by_documents=True)
            dataset = KnowledgeGraphDataset([], combining_data.window_for_entity_bert(train_df), number_of_classes=number_of_classes)
            dataLoader = torch_geometric.loader.DataLoader(dataset, batch_size=64)
            dataset_val = KnowledgeGraphDataset([], combining_data.window_for_entity_bert(val_df), number_of_classes=number_of_classes)
            dataLoader_val = torch_geometric.loader.DataLoader(dataset_val, batch_size=64)
            train(model, dataLoader, dataLoader_val, epochs=epochs, 
                  learning_rate=configuration.training['text']['learning_rate'], 
                  weight_decay=configuration.training['text']['weight_decay'],
                  optimizer=configuration.training['text']['optimizer'],
                  batch_size=64, pretraining=False, save_path=save_path + '_text')
            text_model = torch.load(save_path + '_text' + '/model_best.pt')
            accuracy, f1 = run_test(text_model, dataLoader_test, device)
            print("treniranje tekstovnega modela")
            print("accuracy", accuracy)
            print("f1", f1)
    else:
        save_path = 'checkpoints/battery_' + settings['dataset_name']
        train_df, val_df, test_df = split_data(df, oversample=settings['oversample_dataset'], split_by_documents=True)
        text_model = torch.load(save_path + '_text' + '/model_best.pt')
        accuracy, f1 = run_test(text_model, dataLoader_test, device)
        print("text f1", f1)

    # Prepare realistic graph
    if settings['use_cached_datasets']:
        dataLoader_train = torch.load(save_path + '/dataset_cache' + '_train.pt')
    else:
        train_graph = construct_graph_from_text_only(train_df, configuration, dataset_type="train")
        train_dataset = KnowledgeGraphDataset(train_graph, combining_data.window_for_entity_bert(train_df), configuration=configuration)
        dataLoader_train = torch_geometric.loader.DataLoader(train_dataset, batch_size=batch_size)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(dataLoader_train, save_path + '/dataset_cache' + '_train.pt')

    if settings['use_cached_datasets']:
        dataLoader_val = torch.load(save_path + '/dataset_cache' + '_val.pt')
    else:
        val_graph = construct_graph_from_text_only(val_df, configuration, dataset_type="validation")
        val_dataset = KnowledgeGraphDataset(val_graph, combining_data.window_for_entity_bert(val_df), configuration=configuration)
        dataLoader_val = torch_geometric.loader.DataLoader(val_dataset, batch_size=batch_size)
        torch.save(dataLoader_val, save_path + '/dataset_cache' + '_val.pt')

    if settings['use_cached_datasets']:
        dataLoader_val2 = torch.load(save_path + '/dataset_cache' + '_val2.pt')
    else:
        val2_graph = construct_graph_from_text_only(test_df, configuration, dataset_type="validation2")
        val2_dataset = KnowledgeGraphDataset(val2_graph, combining_data.window_for_entity_bert(test_df), configuration=configuration)
        dataLoader_val2 = torch_geometric.loader.DataLoader(val2_dataset, batch_size=batch_size)
        torch.save(dataLoader_val2, save_path + '/dataset_cache' + '_val2.pt')

    if settings['use_cached_datasets']:
        dataLoader_test = torch.load(save_path + '/dataset_cache' + '_test.pt')
    else:
        test_graph = construct_graph_from_text_only(df_test, configuration, dataset_type="test")
        test_dataset = KnowledgeGraphDataset(test_graph, df_test_windowed, configuration=configuration)
        dataLoader_test = torch_geometric.loader.DataLoader(test_dataset, batch_size=batch_size)
        torch.save(dataLoader_test, save_path + '/dataset_cache' + '_test.pt')

    if settings['train_graph_model']:
        graph_model = GNNRelationPrediction(use_edge_features=True, dropout=0.1, number_of_relations=number_of_classes)
        # graph_model = torch.load('checkpoints/new_best/maccrobat_graph_starter.pt')
        train(graph_model, dataLoader_train, dataLoader_val, epochs=100, 
              learning_rate=configuration.training['graph']['learning_rate'], 
              weight_decay=configuration.training['graph']['weight_decay'],
              optimizer=configuration.training['graph']['optimizer'],
              batch_size=64, pretraining=False, save_path=save_path + '_graph')
    graph_model = torch.load(save_path + '_graph' + '/model_best.pt')
    graph_accuracy, graph_f1 = run_test(graph_model, dataLoader_test, device)
    print("graph F1:", graph_f1)

    bimodal_model = MultiModalPrediction(combine_embeddings=False, number_of_relations=number_of_classes)
    bimodal_model.text_model = text_model
    bimodal_model.graph_model = graph_model
    bimodal_model.combine_embeddings = False
    train(bimodal_model, dataLoader_train, dataLoader_val, epochs=10,
          learning_rate=configuration.training['bimodal']['learning_rate'], 
          weight_decay=configuration.training['bimodal']['weight_decay'],
          optimizer=configuration.training['bimodal']['optimizer'],
          batch_size=64, pretraining=False,
          save_path=save_path + '_bimodal')
    bimodal_model = torch.load(save_path + '_bimodal' + '/model_best.pt')
    train(bimodal_model, dataLoader_train, dataLoader_val, epochs=30, 
          learning_rate=configuration.training['bimodal2']['learning_rate'], 
          weight_decay=configuration.training['bimodal2']['weight_decay'],
          optimizer=configuration.training['bimodal2']['optimizer'],
          batch_size=64, pretraining=False,
          save_path=save_path + '_bimodal')
    bimodal_model = torch.load(save_path + '_bimodal' + '/model_best.pt')
    bimodal_accuracy, bimodal_f1 = run_test(bimodal_model, dataLoader_test, device)
    print("bimodal F1:", bimodal_f1)

if __name__ == '__main__':

    # configuration = Configuration()
    # configuration.use_realistic_graph = True
    # configuration.add_inverse_relations_to_graph = True
    # configuration.add_transitive_relations_to_graph = False
    #
    # configuration.use_relations_from_other_documents = False
    # configuration.realistic_graph['remove_wrong_edges'] = False
    # # train_text_model_on_transitive_relations()
    # # text_model_testing()
    # realistic_test_graph(configuration)


    # text_model_testing_thyme(do_train=False)
    # comparing_graphs()
    # thyme_corpus()
    # graph_model_testing()
    # sys.exit(0)
    # classic_test()
    # global_knowledge_graph_test()

    # generate_threshold_curve()

    # statistical_significance(use_thyme=False)

    configuration = Configuration()
    configuration.use_realistic_graph = True
    configuration.add_inverse_relations_to_graph = True
    configuration.add_transitive_relations_to_graph = False

    configuration.use_relations_from_other_documents = False
    configuration.realistic_graph['remove_wrong_edges'] = False
    configuration.realistic_graph['use_threshold_confidence'] = False
    configuration.realistic_graph['threshold_confidence'] = 0.7
    # train_text_model_on_transitive_relations()
    df = torch.load('thyme_df.pt')
    df_test = torch.load('thyme_test_df.pt')

    df = combining_data.convert_thymre_relations_to_i2b2(df)
    df_test = combining_data.convert_thymre_relations_to_i2b2(df_test)
    df_test = common.oversample(df_test, oversample=True)
    full_test_battery_for_new_dataset(configuration, df, df_test)

    # text_model_testing(df, df_test)
    # realistic_test_graph(configuration, use_cached_dataset=False)
    sys.exit(0)

    # for border in [0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95]:
    #     (raw_predictions, labels) = torch.load("surove_napovedi.pt")
    #     raw_predictions = raw_predictions.cpu()
    #     pred = torch.softmax(raw_predictions, dim=1)
    #     filter = pred.max(dim=1).values > border
    #     labels = torch.tensor(labels)[filter]
    #     pred = pred[filter]
    #     pred = pred.argmax(dim=1)
    #     accuracy = accuracy_score(labels, pred)
    #     print("border", border)
    #     print("accuracy", accuracy)
    #     print("size", len(labels))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # construct_graph_from_text_only()


    # graph_model_testing()
    # text_model_testing()
    # Testing text model only
    # model = EntityBERTRelationExtraction(3, pooling_strategy='both_events', deeper_network=True)
    # df = combining_data.window_for_entity_bert(combining_data.read_i2b2(full_text=True))
    # train_df, val_df, test_df = split_data(df, oversample=True, split_by_documents=True)
    # dataset = KnowledgeGraphDataset([], train_df)
    # dataLoader = torch_geometric.loader.DataLoader(dataset, batch_size=64)
    # dataset_val = KnowledgeGraphDataset([], val_df)
    # dataLoader_val = torch_geometric.loader.DataLoader(dataset_val, batch_size=64)
    # train(model, dataLoader=dataLoader, dataLoader_val=dataLoader_val,
    #       epochs=20, learning_rate=0.1, weight_decay=0.0001, optimizer="sgd",
    #       batch_size=64, pretraining=True)
    #
    # train_df, val_df, test_df = split_data(df, oversample=False, split_by_documents=True)
    # dataset = KnowledgeGraphDataset([], train_df)
    # dataLoader = torch_geometric.loader.DataLoader(dataset, batch_size=64)
    # dataset_val = KnowledgeGraphDataset([], val_df)
    # dataLoader_val = torch_geometric.loader.DataLoader(dataset_val, batch_size=64)
    # train(model, dataLoader=dataLoader, dataLoader_val=dataLoader_val,
    #       epochs=20, learning_rate=0.1, weight_decay=0.0001, optimizer="sgd",
    #       batch_size=64, pretraining=True)




    # generate_hyper_parameters(df)

    # model = GNNRelationPrediction()
    # model = EntityBERTRelationExtraction(3, pooling_strategy='both_events', deeper_network=True)
    # model = torch.load("./checkpoints/EntityBert_relation_extraction.pt")
    # model = MultiModalPrediction()
    # model.graph_model.simulate_mislabeled_relations = True

    configuration = Configuration()
    configuration.add_inverse_relations_to_graph = True
    configuration.add_transitive_relations_to_graph = True

    model = GNNRelationPrediction()
    if False and exists("graph_dataset_train.pt") and exists("graph_dataset_val.pt"):
        train_dataset = torch.load("graph_dataset_train.pt")
        val_dataset = torch.load("graph_dataset_val.pt")
        test_dataset = torch.load("graph_dataset_test.pt")
        train_dataset_balanced = torch.load("graph_dataset_train_balanced.pt")
        val_dataset_balanced = torch.load("graph_dataset_val_balanced.pt")
    else:
        i2b2 = combining_data.window_for_entity_bert(combining_data.read_i2b2(full_text=True))
        train_df, val_df, test_df = split_data(i2b2, oversample=False, split_by_documents=True)
        train_graph = in_memory_kg(train_df)
        val_graph = in_memory_kg(val_df)
        test_graph = in_memory_kg(test_df)
        train_dataset = KnowledgeGraphDataset(train_graph, train_df, configuration=configuration)
        val_dataset = KnowledgeGraphDataset(val_graph, val_df, configuration=configuration)
        test_dataset = KnowledgeGraphDataset(test_graph, test_df, configuration=configuration)
        torch.save(train_dataset, "graph_dataset_train.pt")
        torch.save(val_dataset, "graph_dataset_val.pt")
        torch.save(test_dataset, "graph_dataset_test.pt")

        train_df_balanced, val_df_balanced, test_df_balanced = split_data(i2b2, oversample=True, split_by_documents=True)
        train_dataset_balanced = KnowledgeGraphDataset(train_graph, train_df_balanced, configuration=configuration)
        val_dataset_balanced = KnowledgeGraphDataset(val_graph, val_df_balanced, configuration=configuration)
        torch.save(train_dataset_balanced, "graph_dataset_train_balanced.pt")
        torch.save(val_dataset_balanced, "graph_dataset_val_balanced.pt")

    # i2b2_test_split = combining_data.window_for_entity_bert(combining_data.read_i2b2(full_text=True, use_test_files=True))
    # true_test_graph = in_memory_kg(i2b2_test_split)
    # true_test_dataset = KnowledgeGraphDataset(true_test_graph, i2b2_test_split)

    batch_size = 256
    # batch_size = 32
    dataLoader_balanced = torch_geometric.loader.DataLoader(train_dataset_balanced, batch_size=batch_size)
    dataLoader_val_balanced = torch_geometric.loader.DataLoader(val_dataset_balanced, batch_size=batch_size)
    train(model, dataLoader_balanced, dataLoader_val_balanced, epochs=30, learning_rate=1e-1, weight_decay=0.01)

    dataLoader = torch_geometric.loader.DataLoader(train_dataset, batch_size=batch_size)
    dataLoader_val = torch_geometric.loader.DataLoader(val_dataset, batch_size=batch_size)
    dataLoader_test = torch_geometric.loader.DataLoader(test_dataset, batch_size=batch_size)
    train(model, dataLoader, dataLoader_val, learning_rate=1e-1, weight_decay=0.01)

    # text_model = EntityBERTRelationExtraction(3, pooling_strategy='both_events', deeper_network=False)
    # train(text_model, dataLoader_balanced, dataLoader_val_balanced, epochs=60, learning_rate=0.05, weight_decay=0.0001,
    #       optimizer="sgd",
    #       batch_size=batch_size, pretraining=True)
    # train(text_model, dataLoader, dataLoader_val, epochs=30, learning_rate=0.05, weight_decay=0.0001,
    #       optimizer="sgd",
    #       batch_size=batch_size, pretraining=False)
    #
    # text_model = torch.load("checkpoints/EntityBert_relation_extraction.pt")
    # graph_model = torch.load("checkpoints/graph_relation_extraction.pt")
    # bimodal_model = torch.load("checkpoints/bimodal_relation_extraction.pt")
    # accuracy, f1 = run_test(text_model, dataLoader_test, device)
    # print("text F1:", f1)
    # accuracy, f1 = run_test(graph_model, dataLoader_test, device)
    # print("graph F1:", f1)
    # accuracy, f1 = run_test(bimodal_model, dataLoader_test, device)
    # print("bimodal F1:", f1)

    # model.text_model = text_model
    # model.graph_model = graph_model
    # train(model, dataLoader, dataLoader_val, epochs=30, pretraining=False, save_path="checkpoints/bimodal")

    # train_in_parts(model, dataLoader, dataLoader_val, dataLoader_test, dataLoader_balanced, dataLoader_val_balanced)



    # model = EntityBERTRelationExtraction(3, pooling_strategy='both_events')

    # tokenizer = AutoTokenizer.from_pretrained("./pretrained models/PubmedBERTbase-MimicBig-EntityBERT", max_length=100, padding=True, truncation=True)
    # g = df.groupby('class')
    # df_balanced = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))

    # print("pretrain")
    # train(model, df_balanced, epochs=10)

    # split data

    # train_df, val_df, test_df = split_data(df, oversample=False)

    # print("train")
    # train(model, train_df, val_df, test_df, epochs=300, batch_size=64, learning_rate=1e-1)