import csv
import os
import urllib
import uuid

import pandas
import pandas as pd
from conllu import parse

from rdflib import Graph, URIRef, BNode, Literal, Namespace, RDF, RDFS
from spacy.lang.en import English

print(os.getcwd())

from common import expand_sentence
from dataLoaders.i2b2dataLoader import load_absolute_data, load_data
from spacy.tokenizer import Tokenizer


def read_i2b2(full_text=False, use_test_files=False, include_rows_without_absolute=False):
    keys = ['event1_start_time', 'event2_start_time', 'event1_end_time', 'event2_end_time']
    original_relations = load_absolute_data(full_text=full_text, use_test_files=use_test_files, include_rows_without_absolute=include_rows_without_absolute)
    relations = []
    for i, relation in original_relations.iterrows():
        additional_information = relation['additional_information']
        additional_list = [additional_information[key] for key in keys]
        relations.append([relation['text'], relation['class'], relation['event1_start'], relation['event1_end'], None,
                          relation['event2_start'], relation['event2_end'], None, relation['event1_text'],
                          relation['event2_text'], relation['file_id'], "I2B2",
                          relation['minutes_between_means']] + additional_list)

    df = pandas.DataFrame(relations)
    df.columns = ['text', 'class', 'event1_start', 'event1_end', 'event1_type', 'event2_start', 'event2_end',
                  'event2_type', 'event1_text',
                  'event2_text', 'document_id', 'source', 'minutes_between_means'] + keys
    return df


def build_relation_record(event1, event2, text, parts, document_id, source, reverse=False):
    relation = parts[1]
    if reverse:
        event1, event2 = event2, event1
        relation = relation_inverse(relation)

    event1_start = int(event1[1])
    event1_end = int(event1[2])
    event2_start = int(event2[1])
    event2_end = int(event2[2])
    sentence, event1_start, event1_end, event2_start, event2_end = get_span_of_text_to_include(text, event1_start,
                                                                                               event1_end, event2_start,
                                                                                               event2_end)
    event1_text = sentence[event1_start:event1_end]
    event2_text = sentence[event2_start:event2_end]
    if event1[3] not in event1_text:
        print(event1_text)
    if event2[3] not in event2_text:
        print(event2_text)
    return [sentence, relation, event1_start, event1_end, event1[0], event2_start, event2_end, event2[0], event1_text,
            event2_text,
            document_id, source]


def get_span_of_text_to_include(text, event1_start, event1_end, event2_start, event2_end):
    start, end = expand_sentence(text, min(event1_start, event2_start), max(event1_end, event2_end))
    event1_start -= start
    event2_start -= start
    event1_end -= start
    event2_end -= start
    sentence = text[start:end]
    return sentence, event1_start, event1_end, event2_start, event2_end


def read_macrobat():
    all_overlap_relations = False
    dir = 'data/maccrobat'
    files = os.listdir(dir)
    names = set([name.split('.')[0] for name in files])

    relations = []
    for name in names:
        with open(dir + '/' + name + '.txt') as f:
            text = f.read()
        with open(dir + '/' + name + '.ann') as f:
            annotations = f.readlines()
        # parse annotations
        document_id = name
        mentions = {}
        for annotation in annotations:
            parts = annotation.split()
            if parts[0].startswith("T"):
                parts = [a for a in parts if ';' not in a]
                mentions[parts[0]] = parts[1:]
        events = {}
        for annotation in annotations:
            parts = annotation.split()
            if parts[0].startswith("E"):
                event_type, mention = parts[1].split(":")
                events[parts[0]] = mentions[mention]
        # relations
        for annotation in annotations:
            parts = annotation.split()
            if parts[0].startswith("R") and (parts[1] == "BEFORE" or parts[1] == "AFTER"):
                event1 = events[parts[2].split(":")[1]]
                event2 = events[parts[3].split(":")[1]]
                relations.append(build_relation_record(event1, event2, text, parts, document_id, 'maccrobat'))
                relations.append(
                    build_relation_record(event1, event2, text, parts, document_id, 'maccrobat', reverse=True))
            if parts[0].startswith("*") and parts[1] == "OVERLAP":
                if all_overlap_relations:
                    for i in range(3, len(parts)):
                        for j in range(i + 1, len(parts)):
                            event1 = events[parts[i]]
                            event2 = events[parts[j]]
                            relations.append(build_relation_record(event1, event2, text, parts, document_id, 'maccrobat'))
                else:
                    for i in range(3, len(parts) - 1):
                        event1 = events[parts[i]]
                        event2 = events[parts[i+1]]
                        relations.append(build_relation_record(event1, event2, text, parts, document_id, 'maccrobat'))
        pass

    df = pandas.DataFrame(relations)
    df.columns = ['text', 'class', 'event1_start', 'event1_end', 'event1_type', 'event2_start', 'event2_end',
                  'event2_type', 'event1_text',
                  'event2_text', 'document_id', 'source']
    # g = df.groupby('class')
    # number_of_overlaps = g.size()['BEFORE'] + g.size()['AFTER']
    # df = g.apply(lambda x: x.sample(number_of_overlaps) if x.shape[0] >= number_of_overlaps else x).reset_index(
    #     drop=True)
    return df


def relation_inverse(relation):
    if relation == 'OVERLAP':
        return 'OVERLAP'
    elif relation == 'BEFORE':
        return 'AFTER'
    elif relation == 'AFTER':
        return 'BEFORE'
    elif relation == 'CONTAINED-BY':
        return 'CONTAINS'
    elif relation == 'CONTAINS':
        return 'CONTAINED-BY'
    elif relation == 'BEGINS-ON':
        return 'INITIATES'
    elif relation == 'INITIATES':
        return 'BEGINS-ON'
    elif relation == 'ENDS-ON':
        return 'TERMINATES'
    elif relation == 'TERMINATES':
        return 'ENDS-ON'
    elif relation == 'CONTINUES':
        return 'REINITIATES'
    elif relation == 'REINITIATES':
        return 'CONTINUES'


def read_fine_grained_relations():
    relations = []
    with open('data/en-ud-train.conllu', 'rt', encoding="utf-8") as file:
        data = file.read()
        sentences_train = parse(data)
    with open('data/en-ud-test.conllu', 'rt', encoding="utf-8") as file:
        data = file.read()
        sentences_test = parse(data)
    with open('data/en-ud-dev.conllu', 'rt', encoding="utf-8") as file:
        data = file.read()
        sentences_dev = parse(data)

    with open("data/fine-grained/time_eng_ud_v1.2_2015_10_30.tsv") as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for i, row in enumerate(rd):
            if i == 0:
                continue
            if row[0] == 'train':
                sentences = sentences_train
            if row[0] == 'test':
                sentences = sentences_test
            if row[0] == 'dev':
                sentences = sentences_dev

            sent_id_1 = int(row[2].split(" ")[1])
            sent_id_2 = int(row[6].split(" ")[1])

            sent_id_1, sent_id_2 = sorted((sent_id_1, sent_id_2))

            span_values1 = [int(x) for x in row[3].split("_")]
            ind_1_start = min(span_values1)
            ind_1_end = max(span_values1)

            span_values2 = [int(x) for x in row[7].split("_")]
            ind_2_start = min(span_values2)
            ind_2_end = max(span_values2)

            sentence = ""
            event_1_start_character = 0
            event_1_end_character = 0
            event_2_start_character = 0
            event_2_end_character = 0
            for i, tok in enumerate(sentences[sent_id_1 - 1]):
                if i > 0:
                    sentence += " "
                if i == ind_1_start:
                    event_1_start_character = len(sentence)
                if i == ind_2_start and sent_id_1 == sent_id_2:
                    event_2_start_character = len(sentence)
                sentence += tok['form']
                if i == ind_1_end:
                    event_1_end_character = len(sentence)
                if i == ind_2_end and sent_id_1 == sent_id_2:
                    event_2_end_character = len(sentence)
            if (sent_id_1 != sent_id_2):
                for i, tok in enumerate(sentences[sent_id_2 - 1]):
                    sentence += " "
                    if i == ind_2_start:
                        event_2_start_character = len(sentence)
                    sentence += tok['form']
                    if i == ind_2_end:
                        event_2_end_character = len(sentence)

            event_1_text = sentence[event_1_start_character:event_1_end_character]
            event_2_text = sentence[event_2_start_character:event_2_end_character]

            # calculate class
            event_1_start = int(row[16])
            event_1_end = int(row[17])
            event_2_start = int(row[18])
            event_2_end = int(row[19])
            if event_1_start < event_2_start and event_1_end < event_2_end:
                cls = 'BEFORE'
            elif event_1_start > event_2_start and event_1_end > event_2_end:
                cls = 'AFTER'
            else:
                cls = 'OVERLAP'

            relations.append([sentence, cls, event_1_start_character, event_1_end_character, None,
                              event_2_start_character, event_2_end_character, None, event_1_text, event_2_text,
                              "TreeBank", "fine-grained"])
            relations.append([sentence, relation_inverse(cls), event_2_start_character, event_2_end_character, None,
                              event_1_start_character, event_1_end_character, None, event_2_text, event_1_text,
                              "TreeBank", "fine-grained"])
    return pandas.DataFrame(relations, columns=['text', 'class', 'event1_start', 'event1_end', 'event1_type',
                                                'event2_start', 'event2_end', 'event2_type', 'event1_text',
                                                'event2_text', 'document_id', 'source'])


def generate_thyme_plus_relations(df):
    keys = ['event1_start_time', 'event2_start_time', 'event1_end_time', 'event2_end_time']
    new_relations = []
    for i, row in df.iterrows():
        relation = row['class']
        if row['event1_end_time'] < row['event2_start_time']:
            relation = 'BEFORE'
        elif row['event1_start_time'] > row['event2_end_time']:
            relation = 'AFTER'
        elif row['event1_start_time'] > row['event2_start_time'] and row['event1_end_time'] < row['event2_end_time']:
            relation = 'CONTAINED-BY'
        elif row['event1_start_time'] < row['event2_start_time'] and row['event1_end_time'] > row['event2_end_time']:
            relation = 'CONTAINS'
        elif row['event1_end_time'] == row['event2_start_time']:
            relation = 'BEGINS-ON'
        elif row['event1_start_time'] == row['event2_end_time']:
            relation = 'ENDS-ON'
        else:
            relation = 'OVERLAP'

        if row['event1_start'] < row['event2_start']:
            new_relations.append([
                row['text'],
                relation,
                row['event1_start'],
                row['event1_end'],
                row['event1_type'],
                row['event2_start'],
                row['event2_end'],
                row['event2_type'],
                row['event1_text'],
                row['event2_text'],
                row['document_id'],
                row['source'],
                row['minutes_between_means'],
            ])
        else:
            new_relations.append([
                row['text'],
                relation_inverse(relation),
                row['event2_start'],
                row['event2_end'],
                row['event2_type'],
                row['event1_start'],
                row['event1_end'],
                row['event1_type'],
                row['event2_text'],
                row['event1_text'],
                row['document_id'],
                row['source'],
                row['minutes_between_means'],
            ])
    df = pandas.DataFrame(new_relations)
    df.columns = ['text', 'class', 'event1_start', 'event1_end', 'event1_type', 'event2_start', 'event2_end',
                  'event2_type', 'event1_text',
                  'event2_text', 'document_id', 'source', 'minutes_between_means']
    return df


def get_token_for_char(tokens, char_idx):
    for i, token in enumerate(tokens):
        if char_idx > token.idx:
            continue
        if char_idx == token.idx:
            return i, token
        if char_idx < token.idx:
            return i - 1, tokens[i - 1]
    return len(tokens) - 1, tokens[len(tokens) - 1]


def switch_events(row):
    row['class'] = relation_inverse(row['class'])
    row['event1_start'], row['event2_start'] = row['event2_start'], row['event1_start']
    row['event1_end'], row['event2_end'] = row['event2_end'], row['event1_end']
    row['event1_type'], row['event2_type'] = row['event2_type'], row['event1_type']
    row['event1_text'], row['event2_text'] = row['event2_text'], row['event1_text']

    if "minutes_between_means" in row:
        row["minutes_between_means"] = -row["minutes_between_means"]
    if "event1_start_time" in row:
        row['event1_start_time'], row['event2_start_time'] = row['event2_start_time'], row['event1_start_time']
        row['event1_end_time'], row['event2_end_time'] = row['event2_end_time'], row['event1_end_time']
    return row


def window_for_entity_bert(df, window_size=60, normalize_event_order=True):
    nlp = English()
    tokenizer = Tokenizer(nlp.vocab)
    new_data = []
    for i, row in df.iterrows():
        if normalize_event_order:
            if row['event1_start'] > row['event2_start']:
                row = switch_events(row)
        tokens = tokenizer(row['text'])
        start = min(row['event1_start'], row['event2_start'])
        end = max(row['event1_end'], row['event2_end'])
        start_token, _ = get_token_for_char(tokens, start)
        end_token, _ = get_token_for_char(tokens, end)
        if end_token - start_token > window_size:
            continue
        start_token -= (window_size - (end_token - start_token)) // 2
        end_token += (window_size - (end_token - start_token)) // 2
        end_token += max(0, -start_token)
        start_token = max(0, start_token)
        end_token = min(end_token, len(tokens) - 1)
        start = tokens[start_token].idx
        end = tokens[end_token].idx + len(tokens[end_token])
        row_list = list(row)
        row_list[0] = row['text'][start:end]
        row_list[2] = row['event1_start'] - start
        row_list[3] = row['event1_end'] - start
        row_list[5] = row['event2_start'] - start
        row_list[6] = row['event2_end'] - start
        new_data.append(row_list)
    new_df = pandas.DataFrame(new_data)
    new_df.columns = df.columns
    return new_df


def read_i2b2_plus():
    return generate_thyme_plus_relations(read_i2b2())


def add_inverse_relations(data):
    data = data.reset_index(drop=True)
    start_size = len(data)
    for i in range(start_size):
        row = data.iloc[i].copy()
        row["event1_start"], row["event2_start"] = row["event2_start"], row["event1_start"]
        row["event1_end"], row["event2_end"] = row["event2_end"], row["event1_end"]
        row["event1_text"], row["event2_text"] = row["event2_text"], row["event1_text"]
        row["event1_type"], row["event2_type"] = row["event2_type"], row["event1_type"]
        if "minutes_between_means" in row:
            row["minutes_between_means"] = -row["minutes_between_means"]
        row["class"] = relation_inverse(row["class"])
        data = pd.concat((data, row.to_frame().T))
    return data.drop_duplicates().reset_index(drop=True)


def convert_thymre_relations_to_i2b2(df):
    return df.replace({'class': {
        'OVERLAP': 'OVERLAP',
        'BEGINS-ON': 'AFTER',
        'CONTAINED-BY': 'OVERLAP',
        'CONTAINS': 'OVERLAP',
        'ENDS-ON': 'BEFORE',
        'CONTINUES': 'BEFORE',
        'TERMINATES': 'AFTER',
        'INITIATES': 'BEFORE',
        'REINITIATES': 'BEFORE'
    }})

# def convert_thymre_relations_to_i2b2(df):
#     return df.replace({'class': {
#         'OVERLAP': 'OVERLAP',
#         'BEGINS-ON': 'OVERLAP',
#         'CONTAINED - BY': 'OVERLAP',
#         'CONTAINS': 'OVERLAP',
#         'ENDS-ON': 'OVERLAP',
#         'CONTINUES': 'OVERLAP',
#         'TERMINATES': 'OVERLAP',
#         'INITIATES': 'OVERLAP',
#         'REINITIATES': 'OVERLAP'
#     }})

def add_transitive_to_document(document_data):
    new_rows = []
    document_data = document_data.reset_index(drop=True)
    for i in range(len(document_data)):
        for j in range(len(document_data)):
            if i == j:
                continue
            row1 = document_data.iloc[i].copy()
            row2 = document_data.iloc[j].copy()
            if row1["event2_start"] == row2["event1_start"]:
                new_row = row1
                new_row["event2_start"] = row2["event2_start"]
                new_row["event2_end"] = row2["event2_end"]
                new_row["event2_text"] = row2["event2_text"]
                new_row["event2_type"] = row2["event2_type"]
                if "minutes_between_means" in new_row:
                    new_row["minutes_between_means"] = row1["minutes_between_means"] + row2["minutes_between_means"]
                if row1["class"] == "BEFORE" and row2["class"] != "AFTER":
                    new_row["class"] = "BEFORE"
                elif row1["class"] == "AFTER" and row2["class"] != "BEFORE":
                    new_row["class"] = "AFTER"
                elif row1["class"] == "OVERLAP":
                    new_row["class"] = row2["class"]
                else:
                    new_row["class"] = "OVERLAP"
                new_rows.append(new_row)
    document_data = pd.concat((document_data, pd.DataFrame(new_rows)))
    # print(document_data)
    return document_data


def add_transitive_relations(data):
    document_dfs = []
    for document in set(data["document_id"]):
        document_dfs.append(add_transitive_to_document(data[data["document_id"] == document]))
    return pd.concat(document_dfs).reset_index(drop=True)


if __name__ == '__main__':

    # df = read_macrobat()
    # build_konwledge_graph(read_macrobat())
    # build_konwledge_graph(read_fine_grained_relations())
    # build_konwledge_graph(read_i2b2())
    # build_konwledge_graph(parse_matres.read_matres())
    # i2b2df = load_data(full_text=True, use_test_files=True)
    i2b2df = read_i2b2(full_text=True, use_test_files=True)
    # macrobat_df = read_macrobat()
    # df = pd.concat((read_i2b2(), read_fine_grained_relations()))
    # build_konwledge_graph(df)
    print(df)