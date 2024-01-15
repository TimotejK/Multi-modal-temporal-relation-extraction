import random
import threading
import urllib
import matplotlib.pyplot as plt

import gensim.downloader
import networkx as nx
import numpy as np
import spacy
import scispacy
import torch
import concurrent.futures

from multiprocessing.pool import Pool
from torch_geometric.data import Data
from scispacy.linking import EntityLinker

from common import Configuration, compute_transitive_relation, custom_map
from dataLoaders import combining_data
from dataLoaders.combining_data import read_i2b2
from rdflib import Graph, URIRef, BNode, Literal, Namespace, RDF, RDFS
import torch.nn.functional as F

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

class Document:
    def __init__(self, name, source):
        self.name = name
        self.source = source

nlp = None
glove_vectors = None
def link_entity_to_umls(entity):
    global nlp
    if nlp is None:
        nlp = spacy.load("en_core_sci_sm")
        nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    entities = nlp(entity)
    if len(entities.ents) > 0:
        linked_entities = entities.ents[0]._.kb_ents
        if len(linked_entities) > 0:
            return linked_entities[0][0]
    return entity

def in_memory_kg(df):
    graph = []
    for i, row in df.iterrows():
        graph.append([link_entity_to_umls(row['event1_text']), row['class'], link_entity_to_umls(row['event2_text']), row['document_id']])
    return graph

def get_subgraph(graph, entity1_id, entity2_id, steps=3):
    entities = set()
    entities.add(entity1_id)
    entities.add(entity2_id)
    for k in range(steps):
        new_entities = set()
        for relation in graph:
            if relation[0] in entities:
                new_entities.add(relation[2])
            if relation[2] in entities:
                new_entities.add(relation[0])
        entities = new_entities
    relations = set()
    for relation in graph:
        if relation[0] in entities and relation[2] in entities:
            relations.add(tuple(relation))
    return list(relations)

def get_neighbourhood(graph, initial_entities, steps=3):
    entities = set()
    for e in initial_entities:
        entities.add(e)
    for k in range(steps):
        new_entities = set()
        for relation in graph:
            if relation[0] in entities:
                new_entities.add(relation[2])
            if relation[2] in entities:
                new_entities.add(relation[0])
        entities = entities.union(new_entities)
    return list(entities)

def get_path(graph, entity1_id, entity2_id, max_steps=5, return_multiple=False):
    visited = set()
    visited.add(entity1_id)
    source = {}
    queue = [(entity1_id, 0)]
    paths = []
    shortest_path = max_steps
    while len(queue) > 0:
        q = queue.pop(0)
        if q[1] > max_steps:
            if not return_multiple:
                return None
            else:
                return paths
        if q[1] > shortest_path:
            break
        f = q[0]
        for triplet in graph:
            if triplet[0] == f and triplet[2] not in visited:
                visited.add(triplet[2])
                source[triplet[2]] = f
                queue.append((triplet[2], q[1] + 1))
                if triplet[2] == entity2_id:
                    # we found target
                    node = entity2_id
                    path = [node]
                    while node in source:
                        node = source[node]
                        path.insert(0, node)
                    if not return_multiple:
                        return path
                    paths.append(path)
                    shortest_path = len(path)
    return paths



def build_konwledge_graph(df):
    events = set()
    documents = set()

    relations = []
    event_document_relation = []

    for i, row in df.iterrows():
        document = Document(row['document_id'], row['source'])
        documents.add(document)
        event1 = (row['document_id'], row['event1_text'], row['event1_type'])
        event2 = (row['document_id'], row['event2_text'], row['event2_type'])
        relation = row['class']
        if 'minutes_between_means' in row and row['minutes_between_means'] is not None:
            relations.append((event1, event2, relation, row['minutes_between_means']))
        else:
            relations.append((event1, event2, relation))
        events.add(event1)
        events.add(event2)
        event_document_relation.append((event1, document))
        event_document_relation.append((event2, document))
        # if i > 100:
        #     break

    # Write to KG
    g = Graph()
    events_ns = Namespace("http://example.org/events/")
    documents_ns = Namespace("http://example.org/documents/")
    relations_ns = Namespace("http://example.org/relations/")
    event_type_ns = Namespace("http://example.org/eventtypes/")
    event_object = events_ns['event']
    document_object = documents_ns['document']
    g.add((event_object, RDFS.subClassOf, RDF.object))
    g.add((document_object, RDFS.subClassOf, RDF.object))
    document_ids = {}
    for document in documents:
        document_name = document.name
        document_id = documents_ns[document_name]
        document_ids[document_name] = document_id
        document_name_literal = Literal(document_name)
        document_source_literal = Literal(document.source)
        g.add((document_id, RDFS.subClassOf, document_object))
        g.add((document_id, relations_ns.name, document_name_literal))
        g.add((document_id, relations_ns.source, document_source_literal))
        # g.add((document_id, RDF.type, document_name_literal))
    event_ids = {}
    for event in events:
        # event_id = events_ns[str(uuid.uuid4().int)]
        event_id = events_ns[urllib.parse.quote_plus(event[1])]
        event_ids[event] = event_id
        event_mention = Literal(event[1])
        if event[2] is not None:
            event_type = event_type_ns[event[2]]
            g.add((event_id, relations_ns.eventType, event_type))
        g.add((event_id, RDFS.subClassOf, event_object))
        g.add((event_id, relations_ns.mention, event_mention))
        g.add((event_id, relations_ns.document, document_ids[event[0]]))

    for relation in relations:
        g.add((event_ids[relation[0]], relations_ns[relation[2]], event_ids[relation[1]]))
        if len(relation) > 3:
            g.add((event_ids[relation[0]], relations_ns[str(relation[3]) + '_minutes'], event_ids[relation[1]]))

    v = g.serialize(format="turtle")
    with open('test_small.ttl', 'w') as f:
        f.write(v)

    # Write triplets to file
    f = open("knowledge_graph.txt", "w")
    for relation in relations:
        f.write("{}\t{}\t{}\n".format(event_ids[relation[0]], relation[2], event_ids[relation[1]]))
        # if len(relation) > 3:
        #     g.add((event_ids[relation[0]], relations_ns[str(relation[3]) + '_minutes'], event_ids[relation[1]]))
    f.close()

def get_relations(graph, nodes):
    relations = []
    for g in graph:
        if g[0] in nodes and g[2] in nodes:
            relations.append(g)
    return relations

def get_multiword_word2vec(entity):
    global glove_vectors
    if glove_vectors is None:
        glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')
    vectors = []
    for word in entity.split():
        word = word.lower()
        if word in glove_vectors.key_to_index:
            vectors.append(glove_vectors.vectors[glove_vectors.key_to_index[word]])
    if len(vectors) == 0:
        vectors = [np.random.rand(50)] # TODO you might be able to find better embeddings than random ones
    return np.array(vectors).mean(0)

def get_embedding_for_entity(entity):
    global nlp, glove_vectors
    if nlp is None:
        nlp = spacy.load("en_core_sci_sm")
        nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    linker = nlp.get_pipe("scispacy_linker")
    if glove_vectors is None:
        glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')
    if entity in linker.kb.cui_to_entity:
        canonical_name = linker.kb.cui_to_entity[entity].canonical_name
        definition = linker.kb.cui_to_entity[entity].definition
        vector = get_multiword_word2vec(canonical_name)
    else:
        vector = get_multiword_word2vec(entity)
    return vector

def generate_graph_for_gnn(graph, entity1, entity2, y, text_features=None, use_entire_graph=True, neighbourhood=False):
    # TODO filter the graph
    nodes = []
    if use_entire_graph:
        nodes = [r[0] for r in graph] + [r[2] for r in graph]
    else:
        for path in get_path(graph, entity1, entity2, return_multiple=True):
            nodes += path
    nodes = list(set(nodes))
    if nodes is None or len(nodes) == 0:
        nodes = [entity1, entity2]
    if entity1 not in nodes:
        nodes.append(entity1)
    if entity2 not in nodes:
        nodes.append(entity2)

    if neighbourhood:
        nodes = get_neighbourhood(graph, nodes, steps=1)
    node_to_index = {}
    for i, node in enumerate(nodes):
        node_to_index[node] = i
    relations = get_relations(graph, nodes)
    edge_index = []
    edge_type = []
    edge_attr = []
    for r in relations:
        edge_index.append([node_to_index[r[0]], node_to_index[r[2]]])
        edge_type.append(labels[r[1]])
        if len(r) > 4:
            edge_attr.append(r[4])
        else:
            edge_attr.append(tuple(F.one_hot(torch.tensor(labels[r[1]]), 3).tolist()))
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    embeddings = np.array([get_embedding_for_entity(n) for n in nodes])
    # x = torch.ones(len(nodes), 50)
    x = torch.tensor(embeddings)
    edge_type = torch.tensor(edge_type)
    index1 = node_to_index[entity1]
    index2 = node_to_index[entity2]
    rule_based_prediction = rule_based_model(graph, entity1, entity2)

    # Text features
    if text_features is not None:
        text, event1_start, event1_end, event2_start, event2_end, _, document_id = text_features
        data = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=torch.tensor([y]),
                    event1_index=index1, event2_index=index2, rule_based_prediction=rule_based_prediction,
                    text=text, event1_start=event1_start, event1_end=event1_end, event2_start=event2_start, event2_end=event2_end, document_id=document_id,
                    edge_attr=torch.tensor(edge_attr))
    else:
        data = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=torch.tensor([y]),
                    event1_index=index1, event2_index=index2, rule_based_prediction=rule_based_prediction)
    return data

def add_event_tokens(text, event1_start, event1_end, event2_start, event2_end):
    tag_start1, tag_start2, tag_end1, tag_end2 = "<e1>", "<e2>", "</e1>", "</e2>"
    # tag_start1, tag_start2, tag_end1, tag_end2 = "<e>", "<e>", "</e>", "</e>"
    text = text[:event1_start] + tag_start1 + text[event1_start:]
    if event1_end >= event1_start:
        event1_end += len(tag_start1)
    if event2_start >= event1_start:
        event2_start += len(tag_start1)
    if event2_end >= event1_start:
        event2_end += len(tag_start1)
    if event1_start >= event1_start:
        event1_start += len(tag_start1)

    text = text[:event1_end] + tag_end1 + text[event1_end:]
    if event1_start > event1_end:
        event1_start += len(tag_end1)
    if event2_start > event1_end:
        event2_start += len(tag_end1)
    if event2_end > event1_end:
        event2_end += len(tag_end1)

    if max(event1_start, event2_start) < min(event1_end, event2_end):
        return text, event1_start, event1_end, event1_start, event1_end

    text = text[:event2_start] + tag_start2 + text[event2_start:]
    if event1_start >= event2_start:
        event1_start += len(tag_start2)
    if event1_end >= event2_start:
        event1_end += len(tag_start2)
    if event2_end >= event2_start:
        event2_end += len(tag_start2)
    if event2_start >= event2_start:
        event2_start += len(tag_start2)

    text = text[:event2_end] + tag_end2 + text[event2_end:]
    if event1_start >= event2_end:
        event1_start += len(tag_end2)
    if event1_end >= event2_end:
        event1_end += len(tag_end2)
    if event2_start >= event2_end:
        event2_start += len(tag_end2)
    return text, event1_start, event1_end, event2_start, event2_end

def convert_df_row(row):
    text = row['text']
    y = row['class']
    event1_start = row['event1_start']
    event2_start = row['event2_start']
    event1_end = row['event1_end']
    event2_end = row['event2_end']
    document_id = row['document_id']
    text, event1_start, event1_end, event2_start, event2_end = add_event_tokens(text, event1_start, event1_end,
                                                                                event2_start, event2_end)

    if "<" in text[event1_start:event1_end] or "<" in text[event2_start:event2_end]:
        # print("opozorilo")
        pass
    return text, event1_start, event1_end, event2_start, event2_end, torch.tensor(labels[y]), document_id

def add_inverse_relations(graph):
    set_of_relations = set([tuple(r) for r in graph])
    new_relations = []
    for relation in graph:
        inverse_relation = relation.copy()
        inverse_relation[0], inverse_relation[2] = inverse_relation[2], inverse_relation[0]
        inverse_relation[1] = combining_data.relation_inverse(inverse_relation[1])
        if len(inverse_relation) > 4:
            conf_relation = list(inverse_relation[4])
            conf_relation[0], conf_relation[1] = conf_relation[1], conf_relation[0]
            inverse_relation[4] = tuple(conf_relation)
        if tuple(inverse_relation) not in set_of_relations:
            new_relations.append(inverse_relation)
            set_of_relations.add(tuple(inverse_relation))
    return graph + new_relations


def add_transitive_relations(graph, only_additional=False):
    set_of_relations = set([tuple(r) for r in graph])
    new_relations = []
    for r1 in graph:
        for r2 in graph:
            if r1[3] == r2[3] and r1[2] == r2[0]:
                # relations are neighbours
                new_relation = r1.copy()
                relation_text = compute_transitive_relation(r1[1], r2[1])
                new_relation[1] = relation_text
                new_relation[2] = r2[2]
                if tuple(new_relation) not in set_of_relations:
                    set_of_relations.add(tuple(new_relation))
                    new_relations.append(new_relation)
    if only_additional:
        return new_relations
    else:
        return graph + new_relations


# returns a graph without connections between the nodes in the nodes list
def filter_knowledge_graph(grpah, nodes):
    return list(filter(lambda edge: not (edge[0] in nodes and edge[2] in nodes), grpah))

# returns a part of the graph related to the observed document
def filter_knowledge_graph_document(grpah, document):
    return list(filter(lambda edge: document == edge[3], grpah))

def aggregate_relations(graph):
    tmp_graph = graph.copy()
    aggregated_graph = []
    while len(tmp_graph) > 0:
        event1 = tmp_graph[0][0]
        event2 = tmp_graph[0][2]
        rel = [0,0,0]
        to_remove = []
        for r in tmp_graph:
            if r[0] == event1 and r[2] == event2:
                rel[labels[r[1]]] += 1
            to_remove.append(r)

        aggregated_graph.append([event1, tmp_graph[0][1], event2, tmp_graph[0][3], F.softmax(torch.tensor(rel))])
        for r in to_remove:
            tmp_graph.remove(r)

def create_graph(iteration):
    (i, row), graph, configuration = iteration
    document_id = row["document_id"]
    event1 = link_entity_to_umls(row["event1_text"])
    event2 = link_entity_to_umls(row["event2_text"])
    if configuration.no_document_filtering:
        active_graph = graph
    else:
        active_graph = filter_knowledge_graph_document(graph, row["document_id"])
    if configuration.add_inverse_relations_to_graph:
        active_graph = add_inverse_relations(active_graph)
    if configuration.add_transitive_relations_to_graph:
        transitive_relations = add_transitive_relations(active_graph, only_additional=True)
        active_graph += transitive_relations

    # relations from other files
    if not configuration.no_document_filtering and configuration.use_relations_from_other_documents:
        relations_from_other_files = list(filter(lambda r: r[3] != row["document_id"], get_relations(graph, [event1, event2])))
        if configuration.add_inverse_relations_to_graph:
            relations_from_other_files = add_inverse_relations(relations_from_other_files)

    # remove random relations to simulate wrongly labeled relations
    if configuration.simulated_realistic_graph['use_simulated_realistic_graph']:
        active_graph = random.sample(active_graph, int(len(active_graph) * configuration.simulated_realistic_graph['portion_of_relations_to_keep']))

    if not configuration.no_document_filtering and configuration.use_relations_from_other_documents:
        active_graph = active_graph + relations_from_other_files

    # text features
    text_features = convert_df_row(row)

    if configuration.remove_target_relation:
        active_graph = filter_knowledge_graph(active_graph, [event1, event2])
    subgraph = generate_graph_for_gnn(active_graph, event1, event2,
                                      labels[row["class"]], text_features, use_entire_graph=configuration.use_entire_graph)
    if len(subgraph.x) == 0:
        return None
    return subgraph

class KnowledgeGraphDataset(torch.utils.data.Dataset):
    def __init__(self, graph, df, number_of_classes=3, simulate_wrong_relations=False, configuration=Configuration()):
        self.graphs = []
        self.labels = []

        global nlp, glove_vectors
        if nlp is None:
            nlp = spacy.load("en_core_sci_sm")
            nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        if glove_vectors is None:
            glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')

        # with Pool(8) as pool:
        #     self.graphs = pool.map(create_graph, [(row, graph) for row in df.iterrows()])
        # self.graphs = list(map(create_graph, [(row, graph, configuration) for row in df.iterrows()]))
        self.graphs = list(custom_map(create_graph, [(row, graph, configuration) for row in df.iterrows()]))
        self.graphs = [g for g in self.graphs if g is not None]
        self.num_node_features = self.graphs[0].x.shape[1]
        self.labels = [x.y for x in self.graphs]

        self.num_classes = number_of_classes
        random.Random(1).shuffle(self.graphs)

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.graphs[idx].y

        item = {'graph': self.graphs[idx]}
        item['labels'] = torch.tensor(self.graphs[idx].y)
        return item

def path_to_relations(graph, path):
    relation_sequence = []
    for i in range(len(path) - 1):
        e1 = path[i]
        e2 = path[i + 1]
        for r in graph:
            if r[0] == e1 and r[2] == e2:
                relation_sequence.append(r[1])
                break
    return relation_sequence

def rule_based_model(graph, event1, event2):
    path = get_path(graph, event1, event2)
    if path is None:
        return 2
    relation_sequence = path_to_relations(graph, path)
    relation = 0
    for r in relation_sequence:
        if r == "BEFORE":
            relation -= 1
        if r == "AFTER":
            relation += 1
    if relation > 0:
        return 1
    elif relation < 0:
        return 0
    else:
        return 2

def visualise_graph(graph, marked_nodes=None):
    nodes = set()
    connections = []
    for r in graph:
        connections.append([r[0], r[2]])
        nodes.add(r[0])
        nodes.add(r[2])

    G = nx.Graph()
    G.add_edges_from(connections)
    color_map = ['red' if marked_nodes is not None and node in marked_nodes else 'blue' for node in G]
    nx.draw(G, with_labels=True, node_color=color_map)
    plt.show()

if __name__ == '__main__':
    # get_embedding_for_entity("C0023508")
    i2b2 = read_i2b2(True)
    graph = in_memory_kg(i2b2)
    dataset = KnowledgeGraphDataset(graph, i2b2)


    pass


