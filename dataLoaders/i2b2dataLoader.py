import re
from datetime import datetime, timedelta
from os import listdir
import xml.etree.ElementTree as ET
from os.path import exists

import pandas
import torch
# from torch_geometric.data import Data
from transformers import BertTokenizerFast, BertModel

from common import expand_sentence

def get_event_time(event_id):
    tsv_files = {}
    for file in listdir('data/i2b2-absolute'):
        if 'lock' in file:
            continue
        id = file.split("_")[1].split('.')[0]
        tsv_files[id] = file
        pass
    pass

def convert_datetime_to_minutes(event_time):
    return (event_time-datetime(1900, 1, 1, 0, 0, 0, 0)).total_seconds()/60

def add_padding_for_incomplete_rows(row):
    additional_information = {'event1_start_time': None, 'event2_start_time': None,
                              'event1_end_time': None, 'event2_end_time': None}
    return list(row) + [None, None, None, None, None, additional_information]
def load_absolute_data(full_text=False, use_test_files=False, include_rows_without_absolute=False):
    tsv_files = {}
    for file in listdir('data/i2b2-absolute'):
        if 'lock' in file:
            continue
        id = file.split("_")[1].split('.')[0]
        tsv_files[id] = file
        pass
    data = load_data(full_text=full_text, use_test_files=use_test_files)
    rows = []
    loaded_data = None
    loaded_id = -1
    for i in range(len(data)):
        file_id = data['file_id'][i]
        if file_id not in tsv_files:
            if include_rows_without_absolute:
                rows.append(add_padding_for_incomplete_rows(data.iloc[i]))
            continue
        if file_id != loaded_id:
            loaded_id = file_id
            loaded_data = pandas.read_csv('data/i2b2-absolute/' + tsv_files[file_id], sep='\t', header=None, index_col=False,
                    names=['id', 'start_mode', 'start_min', 'start_max', 'duration_mode', 'duration_min',
                           'duration_max', 'end_mode', 'end_min', 'end_max'])
        event1_duration_mode = loaded_data.loc[loaded_data['id'] == data['event1_id'][i]].reset_index()['duration_mode'][0]
        event1_start_mode = loaded_data.loc[loaded_data['id'] == data['event1_id'][i]].reset_index()['start_mode'][0]
        event1_end_mode = loaded_data.loc[loaded_data['id'] == data['event1_id'][i]].reset_index()['end_mode'][0]
        event2_duration_mode = loaded_data.loc[loaded_data['id'] == data['event2_id'][i]].reset_index()['duration_mode'][0]
        event2_start_mode = loaded_data.loc[loaded_data['id'] == data['event2_id'][i]].reset_index()['start_mode'][0]
        event2_end_mode = loaded_data.loc[loaded_data['id'] == data['event2_id'][i]].reset_index()['end_mode'][0]

        # missing information
        if (not isinstance(event1_start_mode, str) and not isinstance(event1_end_mode, str)) or\
                (not isinstance(event1_start_mode, str) and not isinstance(event1_duration_mode, str)) or\
                (not isinstance(event1_duration_mode, str) and not isinstance(event1_end_mode, str)) or\
                (not isinstance(event2_start_mode, str) and not isinstance(event2_end_mode, str)) or\
                (not isinstance(event2_start_mode, str) and not isinstance(event2_duration_mode, str)) or\
                (not isinstance(event2_duration_mode, str) and not isinstance(event2_end_mode, str)):
            rows.append(add_padding_for_incomplete_rows(data.iloc[i]))
            continue
        event1_start_mode, event1_end_mode, devent1_duration_mode = parse_times(event1_start_mode, event1_end_mode, event1_duration_mode)
        event2_start_mode, event2_end_mode, devent2_duration_mode = parse_times(event2_start_mode, event2_end_mode, event2_duration_mode)

        text = list(data.iloc[i])[0]
        event1_text = text[list(data.iloc[i])[2]: list(data.iloc[i])[3]]
        event2_text = text[list(data.iloc[i])[4]: list(data.iloc[i])[5]]

        additional_information = {'event1_start_time': convert_datetime_to_minutes(event1_start_mode),
            'event2_start_time': convert_datetime_to_minutes(event2_start_mode),
            'event1_end_time': convert_datetime_to_minutes(event1_end_mode),
            'event2_end_time': convert_datetime_to_minutes(event2_end_mode)
        }

        timedelta_between_starts = (event2_start_mode - event1_start_mode)
        minutes_between_starts = timedelta_between_starts.days*24*60 + timedelta_between_starts.seconds // 60
        timedelta_between_ends = (event2_end_mode - event1_end_mode)
        minutes_between_ends = timedelta_between_ends.days*24*60 + timedelta_between_ends.seconds // 60
        minutes_between_means = (minutes_between_starts + minutes_between_ends) / 2
        rows.append(list(data.iloc[i]) + [minutes_between_starts,
                                          minutes_between_ends,
                                          minutes_between_means, event1_text, event2_text,
                                          additional_information])
        pass
    return pandas.DataFrame(rows,
            columns=["text", "class", "event1_start", "event1_end", "event2_start", "event2_end", "event1_id",
                     "event2_id", "file_id", "timexs", "minutes_between_starts", "minutes_between_ends", "minutes_between_means",
                     "event1_text", "event2_text",
                     "additional_information"])


def load_data(full_text=False, use_test_files=False):
    if use_test_files:
        data_folder = 'data/i2b2-test'
    else:
        data_folder = 'data/i2b2'

    rows = []
    for file in listdir(data_folder):
        if not file.endswith('.xml'):
            continue
        with open(data_folder + '/' + file, 'r') as open_file:
            file_string = open_file.read().replace(' & ', '   ').replace('L&D', 'LnD').replace('&', 'n')
        root = ET.fromstring(file_string)
        # root = tree.getroot()
        file_id = file.split(".")[0]
        text = root[0].text
        annotations = root[1]
        events = {}
        links = []
        for a in annotations:
            if a.tag == 'EVENT':
                events[a.attrib['id']] = a.attrib
            if a.tag == 'TIMEX3':
                events[a.attrib['id']] = a.attrib
            if a.tag == 'TLINK':
                links.append(a.attrib)

        for link in links:
            if full_text:
                if not link['fromID'].startswith('E') or not link['toID'].startswith('E'):
                    continue
                if link['type'] == '':
                    continue
                s1 = int(events[link['fromID']]['start'])
                e1 = int(events[link['fromID']]['end'])
                s2 = int(events[link['toID']]['start'])
                e2 = int(events[link['toID']]['end'])
                timexs = []
                for e in events:
                    if e.startswith("T"):
                        timexs.append({'start': int(events[e]['start']),
                                       'end': int(events[e]['end']),
                                       'val': events[e]['val'],
                                       'type': events[e]['type']})

                rows.append(
                    [text, link["type"], s1, e1, s2, e2, link['fromID'], link['toID'], file_id, timexs])
            else:
                if not link['fromID'].startswith('E') or not link['toID'].startswith('E'):
                    continue
                if link['type'] == '':
                    continue
                if int(events[link['fromID']]['start']) < int(events[link['toID']]['end']):
                    start = int(events[link['fromID']]['start'])
                    end = int(events[link['toID']]['end'])
                    s1 = 0
                    e1 = int(events[link['fromID']]['end']) - int(events[link['fromID']]['start'])
                    s2 = int(events[link['toID']]['start']) - int(events[link['fromID']]['start'])
                    e2 = int(events[link['toID']]['end']) - int(events[link['fromID']]['start'])
                else:
                    start = int(events[link['toID']]['start'])
                    end = int(events[link['fromID']]['end'])
                    s1 = int(events[link['fromID']]['start']) - int(events[link['toID']]['start'])
                    e1 = int(events[link['fromID']]['end']) - int(events[link['toID']]['start'])
                    s2 = 0
                    e2 = int(events[link['toID']]['end']) - int(events[link['toID']]['start'])

                while start > 1 and text[start - 1] != ".":
                    s1 += 1
                    e1 += 1
                    s2 += 1
                    e2 += 1
                    start -= 1
                while end < len(text) - 1 and text[end + 1] != ".":
                    end += 1

                timexs = []
                for e in events:
                    if e.startswith("T"):
                        if int(events[e]['start']) >= start and int(events[e]['end']) <= end:
                            timexs.append({'start': int(events[e]['start']) - start,
                                           'end': int(events[e]['end']) - start,
                                           'val': events[e]['val'],
                                           'type': events[e]['type']})

                rows.append([text[start:end], link["type"], s1, e1, s2, e2, link['fromID'], link['toID'], file_id, timexs])
    return pandas.DataFrame(rows, columns=["text", "class", "event1_start", "event1_end", "event2_start", "event2_end", "event1_id", "event2_id", "file_id", "timexs"])

def single_events():
    data_folder = 'data/i2b2'
    rows = []
    for file in listdir(data_folder):
        if not file.endswith('.xml'):
            continue
        with open(data_folder + '/' + file, 'r') as open_file:
            file_string = open_file.read().replace(' & ', '   ').replace('L&D', 'LnD')
        root = ET.fromstring(file_string)
        # root = tree.getroot()
        file_id = file.split(".")[0]
        text = root[0].text
        annotations = root[1]
        events = {}
        for a in annotations:
            if a.tag == 'EVENT':
                events[a.attrib['id']] = a.attrib
            # if a.tag == 'TIMEX3':
            #     events[a.attrib['id']] = a.attrib

        for e in events:
            start = int(events[e]['start'])
            end = int(events[e]['end'])
            sentence_start, sentence_end = expand_sentence(text, start, end)
            rows.append([text[sentence_start : sentence_end], start - sentence_start, end - sentence_start, events[e]['id'], file_id])
            pass
    return pandas.DataFrame(rows, columns=["text", "event_start", "event_end", "event_id", "file_id"])

def parse_times(start, end, duration):
    # make sure start time is available
    if isinstance(start, str):
        e1s = datetime.strptime(start, '%Y-%m-%d %H:%M')
    else:
        e1e = datetime.strptime(end, '%Y-%m-%d %H:%M')
        parsed = re.search(r'(\d+)Y(\d+)M(\d+)D(\d+)H(\d+)m', duration)
        dur = timedelta(days=365 * int(parsed.group(1)) +
                                  30 * int(parsed.group(2)) +
                                  int(parsed.group(3)),
                             hours=int(parsed.group(4)),
                             minutes=int(parsed.group(5)))
        e1s = e1e - dur

    # make sure end time is available
    if isinstance(end, str):
        e1e = datetime.strptime(end, '%Y-%m-%d %H:%M')
    else:
        e1s = datetime.strptime(start, '%Y-%m-%d %H:%M')
        parsed = re.search(r'(\d+)Y(\d+)M(\d+)D(\d+)H(\d+)m', duration)
        dur = timedelta(days=365 * int(parsed.group(1)) +
                                  30 * int(parsed.group(2)) +
                                  int(parsed.group(3)),
                             hours=int(parsed.group(4)),
                             minutes=int(parsed.group(5)))
        e1e = e1s + dur

    # make sure duration is available
    if isinstance(duration, str):
        parsed = re.search(r'(\d+)Y(\d+)M(\d+)D(\d+)H(\d+)m', duration)
        dur = timedelta(days=365 * int(parsed.group(1)) +
                                  30 * int(parsed.group(2)) +
                                  int(parsed.group(3)),
                             hours=int(parsed.group(4)),
                             minutes=int(parsed.group(5)))
    else:
        dur = e1e - e1s

    return e1s, e1e, dur


def load_data_temporal_attributes(max_duration_minutes=10080):
    tsv_files = {}
    for file in listdir('data/i2b2-absolute'):
        if 'lock' in file:
            continue
        id = file.split("_")[1].split('.')[0]
        tsv_files[id] = file
        pass
    data = single_events()
    rows = []
    loaded_data = None
    loaded_id = -1
    for i in range(len(data)):
        file_id = data['file_id'][i]
        if file_id not in tsv_files:
            continue
        if file_id != loaded_id:
            loaded_id = file_id
            loaded_data = pandas.read_csv('data/i2b2-absolute/' + tsv_files[file_id], sep='\t', header=None,
                                          index_col=False,
                                          names=['id', 'start_mode', 'start_min', 'start_max', 'duration_mode',
                                                 'duration_min',
                                                 'duration_max', 'end_mode', 'end_min', 'end_max'])
        event_start_mode = loaded_data.loc[loaded_data['id'] == data['event_id'][i]].reset_index()['start_mode'][0]
        event_end_mode = loaded_data.loc[loaded_data['id'] == data['event_id'][i]].reset_index()['end_mode'][0]
        event_duration_mode = loaded_data.loc[loaded_data['id'] == data['event_id'][i]].reset_index()['duration_mode'][0]

        event_start_min = loaded_data.loc[loaded_data['id'] == data['event_id'][i]].reset_index()['start_min'][0]
        event_end_min = loaded_data.loc[loaded_data['id'] == data['event_id'][i]].reset_index()['end_min'][0]
        event_duration_min = loaded_data.loc[loaded_data['id'] == data['event_id'][i]].reset_index()['duration_min'][0]

        event_start_max = loaded_data.loc[loaded_data['id'] == data['event_id'][i]].reset_index()['start_max'][0]
        event_end_max = loaded_data.loc[loaded_data['id'] == data['event_id'][i]].reset_index()['end_max'][0]
        event_duration_max = loaded_data.loc[loaded_data['id'] == data['event_id'][i]].reset_index()['duration_max'][0]

        # missing information
        if not isinstance(event_start_mode, str) and not isinstance(event_end_mode, str):
            continue
        if not isinstance(event_start_mode, str) and not isinstance(event_duration_mode, str):
            continue
        if not isinstance(event_duration_mode, str) and not isinstance(event_end_mode, str):
            continue

        _, _, duration_mode = parse_times(event_start_mode, event_end_mode, event_duration_mode)
        _, _, duration_min = parse_times(event_start_max, event_end_min, event_duration_min)
        _, _, duration_max = parse_times(event_start_min, event_end_max, event_duration_max)

        minutes_min = duration_min.days * 24 * 60 + duration_min.seconds / 60
        minutes_max = duration_max.days * 24 * 60 + duration_max.seconds / 60
        minutes_mode = duration_mode.days * 24 * 60 + duration_mode.seconds / 60

        minutes_min = max(minutes_min, 0)
        minutes_max = max(minutes_max, 0)
        minutes_mode = max(minutes_mode, 0)

        minutes_min = min(minutes_min/max_duration_minutes, 1)
        minutes_max = min(minutes_max/max_duration_minutes, 1)
        minutes_mode = min(minutes_mode/max_duration_minutes, 1)

        # years = int(duration_mode.days / 365)
        # months = int(duration_mode.days / 30)
        # days = duration_mode.days
        # hours = int(duration_mode.seconds / 3600)
        # minutes = int(duration_mode.seconds / 60)
        # unit = 5 if years != 0 else 4 if months != 0 else 3 if days != 0 else 2 if hours != 0 else 1 if minutes != 0 else 0
        rows.append(list(data.iloc[i]) + [minutes_min, minutes_mode, minutes_max])
        pass
    return pandas.DataFrame(rows,
                            columns=["text", "event_start", "event_end", "event_id", "file_id",
                                     "duration_min", "duration_mode", "duration_max"])


def load_data_event_extraction():
    data_folder = 'data/i2b2'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    bert = BertModel.from_pretrained("bert-base-uncased")
    bert.to(device)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    rows = []
    for file in listdir(data_folder):
        if not file.endswith('.xml'):
            continue
        with open(data_folder + '/' + file, 'r') as file:
            file_string = file.read().replace(' & ', '   ').replace('L&D', 'LnD')
        root = ET.fromstring(file_string)
        # root = tree.getroot()
        text = root[0].text
        annotations = root[1]
        events = []
        for a in annotations:
            if a.tag == 'EVENT':
                events.append(a.attrib)

        sentences = text.split(" .")
        index = 0
        for s in sentences:
            tokens = tokenizer(s, padding='max_length', max_length=512, truncation=True, return_tensors="pt",
                               return_offsets_mapping=True)
            length = torch.argmin(tokens['attention_mask'][0])
            labels = [0 for _ in range(length)]
            for e in events:
                start = int(e['start'])
                end = int(e['end'])
                if start > index and end <= index + len(s):
                    for i in range(tokens.char_to_token(start - index), tokens.char_to_token(end - index - 1) + 1):
                        labels[i] = 1
            index += len(s) + 2

            # to data object
            file_name = 'cache/' + str(abs(hash(s))) + '.pt'
            if not exists(file_name):
                tokens.to(device)
                sequence_output, _ = bert(input_ids=tokens['input_ids'].squeeze(1),
                                          attention_mask=tokens['attention_mask'], return_dict=False)
                tensor_length = torch.argmin(tokens['attention_mask'][0])
                torch.save(sequence_output[0, :tensor_length].clone(), file_name)
                pass

            data = Data(
                x=tokens['input_ids'].squeeze(1)[0],
                file=file_name,
                y=torch.tensor(labels))
            rows.append(data)
    return rows


if __name__ == '__main__':
    # df = load_data()
    # df = load_data_event_extraction()
    df = load_absolute_data(include_rows_without_absolute=True)
    # df = load_data_temporal_attributes()
    pass

