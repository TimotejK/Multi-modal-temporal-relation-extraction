import sys
from models.EntityBERT import *
from dataLoaders import combining_data
from common import Configuration, split_data
from models.EntityBERT.EntityBERT import full_test_battery_for_new_dataset

def test_maccrobat(realistic=True, remove_wrong=False):
    configuration = Configuration()
    configuration.use_realistic_graph = realistic
    configuration.add_inverse_relations_to_graph = True
    configuration.add_transitive_relations_to_graph = False

    configuration.use_relations_from_other_documents = False
    configuration.realistic_graph['remove_wrong_edges'] = remove_wrong
    configuration.realistic_graph['use_threshold_confidence'] = False
    configuration.realistic_graph['threshold_confidence'] = 0.7
    # train_text_model_on_transitive_relations()

    df, _, df_test = split_data(combining_data.read_macrobat(), train_size=0.7, val_size=0)

    maccrobat_training_settings = {
            "text": {
                "learning_rate": 0.01,
                "weight_decay": 0.001,
                "optimizer": "adam",
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

    configuration.training = maccrobat_training_settings

    settings = {
            "dataset_name": 'maccrobat',
            "text_model_pretraining": True,
            "oversample_dataset": False,
            "train_text_model": False,
            "train_graph_model": False,
            "use_cached_datasets": False,
            "use_learned_text_model_for_transitive": False
        }

    full_test_battery_for_new_dataset(configuration, df, df_test, settings=settings)

def test_i2b2(realistic=True, remove_wrong=False):
    configuration = Configuration()
    configuration.use_realistic_graph = realistic
    configuration.add_inverse_relations_to_graph = True
    configuration.add_transitive_relations_to_graph = False

    configuration.use_relations_from_other_documents = False
    configuration.realistic_graph['remove_wrong_edges'] = remove_wrong
    configuration.realistic_graph['use_threshold_confidence'] = False
    configuration.realistic_graph['threshold_confidence'] = 0.7
    # train_text_model_on_transitive_relations()

    df = combining_data.read_i2b2(full_text=True, include_rows_without_absolute=True)
    df_test = combining_data.read_i2b2(full_text=True, use_test_files=True, include_rows_without_absolute=True)

    maccrobat_training_settings = {
            "text": {
                "learning_rate": 0.01,
                "weight_decay": 0.001,
                "optimizer": "adam",
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

    configuration.training = maccrobat_training_settings

    settings = {
            "dataset_name": 'i2b2',
            "text_model_pretraining": True,
            "oversample_dataset": False,
            "train_text_model": False,
            "train_graph_model": True,
            "use_cached_datasets": True,
            "use_learned_text_model_for_transitive": False
        }

    full_test_battery_for_new_dataset(configuration, df, df_test, settings=settings)

def test_thyme_i2b2_relations(realistic=False, learned_transitive=False, remove_wrong=False):
    configuration = Configuration()
    configuration.use_realistic_graph = realistic
    configuration.add_inverse_relations_to_graph = True
    configuration.add_transitive_relations_to_graph = False

    configuration.use_relations_from_other_documents = False
    configuration.realistic_graph['remove_wrong_edges'] = remove_wrong
    configuration.realistic_graph['use_threshold_confidence'] = False
    configuration.realistic_graph['threshold_confidence'] = 0.7

    # configuration.training["bimodal"]["optimizer"] = "sgd"
    # configuration.training["bimodal"]["learning_rate"] = 0.01
    # configuration.training["bimodal"]["weight_decay"] = 0.01
    # train_text_model_on_transitive_relations()

    df = torch.load('thyme_df.pt')
    df_test = torch.load('thyme_test_df.pt')
    df = combining_data.convert_thymre_relations_to_i2b2(df)
    df_test = combining_data.convert_thymre_relations_to_i2b2(df_test)


    settings = {
            "dataset_name": 'thyme-' + ('realistic' if realistic else 'unrealistic') + ('-learned_transitive' if learned_transitive else ''),
            "text_model_pretraining": False,
            "oversample_dataset": True,
            "train_text_model": True,
            "train_graph_model": True,
            "use_cached_datasets": False,
            "use_learned_text_model_for_transitive": learned_transitive
        }
    df_test = common.oversample(df_test, oversample=settings['oversample_dataset'])

    full_test_battery_for_new_dataset(configuration, df, df_test, settings=settings)

def test_thyme_original_full_graph(realistic, remove_wrong=False):
    configuration = Configuration()
    configuration.use_realistic_graph = realistic
    configuration.add_inverse_relations_to_graph = True
    configuration.add_transitive_relations_to_graph = False

    configuration.use_relations_from_other_documents = False
    configuration.realistic_graph['remove_wrong_edges'] = remove_wrong
    configuration.realistic_graph['use_threshold_confidence'] = False
    configuration.realistic_graph['threshold_confidence'] = 0.7
    # train_text_model_on_transitive_relations()

    df = torch.load('thyme_df.pt')
    df_test = torch.load('thyme_test_df.pt')


    settings = {
            "dataset_name": 'thyme-original-' + ('realistic' if realistic else 'unrealistic'),
            "text_model_pretraining": True,
            "oversample_dataset": False,
            "train_text_model": False,
            "train_graph_model": True,
            "use_cached_datasets": False,
            "use_learned_text_model_for_transitive": True
        }

    full_test_battery_for_new_dataset(configuration, df, df_test, settings=settings)

if __name__ == '__main__':
    remove_wrong = False
    if "remove_wrong" in sys.argv:
        remove_wrong = True

    if "i2b2" in sys.argv:
        test_i2b2(True, remove_wrong=remove_wrong)
    if "maccrobat" in sys.argv:
        test_maccrobat(True, remove_wrong=remove_wrong)
    if "maccrobat_unrealistic" in sys.argv:
        test_maccrobat(False, remove_wrong=remove_wrong)
    if "thyme_simple_unrealistic" in sys.argv:
        test_thyme_i2b2_relations(realistic=False, learned_transitive=False, remove_wrong=remove_wrong)
    if "thyme_simple_realistic" in sys.argv:
        test_thyme_i2b2_relations(realistic=True, learned_transitive=False, remove_wrong=remove_wrong)
    if "thyme_simple_unrealistic_transitive" in sys.argv:
        test_thyme_i2b2_relations(realistic=False, learned_transitive=True, remove_wrong=remove_wrong)
    if "thyme_simple_realistic_transitive" in sys.argv:
        test_thyme_i2b2_relations(realistic=True, learned_transitive=True, remove_wrong=remove_wrong)
    if "thyme_original_realistic" in sys.argv:
        test_thyme_original_full_graph(True, remove_wrong=remove_wrong)
    if "thyme_original_unrealistic" in sys.argv:
        test_thyme_original_full_graph(False, remove_wrong=remove_wrong)
