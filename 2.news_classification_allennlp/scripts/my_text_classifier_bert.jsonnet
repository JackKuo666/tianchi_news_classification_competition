{
    "dataset_reader" : {
        "type": "classification-tsv",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": "bert-mini/"
        },
        "token_indexers": {
            "bert": {
                "type": "pretrained_transformer",
                "model_name": "bert-mini/"
            }
        },
        "max_tokens": 256
    },
    "train_data_path": "data/news_classification/train_data.csv",
    "validation_data_path": "data/news_classification/dev_data.csv",
    "vocabulary":{
        "type": "from_files",
        "directory": "data/vocab_model.tar.gz"
    },
    "model": {
        "type": "simple_classifier",
        "embedder": {
            "token_embedders": {
                "bert": {
                    "type": "pretrained_transformer",
                    "model_name": "bert-mini/"
                }
            }
        },
        "encoder": {
            "type": "bert_pooler",
            "pretrained_model": "bert-mini/",
            "requires_grad": true
        }
    },
    "data_loader": {
        "batch_size": 8,
        "max_instances_in_memory": 1800,
       "shuffle": true
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1.0e-5
        },
        "num_epochs": 1
    }
}
