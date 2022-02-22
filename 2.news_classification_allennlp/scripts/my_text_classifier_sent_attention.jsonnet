{
    "dataset_reader" : {
        "type": "sent_attention_reader",
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
    "train_data_path": "data/news_classification/train_data_small.csv",
    "validation_data_path": "data/news_classification/dev_data_small.csv",
    "vocabulary":{
        "type": "from_files",
        "directory": "data/vocab_model.tar.gz"
    },
    "model": {
        "type": "sent_attention_classifier",
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
        "max_instances_in_memory": 180,
        "shuffle": true
    },
    "trainer": {
        "validation_metric": "+f1",
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1.0e-5
        },
        "num_epochs": 5
    }
}
