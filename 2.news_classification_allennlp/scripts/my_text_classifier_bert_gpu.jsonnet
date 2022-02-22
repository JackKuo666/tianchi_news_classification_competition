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
            "requires_grad": true,
            "dropout": 0.1
        }
    },
    "data_loader": {
        "batch_size": 512,
        "cuda_device": 0,
        "max_instances_in_memory": 18000,
        "shuffle": true
    },
    "trainer": {
        "checkpointer":{
            "type": "simple_checkpointer",
            "serialization_dir":"checkpoint/",
            "save_every_num_seconds": 1200
        },
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1.0e-5
        },
        "num_epochs": 3
    }
}
