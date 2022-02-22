{
    "dataset_reader" : {
        "type": "classification-tsv",
        "token_indexers": {
            "tokens": {
                "type": "single_id"
            }
        }
    },
    "train_data_path": "data/news_classification/train_data_small.csv",
    "validation_data_path": "data/news_classification/dev_data_small.csv",
    "vocabulary":{
        "type": "from_files",
        "directory": "data/vocab_model.tar.gz"
    },
    "model": {
        "type": "simple_classifier",
        "embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 10
                }
            }
        },
        "encoder": {
            "type": "bag_of_embeddings",
            "embedding_dim": 10
        }
    },
    "data_loader": {
        "batch_size": 8,
        "max_instances_in_memory": 80,
        "cuda_device": 0,
        "shuffle": true
    },
    "trainer": {
        "optimizer": "adam",
        "num_epochs": 5
    }
}
