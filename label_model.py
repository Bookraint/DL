import tempfile
import random
import torch
import os
import json
import torch.nn as nn
import numpy as np

from absl import logging
from xenon.models.model_base_pt import BaseModel
from xenon.models import register_model_pt
from xenon.utils.flags_core import Flag
from transformers import AutoModel, BertModel, BertConfig
from xenon.utils.hdfs_io import hopen, hmget, hlist_files
from xenon.utils.torch_io import load as torch_load
from xenon.utils.file_utils import is_ptx_available
from xenon.utils.hparams_sets import register_hparams_set


@register_model_pt("label_model")
class LabelModel(BaseModel):
    def __init__(self, args, num_labels,name=None):
        """Initializes a Huggingface Sequence Classification model.
        Args:
            args: A dict, containing the model configuration.
            name: The name of the model.
        """
        super(LabelModel, self).__init__(
            args, name=name or "label_model"
        )
        self.num_labels = num_labels

        self.model_type = args["model_type"]
        self.output_type = args["output_type"]
        self.label_emb = args["label_embedding"]
        self.label_file = args['label_file']

       
        self.classifier_dropout = nn.Dropout(p=args["classifier_dropout"])
        self.classifier = nn.Linear(args["hidden_size"], num_labels)
        self.build_label_representation()
        self._init_weights()
        self.build_model(args)

    def _init_weights(self):
        """Initialize the weights"""
        self.classifier.weight.data.copy_(self.le)


    def build_label_representation(self):
        label_embedding = {} #text_a: embedding:
        with hopen(self.label_emb) as f:
            for _, line in enumerate(f):
                e = json.loads(line.decode().strip())
                label_embedding.update({e['text_a']: e['embedding']})
        
        le = []
        with hopen(self.label_file) as f:
            for _, line in enumerate(f):
                text = line.decode().strip()
                le.append(label_embedding[text])
        self.le = torch.tensor(np.array(le))



    def build_model(self, args):
        pretrain_model_name = args["pretrain_model_name"]
        config_path = args["config_path"]
        if pretrain_model_name is None and config_path is None:
            logging.info("Building model with config.")
            self.build_model_with_config(args)
            return
        if self.model_type == "ptx":
            from ptx.model.deberta.model import DebertaBare

            if pretrain_model_name is None:
                if config_path.startswith("hdfs"):
                    with hopen(config_path) as f:
                        config = json.loads(f.read().decode())
                else:
                    config = json.load(open(config_path))
                self.backbone_model = DebertaBare(**config)
            else:
                config_path = os.path.join(args["pretrain_model_name"], "config.json")
                model_path = os.path.join(
                    args["pretrain_model_name"], "pytorch_model.bin"
                )
                if config_path.startswith("hdfs"):
                    with hopen(config_path) as f:
                        config = json.loads(f.read().decode())
                else:
                    config = json.load(open(config_path))
                self.backbone_model = DebertaBare(**config)
                state_dict = torch_load(model_path)
                new_state_dict = self.backbone_model.state_dict()
                for k, v in state_dict.items():
                    if k in new_state_dict:
                        new_state_dict[k] = v
                self.backbone_model.load_state_dict(new_state_dict)
                logging.info("Load Pretrain model from {}".format(pretrain_model_name))
        else:
            if pretrain_model_name is None:
                if config_path.startswith("hdfs"):
                    with hopen(config_path) as f:
                        config = json.loads(f.read().decode())
                else:
                    config = json.load(open(config_path))
                self.backbone_model = AutoModel.from_config(config)
            else:
                if pretrain_model_name.startswith("hdfs"):
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        files = hlist_files([pretrain_model_name])
                        hmget(files, tmpdirname)
                        self.backbone_model = AutoModel.from_pretrained(tmpdirname)
                else:
                    self.backbone_model = AutoModel.from_pretrained(pretrain_model_name)

    def build_model_with_config(self, args):
        bert_config = BertConfig()
        bert_config.hidden_size = args["hidden_size"]
        if args["intermediate_size"] is not None:
            bert_config.intermediate_size = args["intermediate_size"]
        else:
            bert_config.intermediate_size = args["hidden_size"]
        bert_config.max_position_embeddings = args["max_position_embeddings"]
        bert_config.num_attention_heads = args["num_attention_heads"]
        bert_config.num_hidden_layers = args["num_hidden_layers"]
        bert_config.vocab_size = args["vocab_size"]
        self.backbone_model = BertModel(bert_config)

    @staticmethod
    def class_or_method_args():
        return [
            Flag(
                "config_path", dtype=Flag.TYPE.STRING, default=None, help="Config path"
            ),
            Flag(
                "classifier_dropout",
                dtype=Flag.TYPE.FLOAT,
                default=0.1,
                help="classifier_dropout",
            ),
            Flag(
                "hidden_size",
                dtype=Flag.TYPE.INTEGER,
                default=768,
                help="hidden size",
            ),
            Flag(
                "model_type",
                dtype=Flag.TYPE.STRING,
                default="huggingface",
                help="huggingface/ptx",
            ),
            Flag(
                "output_type",
                dtype=Flag.TYPE.STRING,
                default="cls",
                help="cls/sequence",
            ),
            Flag(
                "pretrain_model_name",
                dtype=Flag.TYPE.STRING,
                default=None,
                help="pretrain model path",
            ),
            Flag(
                "label_embedding",
                dtype=Flag.TYPE.STRING,
                default=None,
                help="label_embedding path",
            ),
            Flag(
                "label_file",
                dtype=Flag.TYPE.STRING,
                default=None,
                help="label_file path",
            ),
        ]

    @classmethod
    def new(cls, args: dict, num_labels: int, name=None):
        model = cls(args, num_labels=num_labels,name=name)
        return model

    def forward(self, input_ids, attention_mask, **kwargs):
        if self.model_type == "ptx":
            result = self.backbone_model(
                input_ids=input_ids, attention_mask=attention_mask, output_pooled=True
            )
            if self.output_type == "cls":
                result = result["pooled_output"]
            else:
                result = result["sequence_output"]
        else:
            result = self.backbone_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            if self.output_type == "cls":
                result = result[1]
            else:
                result = result[0]
        dropped_result = self.classifier_dropout(result)

        classifier_logits = self.classifier(dropped_result)
        return classifier_logits

    def dummy_inputs(self, max_length=10):
        return (
            torch.LongTensor(
                [[random.randint(0, 10) for _ in range(max_length)]]
            ).cuda(),
            torch.LongTensor(
                [[random.randint(0, 1) for _ in range(max_length)]]
            ).cuda(),
        )

