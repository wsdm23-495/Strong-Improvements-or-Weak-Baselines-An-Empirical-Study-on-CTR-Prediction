import torch
from torch import nn
from libctr.pytorch.models import BaseModel
from libctr.pytorch.layers import EmbeddingLayer, MLP_Layer, CrossNet


class DCN(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DCN", 
                 gpu=-1, 
                 task="binary_classification",
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 dnn_hidden_units=[], 
                 dnn_activations="ReLU",
                 crossing_layers=3, 
                 net_dropout=0, 
                 batch_norm=False, 
                 layer_norm=False,
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 **kwargs):
        super(DCN, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        input_dim = feature_map.num_fields * embedding_dim
        self.dnn = MLP_Layer(input_dim=input_dim,
                             output_dim=None, # output hidden layer
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm, 
                             layer_norm=layer_norm,
                             use_bias=True) \
                   if dnn_hidden_units else None # in case of only crossing net used
        self.crossnet = CrossNet(input_dim, crossing_layers)
        final_dim = input_dim
        if isinstance(dnn_hidden_units, list) and len(dnn_hidden_units) > 0: # if use dnn
            final_dim += dnn_hidden_units[-1]
        self.fc = nn.Linear(final_dim, 1) # [cross_part, dnn_part] -> logit
        self.output_activation = self.get_output_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)
        flat_feature_emb = feature_emb.flatten(start_dim=1)
        cross_out = self.crossnet(flat_feature_emb)
        if self.dnn is not None:
            dnn_out = self.dnn(flat_feature_emb)
            final_out = torch.cat([cross_out, dnn_out], dim=-1)
        else:
            final_out = cross_out
        y_pred = self.fc(final_out)
        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict






