from torch import nn
from libctr.pytorch.models import BaseModel
from libctr.pytorch.layers import FM_Layer, EmbeddingLayer


class FM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="FM", 
                 gpu=-1, 
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 regularizer=None, 
                 **kwargs):
        super(FM, self).__init__(feature_map, 
                                 model_id=model_id, 
                                 gpu=gpu, 
                                 embedding_regularizer=regularizer, 
                                 net_regularizer=regularizer,
                                 **kwargs)
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        self.fm_layer = FM_Layer(feature_map, output_activation=self.get_output_activation(task), 
                                 use_bias=True)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()
            
    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)
        y_pred = self.fm_layer(X, feature_emb)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict

