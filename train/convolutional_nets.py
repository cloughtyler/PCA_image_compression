import torch
import pytorch_lightning as pl 
import torchmetrics as tm
from transformers import ConvNextConfig, ConvNextModel, ConvNextFeatureExtractor
"""
    This file contains the code for the convolutional neural network.
    Using pytorch lightning library we can package much of the boilerplate code
    into a single module

    class convolutionat_net:
        PyTorch lightning model - convolutional neural network
        This class is a wrapper around the pytorch lightning model class
        that adds some additional functionality.
        We will use this class to build, train and test models
"""
class convolutional_net(pl.LightningModule):
    def __init__(self, in_channels, num_classes,
            num_filters, kernel_sizes, 
            dropout, learning_rate, image_size):
        super(convolutional_net, self).__init__()
        # init metrics for eval
        self.train_acc = tm.Accuracy()
        self.train_recall = tm.Recall()
        self.train_precision = tm.Precision()
        self.val_acc = tm.Accuracy()
        self.val_recall = tm.Recall()
        self.val_precision = tm.Precision()
        self.test_acc = tm.Accuracy()
        self.test_recall = tm.Recall()
        self.test_precision = tm.Precision()
        # init layers
        self.conv_layers = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_channels = in_channels if kernel_sizes[0] == kernel_size else num_filters, 
                            out_channels = num_filters,
                            kernel_size = kernel_size,
                            stride = 1,
                            padding = 0) for kernel_size in kernel_sizes]
                            )
        # get last kernel size
        # TODO: currently hardcoded
        for i in range(len(kernel_sizes)):
            image_size = image_size - kernel_sizes[i]
            counter = i
        final_size = image_size + counter + 1
        print('correct final size? ' + str(final_size))
        # init fully connected layers
        self.fc_layer = torch.nn.Linear(final_size * final_size * num_filters, 128)
        self.dropout = torch.nn.Dropout(dropout)
        self.output_layer = torch.nn.Linear(128, num_classes)
        # init optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
        # init loss function
        self.loss_fn = torch.nn.CrossEntropyLoss()
    def forward(self, images):
        # forward pass through the network
        x = images
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            x = torch.nn.functional.relu(x)
        # flatten the output
        x = x.view(x.size(0), -1)
        # pass through fully connected layer
        x = self.fc_layer(x)
        x = self.dropout(x)
        x = torch.nn.functional.relu(x)
        # pass through output layer
        x = self.output_layer(x)
        
        return x
    def training_step(self, batch):
        # forward pass
        images, labels = batch
        logits = self.forward(images)
        # compute loss
        loss = self.cross_entropy_loss(logits, labels)
        # compute metrics
        acc = self.train_acc(logits, labels)
        recall = self.train_recall(logits, labels)
        precision = self.train_precision(logits, labels)
        # log metrics
        log = {'train_loss': loss,
                    'train_acc': acc,
                    'train_recall': recall,
                    'train_precision': precision}
        self.log = log
        # return loss
        return loss, log
    def validation_step(self, batch):
        # forward pass
        images, labels = batch
        # no gradient
        with torch.no_grad():
            logits = self.forward(images)
        # compute loss
        loss = self.cross_entropy_loss(logits, labels)
        # compute metrics
        acc = self.train_acc(logits, labels)
        recall = self.train_recall(logits, labels)
        precision = self.train_precision(logits, labels)
        # log metrics
        log = {'val_loss': loss,
                    'val_acc': acc,
                    'val_recall': recall,
                    'val_precision': precision}
        self.log = log
        # return loss
        return loss, log
    def test_step(self, batch):
        # forward pass
        images, labels = batch
        with torch.no_grad():
            logits = self.forward(images)
        # compute loss
        loss = self.cross_entropy_loss(logits, labels)
        # compute metrics
        acc = self.train_acc(logits, labels)
        recall = self.train_recall(logits, labels)
        precision = self.train_precision(logits, labels)
        # log metrics
        log = {'test_loss': loss,
                    'test_acc': acc,
                    'test_recall': recall,
                    'test_precision': precision}
        self.log = log
        # return loss
        return loss, log
    def configure_optimizers(self):
        # get optimizer
        optimizer = self.optimizer
        # return optimizer
        return optimizer
    def cross_entropy_loss(self, logits, labels):
        return torch.nn.functional.cross_entropy(logits, labels)

"""
class convnext_model:
    Facebook convnext model base with linear classification head
    This class is a wrapper around the pytorch lightning model class

"""
class convnext_model(pl.LightningModule):
    def __init__(self, in_channels, num_classes, 
                        dropout, learning_rate, image_size):
        super(convnext_model, self).__init__()
        # init metrics for eval
        self.train_acc = tm.Accuracy()
        self.train_recall = tm.Recall()
        self.train_precision = tm.Precision()
        self.val_acc = tm.Accuracy()
        self.val_recall = tm.Recall()
        self.val_precision = tm.Precision()
        self.test_acc = tm.Accuracy()
        self.test_recall = tm.Recall()
        self.test_precision = tm.Precision()
        # init convnext feature extractor layer
#         convnext_config = ConvNextConfig(num_channels = in_channels)
        self.convnext = ConvNextModel.from_pretrained("facebook/convnext-tiny-224")
#         self.convnext = self.convnext(convnext_config)
        # init fully connected layers
        self.dense = torch.nn.Linear(in_features = 768, out_features = 128)
        # init dropout layer
        self.dropout = torch.nn.Dropout(dropout)
        # init output layer
        self.output_layer = torch.nn.Linear(in_features = 128, out_features = num_classes)
        # init optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
        # init loss function
        self.loss_fn = torch.nn.CrossEntropyLoss()
    def forward(self, images):
        # forward pass through the network
#         x = self.feature_extractor(images, return_tensors = 'pt')
        x = self.convnext(images)
        x = x.pooler_output
        # pass through fully connected layer
        x = self.dense(x)
        x = self.dropout(x)
        x = torch.nn.functional.relu(x)
        # pass through output layer
        x = self.output_layer(x)
        return x
    def training_step(self, batch):
        # forward pass
        images, labels = batch
        logits = self.forward(images)
        # compute loss
        loss = self.cross_entropy_loss(logits, labels)
        # compute metrics
        acc = self.train_acc(logits, labels)
        recall = self.train_recall(logits, labels)
        precision = self.train_precision(logits, labels)
        # log metrics
        log = {'train_loss': loss,
                    'train_acc': acc,
                    'train_recall': recall,
                    'train_precision': precision}
        self.log = log
        # return loss
        return loss, log
    def validation_step(self, batch):
        # forward pass
        images, labels = batch
        # no gradient
        with torch.no_grad():
            logits = self.forward(images)
        # compute loss
        loss = self.cross_entropy_loss(logits, labels)
        # compute metrics
        acc = self.train_acc(logits, labels)
        recall = self.train_recall(logits, labels)
        precision = self.train_precision(logits, labels)
        # log metrics
        log = {'val_loss': loss,
                    'val_acc': acc,
                    'val_recall': recall,
                    'val_precision': precision}
        self.log = log
        # return loss
        return loss, log
    def test_step(self, batch):
        # forward pass
        images, labels = batch
        with torch.no_grad():
            logits = self.forward(images)
        # compute loss
        loss = self.cross_entropy_loss(logits, labels)
        # compute metrics
        acc = self.train_acc(logits, labels)
        recall = self.train_recall(logits, labels)
        precision = self.train_precision(logits, labels)
        # log metrics
        log = {'test_loss': loss,
                    'test_acc': acc,
                    'test_recall': recall,
                    'test_precision': precision}
        self.log = log
        # return loss
        return loss, log
    def configure_optimizers(self):
        # get optimizer
        optimizer = self.optimizer
        # return optimizer
        return optimizer
    def cross_entropy_loss(self, logits, labels):
        return torch.nn.functional.cross_entropy(logits, labels)