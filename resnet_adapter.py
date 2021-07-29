import dtlpy as dl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.linear import Linear
from torch.nn.modules.module import T
from torch.utils.data import Dataset, DataLoader, dataloader
import torchvision
from torchvision import models, transforms, datasets, io
import time
import os
import copy
import numpy as np
from PIL import Image
import json
import glob
from pathlib import Path
import pandas as pd


class ModelAdapter(dl.BaseModelAdapter):
    """
    resnet Model adapter using pytorch.
    The class bind Dataloop model and snapshot entities with model code implementation
    """
    _defaults = {
        'model_fname': 'my_resnet.pth',
        'input_shape': (299, 299, 3),
    }

    def __init__(self, model_entity):
        super(ModelAdapter, self).__init__(model_entity)
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.label_map = {}

    # ===============================
    # NEED TO IMPLEMENT THESE METHODS
    # ===============================
     # READ: https://pytorch.org/tutorials/beginner/saving_loading_models.html

    def load(self, local_path, **kwargs):
        """ Loads model and populates self.model with a `runnable` model

            Virtual method - need to implement

            This function is called by load_from_snapshot (download to local and then loads)

        :param local_path: `str` directory path in local FileSystem
        """
        use_pretrained = getattr(self, 'use_pretrained', False)
        input_shape = getattr(self, 'input_shape', None)

        msg = "Loading the model. pretrained = {}, local_path {}; input_shape {}".format(
                use_pretrained, local_path, input_shape)
        print(msg)
        self.logger.info(msg)

        if local_path is not None:
            self.logger.info("Loading a model from {}".format(local_path))
            #  TODO: Is it better to use model.load_state_dict
            # model = TheModelClass(*args, **kwargs)
            # model.load_state_dict(torch.load(PATH))
            model_path = "{d}/{f}".format(d=local_path, f=self.model_name)
            self.model = torch.load(model_path)
            self.model.eval()
            # How to load the label_map from loaded model
            self.logger.info("Loaded model from {} successfully".format(model_path))

        elif use_pretrained:
            self.logger.info('Using the pytorch pretrained model')
            self.model = models.resnet50(pretrained=True)
            self.label_map = json.load(open('imagenet_labels.json', 'r'))
            self.model.eval()
        else:
            self.logger.info("Build a dedicated model for specific labels. This requires `label_map`")
            if not hasattr(self, 'label_map'):
                raise RuntimeError("Cannot create specific model w/o a label_map")
            if not hasattr(self, 'nof_classes'):
                self.nof_classes = len(self.label_map)

            self.model = models.resnet50(pretrained=True)

            # optional is to freeze all previous layers - which is better for smaller data
            for param in self.model.parameters():
                param.requires_grad = False

            num_ftrs = self.model.fc.in_features
            # New layer added is by default requires_grad=True
            self.model.fc = nn.Linear(in_features=num_ftrs, out_features=self.nof_classes)

            self.logger.info("Created new trainable resnet50 model with {} classes. ({})".
                             format(self.nof_classes, self.model_entity.name))

        # Save the pytorch preprocess
        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                # transforms.CenterCrop(224),
                transforms.CenterCrop(self.input_shape[:2]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def save(self, local_path, **kwargs):
        """ saves configuration and weights locally

            Virtual method - need to implement

            the function is called in save_to_snapshot which first save locally and then uploads to snapshot entity

        :param local_path: `str` directory path in local FileSystem
        """
        raise NotImplementedError("Please implement 'save' method in {}".format(self.__class__.__name__))
        # torch.save(model, local_path)

    def train(self, data_path, dump_path, **kwargs):
        """ Train the model according to data in local_path and save the snapshot to dump_path

            Virtual method - need to implement
        :param data_path: `str` local File System path to where the data was downloaded and converted at
        :param dump_path: `str` local File System path where to dump training mid-results (checkpoints, logs...)
        """
        num_epochs = kwargs.get('num_epocs', 10)
        
        # DATA TRANSFORMERS
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                # transforms.RandomResizedCrop(self.input_shape[:2]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        criterion = nn.CrossEntropyLoss()
        # Only last fully connected layer is being updated
        optimizer = optim.SGD(self.model.fc.parameters(), lr=0.001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # Prepare the data:
        # TODO: how to use  different loaders (train / val)
        dataset = DlpClassDataset(data_path=data_path, label_map=self.label_map, transform=data_transforms['train'])
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        dataset_sizes = {x: len(dataset) for x in ['train', 'val']}
        

        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloader:  #dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    exp_lr_scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                msg  = '{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc)
                print(msg)
                self.logger.info(msg)

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        self.logger.info('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        
    def predict(self, batch, **kwargs):
        """ Model inference (predictions) on batch of images

            Virtual method - need to implement

        :param batch: `np.ndarray`
        :return: `list[dl.AnnotationCollection]` each collection is per each image / item in the batch
        """
        img_tensors = [self.preprocess(img.astype('uint8')) for img in batch]
        batch_tensor = torch.stack(img_tensors)
        self.model.eval() # set dropout and batch normalization layers to evaluation mode
        preds_out = self.model(batch_tensor)
        percentages = torch.nn.functional.softmax(preds_out, dim=1)

        batch_predictions = []
        # scores, predictions_idxs = torch.max(percentages, 1)
        # for pred_score, high_pred_index in zip(scores, predictions_idxs):
        for img_pred in percentages:
            pred_score, high_pred_index = torch.max(img_pred, 0)
            pred_label = self.label_map.get(str(high_pred_index.item()), 'UNKNOWN')

            item_pred = dl.ml.predictions_utils.add_classification(
                label=pred_label,
                score=pred_score.item(),
                adapter=self,
                collection=None
            )
            self.logger.debug("Predicted {:20} ({:1.3f})".format(pred_label, pred_score))
            batch_predictions.append(item_pred)
        return batch_predictions

    def convert(self, data_path, **kwargs):
        """ Convert Dataloop structure data to model structured

            Virtual method - need to implement

            e.g. take dlp dir structure and construct annotation file

        :param data_path: `str` local File System directory path where
                           we already downloaded the data from dataloop platform
        :return:
        """

        # Create a dataset
        self.dataset = DlpClassDataset(data_path=data_path, label_map=self.label_map, transform=None)

        # optional use a datalopader
        self.dataloader = DataLoader(self.dataset, batch_size=4, shuffle=True)
        # TODO: how to differ between train and val datasets
        
    # =============
    # DTLPY METHODS
    # Do not change
    # =============
    # function are here to ease the traceback...
    def load_from_snapshot(self, local_path, snapshot_id, **kwargs):
        """ Loads a model from given `snapshot`.
            Reads configurations and instantiate self.snapshot
            Downloads the snapshot bucket (if available)

        :param local_path:  `str` directory path in local FileSystem to download the snapshot to
        :param snapshot_id:  `str` snapshot id
        """
        return super(ModelAdapter, self).load_from_snapshot(local_path=local_path, snapshot_id=snapshot_id, **kwargs)

    def save_to_snapshot(self, local_path, snapshot_name=None, description=None, cleanup=False, **kwargs):
        """ Saves configuration and weights to new snapshot bucket
            loads only applies for remote buckets

        :param local_path: `str` directory path in local FileSystem
        :param snapshot_name: `str` name for the new snapshot
        :param description:  `str` description for the new snapshot
        :param cleanup: `bool` if True (default) remove the data from local FileSystem after upload
        :return:
        """
        return super(ModelAdapter, self).save_to_snapshot(local_path=local_path, snapshot_name=snapshot_name,
                                                          description=description, cleanup=cleanup,
                                                          **kwargs)

    def prepare_trainset(self, data_path, partitions=None, filters=None, **kwargs):
        """
        Prepares train set for train.
        download the specific partition selected to data_path and preforms `self.convert` to the data_path dir

        :param data_path: `str` directory path to use as the root to the data from Dataloop platform
        :param partitions: `dl.SnapshotPartitionType` or list of partitions, defaults for all partitions
        :param filters: `dl.Filter` in order to select only part of the data
        """
        return super(ModelAdapter, self).prepare_trainset(data_path=data_path, partitions=partitions,
                                                          filters=filters,
                                                          **kwargs)

    def predict_items(self, items: list, with_upload=True, cleanup=False, batch_size=16, output_shape=None, **kwargs):
        """
        Predicts all items given

        :param items: `List[dl.Item]`
        :param with_upload: `bool` uploads the predictions back to the given items
        :param cleanup: `bool` if set removes existing predictions with the same model-snapshot name (default: False)
        :param batch_size: `int` size of batch to run a single inference
        :param output_shape: `tuple` (width, height) of resize needed per image
        :return: `List[Prediction]' `Prediction is set by model.output_type
        """
        return super(ModelAdapter, self).predict_items(items=items, with_upload=with_upload,
                                                       cleanup=cleanup, batch_size=batch_size, output_shape=output_shape,
                                                       **kwargs)


class DlpClassDataset(Dataset):

    def __init__(self, data_path, label_map, transform=None) -> None:
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            data_path (string): Directory with all the images (dataloop root dir).
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = data_path
        self.transform = transform
        self.label_map = label_map
        self.label_to_id = {v: k for k, v in label_map.items()}

        self.image_paths = []
        self.image_labels = []

        for ann_json in Path(self.root_dir).rglob('*.json'):
        # for ann_json in glob.glob("{}/json/**/*.json".format(self.root_dir)):
            dlp_ann = json.load(open(ann_json, 'r'))
            self.image_paths.append(self.root_dir + '/items/' + dlp_ann['filename'])
            self.image_labels.append(dlp_ann['annotations'][0]['label'])
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.image_paths[idx])
        # image = io.read_image(img_name)
        image = Image.open(img_name)
        label = int(self.label_to_id(self.image_labels[idx], -1))
        if self.transform:
            image = self.transform(image)

        return image, label
