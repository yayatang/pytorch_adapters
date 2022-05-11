from torch.utils.data import DataLoader
from imgaug import augmenters as iaa
import torchvision.transforms
import torch.nn.functional
import torch.nn
import logging
import torch
import json
import time
import copy
import tqdm
import os
import torch.optim as optim
import imgaug as ia
import pandas as pd
import numpy as np
import dtlpy as dl

from dtlpy.utilities.dataset_generators.dataset_generator import collate_torch
from dtlpy.utilities.dataset_generators.dataset_generator_torch import DatasetGeneratorTorch

logger = logging.getLogger('resnet-adapter')


class ModelAdapter(dl.BaseModelAdapter):
    """
    resnet Model adapter using pytorch.
    The class bind Dataloop model and snapshot entities with model code implementation
    """

    def __init__(self, model_entity):
        super(ModelAdapter, self).__init__(model_entity)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        weights_filename = self.snapshot.configuration.get('weights_filename', 'model.pth')
        classes_filename = self.snapshot.configuration.get('classes_filename', 'classes.json')
        # load classes
        with open(os.path.join(local_path, classes_filename)) as f:
            self.label_map = json.load(f)

        # load model arch and state
        model_path = os.path.join(local_path, weights_filename)
        logger.info("Loading a model from {}".format(local_path))
        self.model = torch.load(model_path)
        self.model.to(self.device)
        self.model.eval()
        # How to load the label_map from loaded model
        logger.info("Loaded model from {} successfully".format(model_path))

    def save(self, local_path, **kwargs):
        """ saves configuration and weights locally

            Virtual method - need to implement

            the function is called in save_to_snapshot which first save locally and then uploads to snapshot entity

        :param local_path: `str` directory path in local FileSystem
        """
        weights_filename = kwargs.get('weights_filename', 'model.pth')
        torch.save(self.model, os.path.join(local_path, weights_filename))
        self.snapshot.configuration['weights_filename'] = weights_filename
        self.snapshot.update()

    def train(self, data_path, output_path, **kwargs):
        """ Train the model according to data in local_path and save the snapshot to dump_path

            Virtual method - need to implement
        :param data_path: `str` local File System path to where the data was downloaded and converted at
        :param output_path: `str` local File System path where to dump training mid-results (checkpoints, logs...)
        """
        configuration = self.configuration
        num_epochs = configuration.get('num_epochs', 10)
        batch_size = configuration.get('batch_size', 64)
        input_size = configuration.get('input_size', 256)

        # sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        # DATA TRANSFORMERS
        def gray_to_rgb(x):
            return x.convert('RGB')

        data_transforms = {
            'train': [
                iaa.Resize({"height": input_size, "width": input_size}),
                iaa.flip.Fliplr(p=0.5),
                iaa.flip.Flipud(p=0.2),
                torchvision.transforms.ToPILImage(),
                gray_to_rgb,
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ],
            'val': [torchvision.transforms.ToPILImage(),
                    gray_to_rgb,
                    torchvision.transforms.Resize((input_size, input_size)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]

        }

        ####################
        # Prepare the data #
        ####################
        train_dataset = DatasetGeneratorTorch(data_path=os.path.join(data_path, 'train'),
                                              dataset_entity=self.snapshot.dataset,
                                              annotation_type=dl.AnnotationType.CLASSIFICATION,
                                              transforms=data_transforms['val'],
                                              id_to_label_map=self.snapshot.configuration['id_to_label_map'],
                                              class_balancing=False,
                                              #   to_categorical=True,
                                              #   collate_fn=collate_torch
                                              )
        val_dataset = DatasetGeneratorTorch(data_path=os.path.join(data_path, 'validation'),
                                            dataset_entity=self.snapshot.dataset,
                                            annotation_type=dl.AnnotationType.CLASSIFICATION,
                                            transforms=data_transforms['val'],
                                            id_to_label_map=self.snapshot.configuration['id_to_label_map'],
                                            # to_categorical=True,
                                            # collate_fn=collate_torch
                                            )

        dataloaders = {'train': DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True),
                       'val': DataLoader(val_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)}
        #################
        # prepare model #
        #################
        n_classes = len(train_dataset.id_to_label_map)
        logger.info('Setting last layer for {} classes'.format(n_classes))
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, n_classes)

        criterion = torch.nn.CrossEntropyLoss()
        # Observe that all parameters are being optimized
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())

        # early stopping params
        best_loss = np.inf
        best_acc = 0.0
        not_improving_epochs = 0
        patience_epochs = 7
        end_training = False
        self.metrics = {'history': list()}
        #####
        self.model.to(device=self.device)
        for epoch in range(num_epochs):
            if end_training:
                break
            logger.info('Epoch {}/{} Start...'.format(epoch, num_epochs))
            epoch_time = time.time()
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                total = 0

                # Iterate over data.
                with tqdm.tqdm(dataloaders[phase], unit="batch") as tepoch:
                    for batch in tepoch:
                        inputs = torch.stack(tuple(batch['image']), 0).to(self.device)
                        labels = torch.stack(tuple(batch['annotations']), 0).to(self.device).long().squeeze(1)

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
                        total += inputs.size(0)
                        running_loss += (loss.item() * inputs.size(0))
                        running_corrects += torch.sum(preds == labels.data).double().cpu().numpy()
                        epoch_acc = running_corrects / total
                        epoch_loss = running_loss / total
                        tepoch.set_postfix(loss=epoch_loss, accuracy=100. * epoch_acc)

                if phase == 'train':
                    exp_lr_scheduler.step()

                logger.info(
                    f'Epoch {epoch}/{num_epochs} - {phase} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Duration {(time.time() - epoch_time):.2f}')
                # deep copy the model
                self.metrics['history'].append({'phase': phase,
                                                'epoch': epoch,
                                                'loss': epoch_loss,
                                                'acc': epoch_acc})
                if phase == 'val':
                    if epoch_loss < best_loss:
                        not_improving_epochs = 0
                        best_loss = epoch_loss
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        logger.info(
                            f'Validation loss decreased ({best_loss:.6f} --> {epoch_loss:.6f}).  Saving model ...')
                    else:
                        not_improving_epochs += 1
                    if not_improving_epochs > patience_epochs:
                        end_training = True
                        logger.info(f'EarlyStopping counter: {not_improving_epochs} out of {patience_epochs}')
                ###############
                # save debugs #
                ###############

        time_elapsed = time.time() - since
        logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        logger.info('Best val loss: {:4f}, best acc: {:4f}'.format(best_loss, best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        self.label_map = train_dataset.id_to_label_map

        #############
        # Confusion #
        #############
        self.metrics['confusion'] = dict()
        y_true_total = list()
        y_pred_total = list()
        labels = list(self.label_map.values())
        confusion_dict = dict()
        for phase in ['train', 'val']:
            y_true = list()
            y_pred = list()
            filepaths = list()
            for batch in dataloaders[phase]:
                xs = torch.stack(tuple(batch['image']), 0).to(self.device)
                ys = torch.stack(tuple(batch['annotations']), 0).to(self.device).long().squeeze()
                y_true.extend([self.label_map[int(y)] for y in ys])
                with torch.set_grad_enabled(False):
                    outputs = self.model(xs)
                    _, preds = torch.max(outputs, 1)
                y_pred.extend([self.label_map[int(l)] for l in preds])
                filepaths.extend(batch['image_filepath'])
            for t, p, file in zip(y_true, y_pred, filepaths):
                if t not in confusion_dict:
                    confusion_dict[t] = dict()
                if p not in confusion_dict[t]:
                    confusion_dict[t][p] = list()
                confusion_dict[t][p].append(file)
            s_true = pd.Series(y_true, name='Actual')
            s_pred = pd.Series(y_pred, name='Predicted')
            self.metrics['confusion'][phase] = pd.crosstab(s_true, s_pred).reindex(columns=labels,
                                                                                   index=labels,
                                                                                   fill_value=0)
            y_true_total.extend(y_true)
            y_pred_total.extend(y_pred)
        s_true = pd.Series(y_true_total, name='Actual')
        s_pred = pd.Series(y_pred_total, name='Predicted')
        self.metrics['confusion']['overall'] = pd.crosstab(s_true, s_pred).reindex(columns=labels,
                                                                                   index=labels,
                                                                                   fill_value=0)

    # def debug(self):
    #     import matplotlib.pyplot as plt
    #     if idx is None:
    #         idx = np.random.randint(self.__len__())
    #     data_item = train_dataset.__getitem__(idx)
    #     image = data_item.get('image')
    #     labels = data_item.get('labels')
    #     targets = data_item.get('annotations')
    #     annotations = train_dataset._to_dtlpy(targets=targets, labels=labels)
    #     marked_image = annotations.show(image=image,
    #                                     height=image.shape[0],
    #                                     width=image.shape[1],
    #                                     # alpha=0.8,
    #                                     with_text=False)
    #     if plot:
    #         plt.figure()
    #         plt.imshow(marked_image)
    #     if return_output:
    #         return marked_image, annotations

    def predict(self, batch, **kwargs):
        """ Model inference (predictions) on batch of images

            Virtual method - need to implement

        :param batch: `np.ndarray`
        :return: `list[dl.AnnotationCollection]` each collection is per each image / item in the batch
        """
        input_size = self.snapshot.configuration.get('input_size', 256)
        preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize(input_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
            ]
        )
        img_tensors = [preprocess(img.astype('uint8')) for img in batch]
        batch_tensor = torch.stack(img_tensors).to(self.device)
        preds_out = self.model(batch_tensor)
        percentages = torch.nn.functional.softmax(preds_out, dim=1)
        batch_predictions = list()
        for img_pred in percentages:
            pred_score, high_pred_index = torch.max(img_pred, 0)
            pred_label = self.label_map.get(str(high_pred_index.item()), 'UNKNOWN')
            collection = dl.AnnotationCollection()
            collection.add(annotation_definition=dl.Classification(label=pred_label),
                           model_info={'name': self.model_entity.name,
                                       'confidence': pred_score.item(),
                                       'model_id': self.model_entity.id,
                                       'snapshot_id': self.snapshot.id})
            logger.debug("Predicted {:20} ({:1.3f})".format(pred_label, pred_score))
            batch_predictions.append(collection)
        return batch_predictions

    def convert_from_dtlpy(self, data_path, **kwargs):
        """ Convert Dataloop structure data to model structured

            Virtual method - need to implement

            e.g. take dlp dir structure and construct annotation file

        :param data_path: `str` local File System directory path where
                           we already downloaded the data from dataloop platform
        :return:
        """

        ...


def _get_imagenet_label_json():
    import json
    with open('imagenet_labels_list.json', 'r') as fh:
        labels = json.load(fh)
    return labels


def model_creation(env: str = 'prod'):
    dl.setenv(env)
    project = dl.projects.get('DataloopModels')

    codebase = dl.GitCodebase(git_url='https://github.com/dataloop-ai/pytorch_adapters',
                              git_tag='master')
    model = project.models.create(model_name='ResNet',
                                  description='Global Dataloop ResNet implemeted in pytorch',
                                  output_type=dl.AnnotationType.CLASSIFICATION,
                                  is_global=True,
                                  codebase=codebase,
                                  tags=['torch'],
                                  default_configuration={
                                      'weights_filename': 'model.pth',
                                      'input_size': 256,
                                  },
                                  entry_point='resnet_adapter.py')
    return model


def snapshot_creation(project_name, model: dl.Model, env: str = 'prod', resnet_ver='50'):
    dl.setenv(env)
    project = dl.projects.get(project_name)
    bucket = dl.buckets.create(dl.BucketType.GCS,
                               gcs_project_name='viewo-main',
                               gcs_bucket_name='model-mgmt-snapshots',
                               gcs_prefix='ResNet{}'.format(resnet_ver))
    snapshot = model.snapshots.create(snapshot_name='pretrained-resnet{}'.format(resnet_ver),
                                      description='resnset{} pretrained on imagenet'.format(resnet_ver),
                                      tags=['pretrained', 'imagenet'],
                                      dataset_id=None,
                                      is_global=True,
                                      # status='trained',
                                      configuration={'weights_filename': 'model.pth',
                                                     'classes_filename': 'classes.json'},
                                      project_id=project.id,
                                      bucket=bucket,
                                      labels=_get_imagenet_label_json()
                                      )
    return snapshot


def model_and_snapshot_creation(env: str = 'prod', resnet_ver='50'):
    model = model_creation(env=env)
    print("Model : {} - {} created".format(model.name, model.id))
    snapshot = snapshot_creation(project_name, model=model, env=env, resnet_ver=resnet_ver)
    print("Snapshot : {} - {} created".format(snapshot.name, snapshot.id))
