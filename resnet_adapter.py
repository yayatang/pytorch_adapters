import json
import torch
import torch.nn
import torch.nn.functional
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import copy
import os
import dtlpy as dl
from dtlpy.ml.dataset_generators.torch_dataset_generator import DataGenerator


class ModelAdapter(dl.BaseModelAdapter):
    """
    resnet Model adapter using pytorch.
    The class bind Dataloop model and snapshot entities with model code implementation
    """
    configuration = {
        'weights_filename': 'my_resnet.pth',
        'input_shape': (299, 299, 3),
    }

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
        self.logger.info("Loading a model from {}".format(local_path))
        self.model = torch.load(model_path)
        self.model.to(self.device)
        self.model.eval()
        # How to load the label_map from loaded model
        self.logger.info("Loaded model from {} successfully".format(model_path))

    def save(self, local_path, **kwargs):
        """ saves configuration and weights locally

            Virtual method - need to implement

            the function is called in save_to_snapshot which first save locally and then uploads to snapshot entity

        :param local_path: `str` directory path in local FileSystem
        """
        weights_filename = kwargs.get('weights_filename', 'model.pth')
        classes_filename = kwargs.get('classes_filename', 'classes.json')

        torch.save(self.model, os.path.join(local_path, weights_filename))
        with open(os.path.join(local_path, classes_filename), 'w') as f:
            json.dump(self.label_map, f)
        self.snapshot.configuration['weights_filename'] = weights_filename
        self.snapshot.configuration['label_map'] = self.label_map
        self.snapshot.update()

    def train(self, data_path, output_path, **kwargs):
        """ Train the model according to data in local_path and save the snapshot to dump_path

            Virtual method - need to implement
        :param data_path: `str` local File System path to where the data was downloaded and converted at
        :param output_path: `str` local File System path where to dump training mid-results (checkpoints, logs...)
        """
        configuration = self.configuration
        configuration.update(self.snapshot.configuration)
        num_epochs = configuration.get('num_epochs', 10)
        batch_size = configuration.get('batch_size', 64)
        input_size = configuration.get('input_size', 256)

        # DATA TRANSFORMERS
        data_transforms = {
            'train': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'val': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
        }

        ####################
        # Prepare the data #
        ####################
        train_dataset = DataGenerator(data_path=os.path.join(data_path, 'train'),
                                      dataset_entity=self.snapshot.dataset,
                                      annotation_type=dl.AnnotationType.CLASSIFICATION,
                                      transforms=data_transforms['train'])
        val_dataset = DataGenerator(data_path=os.path.join(data_path, 'validation'),
                                    dataset_entity=self.snapshot.dataset,
                                    annotation_type=dl.AnnotationType.CLASSIFICATION,
                                    transforms=data_transforms['val'])

        dataloaders = {'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
                       'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=True)}
        dataset_sizes = {phase: len(dataloaders[phase]) for phase in ['train', 'val']}

        #################
        # prepare model #
        #################
        n_classes = len(train_dataset.id_to_label_map)
        self.logger.info('Setting last layer for {} classes'.format(n_classes))
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, n_classes)

        criterion = torch.nn.CrossEntropyLoss()
        # Observe that all parameters are being optimized
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        since = time.time()
        epoch_time = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        self.model.to(device=self.device)
        for epoch in range(num_epochs):
            self.logger.info('Epoch {}/{}  duration {:1.2f}'.format(epoch, num_epochs - 1, time.time() - epoch_time))
            epoch_time = time.time()

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device).long()

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

                self.logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

        time_elapsed = time.time() - since
        self.logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        self.logger.info('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        self.label_map = train_dataset.id_to_label_map

    def predict(self, batch, **kwargs):
        """ Model inference (predictions) on batch of images

            Virtual method - need to implement

        :param batch: `np.ndarray`
        :return: `list[dl.AnnotationCollection]` each collection is per each image / item in the batch
        """
        input_shape = self.snapshot.configuration.get('input_shape', (256, 256))
        preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(input_shape),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
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
            self.logger.debug("Predicted {:20} ({:1.3f})".format(pred_label, pred_score))
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
    with open(os.path.join(os.path.dirname(__file__), 'imagenet_labels_list.json'), 'r') as fh:
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
                                  entry_point='resnet_adapter.py')
    return model


def snapshot_creation(model: dl.Model, env: str = 'prod', resnet_ver='50'):
    dl.setenv(env)
    project = dl.projects.get('DataloopModels')
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
    snapshot = snapshot_creation(model=model, env=env, resnet_ver=resnet_ver)
    print("Snapshot : {} - {} created".format(snapshot.name, snapshot.id))
