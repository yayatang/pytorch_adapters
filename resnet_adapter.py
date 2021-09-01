import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import copy
import os
import dtlpy as dl
from dtlpy.ml.ml_dataset import get_torch_dataset


class ModelAdapter(dl.BaseModelAdapter):
    """
    resnet Model adapter using pytorch.
    The class bind Dataloop model and snapshot entities with model code implementation
    """
    configuration = {
        'model_fname': 'my_resnet.pth',
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
        input_shape = self.snapshot.configuration.get('input_shape', (256, 256))
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
        # Save the pytorch preprocess
        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(input_shape),
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
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        ####################
        # Prepare the data #
        ####################
        train_dataset = get_torch_dataset()(data_path=os.path.join(data_path, 'train'),
                                            dataset_entity=self.snapshot.dataset,
                                            annotation_type=dl.AnnotationType.CLASSIFICATION,
                                            transforms=data_transforms['train'])
        val_dataset = get_torch_dataset()(data_path=os.path.join(data_path, 'validation'),
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
        self.model.fc = nn.Linear(num_ftrs, n_classes)

        criterion = nn.CrossEntropyLoss()
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
            self.logger.info('-' * 25)
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
        img_tensors = [self.preprocess(img.astype('uint8')) for img in batch]
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
    with open (os.path.join(os.path.dirname(__file__), 'imagenet_labels_list.json'), 'r') as fh:
        labels = json.load(fh)
    return labels

def model_and_snapshot_creation(env: str = 'prod', resnet_ver='50'):
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

    bucket = dl.buckets.create(dl.BucketType.GCS,
                               gcs_project_name='viewo-main',
                               gcs_bucket_name='model-mgmt-snapshots',
                               gcs_prefix='ResNet{}'.format(resnet_ver))
    snapshot = model.snapshots.create(snapshot_name='pretrained-resnset{}'.format(resnet_ver),
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


def _get_imagenet_labels():
    return ['tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead', 'electric ray',
            'stingray', 'cock', 'hen', 'ostrich', 'brambling', 'goldfinch', 'house finch', 'junco',
            'indigo bunting', 'robin', 'bulbul', 'jay', 'magpie', 'chickadee', 'water ouzel',
            'kite', 'bald eagle', 'vulture', 'great grey owl', 'European fire salamander',
            'common newt', 'eft', 'spotted salamander', 'axolotl', 'bullfrog', 'tree frog',
            'tailed frog', 'loggerhead', 'leatherback turtle', 'mud turtle', 'terrapin',
            'box turtle', 'banded gecko', 'common iguana', 'American chameleon', 'whiptail',
            'agama', 'frilled lizard', 'alligator lizard', 'Gila monster', 'green lizard',
            'African chameleon', 'Komodo dragon', 'African crocodile', 'American alligator',
            'triceratops', 'thunder snake', 'ringneck snake', 'hognose snake', 'green snake',
            'king snake', 'garter snake', 'water snake', 'vine snake', 'night snake',
            'boa constrictor', 'rock python', 'Indian cobra', 'green mamba', 'sea snake',
            'horned viper', 'diamondback', 'sidewinder', 'trilobite', 'harvestman', 'scorpion',
            'black and gold garden spider', 'barn spider', 'garden spider', 'black widow',
            'tarantula', 'wolf spider', 'tick', 'centipede', 'black grouse', 'ptarmigan',
            'ruffed grouse', 'prairie chicken', 'peacock', 'quail', 'partridge',
            'African grey', 'macaw', 'sulphur-crested cockatoo', 'lorikeet', 'coucal',
            'bee eater', 'hornbill', 'hummingbird', 'jacamar', 'toucan', 'drake',
            'red-breasted merganser', 'goose', 'black swan', 'tusker', 'echidna', 'platypus',
            'wallaby', 'koala', 'wombat', 'jellyfish', 'sea anemone', 'brain coral',
            'flatworm', 'nematode', 'conch', 'snail', 'slug', 'sea slug', 'chiton',
            'chambered nautilus', 'Dungeness crab', 'rock crab', 'fiddler crab', 'king crab',
            'American lobster', 'spiny lobster', 'crayfish', 'hermit crab', 'isopod',
            'white stork', 'black stork', 'spoonbill', 'flamingo', 'little blue heron',
            'American egret', 'bittern', 'crane', 'limpkin', 'European gallinule',
            'American coot', 'bustard', 'ruddy turnstone', 'red-backed sandpiper',
            'redshank', 'dowitcher', 'oystercatcher', 'pelican', 'king penguin', 'albatross',
            'grey whale', 'killer whale', 'dugong', 'sea lion', 'Chihuahua',
            'Japanese spaniel', 'Maltese dog', 'Pekinese', 'Shih-Tzu', 'Blenheim spaniel',
            'papillon', 'toy terrier', 'Rhodesian ridgeback', 'Afghan hound', 'basset',
            'beagle', 'bloodhound', 'bluetick', 'black-and-tan coonhound', 'Walker hound',
            'English foxhound', 'redbone', 'borzoi', 'Irish wolfhound', 'Italian greyhound',
            'whippet', 'Ibizan hound', 'Norwegian elkhound', 'otterhound', 'Saluki',
            'Scottish deerhound', 'Weimaraner', 'Staffordshire bullterrier',
            'American Staffordshire terrier', 'Bedlington terrier', 'Border terrier',
            'Kerry blue terrier', 'Irish terrier', 'Norfolk terrier', 'Norwich terrier',
            'Yorkshire terrier', 'wire-haired fox terrier', 'Lakeland terrier',
            'Sealyham terrier', 'Airedale', 'cairn', 'Australian terrier', 'Dandie Dinmont',
            'Boston bull', 'miniature schnauzer', 'giant schnauzer', 'standard schnauzer',
            'Scotch terrier', 'Tibetan terrier', 'silky terrier',
            'soft-coated wheaten terrier', 'West Highland white terrier', 'Lhasa',
            'flat-coated retriever', 'curly-coated retriever', 'golden retriever',
            'Labrador retriever', 'Chesapeake Bay retriever', 'German short-haired pointer',
            'vizsla', 'English setter', 'Irish setter', 'Gordon setter', 'Brittany spaniel',
            'clumber', 'English springer', 'Welsh springer spaniel', 'cocker spaniel',
            'Sussex spaniel', 'Irish water spaniel', 'kuvasz', 'schipperke', 'groenendael',
            'malinois', 'briard', 'kelpie', 'komondor', 'Old English sheepdog',
            'Shetland sheepdog', 'collie', 'Border collie', 'Bouvier des Flandres',
            'Rottweiler', 'German shepherd', 'Doberman', 'miniature pinscher',
            'Greater Swiss Mountain dog', 'Bernese mountain dog', 'Appenzeller',
            'EntleBucher', 'boxer', 'bull mastiff', 'Tibetan mastiff', 'French bulldog',
            'Great Dane', 'Saint Bernard', 'Eskimo dog', 'malamute', 'Siberian husky',
            'dalmatian', 'affenpinscher', 'basenji', 'pug', 'Leonberg', 'Newfoundland',
            'Great Pyrenees', 'Samoyed', 'Pomeranian', 'chow', 'keeshond',
            'Brabancon griffon', 'Pembroke', 'Cardigan', 'toy poodle', 'miniature poodle',
            'standard poodle', 'Mexican hairless', 'timber wolf', 'white wolf', 'red wolf',
            'coyote', 'dingo', 'dhole', 'African hunting dog', 'hyena', 'red fox', 'kit fox',
            'Arctic fox', 'grey fox', 'tabby', 'tiger cat', 'Persian cat', 'Siamese cat',
            'Egyptian cat', 'cougar', 'lynx', 'leopard', 'snow leopard', 'jaguar', 'lion',
            'tiger', 'cheetah', 'brown bear', 'American black bear', 'ice bear',
            'sloth bear', 'mongoose', 'meerkat', 'tiger beetle', 'ladybug', 'ground beetle',
            'long-horned beetle', 'leaf beetle', 'dung beetle', 'rhinoceros beetle',
            'weevil', 'fly', 'bee', 'ant', 'grasshopper', 'cricket', 'walking stick',
            'cockroach', 'mantis', 'cicada', 'leafhopper', 'lacewing', 'dragonfly',
            'damselfly', 'admiral', 'ringlet', 'monarch', 'cabbage butterfly',
            'sulphur butterfly', 'lycaenid', 'starfish', 'sea urchin', 'sea cucumber',
            'wood rabbit', 'hare', 'Angora', 'hamster', 'porcupine', 'fox squirrel',
            'marmot', 'beaver', 'guinea pig', 'sorrel', 'zebra', 'hog', 'wild boar',
            'warthog', 'hippopotamus', 'ox', 'water buffalo', 'bison', 'ram', 'bighorn',
            'ibex', 'hartebeest', 'impala', 'gazelle', 'Arabian camel', 'llama', 'weasel',
            'mink', 'polecat', 'black-footed ferret', 'otter', 'skunk', 'badger',
            'armadillo', 'three-toed sloth', 'orangutan', 'gorilla', 'chimpanzee', 'gibbon',
            'siamang', 'guenon', 'patas', 'baboon', 'macaque', 'langur', 'colobus',
            'proboscis monkey', 'marmoset', 'capuchin', 'howler monkey', 'titi',
            'spider monkey', 'squirrel monkey', 'Madagascar cat', 'indri', 'Indian elephant',
            'African elephant', 'lesser panda', 'eel', 'coho', 'rock beauty', 'giant panda',
            'barracouta', 'anemone fish', 'sturgeon', 'gar', 'lionfish', 'puffer', 'abacus',
            'abaya', 'academic gown', 'accordion', 'acoustic guitar', 'aircraft carrier',
            'airliner', 'airship', 'altar', 'ambulance', 'amphibian', 'analog clock',
            'apiary', 'apron', 'ashcan', 'assault rifle', 'backpack', 'bakery',
            'balance beam', 'balloon', 'ballpoint', 'Band Aid', 'banjo', 'bannister',
            'barbell', 'barber chair', 'barbershop', 'barn', 'barometer', 'barrel', 'barrow',
            'baseball', 'basketball', 'bassinet', 'bassoon', 'bathing cap', 'bath towel',
            'bathtub', 'beach wagon', 'beacon', 'beaker', 'bearskin', 'beer bottle',
            'beer glass', 'bell cote', 'bib', 'bicycle-built-for-two', 'bikini', 'binder',
            'binoculars', 'birdhouse', 'boathouse', 'bobsled', 'bolo tie', 'bonnet',
            'bookcase', 'bookshop', 'bottlecap', 'bow', 'bow tie', 'brass', 'brassiere',
            'breakwater', 'breastplate', 'broom', 'bucket', 'buckle', 'bulletproof vest',
            'bullet train', 'butcher shop', 'cab', 'caldron', 'candle', 'cannon', 'canoe',
            'can opener', 'cardigan', 'car mirror', 'carousel', "carpenter's kit", 'carton',
            'car wheel', 'cash machine', 'cassette', 'cassette player', 'castle',
            'catamaran', 'CD player', 'cello', 'cellular telephone', 'chain',
            'chainlink fence', 'chain mail', 'chain saw', 'chest', 'chiffonier', 'chime',
            'china cabinet', 'Christmas stocking', 'church', 'cinema', 'cleaver',
            'cliff dwelling', 'cloak', 'clog', 'cocktail shaker', 'coffee mug', 'coffeepot',
            'coil', 'combination lock', 'computer keyboard', 'confectionery',
            'container ship', 'convertible', 'corkscrew', 'cornet', 'cowboy boot',
            'cowboy hat', 'cradle', 'crane', 'crash helmet', 'crate', 'crib', 'Crock Pot',
            'croquet ball', 'crutch', 'cuirass', 'dam', 'desk', 'desktop computer',
            'dial telephone', 'diaper', 'digital clock', 'digital watch', 'dining table',
            'dishrag', 'dishwasher', 'disk brake', 'dock', 'dogsled', 'dome', 'doormat',
            'drilling platform', 'drum', 'drumstick', 'dumbbell', 'Dutch oven',
            'electric fan', 'electric guitar', 'electric locomotive', 'entertainment center',
            'envelope', 'espresso maker', 'face powder', 'feather boa', 'file', 'fireboat',
            'fire engine', 'fire screen', 'flagpole', 'flute', 'folding chair',
            'football helmet', 'forklift', 'fountain', 'fountain pen', 'four-poster',
            'freight car', 'French horn', 'frying pan', 'fur coat', 'garbage truck',
            'gasmask', 'gas pump', 'goblet', 'go-kart', 'golf ball', 'golfcart', 'gondola',
            'gong', 'gown', 'grand piano', 'greenhouse', 'grille', 'grocery store',
            'guillotine', 'hair slide', 'hair spray', 'half track', 'hammer', 'hamper',
            'hand blower', 'hand-held computer', 'handkerchief', 'hard disc', 'harmonica',
            'harp', 'harvester', 'hatchet', 'holster', 'home theater', 'honeycomb', 'hook',
            'hoopskirt', 'horizontal bar', 'horse cart', 'hourglass', 'iPod', 'iron',
            "jack-o'-lantern", 'jean', 'jeep', 'jersey', 'jigsaw puzzle', 'jinrikisha',
            'joystick', 'kimono', 'knee pad', 'knot', 'lab coat', 'ladle', 'lampshade',
            'laptop', 'lawn mower', 'lens cap', 'letter opener', 'library', 'lifeboat',
            'lighter', 'limousine', 'liner', 'lipstick', 'Loafer', 'lotion', 'loudspeaker',
            'loupe', 'lumbermill', 'magnetic compass', 'mailbag', 'mailbox', 'maillot',
            'maillot', 'manhole cover', 'maraca', 'marimba', 'mask', 'matchstick', 'maypole',
            'maze', 'measuring cup', 'medicine chest', 'megalith', 'microphone', 'microwave',
            'military uniform', 'milk can', 'minibus', 'miniskirt', 'minivan', 'missile',
            'mitten', 'mixing bowl', 'mobile home', 'Model T', 'modem', 'monastery',
            'monitor', 'moped', 'mortar', 'mortarboard', 'mosque', 'mosquito net',
            'motor scooter', 'mountain bike', 'mountain tent', 'mouse', 'mousetrap',
            'moving van', 'muzzle', 'nail', 'neck brace', 'necklace', 'nipple', 'notebook',
            'obelisk', 'oboe', 'ocarina', 'odometer', 'oil filter', 'organ', 'oscilloscope',
            'overskirt', 'oxcart', 'oxygen mask', 'packet', 'paddle', 'paddlewheel',
            'padlock', 'paintbrush', 'pajama', 'palace', 'panpipe', 'paper towel',
            'parachute', 'parallel bars', 'park bench', 'parking meter', 'passenger car',
            'patio', 'pay-phone', 'pedestal', 'pencil box', 'pencil sharpener', 'perfume',
            'Petri dish', 'photocopier', 'pick', 'pickelhaube', 'picket fence', 'pickup',
            'pier', 'piggy bank', 'pill bottle', 'pillow', 'ping-pong ball', 'pinwheel',
            'pirate', 'pitcher', 'plane', 'planetarium', 'plastic bag', 'plate rack', 'plow',
            'plunger', 'Polaroid camera', 'pole', 'police van', 'poncho', 'pool table',
            'pop bottle', 'pot', "potter's wheel", 'power drill', 'prayer rug', 'printer',
            'prison', 'projectile', 'projector', 'puck', 'punching bag', 'purse', 'quill',
            'quilt', 'racer', 'racket', 'radiator', 'radio', 'radio telescope',
            'rain barrel', 'recreational vehicle', 'reel', 'reflex camera', 'refrigerator',
            'remote control', 'restaurant', 'revolver', 'rifle', 'rocking chair',
            'rotisserie', 'rubber eraser', 'rugby ball', 'rule', 'running shoe', 'safe',
            'safety pin', 'saltshaker', 'sandal', 'sarong', 'sax', 'scabbard', 'scale',
            'school bus', 'schooner', 'scoreboard', 'screen', 'screw', 'screwdriver',
            'seat belt', 'sewing machine', 'shield', 'shoe shop', 'shoji', 'shopping basket',
            'shopping cart', 'shovel', 'shower cap', 'shower curtain', 'ski', 'ski mask',
            'sleeping bag', 'slide rule', 'sliding door', 'slot', 'snorkel', 'snowmobile',
            'snowplow', 'soap dispenser', 'soccer ball', 'sock', 'solar dish', 'sombrero',
            'soup bowl', 'space bar', 'space heater', 'space shuttle', 'spatula',
            'speedboat', 'spider web', 'spindle', 'sports car', 'spotlight', 'stage',
            'steam locomotive', 'steel arch bridge', 'steel drum', 'stethoscope', 'stole',
            'stone wall', 'stopwatch', 'stove', 'strainer', 'streetcar', 'stretcher',
            'studio couch', 'stupa', 'submarine', 'suit', 'sundial', 'sunglass',
            'sunglasses', 'sunscreen', 'suspension bridge', 'swab', 'sweatshirt',
            'swimming trunks', 'swing', 'switch', 'syringe', 'table lamp', 'tank',
            'tape player', 'teapot', 'teddy', 'television', 'tennis ball', 'thatch',
            'theater curtain', 'thimble', 'thresher', 'throne', 'tile roof', 'toaster',
            'tobacco shop', 'toilet seat', 'torch', 'totem pole', 'tow truck', 'toyshop',
            'tractor', 'trailer truck', 'tray', 'trench coat', 'tricycle', 'trimaran',
            'tripod', 'triumphal arch', 'trolleybus', 'trombone', 'tub', 'turnstile',
            'typewriter keyboard', 'umbrella', 'unicycle', 'upright', 'vacuum', 'vase',
            'vault', 'velvet', 'vending machine', 'vestment', 'viaduct', 'violin',
            'volleyball', 'waffle iron', 'wall clock', 'wallet', 'wardrobe', 'warplane',
            'washbasin', 'washer', 'water bottle', 'water jug', 'water tower', 'whiskey jug',
            'whistle', 'wig', 'window screen', 'window shade', 'Windsor tie', 'wine bottle',
            'wing', 'wok', 'wooden spoon', 'wool', 'worm fence', 'wreck', 'yawl', 'yurt',
            'web site', 'comic book', 'crossword puzzle', 'street sign', 'traffic light',
            'book jacket', 'menu', 'plate', 'guacamole', 'consomme', 'hot pot', 'trifle',
            'ice cream', 'ice lolly', 'French loaf', 'bagel', 'pretzel', 'cheeseburger',
            'hotdog', 'mashed potato', 'head cabbage', 'broccoli', 'cauliflower', 'zucchini',
            'spaghetti squash', 'acorn squash', 'butternut squash', 'cucumber', 'artichoke',
            'bell pepper', 'cardoon', 'mushroom', 'Granny Smith', 'strawberry', 'orange',
            'lemon', 'fig', 'pineapple', 'banana', 'jackfruit', 'custard apple',
            'pomegranate', 'hay', 'carbonara', 'chocolate sauce', 'dough', 'meat loaf',
            'pizza', 'potpie', 'burrito', 'red wine', 'espresso', 'cup', 'eggnog', 'alp',
            'bubble', 'cliff', 'coral reef', 'geyser', 'lakeside', 'promontory', 'sandbar',
            'seashore', 'valley', 'volcano', 'ballplayer', 'groom', 'scuba diver',
            'rapeseed', 'daisy', "yellow lady's slipper", 'corn', 'acorn', 'hip', 'buckeye',
            'coral fungus', 'agaric', 'gyromitra', 'stinkhorn', 'earthstar',
            'hen-of-the-woods', 'bolete', 'ear', 'toilet tissue']