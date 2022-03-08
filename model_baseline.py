import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tr
import random
import types
import time
import os

class cancer_classifier(nn.Module):
    def __init__(self, definition):
        super(cancer_classifier, self).__init__()
        if isinstance(definition, list): # creating from scratch (training)
            self.definition = definition
            self.define()
        elif isinstance(definition, str): # loading trained model
            state = torch.load(definition)
            self.definition = state["definition"]
            self.define()
            self.cuda()
            self.load_state_dict(state["state_dict"])
            self.optim.load_state_dict(state["optim_state_dict"])
        else:
            import sys
            sys.exit("ERROR: Wrong model initialization")

    # defines the calling model (i.e. initializes its layers and creates
    # its optimizator)
    def define(self):
        self.cost = 0
        self.layers = []
        i = 0
        for layer_def in self.definition:
            self.cost += layer_def.cost
            layer = layer_def.initialize()
            self.add_module("layer" + str(i), layer.inner)
            self.layers.append(layer)
            i += 1
        self.optim = torch.optim.Adam(self.parameters())

    def forward(self, x):
        conv = True
        for layer in self.layers:
            if conv and not layer.is_conv: # first linear layer
                x = x.reshape([-1, layer.in_features])
                conv = False
            x = layer.inner(x)
        return x

    # stores the calling model's state_dict, definition (layers) and
    # optimizator in the given path as [name].pt
    def save(self, path: str, name: str):
        torch.save({"definition": self.definition,
                    "state_dict": self.state_dict(),
                    "optim_state_dict": self.optim.state_dict(),
                    "model": self
                    }, path + "/" + name + ".pt")

# Helper class used to store information about a cancer_classifier
class Classifier:
    def __init__(self, classifier: cancer_classifier, id: int, tmp_path: str):
        self.cost = classifier.cost
        self.loss = classifier.loss
        self.accuracy = classifier.accuracy
        self.id = "state" + str(id)
        classifier.save(tmp_path, self.id)
        self.definition = classifier.definition
        classifier.definition = None
        del classifier

    # destroys the underlyinig data of the classifier (state)
    def delete(self, path: str):
        os.remove(path + "/" + self.id + ".pt")

# cancer-classifier layer
class Layer:
    def __init__(self, inner, in_features = None):
        assert(inner is not None)
        self.inner = inner
        if in_features is None: # convolutional layer
            self.is_conv = True
        else: # linear layer
            self.is_conv = False
            self.in_features = in_features

# cancer-classifier layer-definition class
class LayerDefinition:
    # layer-initializer functions
    initializers = {"conv": nn.Conv2d, "maxpool": nn.MaxPool2d, "relu": nn.ReLU,
                    "batchnorm": nn.BatchNorm2d, "avgpool": nn.AvgPool2d}

    # returns the new shape of a tensor after applying a convolutional layer to it
    def new_shape(shape: int, padding: int, k_size: int, stride: int):
        return int(((shape + 2*padding - k_size) / stride) + 1)

    # returns the arguments required to create a batchnorm layer and
    # the output shape it produces
    def generate_batchnorm(shape: int, out_channels: int):
        return ({"num_features": out_channels}, shape)

    # allowed normalization layers
    norm = {"batchnorm": generate_batchnorm}

    # returns the arguments required to create a pooling layer and
    # the output shape it produces
    def generate_pool(old_shape: int, out_channels: int):
        # try a max of 10 times, otherwise do not create pool
        shape = old_shape
        for _ in range(10):
            k_size = random.randrange(2, 6) ########## Change max?
            stride = random.randrange(2, 6) ########## Change max?
            shape = LayerDefinition.new_shape(shape, 0, k_size, stride)
            if shape > 0:
                break
        if shape > 0:
            return ({"kernel_size": k_size, "stride": stride}, shape)
        else:
            return (None, old_shape)

    # allowed pooling layers
    pooling = {"maxpool": generate_pool, "avgpool": generate_pool}

    # Creates a Layer definition given that:
    # random_init: Bool: Initialize a random layer
    # prev_layer: Layer: Previous layer in NN (if any, otherwise None)
    # last_layer: Bool: Is NN's last layer
    # conv_layer: Bool: Create a convolutional layer
    def __init__(self, random_init: bool, prev_layer = None, last_layer = False, conv_layer = False):
        if random_init:
            if prev_layer is None or conv_layer: # convolutional layer
                assert(not last_layer)
                self.randomized_conv(prev_layer)
            else: # linear layer
                out_features = 2 if last_layer else random.randint(2, 70) ########## Change max?
                self.linear(prev_layer, out_features)

    # initializes the defined layer and returns it
    def initialize(self):
        if self.is_conv:
            layers = []
            for i in range(len(self.layers)):
                layer = self.layers[i]
                layers.append(LayerDefinition.initializers[layer](**self.args[layer]))
            return Layer(nn.Sequential(*layers))
        else: # linear layer
            return Layer(nn.Linear(**self.args), self.args["in_features"])

    # turns the calling layer definition into a randomized convolutional one
    def randomized_conv(self, prev_layer):
        k_size = random.randrange(3, 6, 2) # 3x3 or 5x5
        out_channels = random.randint(1, 70) ########## Change max?
        shape = 256 if prev_layer is None else prev_layer.shape
        # randomize normalization layer
        norm = None
        if bool(random.getrandbits(1)):
            # randomly choose a normalization layer and get its
            # initializator's arguments
            layer = random.choice(list(LayerDefinition.norm))
            (args, shape) = LayerDefinition.norm[layer](shape, out_channels)
            norm = (layer, args)
        # randomize pooling layer
        pooling = None
        if bool(random.getrandbits(1)):
            # randomly choose a pooling layer and get its
            # initializator's arguments
            layer = random.choice(list(LayerDefinition.pooling))
            (args, shape) = LayerDefinition.pooling[layer](shape, out_channels)
            if args is not None:
                pooling = (layer, args)
        self.conv(prev_layer, k_size, out_channels, norm, pooling, shape)

    # turns the calling layer into a convolutional one given its parameters
    def conv(self, prev_layer, k_size: int, out_channels: int, norm, pooling, shape: int):
        self.is_conv = True
        self.cost = out_channels * (shape ** 2) ########## Change?
        self.shape = shape
        self.args = {"relu": {}}
        self.layers = ["conv"]
        # Convolution layer
        padding = 1 if k_size == 3 else 2 ########## Change?
        if prev_layer is None: # First NN layer
            in_channels = 3
        else:
            assert(prev_layer.is_conv)
            in_channels = prev_layer.args["conv"]["out_channels"]
        self.args["conv"] = {"in_channels": in_channels, "out_channels": out_channels,
                             "kernel_size": k_size, "padding": padding}
        # Normalization layer
        if norm is not None:
            (layer, args) = norm
            self.layers.append(layer)
            self.args[layer] = args
        # Activation layer
        self.layers.append("relu")
        # Pooling layer
        if pooling is not None:
            (layer, args) = pooling
            self.layers.append(layer)
            self.args[layer] = args

    # turns the calling layer into a linear one given its parameters
    def linear(self, prev_layer, out_features: int):
        self.is_conv = False
        self.cost = out_features
        if prev_layer.is_conv:
            in_features = prev_layer.args["conv"]["out_channels"] * (prev_layer.shape ** 2)
        else:
            in_features = prev_layer.cost
        self.in_features = in_features
        self.args = {"in_features": in_features, "out_features": out_features}

    # tries to reproduce the layer that calls it while taking into consideration
    # the given previous layer (if any). last_layer is true if the calling
    # layer is the last one in the NN
    # Returns a 2-tuple containing:
    #   (The new layer, True) if sucessfully inherited
    #   (The previous layer, False) otherwise
    def try_inherit(self, prev_layer, last_layer: bool):
        if self.is_conv:
            assert(not last_layer)
            if prev_layer is None or prev_layer.is_conv:
                copy = LayerDefinition(False)
                k_size = self.args["conv"]["kernel_size"]
                out_channels = self.args["conv"]["out_channels"]
                (norm, pooling) = self.get_norm_pooling()
                copy.conv(prev_layer, k_size, out_channels, norm, pooling, self.shape)
            else:
                return (prev_layer, False)
        else:
            copy = LayerDefinition(False)
            out_features = 2 if last_layer else self.cost
            copy.linear(prev_layer, out_features)
        return (copy, True)

    # gets and returns the calling layer's normalization and pooling
    # layers, if any
    def get_norm_pooling(self):
        assert(self.is_conv)
        norm = None
        pooling = None
        for layer_type in self.layers:
            if layer_type in LayerDefinition.norm:
                norm = (layer_type, self.args[layer_type])
            elif layer_type in LayerDefinition.pooling:
                pooling = (layer_type, self.args[layer_type])
        return (norm, pooling)

    # mutates the calling layer definition
    def mutate(self, prev_layer, next_layer):
        if next_layer is None:
            # do not mutate last layer in NN (to maintain out shape)
            assert(not self.is_conv)
        elif self.is_conv:
            self.randomized_conv(prev_layer)
            # update next layer
            if next_layer.is_conv:
                k_size = next_layer.args["conv"]["kernel_size"]
                out_channels = next_layer.args["conv"]["out_channels"]
                (norm, pooling) = next_layer.get_norm_pooling()
                next_layer.conv(self, k_size, out_channels, norm, pooling, next_layer.shape)
            else:
                next_layer.linear(self, next_layer.cost)
        else:
            assert(not next_layer.is_conv)
            # randomize out_features
            out_features = random.randint(2, 70) ########## Change?
            self.args["out_features"] = out_features
            self.cost = out_features
            # update next layer
            next_layer.args["in_features"] = out_features

# cancer-classifier trainer class
class Trainer:
    def __init__(self, pop_size: int, fitness_eval: int, offsprings: float, k: float, mut: float,
                 max_depth: int, epochs: int, timeout: int,
                 train_loader: torch.utils.data.dataloader.DataLoader,
                 valid_loader: torch.utils.data.dataloader.DataLoader,
                 loss_criterion: nn.modules.loss.CrossEntropyLoss,
                 acc_criterion: types.FunctionType):
        assert(pop_size > 1)
        assert(fitness_eval >= pop_size)
        assert(offsprings > 0 and offsprings <= 1)
        assert(k > 0 and k <= 1)
        assert(mut >= 0 and mut <= 1)
        assert(max_depth > 1)
        assert(epochs > 0)
        assert(timeout > 0)

        self.pop_size = pop_size
        self.fitness_eval = fitness_eval
        self.offsprings = int(offsprings * self.pop_size)
        # at least one offspring per generation
        self.offsprings = self.offsprings if self.offsprings > 0 else 1
        self.k = int(k * self.pop_size)
        # at least two individuals used in tournament selection
        self.k = self.k if self.k > 1 else 2
        self.mut = mut
        self.max_depth = max_depth
        self.epochs = epochs
        self.timeout = timeout
        self.train_set = []
        transforms = [tr.GaussianBlur(5), tr.RandomVerticalFlip(), tr.RandomRotation(180),
                      tr.RandomPerspective(), tr.RandomHorizontalFlip(), tr.Grayscale(3),
                      tr.RandomAffine(0), tr.ColorJitter()]
        for _, (batch, label) in enumerate(train_loader):
            self.train_set.append((batch, label))
            # apply a random transformation to each batch
            transform = random.choice(transforms)
            self.train_set.append((transform(batch), label))
        self.valid_set = []
        for _, (batch, label) in enumerate(valid_loader):
            self.valid_set.append((batch, label))
            # apply a random transformation to each batch
            transform = random.choice(transforms)
            self.train_set.append((transform(batch), label))
        print("FINISHED LOADING DATA")
        self.loss_criterion = loss_criterion
        self.accuracy_criterion = acc_criterion
        self.tmp_path = os.getcwd() + "/temp" # temporary folder used to store models
        if not os.path.exists(self.tmp_path):
            os.mkdir(self.tmp_path)
        self.classifier_cnt = 0 # used to track created classifiers

    # given the arguments given to the trainer constructor, this function
    # tranis a population of classifiers and returns the best one once
    # the max amount of fitness evaluations have been performed or there
    # is a timaout
    def train(self):
        # Generate the initial population of classifiers
        self.population = []
        population = []
        self.start_time = time.time()
        for _ in range(self.pop_size):
            population.append([None, self.classifier_generator()])
            time_elapsed = (time.time() - self.start_time) / 60
            if time_elapsed >= self.timeout: # check timeout
                break
        self.evaluate(population)
        while len(population) != 0:
            self.insert_ordered(population.pop())
        if time_elapsed >= self.timeout:  # check timeout
            return self.finish()
        self.fitness_eval -= self.pop_size
        # Evolve until all fitness evaluations have been performed
        while self.fitness_eval > 0:
            self.fitness_eval -= self.next_generation()
        return self.finish()

    # uses the NSGA-II (fast non-dominated sort) algorithm to assign a fitness value to each
    # classifier based on its loss, accuracy and 'cost'
    def evaluate(self, population: list):
        fronts = [[]]
        dom_solutions = []
        dom_cnt = []
        for p in range(len(population)):
            dom_solutions.append([])
            dom_cnt.append(0)
            for q in range(len(population)):
                if q != p:
                    dom = self.dominates(population[p][1], population[q][1])
                    if dom > 0: # p dominates q
                        dom_solutions[p].append(q)
                    elif dom < 0: # q dominates p
                        dom_cnt[p] += 1
            population[p][0] = dom_cnt[p]
            if dom_cnt[p] == 0: # p belongs to the first front
                fronts[0].append(p)
        i = 0
        while len(fronts[i]) != 0:
            front = []
            for p in fronts[i]:
                for q in dom_solutions[p]:
                    dom_cnt[q] -= 1
                    if dom_cnt[q] == 0: # q belongs to next front
                        population[q][0] = i + 1
                        front.append(q)
            i += 1
            fronts.append(front)

    # compares two classifiers c1 and c2. Returns a number 'x' s.t.:
    #  x>0 if c1 dominates c2
    #  x<0 if c2 dominates c1
    #  x=0 otherwise
    def dominates(self, c1: Classifier, c2: Classifier):
        result = 0
        if c1.accuracy is None:
            if c2.accuracy is None:
                return 0
            else:
                return -1
        elif c2.accuracy is None:
            return 1
        if c1.loss < c2.loss:
            result += 1
        elif c1.loss > c2.loss:
            result -= 1
        if c1.accuracy > c2.accuracy:
            result += 2
        elif c1.accuracy < c2.accuracy:
            result -= 2
        if c1.cost < c2.cost:
            result += 1
        elif c1.cost > c2.cost:
            result -= 1
        return result

    # inserts a given solution (i.e. pair of fitness and a classifier)
    # in the population so that it remains ordered
    def insert_ordered(self, solution: tuple):
        (fitness, _) = solution
        idx = self.binary_search(fitness, 0, len(self.population) - 1)
        self.population.insert(idx, solution)

    # binary search tool used by insert_ordered()
    def binary_search(self, fitness: float, start: int, end: int):
        if start == end:
            return start if (self.population[start][0] > fitness) else (start + 1)
        elif start > end:
            return start
        mid = int((start + end) / 2)
        mid_fitness = self.population[mid][0]
        if mid_fitness < fitness:
            return self.binary_search(fitness, mid + 1, end)
        elif mid_fitness > fitness:
            return self.binary_search(fitness, start, mid - 1)
        return mid

    # Randomly generates a trained classifier
    def classifier_generator(self):
        print("Generating random classifier.")
        layer_cnt = random.randint(2, self.max_depth) # number of layers (at least two)
        conv_layer_cnt = int(random.random() * layer_cnt) # number of conv layers
        conv_layer_cnt = conv_layer_cnt if conv_layer_cnt > 0 else 1 # (at least one)
        # create layers
        definition = []
        layer_def = None
        for i in range(layer_cnt):
            last_layer = (i == layer_cnt - 1)
            conv_layer = (i < conv_layer_cnt)
            layer_def = LayerDefinition(True, layer_def, last_layer, conv_layer)
            definition.append(layer_def)
        # create and train classifier
        classifier = cancer_classifier(definition).cuda()
        classifier = Classifier(self.train_classifier(classifier), self.classifier_cnt, self.tmp_path)
        self.classifier_cnt += 1
        print("Done.")
        return classifier

    # produces the next generation and updates the population
    # returns the number of generated offsprings or, on timeout,
    # the remaining amount of fitness evaluations
    def next_generation(self):
        # set number of offsprings to create
        n = self.offsprings if self.offsprings < self.fitness_eval else self.fitness_eval
        # generate the offsprings and update the population
        population = self.population[:-n] # discard worst n parents
        for _ in range(n):
            population.append([None, self.offspring_generator()])
            time_elapsed = (time.time() - self.start_time) / 60
            if time_elapsed >= self.timeout: # check timeout
                break
        self.evaluate(population)
        # clean discarded classifiers
        for (_, classifier) in self.population[-n:]:
            classifier.delete(self.tmp_path)
        self.population = []
        while len(population) != 0:
            self.insert_ordered(population.pop())
        return self.fitness_eval if time_elapsed >= self.timeout else n  # check timeout

    # generates a trained offspring classifier out of the current population
    def offspring_generator(self):
        print("Generating offspring.")
        (parent1, parent2) = self.get_parents()
        # set the number of layers to inherit from each parent
        # (at least 1 for each one)
        ratio = random.random()
        p1_layers = int(ratio * len(parent1))
        p1_layers = p1_layers if p1_layers > 0 else 1
        p2_len = len(parent2)
        p2_layers = int((1-ratio) * p2_len)
        p2_layers = p2_layers if p2_layers > 0 else 1
        mutation_cnt = int(self.mut * (p1_layers + p2_layers))
        if mutation_cnt < 1 and self.mut > 0:
            # at least one mutation if non-zero probability
            mutation_cnt = 1
        definition = []
        # inherit from parent1
        layer_def = None
        for i in range(p1_layers):
            (layer_def, success) = parent1[i].try_inherit(layer_def, False)
            assert(success)
            definition.append(layer_def)
        # inherit from parent two
        layer_def = definition[-1]
        for i in range(p2_len - p2_layers, p2_len):
            last_layer = (i == p2_len - 1)
            (layer_def, success) = parent2[i].try_inherit(layer_def, last_layer)
            if success:
                definition.append(layer_def)
            else:
                mutation_cnt -= 1 # Not inherited layer considered mutation
        # perform the remaining mutations
        for _ in range(mutation_cnt):
            i = random.randrange(len(definition)) # choose random layer
            layer_def = definition[i]
            prev_layer = definition[i-1] if i > 0 else None
            next_layer = definition[i+1] if i < (len(definition) - 1) else None
            layer_def.mutate(prev_layer, next_layer)
        # create and train offspring
        classifier = cancer_classifier(definition).cuda()
        classifier = Classifier(self.train_classifier(classifier), self.classifier_cnt, self.tmp_path)
        self.classifier_cnt += 1
        print("Done.")
        return classifier

    # selects and returns 2 parents (their lists of LayerDefinitions) out of the current
    # population
    def get_parents(self):
        indexes = list(range(len(self.population))) # list of available parent indexes
        random.shuffle(indexes)
        # select the 2 -out of K- best parents
        if indexes[0] < indexes[1]:
            p1 = indexes[0]
            p2 = indexes[1]
        else:
            p1 = indexes[1]
            p2 = indexes[0]
        for idx in indexes[2 : self.k]:
            if idx < p1:
                p2 = p1
                p1 = idx
            elif idx < p2:
                p2 = idx
        return (self.population[p1][1].definition, self.population[p2][1].definition)

    # trains a classifier, sets its loss and accuracy and returns it
    def train_classifier(self, classifier: cancer_classifier):
        optim = classifier.optim
        error = False
        try:
            for epoch in range(self.epochs):
                valid_accuracy_list = []
                valid_total_loss = 0.0
                # TRAINING
                classifier.train()
                # load mini-batches and do training
                for (batch, label) in self.train_set:
                    batch = batch.cuda()
                    label = label.cuda()
                    prediction = classifier(batch)
                    loss = self.loss_criterion(prediction, label)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    del batch
                    del label
                # VALIDATION
                with torch.no_grad():
                    classifier.eval()
                    # load mini-batches and do validation
                    for (batch, label) in self.valid_set:
                        batch = batch.cuda()
                        label = label.cuda()
                        prediction = classifier(batch)
                        loss = self.loss_criterion(prediction, label)
                        valid_accuracy_list.append(self.accuracy_criterion(prediction, label))
                        valid_total_loss += loss.item()
                        del batch
                        del label
                    valid_total_accuracy = sum(valid_accuracy_list) / len(valid_accuracy_list)
        except (ZeroDivisionError, RuntimeError):
        #except ZeroDivisionError:
            error = True
            print("Error while training classifier.")
        if error:
            classifier.loss = None
            classifier.accuracy = None
        else:
            classifier.loss = round(valid_total_loss, 3)
            classifier.accuracy = round(valid_total_accuracy*100, 3)
            print("Classifier trained.")
        return classifier

    # finishes up and returns the best classifier found
    def finish(self):
        classifier = self.population[0][1] # wrapper
        best_classifier = cancer_classifier(self.tmp_path + "/" + classifier.id + ".pt").cuda()
        best_classifier.accuracy = classifier.accuracy
        # clean up
        for (_, classifier) in self.population:
            classifier.delete(self.tmp_path)
        return best_classifier
