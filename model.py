from keras.models import Model, load_model
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.optimizers import SGD


def res50(lr=0.01, decay=1e-4, momentum=0.9, nb_classes=100, input_shape=(512,300,1)):
    base_model = ResNet50(weights=None, include_top=False, pooling=None, input_shape=input_shape, classes=100)
    x = base_model.output
    x = Flatten()(x)
   #x = Dense(1024, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    return model
