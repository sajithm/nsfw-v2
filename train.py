from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dense, Dropout, Flatten
#from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 128
EPOCH_COUNT = 50

MODEL_PATH = 'model.hdf5'
WEIGHTS_PATH = 'weights.hdf5'
IMAGE_DEPTH = 3
IMAGE_WIDTH = 192
IMAGE_HEIGHT = 192
IMAGE_SHAPE = (IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH)

if K.image_data_format() == 'channels_last':
    IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH)

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = IMAGE_SHAPE))
model.add(ZeroPadding2D(padding = (1, 1)))
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides=(2, 2)))
model.add(Dropout(rate = 0.25))

model.add(ZeroPadding2D(padding = (1, 1)))
model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
model.add(ZeroPadding2D(padding = (1, 1)))
model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides=(2, 2)))
model.add(Dropout(rate = 0.25))

model.add(ZeroPadding2D(padding = (1, 1)))
model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu'))
model.add(ZeroPadding2D(padding = (1, 1)))
model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides=(2, 2)))
model.add(Dropout(rate = 0.25))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(rate = 0.5))
model.add(Dense(5, activation = 'softmax'))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/train',
        target_size = (IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size = BATCH_SIZE,
        class_mode = 'categorical')
test_set = test_datagen.flow_from_directory(
        'dataset/test',
        target_size = (IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size = BATCH_SIZE,
        class_mode = 'categorical')

earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=3, verbose=1, mode='auto')
checkpoint = ModelCheckpoint(WEIGHTS_PATH, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
model.fit_generator(
        training_set,
        epochs = EPOCH_COUNT,
        validation_data = test_set,
        callbacks = [earlystop, checkpoint])
model.save(MODEL_PATH, True, True)
