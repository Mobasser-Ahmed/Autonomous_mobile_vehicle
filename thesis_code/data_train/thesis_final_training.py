from matplotlib import pyplot as plt
import os
from sklearn.model_selection import train_test_split
import thesis_final_utils as utils
import tensorflow as tf

print('Setting UP')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)




#### STEP 1 - INITIALIZE DATA
# path = 'DataCollectedTest1'
path = 'BlackTrackPortable2'
data = utils.importDataInfo(path)
print(data.head())

#### STEP 2 - VISUALIZE AND BALANCE DATA
data = utils.balanceData(data, display=True)

#### STEP 3 - PREPARE FOR PROCESSING
imagesPath, steerings = utils.loadData(path, data)
# print('No of Path Created for Images ',len(imagesPath),len(steerings))
# cv2.imshow('Test Image',cv2.imread(imagesPath[5]))
# cv2.waitKey(0)

#### STEP 4 - SPLIT FOR TRAINING AND VALIDATION
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings,
                                              test_size=0.3,random_state=10)

print('Total Training Images: ',len(xTrain))
print('Total Validation Images: ',len(xVal))

#### STEP 5 - AUGMENT DATA

#### STEP 6 - PREPROCESS

#### STEP 7 - CREATE MODEL
model = utils.createModel()

#### STEP 8 - TRAINNING
# history = model.fit(utils.dataGen(xTrain, yTrain, 32, 1),
#                     steps_per_epoch=200,
#                     epochs=30,
#                     validation_data=utils.dataGen(xVal, yVal, 16, 0),
#                     validation_steps=16,
#                     verbose=1,
#                     shuffle=1,
#                     )
# history = model.fit(utils.dataGen(xTrain, yTrain, 128, 1),
#                     steps_per_epoch=200,
#                     epochs=25,
#                     validation_data=utils.dataGen(xVal, yVal, 128, 0),
#                     validation_steps=200,
#                     verbose=1,
#                     shuffle=1,
#                     )
# feels like batch of 100 with 300 steps_per_epoch worked with 30 epoch works
model.summary()
#####performs well########
# history = model.fit_generator(utils.dataGen(xTrain, yTrain, 100, 1),
#                     steps_per_epoch=300,
#                     epochs=30,
#                     validation_data=utils.dataGen(xVal, yVal, 100, 0),
#                     validation_steps=200,
#                     verbose=1,
#                     shuffle=1,
#                     )
#####performs well########
history = model.fit_generator(utils.dataGen(xTrain, yTrain, 100, 1),
                    steps_per_epoch=300,
                    epochs=430,
                    validation_data=utils.dataGen(xVal, yVal, 100, 0),
                    validation_steps=200,
                    verbose=1,
                    shuffle=1,
                    )


#### STEP 9 - SAVE THE MODEL
model.save('original_2.h5')
print('Model Saved')

#### STEP 10 - PLOT THE RESULTS
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()























