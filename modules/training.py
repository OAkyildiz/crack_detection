#!/usr/bin/env python3

K.set_floatx(ftype)

##RENAME, (LOAD?) put to  appropriate module.
# need it for retraining, comparing AND for the actual RUNS
#soo:
import nnetworks
#oooh wait, these are model relevant =>will go to nnetworks!
#def modelFromSaved() --> move to the module above
# training routine:
def train(model,data)
mdl= keras.models.load_model(filepath)
model = model_from_json(json_string)
#Input-> load data
#(so folders?)

#Preprocess?
#Keras handles that in the previous step-> update


# select  model
# compile model
model.compile(..)

# train model
model.fit_generator (...
# test model

#..model.fit() is that after you load?
# save json, weights arch
model.save_weights('first_try.h5')
model.save(file)


# save plot data/ plot  if u categorical_crossentropy
#use your results module

#return model? Do i want a copied instance?
