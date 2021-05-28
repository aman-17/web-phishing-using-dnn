from neural_netwrk import model
import numpy as np
import pandas as pd
import keras
input_data = pd.read_csv('phishing.csv')

labels=input_data['class']
labels = [(1,0) if i==1 else (0,1) for i in labels]
labels = np.array(labels)

#print(len(input_data.columns))

data = []
for i in range(len(labels)):
    row=[]
    for j in range(1,len(input_data.columns)-1):
        row.append(input_data[input_data.columns[j]][i])
    
    data.append(row)

data = np.array(data)
#X_train = data[:int(0.9*len(labels))]
#X_test = data[int(0.9*len(labels)):]

#Y_train = labels[:int(0.9*len(labels))]
#Y_test = labels[int(0.9*len(labels)):]
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(1e-2), metrics=['accuracy'])
callbacks = [
    keras.callbacks.ModelCheckpoint("./checkpoints/save_at_{epoch}.h5"),
]
model.fit(data, labels, validation_split=0.1, epochs=50, batch_size=1000,callbacks=callbacks)

model.save_weights('urls-lstm-weights.h5')
