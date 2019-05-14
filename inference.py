import keras
import numpy as np
import librosa

# model_path = 'D:\\datas\\SON\\freesound-audio-tagging-2019\\best_1.h5'
model_path = 'D:\\datas\\SON\\OUAKAM_AVRIL_MAI_2018\\2018-05\\best_1.h5'

# sound_path = 'D:\\datas\\SON\\freesound-audio-tagging-2019\\train_curated\\00c40a6d.wav'
sound_path = 'D:\\datas\\SON\\OUAKAM_AVRIL_MAI_2018\\2018-04\\06115005.wav'

class Config(object):
    def __init__(self,
                 sampling_rate=16000, audio_duration=2,
                 n_classes=80,
                 use_mfcc=False, n_folds=10, learning_rate=0.0001,
                 max_epochs=50, n_mfcc=20):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        self.audio_length = self.sampling_rate * self.audio_duration
        if self.use_mfcc:
            self.dim = (self.n_mfcc, 1 + int(np.floor(self.audio_length/512)), 1)
        else:
            self.dim = (self.audio_length, 1)

#Should correspond to the config the model trained with
config = Config(sampling_rate=16000, audio_duration=2, n_folds=10, max_epochs=10)

def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+1e-6)
    return data -0.5

def data_prep(config,path):
    data,_ = librosa.core.load(path,sr=config.sampling_rate,
                    res_type='kaiser_fast')

    print('len data : ', len(data))
    print('audio length : ', config.audio_length)

    if len(data) > config.audio_length:
        max_offset=len(data)-config.audio_length
        offset = np.random.randint(max_offset)
        data = data[offset:(config.audio_length+offset)]
    else:
        if config.audio_length > len(data):
            max_offset = config.audio_length - len(data)
            offset = np.random.randint(max_offset)
        else:
            offset=0
        print('offset : ',offset )
        print('len data : ', len(data))
        print('audio length : ', config.audio_length)
        data = np.pad(data,(offset,config.audio_length - len(data) - offset), "constant")

    if config.use_mfcc:
        data = librosa.feature.mfcc(data,sr=config.sampling_rate,
                                            n_mfcc=config.n_mfcc)
        data = np.expand_dims(data,axis=1)
    else:
        data = audio_norm(data)[:,np.newaxis]

    return data

data = np.expand_dims(data_prep(config,sound_path),axis=1)
print('shape : ', data.shape)
data=data.reshape((1,32000,1))
print('shape : ', data.shape)

model = keras.models.load_model(model_path)
model.summary()
res =model.predict(data)
print (res)
