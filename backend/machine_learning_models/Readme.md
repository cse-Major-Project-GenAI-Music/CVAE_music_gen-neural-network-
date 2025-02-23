# Music Generation with CVAE and CLSTM

This document provides an overview of two model architectures used for music generation: Conditional Variational Autoencoder (CVAE) and Contextual LSTM (CLSTM). You can view the detailed implementations and download the models from the following links:

- [Click to view 'Building CVAE'](https://www.kaggle.com/code/irohiwin/800-cvae-gen-music-1?scriptVersionId=222120120)
- [Click to view 'Metrics for CVAE'](https://www.kaggle.com/code/irohiwin/measure-uniqueness-of-generated-samples-cvae/)
- [Click to view 'CLSTM and its Metrics'](https://www.kaggle.com/code/irohiwin/600-clstm-decoder-gen-music-2)

---

## Model Architectures

### Conditional Variational Autoencoder (CVAE)

The CVAE model architecture is designed to generate music sequences by learning a compressed latent representation conditioned on specific inputs. The architecture uses convolutional and transposed convolutional layers to encode and decode the data efficiently.

#### Architecture Overview:
```
===================================================================================================================
Layer (type:depth-idx)                   Input Shape               Output Shape              Param #
===================================================================================================================
CVAE                                     [1, 4, 300, 128]          [1, 4, 300, 128]          --
├─Conv2d: 1-1                            [1, 4, 300, 128]          [1, 32, 150, 64]          1,184
├─ReLU: 1-2                              [1, 32, 150, 64]          [1, 32, 150, 64]          --
├─Conv2d: 1-3                            [1, 32, 150, 64]          [1, 64, 75, 32]           18,496
├─ReLU: 1-4                              [1, 64, 75, 32]           [1, 64, 75, 32]           --
├─Conv2d: 1-5                            [1, 64, 75, 32]           [1, 128, 38, 16]          73,856
├─ReLU: 1-6                              [1, 128, 38, 16]          [1, 128, 38, 16]          --
├─Linear: 1-7                            [1, 77828]                [1, 64]                   4,981,056
├─Linear: 1-8                            [1, 77828]                [1, 64]                   4,981,056
├─Linear: 1-9                            [1, 68]                   [1, 77824]                5,369,856
├─ConvTranspose2d: 1-10                  [1, 128, 38, 16]          [1, 64, 75, 32]           73,792
├─ReLU: 1-11                             [1, 64, 75, 32]           [1, 64, 75, 32]           --
├─ConvTranspose2d: 1-12                  [1, 64, 75, 32]           [1, 32, 150, 64]          18,464
├─ReLU: 1-13                             [1, 32, 150, 64]          [1, 32, 150, 64]          --
├─ConvTranspose2d: 1-14                  [1, 32, 150, 64]          [1, 4, 300, 128]          1,156
===================================================================================================================
Total params: 15,518,916
Trainable params: 15,518,916
Non-trainable params: 0
Total mult-adds (M): 514.74
===================================================================================================================
```

### Contextual LSTM (CLSTM)

The CLSTM model is designed to generate music sequences by leveraging the temporal dependencies in the data. It combines LSTM layers with transposed convolutional layers for efficient decoding.

#### Architecture Overview:
```
===================================================================================================================
Layer (type:depth-idx)                   Input Shape               Output Shape              Param #
===================================================================================================================
CLSTM_Decoder                             [1, 64]                   [1, 4, 200, 128]          --
├─LSTM: 1-1                              [1, 1, 68]                [1, 1, 64]                34,304
├─Linear: 1-2                            [1, 68]                   [1, 51200]                3,532,800
├─ConvTranspose2d: 1-3                   [1, 128, 25, 16]          [1, 64, 50, 32]           73,792
├─ReLU: 1-4                              [1, 64, 50, 32]           [1, 64, 50, 32]           --
├─ConvTranspose2d: 1-5                   [1, 64, 50, 32]           [1, 32, 100, 64]          18,464
├─ReLU: 1-6                              [1, 32, 100, 64]          [1, 32, 100, 64]          --
├─ConvTranspose2d: 1-7                   [1, 32, 100, 64]          [1, 4, 200, 128]          1,156
===================================================================================================================
Total params: 3,660,516
Trainable params: 3,660,516
Non-trainable params: 0
Total mult-adds (M): 269.40
===================================================================================================================
```

You can explore the model implementations and download the trained models from the links provided above.

