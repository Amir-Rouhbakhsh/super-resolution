# https://youtu.be/20af-_AQCBM
"""
@author: Sreenivas Bhattiprolu

convolutional, encoder, decoder blocks to build autoencoder and unet models

- Encoder would be the same for both autoencoder and Unet.
- Decoder path would be different for U-Net as we need to add concatenation. 


"""
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras.optimizers import Adam
from keras.layers import Activation, MaxPool2D, Concatenate
from tensorflow.keras import layers


'''
#Convolutional block to be used in autoencoder and U-Net
def conv_block(x, num_filters):
    x = layers.Conv2D(num_filters, (3, 3), padding='same', activation='relu')(x)
    
    x = BatchNormalization()(x)   #Not in the original network. 
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)  #Not in the original network
    x = Activation("relu")(x)

    return x

'''

def res_conv_block(x, size, dropout=0.0, batch_norm=True):
    '''
    Residual convolutional layer.
    Two variants....
    Either put activation function before the addition with shortcut
    or after the addition (which would be as proposed in the original resNet).
    
    1. conv - BN - Activation - conv - BN - Activation 
                                          - shortcut  - BN - shortcut+BN
                                          
    2. conv - BN - Activation - conv - BN   
                                     - shortcut  - BN - shortcut+BN - Activation                                     
    
    Check fig 4 in https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf

    '''
    filter_size=3

    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation('relu')(conv)
    
    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    #conv = layers.Activation('relu')(conv)    #Activation before addition with shortcut
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    shortcut = layers.Conv2D(size, kernel_size=(1, 1), padding='same')(x)
    if batch_norm is True:
        shortcut = layers.BatchNormalization(axis=3)(shortcut)

    res_path = layers.add([shortcut, conv])
    res_path = layers.Activation('relu')(res_path)    #Activation after addition with shortcut (Original residual block)
    return res_path




from tensorflow.keras import layers, models, Input

# Define encoder block
def encoder_block(x, filters):
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

# Define decoder block
def decoder_block(x, filters):
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    return x

# Updated conv_block function
def conv_block(x, num_filters):
    x = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)  # Add batch normalization
    x = Activation("relu")(x)
    
    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)  # Add batch normalization
    x = Activation("relu")(x)

    return x

# Encoder function
def build_encoder(input_tensor):
    # Use the input_tensor passed to the function
    x = layers.Conv2D(64, (1, 1), padding='same')(input_tensor)  # Expand channels to 64

    s1, p1 = encoder_block(x, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    
    encoded = conv_block(p4, 1024)  # Bridge
    
    return encoded

# Decoder function for Autoencoder ONLY
def build_decoder(encoded):
    d1 = decoder_block(encoded, 512)
    d2 = decoder_block(d1, 256)
    d3 = decoder_block(d2, 128)
    d4 = decoder_block(d3, 64)
    
    decoded = layers.Conv2D(3, (3, 3), padding="same", activation="sigmoid")(d4)
    return decoded

# Autoencoder model
def build_autoencoder(input_shape):
    input_img = Input(shape=input_shape)
    encoded = build_encoder(input_img)  # Pass the input tensor to the encoder
    decoded = build_decoder(encoded)  # Pass encoder output to decoder
    autoencoder = models.Model(inputs=input_img, outputs=decoded)  # Define the model
    return autoencoder

# model=build_autoencoder((256, 256, 3))
# print(model.summary())





'''

#Encoder will be the same for Autoencoder and U-net
#We are getting both conv output and maxpool output for convenience.
#we will ignore conv output for Autoencoder. It acts as skip connections for U-Net
def build_encoder(input_image):
    #inputs = Input(input_shape)

    s1, p1 = encoder_block(input_image, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    
    encoded = conv_block(p4, 1024) #Bridge
    
    return encoded

#Decoder for Autoencoder ONLY. 
def build_decoder(encoded):
    d1 = decoder_block(encoded, 512)
    d2 = decoder_block(d1, 256)
    d3 = decoder_block(d2, 128)
    d4 = decoder_block(d3, 64)
    
    decoded = Conv2D(3, 3, padding="same", activation="tanh")(d4)
    return decoded

#Use encoder and decoder blocks to build the autoencoder. 
def build_autoencoder(input_shape):
    input_img = Input(shape=input_shape)
    autoencoder = Model(input_img, build_decoder(build_encoder(input_img)))
    return(autoencoder)


'''


#Decoder block for unet
#skip features gets input from encoder for concatenation
def decoder_block_for_unet(x, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(x)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def decoder_block_for_resunet(x, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(x)
    x = Concatenate()([x, skip_features])
    x = res_conv_block(x, num_filters)
    return x


#Build Unet using the blocks
def build_unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024) #Bridge

    d1 = decoder_block_for_unet(b1, s4, 512)
    d2 = decoder_block_for_unet(d1, s3, 256)
    d3 = decoder_block_for_unet(d2, s2, 128)
    d4 = decoder_block_for_unet(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)  #Binary (can be multiclass)

    model = Model(inputs, outputs, name="U-Net")
    print(model.summary())
    return model

#Build Unet using the blocks
def build_resunet(input_shape):
    #inputs = Input(input_shape)
    #input_img = layers.Input(shape=(64, 64, 3))  # Input shape (64, 64, 3)
    inputs = layers.Input(shape=input_shape)
    # Step 1: Expand channels to 64 using a 1x1 convolution
    x = layers.Conv2D(64, (1, 1), padding='same')(inputs)

    s1, p1 = encoder_block(x, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024) #Bridge

    d1 = decoder_block_for_resunet(b1, s4, 512)
    d2 = decoder_block_for_resunet(d1, s3, 256)
    d3 = decoder_block_for_resunet(d2, s2, 128)
    d4 = decoder_block_for_resunet(d3, s1, 64)

    outputs = Conv2D(3, 1, padding="same", activation="sigmoid")(d4)  #Binary (can be multiclass)

    model = Model(inputs, outputs, name="U-Net")
    print(model.summary())
    return model


















