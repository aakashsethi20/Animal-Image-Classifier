import os, sys, subprocess, glob

import numpy as np
import matplotlib.pyplot as plt
import caffe

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

# Setting the path for the models
models_path = './models/'

# checking if the model exists, else it will exit the program
# and ask the user to put the caffemodel at the appropriate location
if os.path.isfile(models_path + 'bvlc_reference_caffenet.caffemodel'):
    print('Caffemodel found!')
else:
    print('Caffemodel not at the correct place. Please download it and place it at ./models')
    sys.exit(1)

# setting the caffe to use CPU since my computer doesn't have
# a CUDA enabled GPU
caffe.set_mode_cpu()

# Setting the path to model definition and trained model weights
model_def = models_path + 'deploy.prototxt'                         # Deliverable B
model_weights = models_path + 'bvlc_reference_caffenet.caffemodel'  # Deliverable C

# Initializing caffe's Net class with the pretrained model and definition
net = caffe.Net(model_def, model_weights, caffe.TEST) # caffe.NET sets the application in Test mode

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load('./imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print ('mean-subtracted values:', list(zip('BGR', mu)))

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))      # move image channels to outermost dimension
transformer.set_mean('data', mu)                # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)          # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))   # swap channels from RGB to BGR

# Creating a list of images to be analyzed.
# We will use python's glob library for this task
# Some images are .jpeg and some are .jpg, so need to take care of both
# Assuming that all images are either .jpg or .jpeg. But it is simple to
# extend this functionality to other extensions too.
image_filenames = glob.glob('./images/*.jpeg') + glob.glob('./images/*.jpg')

# set the size of the input
net.blobs['data'].reshape(len(image_filenames),         # batch size same as the number of images we have
                          3,                            # 3-channel (BGR) images
                          227, 227)                     # image size is 227x227

# The final_<something> lists declared under will contain the input
# and the output in a nice format for us to display on the console at 
# the end of the program
final_input = []
final_output = []
final_possible_classification = []

# This loop will go through each image, process it
# and save the output in our lists declared above
for image_filename in sorted(image_filenames):
    image = caffe.io.load_image(image_filename)         # inputting the image to caffe
    transformed_image = transformer.preprocess('data', image)
    plt.imshow(image)           # This doesn't actually prints the image but simply just draws it

    # Adding image name to the final_input
    final_input.append(image_filename.split('/')[2])

    # copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = transformed_image

    # perform classification
    output = net.forward()

    output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

    # load ImageNet labels
    labels_file = './data/ilsvrc12/synset_words.txt'
    if not os.path.exists(labels_file):
        subprocess.call(['./data/ilsvrc12/get_ilsvrc_aux.sh'])
        
    labels = np.loadtxt(labels_file, str, delimiter='\t')

    # Adding the most probable classification to the appropriate list
    final_possible_classification.append(' '.join(labels[output_prob.argmax()].split()[1:])[:-1])

    # sort top five predictions from softmax output
    top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

    # Adding the whole processing result to the appropriate list to display later
    final_output.append(list(zip(output_prob[top_inds], labels[top_inds])))

# clear the terminal using the bash command
bash_command_clear = 'clear'
process = subprocess.call(bash_command_clear.split())

# This loop is to print out the whole analysis in a nice format
for result in zip(final_input, final_output, final_possible_classification):
    current_row = tuple(result)

    # Prints out the image name
    print ('Input: ' + str(current_row[0]))

    print ('Label\t\t\tProbability')
    for prob, label in current_row[1]:
        # This line might seem complicated but it's to diplay results nicely
        print(' '.join(label.split()[1:])[:-1].split(',')[0]+'\t\t\t'+str(prob))
    
    # Print the most probable classification
    print('Output: It is a ' + str(current_row[2])+ '\n\n')
