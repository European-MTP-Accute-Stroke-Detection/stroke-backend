from werkzeug.utils import secure_filename
from pydicom import dcmread
import torch
from torch import nn
import pandas as pd  
import pydicom, numpy as np
import matplotlib.pylab as plt
import concurrent.futures

from random import randrange
from tqdm import tqdm
import platform
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import sklearn
import math
import copy
from scipy.ndimage import binary_erosion, binary_dilation
import multiprocessing

from scipy.ndimage import label
from skimage.measure import regionprops
from scipy.ndimage import binary_fill_holes

from keras.preprocessing import image
from numpy import asarray
import matplotlib.cm as cm
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries
import shap
from functools import partial
from sklearn.utils import check_random_state
from lime import lime_base

from scipy.ndimage.measurements import label
from scipy.spatial.distance import pdist, squareform

prefix = '/' if platform.system() == 'Windows' else ''


def remove_noise(binary_array, erosion_iterations=2, dilation_iterations=1):
    # Perform erosion to remove small foreground regions
    eroded_array = binary_erosion(binary_array, iterations=erosion_iterations)
    
    # Perform dilation to restore the remaining foreground regions
    dilated_array = binary_dilation(eroded_array, iterations=dilation_iterations)
    
    return dilated_array

def remove_small_regions(binary_array, min_size=100):
    # Label connected regions in the binary array
    labeled_array, num_labels = label(binary_array)
    
    # Get properties of each labeled region
    region_props = regionprops(labeled_array)
    
    # Create an empty array for the output
    filtered_array = np.zeros_like(binary_array, dtype=bool)
    
    # Iterate over the regions and filter based on size
    for region in region_props:
        if region.area >= min_size:
            filtered_array[labeled_array == region.label] = True
    
    return filtered_array

def create_segmentation_mask(img):

    mask = np.full((img.shape[0], img.shape[1]), False, dtype=bool)
    #removes background and bone
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j][0] != 0 and img[i][j][1] != 0:
                if img[i][j][0] != 1 and img[i][j][1] != 1:
                    mask[i][j] = True

    #removes noise
    mask = remove_noise(mask)
    #removes additional small noisy regions
    mask = remove_small_regions(mask, min_size=200)
    #removes small areas inside positive regions
    final_mask = binary_fill_holes(mask)

    return final_mask

def calculate_diameter(binary_array):
    # Label connected components in the binary array
    labeled_array, num_features = label(binary_array)
    
    # Find the labels of the connected components
    labels = np.unique(labeled_array)[1:]  # Exclude background label 0
    
    max_diameter = 0
    
    # Iterate through each label and calculate the diameter
    for label_val in labels:
        # Extract the connected component for the current label
        component = np.where(labeled_array == label_val, 1, 0)
        
        # Find the boundary points of the connected component
        boundary_points = np.argwhere(component == 1)
        
        # Calculate the pairwise distances between the boundary points
        pairwise_distances = pdist(boundary_points, metric='euclidean')
        
        # Find the maximum distance between any two boundary points
        if len(pairwise_distances) > 0:
            max_component_diameter = np.max(pairwise_distances)
            max_diameter = max(max_diameter, max_component_diameter)
    
    return max_diameter

def create_crucial_segmentation_mask(img):
    
    mask = create_segmentation_mask(img)

    final_mask = copy.deepcopy(mask)

    zero_coords = np.where(mask == False)

    _threshold = int(calculate_diameter(mask)*1/5)
    print(_threshold)

    for x, y in np.ndindex(mask.shape):
        # Calculate the distances to the pixels with value 0
        distances = np.sqrt((x - zero_coords[0])**2 + (y - zero_coords[1])**2)
        
        # Check if the minimum distance is greater than the threshold distance
        if np.min(distances) >= _threshold:
            # Change the value of the pixel
            final_mask[x, y] = False

    return final_mask



def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        print(preds)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def calculate_heatmap(cam, img, predict_id):
    img = (img*255)
    img = keras.preprocessing.image.img_to_array(img[0])
    
    #img = img[0]
    #img = keras.preprocessing.image.img_to_array(img[0])
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * cam)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    # Display Grad CAM
    #print(np.array(superimposed_img))#
    plt.figure(frameon=False)
    #imgplot = plt.imshow(superimposed_img)

    #plt.axis('off')
    #plt.savefig(prefix + 'static/' + predict_id + '/cam.png', bbox_inches = 'tight')
    #plt.close()
    
    return np.asarray(jet_heatmap)


def visualize_heatmap(heatmap, img, path):
    
    #print(heatmap.shape)
    heatmap = np.array(Image.fromarray(np.uint8(heatmap * 255) , 'L'))
    #print(heatmap.shape)
    plt.figure(frameon=False)
    plt.imshow(heatmap)
    plt.axis('off')
    #plt.savefig('cam.png')
    
    plt.show()
    plt.close()


def plotLIME(model, data, prediction, complexity, torch = False):
    if torch:
        explainer = LimeImageExplainer_torch() 
        m = nn.Softmax()
        prediction = m(prediction).cpu().detach().numpy()
        data_seg = data.detach().numpy()
        data_seg = np.transpose(data_seg, (1, 2, 0))
    else:
        explainer = LimeImageExplainer()        
        data_seg = data

    segmenter = SegmentationAlgorithm('slic', n_segments=16, compactness=10, sigma=2,
                     start_label=1, mask = create_crucial_segmentation_mask(data_seg))
    explanation_1 = explainer.explain_instance(data, 
                                         classifier_fn = model, 
                                         top_labels=5, 
                                         num_features = 64,
                                         batch_size =128,
                                         hide_color=0, # 0 - gray 
                                         num_samples=complexity,
                                         segmentation_fn=segmenter
                                        )
    temp, mask = explanation_1.get_image_and_mask(explanation_1.top_labels[0], 
                                                positive_only=True,
                                                negative_only=False,
                                                num_features=2, 
                                                hide_rest=False)

    lime_res = mark_boundaries(temp, mask, mode = "thick", color = (255,0,0))
    
    #imgplot = plt.imshow(lime_res)
    
    #plt.axis('off')
    #plt.savefig(prefix + 'static/' + predict_id + '/'+name+'.png', bbox_inches = 'tight')
    #plt.close()
    return mask

class ImageExplanation(object):
    def __init__(self, image, segments):
        """Init function.
        Args:
            image: 3d numpy array
            segments: 2d numpy array, with the output from skimage.segmentation
        """
        if torch.is_tensor(image):
            self.image = np.moveaxis(image.numpy(), 0, -1)
        else:
            self.image = image
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = {}
        self.score = {}

    def get_image_and_mask(self, label, positive_only=True, negative_only=False, hide_rest=False,
                           num_features=5, min_weight=0.):
        """Init function.
        Args:
            label: label to explain
            positive_only: if True, only take superpixels that positively contribute to
                the prediction of the label.
            negative_only: if True, only take superpixels that negatively contribute to
                the prediction of the label. If false, and so is positive_only, then both
                negativey and positively contributions will be taken.
                Both can't be True at the same time
            hide_rest: if True, make the non-explanation part of the return
                image gray
            num_features: number of superpixels to include in explanation
            min_weight: minimum weight of the superpixels to include in explanation
        Returns:
            (image, mask), where image is a 3d numpy array and mask is a 2d
            numpy array that can be used with
            skimage.segmentation.mark_boundaries
        """
        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        if positive_only & negative_only:
            raise ValueError("Positive_only and negative_only cannot be true at the same time.")
        segments = self.segments
        image = self.image
        exp = self.local_exp[label]
        mask = np.zeros(segments.shape, segments.dtype)
        
        if hide_rest:
            temp = np.zeros(self.image.shape)
        else:
            temp = self.image.copy()
        if positive_only:
            fs = [x[0] for x in exp
                  if x[1] > 0 and x[1] > min_weight][:num_features]
        if negative_only:
            fs = [x[0] for x in exp
                  if x[1] < 0 and abs(x[1]) > min_weight][:num_features]
        if positive_only or negative_only:
            for f in fs:
                temp[segments == f] = image[segments == f].copy()
                mask[segments == f] = 1
            return temp, mask
        else:
            for f, w in exp[:num_features]:
                if np.abs(w) < min_weight:
                    continue
                c = 0 if w < 0 else 1
                mask[segments == f] = -1 if w < 0 else 1
                temp[segments == f] = image[segments == f].copy()
                temp[segments == f, c] = np.max(image)
            return temp, mask
        
class LimeImageExplainer(object):
    """Explains predictions on Image (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self, kernel_width=.25, kernel=None, verbose=False,
                 feature_selection='auto', random_state=None):
        """Init function.
        Args:
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel_fn, verbose, random_state=self.random_state)

    def explain_instance(self, image, classifier_fn, labels=(1,),
                         hide_color=None,
                         top_labels=5, num_features=100000, num_samples=1000,
                         batch_size=10,
                         segmentation_fn=None,
                         distance_metric='cosine',
                         model_regressor=None,
                         random_seed=None,
                         progress_bar=True):
        """Generates explanations for a prediction.
        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).
        Args:
            image: 3 dimension RGB image. If this is only two dimensional,
                we will assume it's a grayscale image and call gray2rgb.
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            hide_color: If not None, will hide superpixels with this color.
                Otherwise, use the mean pixel color of the image.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            batch_size: batch size for model predictions
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
            segmentation_fn: SegmentationAlgorithm, wrapped skimage
            segmentation function
            random_seed: integer used as random seed for the segmentation
                algorithm. If None, a random integer, between 0 and 1000,
                will be generated using the internal random number generator.
            progress_bar: if True, show tqdm progress bar.
        Returns:
            An ImageExplanation object (see lime_image.py) with the corresponding
            explanations.
        """
        if len(image.shape) == 2:
            image = gray2rgb(image)
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        if segmentation_fn is None:
            segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                    max_dist=200, ratio=0.2,
                                                    random_seed=random_seed)
        segments = segmentation_fn(image)

        fudged_image = image.copy()
        if hide_color is None:
            for x in np.unique(segments):
                fudged_image[segments == x] = (
                    np.mean(image[segments == x][:, 0]),
                    np.mean(image[segments == x][:, 1]),
                    np.mean(image[segments == x][:, 2]))
        else:
            fudged_image[:] = hide_color

        top = labels

        data, labels = self.data_labels(image, fudged_image, segments,
                                        classifier_fn, num_samples,
                                        batch_size=batch_size,
                                        progress_bar=progress_bar)

        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        ret_exp = ImageExplanation(image, segments)
        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()
        for label in top:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                data, labels, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp

    def data_labels(self,
                    image,
                    fudged_image,
                    segments,
                    classifier_fn,
                    num_samples,
                    batch_size=10,
                    progress_bar=True):
        """Generates images and predictions in the neighborhood of this image.
        Args:
            image: 3d numpy array, the image
            fudged_image: 3d numpy array, image to replace original image when
                superpixel is turned off
            segments: segmentation of the image
            classifier_fn: function that takes a list of images and returns a
                matrix of prediction probabilities
            num_samples: size of the neighborhood to learn the linear model
            batch_size: classifier_fn will be called on batches of this size.
            progress_bar: if True, show tqdm progress bar.
        Returns:
            A tuple (data, labels), where:
                data: dense num_samples * num_superpixels
                labels: prediction probabilities matrix
        """
        n_features = np.unique(segments).shape[0]
        data = self.random_state.randint(0, 2, num_samples * n_features)\
            .reshape((num_samples, n_features))
        labels = []
        data[0, :] = 1
        imgs = []
        rows = tqdm(data) if progress_bar else data
        for row in rows:
            temp = copy.deepcopy(image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            #print(mask.shape)
            #print(image.shape)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = fudged_image[mask]
            imgs.append(temp)
            if len(imgs) == batch_size:
                preds = classifier_fn(np.array(imgs))
                labels.extend(preds)
                imgs = []
        if len(imgs) > 0:
            preds = classifier_fn(np.array(imgs))
            labels.extend(preds)
        return data, np.array(labels)

class LimeImageExplainer_torch(object):
    """Explains predictions on Image (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self, kernel_width=.25, kernel=None, verbose=False,
                 feature_selection='auto', random_state=None):
        """Init function.
        Args:
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel_fn, verbose, random_state=self.random_state)

    def explain_instance(self, image, classifier_fn, labels=(1,),
                         hide_color=None,
                         top_labels=5, num_features=100000, num_samples=1000,
                         batch_size=10,
                         segmentation_fn=None,
                         distance_metric='cosine',
                         model_regressor=None,
                         random_seed=None,
                         progress_bar=True):
        """Generates explanations for a prediction.
        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).
        Args:
            image: 3 dimension RGB image. If this is only two dimensional,
                we will assume it's a grayscale image and call gray2rgb.
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            hide_color: If not None, will hide superpixels with this color.
                Otherwise, use the mean pixel color of the image.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            batch_size: batch size for model predictions
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
            segmentation_fn: SegmentationAlgorithm, wrapped skimage
            segmentation function
            random_seed: integer used as random seed for the segmentation
                algorithm. If None, a random integer, between 0 and 1000,
                will be generated using the internal random number generator.
            progress_bar: if True, show tqdm progress bar.
        Returns:
            An ImageExplanation object (see lime_image.py) with the corresponding
            explanations.
        """
        if len(image.shape) == 2:
            image = gray2rgb(image)
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        if segmentation_fn is None:
            segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                    max_dist=200, ratio=0.2,
                                                    random_seed=random_seed)
    
        segments = segmentation_fn(np.moveaxis(image.cpu().detach().numpy(), 0, -1))
        

        fudged_image = image.numpy().copy()
        fudged_image = np.transpose(fudged_image, (1, 2, 0))
        if hide_color is None:
            for x in np.unique(segments):
                fudged_image[segments == x] = (
                    np.mean(image[segments == x][:, 0]),
                    np.mean(image[segments == x][:, 1]),
                    np.mean(image[segments == x][:, 2]))
        else:
            fudged_image[:] = hide_color
            

        fudged_image = np.moveaxis(fudged_image, 0, -1)

        top = labels

        data, labels = self.data_labels(image, fudged_image, segments,
                                        classifier_fn, num_samples,
                                        batch_size=batch_size,
                                        progress_bar=progress_bar)
        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        ret_exp = ImageExplanation(image, segments)

        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()

        for label in top:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                data, labels, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
            
        return ret_exp

    def data_labels(self,
                    image,
                    fudged_image,
                    segments,
                    classifier_fn,
                    num_samples,
                    batch_size=10,
                    progress_bar=True):
        """Generates images and predictions in the neighborhood of this image.
        Args:
            image: 3d numpy array, the image
            fudged_image: 3d numpy array, image to replace original image when
                superpixel is turned off
            segments: segmentation of the image
            classifier_fn: function that takes a list of images and returns a
                matrix of prediction probabilities
            num_samples: size of the neighborhood to learn the linear model
            batch_size: classifier_fn will be called on batches of this size.
            progress_bar: if True, show tqdm progress bar.
        Returns:
            A tuple (data, labels), where:
                data: dense num_samples * num_superpixels
                labels: prediction probabilities matrix
        """
        def predict(imgs):
            m = nn.Softmax()
            preds = m(classifier_fn(torch.squeeze(torch.from_numpy(np.array(imgs)))))
            return preds

        n_features = np.unique(segments).shape[0]
        data = self.random_state.randint(0, 2, num_samples * n_features)\
            .reshape((num_samples, n_features))
        labels = []
        data[0, :] = 1
        imgs = []
        rows = tqdm(data) if progress_bar else data
        for row in rows:
            temp = copy.deepcopy(np.moveaxis(image.cpu().detach().numpy(), 0, -1))
            fudged_image_ = np.transpose(copy.deepcopy(fudged_image), (0,2,1))
            #temp_ = copy.deepcopy(image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                if z != 0:
                    mask[segments == z] = True
            #if num % 2 == 0:
            #    temp[mask] = temp[mask]*[0.2, 0.8, 0.8]#*(fudged_image_[mask].25)
            #else:
            #    temp[mask] = temp[mask]*[1.1, 1.1, 1.1]#*(fudged_image_[mask].25)
            temp[mask] = temp[mask] - ((1/(1+np.exp(-temp[mask]+fudged_image_[mask])))-0.5)*2


            imgs.append(np.transpose(np.array(temp),(2,1,0)))
            if len(imgs) == batch_size:
                num_processes = multiprocessing.cpu_count()
                chunk_size = len(imgs) // num_processes
                chunks = [imgs[i:i+chunk_size] for i in range(0, len(imgs), chunk_size)]

                # Create a thread pool executor
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_processes)

                # Submit the predict function to the executor for each chunk
                results = [executor.submit(predict, chunk) for chunk in chunks]

                # Retrieve the results from the executor
                predictions = []
                for future in concurrent.futures.as_completed(results):
                    prediction = future.result()
                    predictions.extend(prediction)

                labels.extend(predictions)
                imgs = []
        if len(imgs) > 0:
            num_processes = multiprocessing.cpu_count()
            chunk_size = len(imgs) // num_processes
            chunks = [imgs[i:i+chunk_size] for i in range(0, len(imgs), chunk_size)]
                
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_processes)

                # Submit the predict function to the executor for each chunk
            results = [executor.submit(predict, chunk) for chunk in chunks]

                # Retrieve the results from the executor
            predictions = []
            for future in concurrent.futures.as_completed(results):
                prediction = future.result()
                predictions.extend(prediction)
            labels.extend(predictions)
            
        final_labels = []
        m = nn.Softmax()
        for torch_label in labels:
            final_labels.append(m(torch_label).cpu().detach().numpy())
        
        #print(final_labels)
        return data, np.array(final_labels)

def get_SHAP_values(explainer, data):
    shap_values = explainer.shap_values(data)

    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(data.numpy(), 1, -1), 1, 2)

    result = shap.image_plot(shap_numpy[0], -test_numpy)

    return result


def get_XAI_info(xai_id):
    xai_id_method, xai_id_complexity = xai_id.split('_')

    if xai_id_complexity == "low":
        xai_id_complexity = 100
    elif xai_id_complexity == "medium":
        xai_id_complexity = 500
    else:
        xai_id_complexity = 1000

    return xai_id_method, xai_id_complexity




######### Uncertainty Quantification

def calculate_uncertainty(model, data, num_samples):
    """
    The calculate_uncertainty method is designed to assess the uncertainty of the provided data using a 
    specified pytorch machine learning model. The parameter num_samples denotes the number of iterations 
    the model will be run to compute the uncertainty measure. We apply the monte carlo dropout.

    Parameters:

    - data: The input data for which uncertainty needs to be evaluated. This data should 
    be compatible with the model's input format.
    - num_samples: The number of iterations or samples to be used in the uncertainty estimation 
    process. A higher value of num_samples typically leads to more accurate uncertainty assessments 
    but may also increase computation time.
    - model: The pre-trained machine learning model used to make predictions on the given data. 
    This model should have dropout layers for a correct calculation.
    
    Return Value:

    The method returns a measure of uncertainty for each data point in the input, 
    providing valuable insights into the reliability of the model's predictions. 
    Higher uncertainty values indicate less confidence in the predictions, while 
    lower values suggest more reliable predictions.   
    """
    model_uncertainty = copy.deepcopy(model)
    predictions = []

    for _ in tqdm(range(num_samples)):
        model_uncertainty.apply(lambda module: setattr(module, 'training', True))
        # Forward pass
        m = nn.Softmax()
        output = m(model_uncertainty(data))
        predictions.append(output)

    # Convert predictions to a tensor
    predictions = torch.stack(predictions)

    # Calculate uncertainty by standard deviation
    uncertainty = torch.std(predictions, dim=0, unbiased=False)
    uncertainty_numpy = uncertainty.detach().numpy()

    print(uncertainty_numpy)

    # Uncertainty tensor will have the same shape as the model's output
    return uncertainty_numpy