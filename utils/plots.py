import numpy as np

import random

import pandas as pd
from tqdm import tqdm_notebook as tqdm
from skimage import util 

import gc

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from skimage.transform import resize

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.metrics import accuracy_score

def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, linestyle, label=label, linewidth=2)

def plot_subfigure(X, Y, subplot, transform):
    if transform == "pca":
        X = PCA(n_components=2).fit_transform(X)
    elif transform == "cca":
        X = CCA(n_components=2).fit(X, Y).transform(X)
    else:
        raise ValueError

    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])

    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])

    classif = OneVsRestClassifier(LogisticRegression())
    classif.fit(X, Y)
    y_pred = classif.predict(X)
    
    print('{} + OneVsRestClassifier + LogisticRegression accuracy_score {}'.format(transform, accuracy_score(Y, y_pred)))

    plt.subplot(1, 2, subplot)    
    plt.scatter(X[:, 0], X[:, 1], s=15, c='gray', edgecolors=(0, 0, 0))
    
    for i in np.unique(Y.argmax(axis=1)):
        class_ = np.where(Y[:, i])
        plt.scatter(X[class_, 0], X[class_, 1], s=25,  linewidths=2, label='Class {}'.format(str(i)))
    
    for i in range(len(classif.estimators_)):
        plot_hyperplane(classif.estimators_[i], min_x, max_x, 'k--', 'Boundary\nfor class {}'.format(str(i)))

    plt.xticks(())
    plt.yticks(())

    plt.xlim(min_x - .1 * max_x, max_x + .1 * max_x)
    plt.ylim(min_y - .1 * max_y, max_y + .1 * max_y)
    
def plot_transform_hyperplanes(X, y, save=None):
    
    plt.figure(figsize=(27,9), dpi=100)

    plot_subfigure(X, y, 1, 'cca')
    plot_subfigure(X, y, 2, 'pca')

    
    plt.subplots_adjust(.04, .02, .97, .94, .09, .2)

    if save is not None:
        print('Saving Image {} ...'.format(save))
        plt.savefig(save)
        plt.close()
    else:
        plt.show()
    del X,y
    
    gc.collect()
        

from matplotlib import offsetbox
#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_dataset3d(X, y, save=None):
    print('data size {}'.format(X.shape))
    fig = plt.figure(figsize=(27, 18), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    uni_y = len(np.unique(y))
    for yi in tqdm(range(uni_y)):
        ax.scatter(X[:, 0][y == yi], X[:, 1][y == yi], X[:, 2][y == yi], color=plt.cm.Set1(yi / uni_y), marker='o')

    if save is not None:
        print('Saving Image {} ...'.format(save))
        plt.title('epoch ' + save.split('.')[0].split()[-1], fontdict={'fontsize': 20}, loc='left')
        plt.savefig(save)
        plt.close()
    else:
        plt.show()
    del X, y, fig, ax

    gc.collect()

def plot_dataset(X, y, images=None, labels=None, gray=False, save=None, y_original=None):
    print("Plotting Dataset")
    plt.cla()
    print('data size {}'.format(X.shape))
    uni_y = len(np.unique(y))
    
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    fig = plt.figure(figsize=(27,18), dpi=100)
    ax = plt.subplot(111)

    for i in tqdm(range(X.shape[0])):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / uni_y),
                 fontdict={'weight': 'bold', 'size': 9})
            

    if images is not None:
        if hasattr(offsetbox, 'AnnotationBbox'):
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(X.shape[0]):
                dist = np.sum((X[i] - shown_images) ** 2, 1)
                if np.min(dist) < 4e-3:
                    # don't show points that are too close
                    continue
                
                if labels is not None:
                    if y_original is not None:
                        plt.text(X[i, 0]-0.01, X[i, 1]-0.033, labels[y_original[i]], fontdict={'weight': 'bold', 'size': 15})
                    else:
                        plt.text(X[i, 0]-0.01, X[i, 1]-0.033, labels[y[i]], fontdict={'weight': 'bold', 'size': 15})
                
                shown_images = np.r_[shown_images, [X[i]]]
                if gray:
                    image_ =  offsetbox.OffsetImage(np.expand_dims(util.invert(images[i]), axis=0))
                else:
                    image_ =  offsetbox.OffsetImage(images[i], cmap=plt.cm.gray_r)
                
                imagebox = offsetbox.AnnotationBbox(image_ , X[i])
                
                ax.add_artist(imagebox)
        
    plt.xticks([]), plt.yticks([])
    
    for item in [fig, ax]:
        item.patch.set_visible(False)

    ax.axis('off')
        
    if save is not None:
        print('Saving Image {} ...'.format(save))
        plt.title('epoch '+ save.split('.')[0].split()[-1], fontdict={'fontsize': 20}, loc='left')
        plt.savefig(save)
        plt.close()
    else:    
        plt.show()  
    del X, y, fig, ax
 
    gc.collect()

def plot_samples(samples, scale=10, save=None):
    print(samples)
    print(type(samples))
    samples.clip(0,1);
    print("Plotting Samples...")
    im = merge(samples, (10,10))

    fig_width = int(im.shape[0] * scale)
    fig_height = int(im.shape[1] * scale)

    im = resize(im, (fig_width, fig_height), anti_aliasing=True)

    fig = plt.figure(dpi=150)
    ax = plt.subplot(111)
    plt.imshow(im)
    for item in [fig, ax]:
        item.patch.set_visible(False)
    plt.axis('off')

    if save is not None:
        print('Saving Image ', save)
        plt.title('epoch '+ save.split('.')[0].split()[-1], fontdict={'fontsize': 8}, loc='left')
        plt.savefig(save)
        plt.close()
    del im, samples, fig, ax
    gc.collect()
    
def pick_n(X, n):
    samples = list()
    for _ in range(n):
         samples.append(random.randint(0,len(X)-1))
    return X[samples]
    
def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    images = pick_n(images, 100)
    if (images.shape[3] in (3,4)):
        c = images.shape[-1:][0]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')    


def resize_gif(path, save_as=None, resize_to=None):
    """
    Resizes the GIF to a given length:

    Args:
        path: the path to the GIF file
        save_as (optional): Path of the resized gif. If not set, the original gif will be overwritten.
        resize_to (optional): new size of the gif. Format: (int, int). If not set, the original GIF will be resized to
                              half of its size.
    """
    all_frames = extract_and_resize_frames(path, resize_to)

    if not save_as:
        save_as = path

    if len(all_frames) == 1:
        print("Warning: only 1 frame found")
        all_frames[0].save(save_as, optimize=True)
    else:
        all_frames[0].save(save_as, optimize=True, save_all=True, append_images=all_frames[1:], loop=1000)


def analyseImage(path):
    """
    Pre-process pass over the image to determine the mode (full or additive).
    Necessary as assessing single frames isn't reliable. Need to know the mode
    before processing all frames.
    """
    im = Image.open(path)
    results = {
        'size': im.size,
        'mode': 'full',
    }
    try:
        while True:
            if im.tile:
                tile = im.tile[0]
                update_region = tile[1]
                update_region_dimensions = update_region[2:]
                if update_region_dimensions != im.size:
                    results['mode'] = 'partial'
                    break
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    return results


def extract_and_resize_frames(path, resize_to=None):
    """
    Iterate the GIF, extracting each frame and resizing them

    Returns:
        An array of all frames
    """
    mode = analyseImage(path)['mode']

    im = Image.open(path)

    if not resize_to:
        resize_to = (im.size[0] // 2, im.size[1] // 2)

    i = 0
    p = im.getpalette()
    last_frame = im.convert('RGBA')

    all_frames = []

    try:
        while True:
            # print("saving %s (%s) frame %d, %s %s" % (path, mode, i, im.size, im.tile))

            '''
            If the GIF uses local colour tables, each frame will have its own palette.
            If not, we need to apply the global palette to the new frame.
            '''
            if not im.getpalette():
                im.putpalette(p)

            new_frame = Image.new('RGBA', im.size)

            '''
            Is this file a "partial"-mode GIF where frames update a region of a different size to the entire image?
            If so, we need to construct the new frame by pasting it on top of the preceding frames.
            '''
            if mode == 'partial':
                new_frame.paste(last_frame)

            new_frame.paste(im, (0, 0), im.convert('RGBA'))

            new_frame.thumbnail(resize_to, Image.ANTIALIAS)
            all_frames.append(new_frame)

            i += 1
            last_frame = new_frame
            im.seek(im.tell() + 1)
    except EOFError:
        pass

    return all_frames