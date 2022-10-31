"""
Script de départ de la problématique
Problématique APP2 Module IA S8
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import numpy as np
import random

from ImageCollection import ImageCollection
import NN
import Bayes
import PPV


#######################################
def main():
    # Génère une liste de N images, les visualise et affiche leur histo de couleur
    # TODO: voir L1.E3 et problématique
    #N = 5
    #im_list = np.sort(random.sample(range(np.size(ImageCollection.image_list, 0)), N))
    #print(im_list)
    # ImageCollection.view_scatter()
    # data extract
    NN.NN()
    #Bayes.Bayes()
    #PPV.PPV(1, 20)
    #ImageCollection.covariance()
    # ImageCollection.images_display(im_list)
    # ImageCollection.view_histogrammes(im_list)
    plt.show()


######################################
if __name__ == '__main__':
    main()