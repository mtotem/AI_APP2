"""
Classe "ImageCollection" statique pour charger et visualiser les images de la problématique
Membres statiques:
    image_folder: le sous-répertoire d'où les images sont chargées
    image_list: une énumération de tous les fichiers .jpg dans le répertoire ci-dessus
    images: une matrice de toutes les images, (optionnelle, décommenter le code)
    all_images_loaded: un flag qui indique si la matrice ci-dessus contient les images ou non
Méthodes statiques: TODO JB move to helpers
    images_display: affiche quelques images identifiées en argument
    view_histogrammes: affiche les histogrammes de couleur de qq images identifiées en argument
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from skimage import color as skic
from skimage import io as skiio
import helpers.analysis as an
import helpers.classifiers as classifiers
from functions import *
import functions
import matplotlib.colors as mcolors

class ImageCollection:
    """
    Classe globale pour regrouper les infos utiles et les méthodes de la collection d'images
    """
    # os.listdir()
    # liste de toutes les images
    image_folder = r"." + os.sep + "baseDeDonneesImages"

    _path = glob.glob(image_folder + os.sep + r"*.jpg")

    #Get a list of all labels
    labellist=[]
    for path in _path:
        name=path.split('\\')[-1].split('.jpg')[0]
        label = ''.join(i for i in name if not i.isdigit())
        labellist.append(label)
    #All labels
    labellist = list(dict.fromkeys(labellist))
    #Main labels
    labellist=['coast','forest','street']
    #Certain labels
    labellist=['coast','coast_sun','forest','forest_for','forest_nat','street','street_urb','street_gre']

    pathlist=dict()
    for label in labellist:
        pathlist[label]=(glob.glob(image_folder + os.sep + label + r"*.jpg"))
    if len(labellist)==8:
        pathlist['coast']=[x for x in pathlist['coast'] if 'coast_sun' not in x]
        pathlist['forest']=[x for x in pathlist['forest'] if 'forest_for' not in x]
        pathlist['forest']=[x for x in pathlist['forest'] if 'forest_nat' not in x]
        pathlist['street']=[x for x in pathlist['street'] if 'street_urb' not in x]
        pathlist['street']=[x for x in pathlist['street'] if 'street_gre' not in x]

    if len(labellist)==3:
        colors = ['red', 'green','blue']
    if len(labellist)==8:
        colors=['red','pink','green','limegreen','darkgreen','blue','cornflowerblue','navy']
        # colors=list(mcolors.CSS4_COLORS[0:len(labellist)])

    image_list = os.listdir(image_folder)
    # Filtrer pour garder juste les images
    image_list = [i for i in image_list if '.jpg' in i]
    all_images_loaded = False
    images = []

    # # Créer un array qui contient toutes les images
    # # Dimensions [980, 256, 256, 3]
    # #            [Nombre image, hauteur, largeur, RGB]
    # # TODO décommenter si voulu pour charger TOUTES les images
    # images = np.array([np.array(skiio.imread(image)) for image in _path])
    # all_images_loaded = True

    fig1, ax1 = (None,None)
    handles = []

    def view_scatter():
        """
        Creates scatter plots for different component combinations
        """
        funclist=[avgBlue, avgGreen, avgRed, avgY, avgcb, avgcr, frequencyPeakBlueRGB, frequencyPeakGreenRGB,
                  frequencyPeakRedRGB, frequencyPeakY, frequencyPeakcb, frequencyPeakcr, lowerLeftAvgBlue, lowerLeftAvgGreen,
                  lowerLeftAvgRed, lowerLeftHFBlue, lowerLeftHFGreen, lowerLeftHFRed, lowerLeftHistBlue, lowerLeftHistGreen,
                  lowerLeftHistRed, lowerRightAvgBlue, lowerRightAvgGreen, lowerRightAvgRed, lowerRightHFBlue, lowerRightHFGreen,
                  lowerRightHFRed, lowerRightHistBlue, lowerRightHistGreen, lowerRightHistRed,
                  mean, meanYcbcr, std, stdYcbcr, upperLeftAvgBlue, upperLeftAvgGreen, upperLeftAvgRed, upperLeftHFBlue,
                  upperLeftHFGreen, upperLeftHFRed, upperLeftHistBlue, upperLeftHistGreen, upperLeftHistRed, upperRightAvgBlue,
                  upperRightAvgGreen, upperRightAvgRed, upperRightHFBlue, upperRightHFGreen, upperRightHFRed, upperRightHistBlue,
                  upperRightHistGreen, upperRightHistRed]
        func_list1=funclist[0:int(len(funclist)/2)]
        func_list2=funclist[int(len(funclist)/2):len(funclist)-1]

        func_list1=[nbedges]
        func_list2=[corner]

        for i in range(len(func_list1)):
            func1=func_list1[i]
            func2=func_list2[i]

            ImageCollection.fig1, ImageCollection.ax1 = plt.subplots(1, 1)
            colors = ImageCollection.colors
            ImageCollection.handles=[]

            # allcomponents=np.zeros(len(ImageCollection.pathlist.keys()))
            allcomponents=[]
            for keyid, key in enumerate(ImageCollection.pathlist.keys()):
                imgs=np.array([np.array(skiio.imread(image)) for image in ImageCollection.pathlist[key]])
                component=np.zeros((imgs.shape[0], 2))
                for id, img in enumerate(imgs):
                    component[id][0] = func1(img)
                    component[id][1] = func2(img)
                ImageCollection.view_classes(component,colors[keyid])
                allcomponents.append(component)

            extent=dict()
            extent['xmin']=np.min([np.min(elem[:,0]) for elem in allcomponents])
            extent['ymin']=np.min([np.min(elem[:,1]) for elem in allcomponents])
            extent['xmax']=np.max([np.max(elem[:,0]) for elem in allcomponents])
            extent['ymax']=np.max([np.max(elem[:,1]) for elem in allcomponents])

            ImageCollection.ax1.set_xlim([0,300])
            ImageCollection.ax1.set_ylim([0,300])
            title = f"{func2.__name__} en fonction de {func1.__name__}"
            axes=[func1.__name__,func2.__name__]
            ImageCollection.graphinfo(title=title,axes=axes,labels=ImageCollection.pathlist.keys(),extent=extent)


    def view_classes(data,color):
        """
        Affichage des classes dans data
        *** Fonctionne pour des classes 2D

        data: tableau des classes à afficher. La première dimension devrait être égale au nombre de classes.
        extent: bornes du graphique
        border_coeffs: coefficient des frontières, format des données voir helpers.classifiers.get_borders()
            coef order: [x**2, xy, y**2, x, y, cst (cote droit log de l'equation de risque), cst (dans les distances de mahalanobis)]
        """
        dims = np.asarray(data).shape

        colorpoints = color
        colorfeatures = color

        # Plot les points
        ImageCollection.handles.append(ImageCollection.ax1.scatter(data[:, 0], data[:, 1], s=5, c=colorpoints))
        if data.shape[0] > 1:
            an.viewEllipse(data, ImageCollection.ax1, edgecolor=colorfeatures)

        #Plot les moyennes
        if data.shape[0]>1:
            m, cov, valpr, vectprop = an.calcModeleGaussien(data)
            ImageCollection.ax1.scatter(m[0], m[1], c=colorfeatures, s=60)
    def graphinfo(title,axes,labels,extent):
        ImageCollection.ax1.legend(ImageCollection.handles,
                   labels,
                   scatterpoints=1,
                   loc="lower left",
                   title="Classes",
                   ncol=1,
                   fontsize=8)
        ImageCollection.ax1.set_title(title)
        ImageCollection.ax1.set_xlim([extent['xmin'], extent['xmax']])
        ImageCollection.ax1.set_ylim([extent['ymin'], extent['ymax']])

        ImageCollection.ax1.set_xlabel(axes[0])
        ImageCollection.ax1.set_ylabel(axes[1])

    def covariance():
        functions = [mean, std, avgRed, avgGreen, avgBlue, avgY, avgcb, avgcr, frequencyPeakBlueRGB,
                     frequencyPeakRedRGB, frequencyPeakGreenRGB, frequencyPeakY, frequencyPeakcb,
                     frequencyPeakcr, maxPeakRed, maxPeakBlue, maxPeakGreen, upperRightAvgBlue,
                     upperRightAvgGreen, upperRightAvgRed, upperRightHFBlue, upperRightHFGreen, upperRightHFRed,
                     upperRightHistGreen, upperRightHistGreen, upperRightHistBlue, upperLeftAvgRed, upperLeftAvgGreen,
                     upperLeftAvgBlue, upperLeftHFGreen, upperLeftHFBlue, upperLeftHFRed, upperLeftHistBlue,
                     upperLeftHistGreen, upperLeftHistRed, lowerRightAvgBlue, lowerRightAvgGreen, lowerRightAvgRed,
                     lowerRightHFGreen, lowerRightHFRed, lowerRightHFBlue, lowerRightHistBlue, lowerRightHistGreen,
                     lowerRightHistRed, lowerLeftAvgGreen, lowerLeftAvgBlue, lowerLeftAvgRed, lowerLeftHFGreen,
                     lowerLeftHFBlue, lowerLeftHFRed, lowerLeftHistBlue, lowerLeftHistGreen, lowerLeftHistRed]
        functions_text = ["mean, std, avgRed, avgGreen, avgBlue, avgY, avgcb, avgcr, frequencyPeakBlueRGB,"
                     "frequencyPeakRedRGB, frequencyPeakGreenRGB, frequencyPeakY, frequencyPeakcb,"
                     "frequencyPeakcr, maxPeakRed, maxPeakBlue, maxPeakGreen, upperRightAvgBlue,"
                     "upperRightAvgGreen, upperRightAvgRed, upperRightHFBlue, upperRightHFGreen, upperRightHFRed,"
                     "upperRightHistGreen, upperRightHistGreen, upperRightHistBlue, upperLeftAvgRed, upperLeftAvgGreen,"
                     "upperLeftAvgBlue, upperLeftHFGreen, upperLeftHFBlue, upperLeftHFRed, upperLeftHistBlue,"
                     "upperLeftHistGreen, upperLeftHistRed, lowerRightAvgBlue, lowerRightAvgGreen, lowerRightAvgRed,"
                     "lowerRightHFGreen, lowerRightHFRed, lowerRightHFBlue, lowerRightHistBlue, lowerRightHistGreen,"
                     "lowerRightHistRed, lowerLeftAvgGreen, lowerLeftAvgBlue, lowerLeftAvgRed, lowerLeftHFGreen,"
                     "lowerLeftHFBlue, lowerLeftHFRed, lowerLeftHistBlue, lowerLeftHistGreen, lowerLeftHistRed"]
        coast_img = np.array([np.array(skiio.imread(image)) for image in ImageCollection._pathCoast])
        forest_img = np.array([np.array(skiio.imread(image)) for image in ImageCollection._pathForest])
        street_img = np.array([np.array(skiio.imread(image)) for image in ImageCollection._pathStreet])
        street_component = np.zeros((street_img.shape[0]))
        forest_component = np.zeros((forest_img.shape[0]))
        coast_component = np.zeros((coast_img.shape[0]))
        values_s = np.empty(street_img.shape[0])
        values_f = np.empty(forest_img.shape[0])
        values_c = np.empty(coast_img.shape[0])
        for func in functions:
            for id, img in enumerate(street_img):
                street_component[id] = func(img)
            for id, img in enumerate(street_img):
                forest_component[id] = func(img)
            for id, img in enumerate(street_img):
                coast_component[id] = func(img)
            values_s = np.vstack([values_s, street_component])
            values_f = np.vstack([values_f, forest_component])
            values_c = np.vstack([values_c, coast_component])
        mat_cov_coast = np.cov(values_c)
        mat_cov_forest = np.cov(values_f)
        mat_cov_street = np.cov(values_s)
        mat_p_coast = np.zeros_like(mat_cov_coast)
        mat_p_forest = np.zeros_like(mat_cov_forest)
        mat_p_street = np.zeros_like(mat_cov_street)
        for i in range(mat_cov_street.shape[0]):
            for j in range(mat_cov_street.shape[1]):
                mat_p_coast[i][j] = 1 - (mat_cov_coast[i][j] / (np.sqrt(mat_cov_coast[i][i]) * (np.sqrt(mat_cov_coast[j][j]))))
                mat_p_forest[i][j] = 1 - (mat_cov_forest[i][j] / (np.sqrt(mat_cov_forest[i][i]) * (np.sqrt(mat_cov_forest[j][j]))))
                mat_p_street[i][j] = 1 - (mat_cov_street[i][j] / (np.sqrt(mat_cov_street[i][i]) * (np.sqrt(mat_cov_street[j][j]))))
        print(functions_text)
        ax1 = plt.subplot(3, 1, 1)
        ax1.matshow(mat_p_coast)
        ax1.title.set_text('1-p coast correlation')
        ax2 = plt.subplot(3, 1, 2)
        ax2.matshow(mat_p_forest)
        ax2.title.set_text('1-p forest correlation')
        ax3 = plt.subplot(3, 1, 3)
        ax3.matshow(mat_p_street)
        ax3.title.set_text('1-p street correlation')

    def images_display(indexes):
        """
        fonction pour afficher les images correspondant aux indices
        indexes: indices de la liste d'image (int ou list of int)
        """

        # Pour qu'on puisse traiter 1 seule image
        if type(indexes) == int:
            indexes = [indexes]

        fig2 = plt.figure()
        ax2 = fig2.subplots(len(indexes), 1)

        for i in range(len(indexes)):
            if ImageCollection.all_images_loaded:
                im = ImageCollection.images[i]
            else:
                im = skiio.imread(ImageCollection.image_folder + os.sep + ImageCollection.image_list[indexes[i]])
            ax2[i].imshow(im)

    def view_histogrammes(indexes):
        """
        Affiche les histogrammes de couleur de quelques images
        indexes: int or list of int des images à afficher
        """

        # helper function pour rescaler le format lab
        def rescaleHistLab(LabImage, n_bins):
            """
            Helper function
            La représentation Lab requiert un rescaling avant d'histogrammer parce que ce sont des floats!
            """
            # Constantes de la représentation Lab
            class LabCte:      # TODO JB : utiliser an.Extent?
                min_L: int = 0
                max_L: int = 100
                min_ab: int = -110
                max_ab: int = 110
            # Création d'une image vide
            imageLabRescale = np.zeros(LabImage.shape)
            # Quantification de L en n_bins niveaux
            imageLabRescale[:, :, 0] = np.round(
                (LabImage[:, :, 0] - LabCte.min_L) * (n_bins - 1) / (
                        LabCte.max_L - LabCte.min_L))  # L has all values between 0 and 100
            # Quantification de a et b en n_bins niveaux
            imageLabRescale[:, :, 1:3] = np.round(
                (LabImage[:, :, 1:3] - LabCte.min_ab) * (n_bins - 1) / (
                        LabCte.max_ab - LabCte.min_ab))  # a and b have all values between -110 and 110
            return imageLabRescale


        ###########################################
        # view_histogrammes starts here
        ###########################################
        # TODO JB split calculs et view en 2 fonctions séparées
        # Pour qu'on puisse traiter 1 seule image
        if type(indexes) == int:
            indexes = [indexes]

        fig = plt.figure()
        ax = fig.subplots(len(indexes), 3)

        for num_images in range(len(indexes)):
            # charge une image si nécessaire
            if ImageCollection.all_images_loaded:
                imageRGB = ImageCollection.images[num_images]
            else:
                imageRGB = skiio.imread(
                    ImageCollection.image_folder + os.sep + ImageCollection.image_list[indexes[num_images]])

            # Exemple de conversion de format pour Lab et HSV
            imageLab = skic.rgb2lab(imageRGB)  # TODO L1.E3.5: afficher ces nouveaux histogrammes
            imageHSV = skic.rgb2hsv(imageRGB)  # TODO problématique: essayer d'autres espaces de couleur

            # Number of bins per color channel pour les histogrammes (et donc la quantification de niveau autres formats)
            n_bins = 256

            # Lab et HSV requiert un rescaling avant d'histogrammer parce que ce sont des floats au départ!
            imageLabhist = rescaleHistLab(imageLab, n_bins) # External rescale pour Lab
            imageHSVhist = np.round(imageHSV * (n_bins - 1))  # HSV has all values between 0 and 100

            # Construction des histogrammes
            # 1 histogram per color channel
            pixel_valuesRGB = np.zeros((3, n_bins))
            pixel_valuesLab = np.zeros((3, n_bins))
            pixel_valuesHSV = np.zeros((3, n_bins))
            for i in range(n_bins):
                for j in range(3):
                    pixel_valuesRGB[j, i] = np.count_nonzero(imageRGB[:, :, j] == i)
                    pixel_valuesLab[j, i] = np.count_nonzero(imageLabhist[:, :, j] == i)
                    pixel_valuesHSV[j, i] = np.count_nonzero(imageHSVhist[:, :, j] == i)

            # permet d'omettre les bins très sombres et très saturées aux bouts des histogrammes
            skip = 5
            start = skip
            end = n_bins - skip

            # affichage des histogrammes
            ax[num_images, 0].plot(range(start, end), pixel_valuesRGB[0, start:end], c='red')
            ax[num_images, 0].plot(range(start, end), pixel_valuesRGB[1, start:end], c='green')
            ax[num_images, 0].plot(range(start, end), pixel_valuesRGB[2, start:end], c='blue')
            ax[num_images, 0].set(xlabel='pixels', ylabel='compte par valeur d\'intensité')
            # ajouter le titre de la photo observée dans le titre de l'histogramme
            image_name = ImageCollection.image_list[indexes[num_images]]
            ax[num_images, 0].set_title(f'histogramme RGB de {image_name}')

            # 2e histogramme
            # TODO L1.E3 afficher les autres histogrammes de Lab ou HSV dans la 2e colonne de subplots
            ax[num_images, 1].plot(range(start, end), pixel_valuesHSV[0, start:end], c='red')
            ax[num_images, 1].plot(range(start, end), pixel_valuesHSV[1, start:end], c='green')
            ax[num_images, 1].plot(range(start, end), pixel_valuesHSV[2, start:end], c='blue')
            ax[num_images, 1].set(xlabel='pixels', ylabel='compte par valeur d\'intensité')
            # ajouter le titre de la photo observée dans le titre de l'histogramme
            image_name = ImageCollection.image_list[indexes[num_images]]
            ax[num_images, 1].set_title(f'histogramme HSV de {image_name}')

            # 3e histogramme
            # TODO L1.E3 afficher les autres histogrammes de Lab ou HSV dans la 2e colonne de subplots
            ax[num_images, 2].plot(range(start, end), pixel_valuesLab[0, start:end], c='red')
            ax[num_images, 2].plot(range(start, end), pixel_valuesLab[1, start:end], c='green')
            ax[num_images, 2].plot(range(start, end), pixel_valuesLab[2, start:end], c='blue')
            ax[num_images, 2].set(xlabel='pixels', ylabel='compte par valeur d\'intensité')
            # ajouter le titre de la photo observée dans le titre de l'histogramme
            image_name = ImageCollection.image_list[indexes[num_images]]
            ax[num_images, 2].set_title(f'histogramme LAB de {image_name}')