
import time
import glob
import os
from functions import *
import helpers.analysis as an
from skimage import io as skiio

class Bayes:
    def __init__(self):
        start = time.time()
        print('Starting Bayes')

        image_folder = r"." + os.sep + "baseDeDonneesImages"
        _path = glob.glob(image_folder + os.sep + r"*.jpg")
        #labellist = ['coast', 'coast_sun', 'forest', 'forest_for', 'forest_nat', 'street', 'street_urb', 'street_gre']
        labellist = ['coast', 'forest', 'street']

        pathlist = []
        for label in labellist:
            pathlist = pathlist + glob.glob(image_folder + os.sep + label + r"*.jpg")

        target = np.zeros(len(pathlist))
        for i, path in enumerate(pathlist):
            for j, label in enumerate(labellist):
                if label in path:
                    target[i] = j
        # Remove to process all data
        target = target
        pathlist = pathlist

        funclist = [avgBlue, avgRed, corner, nbedges, entropy]

        forest_t = []
        coast_t = []
        street_t = []
        for id, path in enumerate(pathlist):
            img = skiio.imread(path)
            data = np.zeros(len(funclist))
            for funcID, func in enumerate(funclist):
                data[funcID] = func(img)
            if "forest" in path:
                forest_t.append(data)
            elif "coast" in path:
                coast_t.append(data)
            elif "street" in path:
                street_t.append(data)
        print(f'Fetched components : {time.time() - start} seconds')

        minimal = min(len(forest_t), len(coast_t), len(street_t))
        forest = np.zeros((minimal, len(funclist)))
        for i in range(minimal):
            for val_id, val in enumerate(forest_t[i]):
                forest[i][val_id] = val
        coast = np.zeros((minimal, len(funclist)))
        for i in range(minimal):
            for val_id, val in enumerate(coast_t[i]):
                coast[i][val_id] = val
        street = np.zeros((minimal, len(funclist)))
        for i in range(minimal):
            for val_id, val in enumerate(street_t[i]):
                street[i][val_id] = val

        AllClass = np.array([forest, coast, street])
        _x, _y, _z = AllClass.shape
        # Chaque ligne de data contient 1 point en 2D
        # Les points des 3 classes sont mis ?? la suite en 1 seul long array
        data = AllClass.reshape(_x * _y, _z)
        ndata = len(data)
        # assignation des classes d'origine 0 ?? 2 pour C1 ?? C3 respectivement
        class_labels = np.zeros([ndata, 1])
        class_labels[range(len(forest), 2 * len(forest))] = 1
        class_labels[range(2 * len(forest), ndata)] = 2
        # Min et max des donn??es
        extent = an.Extent_5d(ptList=data)

        ndonnees = 5000
        donneesTest = an.genDonneesTest_5d(ndonnees, extent)

        full_Bayes_risk(AllClass, class_labels, donneesTest, 'bayes risque test', extent, data, class_labels)
        print("hop")

def full_Bayes_risk(train_data, train_classes, donnee_test, title, extent, test_data, test_classes):
    """
    Classificateur de Bayes complet pour des classes ??quiprobables (apriori ??gal)
    Selon le calcul direct du risque avec un mod??le gaussien
    Produit un graphique pertinent et calcule le taux d'erreur moyen

    train_data: donn??es qui servent ?? b??tir les mod??les
    train_classes: ??tiquettes de train_data
    test_data: donn??es ??tiquet??es dans "classes" ?? classer pour calculer le taux d'erreur
    test_classes: ??tiquettes de test_data
    donnee_test: donn??es al??atoires pour visualiser la fronti??re
    title: titre ?? utiliser pour la figure
    """

    # calcule p(x|Ci) pour toutes les donn??es ??tiquet??es
    # rappel (c.f. exercice pr??paratoire)
    # ici le risque pour la classe i est pris comme 1 - p(x|Ci) au lien de la somme du risque des autres classes
    prob_dens, prob_dens2 = compute_prob_dens_gaussian(train_data, donnee_test, test_data)
    # donc minimiser le risque revient ?? maximiser p(x|Ci)
    classified = np.argmax(prob_dens, axis=1).reshape(len(donnee_test), 1)
    classified2 = np.argmax(prob_dens2, axis=1).reshape(test_classes.shape)

    # calcule le taux de classification moyen
    error_class = 6  # optionnel, assignation d'une classe diff??rente ?? toutes les donn??es en erreur, aide pour la visualisation
    error_indexes = calc_erreur_classification(test_classes, classified2)
    classified2[error_indexes] = error_class
    print(
        f'Taux de classification moyen sur l\'ensemble des classes, {title}: {100 * (1 - len(error_indexes) / len(classified2))}%')

    train_data = np.array(train_data)
    x, y, z = train_data.shape
    train_data = train_data.reshape(x*y, z)
    #  view_classification_results(train_data, test1, c1, c2, glob_title, title1, title2, extent, test2=None, c3=None, title3=None)
    an.view_classification_results(train_data, donnee_test, train_classes, classified / error_class / .75,
                                   f'Classification de Bayes, {title}', 'Donn??es originales', 'Donn??es al??atoires',
                                   extent, test_data, classified2 / error_class / .75, 'Donn??es d\'origine reclass??es')

def calc_erreur_classification(original_data, classified_data):
    """
    Retourne le nombre d'??l??ments diff??rents entre deux vecteurs
    """
    # g??n??re le vecteur d'erreurs de classification
    vect_err = np.absolute(original_data - classified_data).astype(bool)
    indexes = np.array(np.where(vect_err == True))[0]
    print(f'\n\n{len(indexes)} erreurs de classification sur {len(original_data)} donn??es')
    # print(indexes)
    return indexes

def compute_prob_dens_gaussian(train_data, test_data1, test_data2):
    """
    Construit les mod??les gaussien de chaque classe (premi??re dimension de train_data)
    puis calcule la densit?? de probabilit?? de chaque point dans test_data par rapport ?? chaque classe

    retourne un tableau de la valeur de la densit?? de prob pour chaque point dans test_data1 et un autre pour
        test_data2 par rapport ?? chaque classe
    """
    train_data = np.array(train_data)
    x, y, z = train_data.shape

    # b??tit la liste de toutes les stats
    # i.e. un mod??le
    # donc ceci correspond ?? .fit dans la logique sklearn
    mean_list = []
    cov_list = []
    det_list = []
    inv_cov_list = []
    for i in range(x):
        mean, cov, pouet, pouet  = an.calcModeleGaussien(train_data[i])
        mean_list.append(mean)
        inv_cov = np.linalg.inv(cov)
        cov_list.append(cov)
        inv_cov_list.append(inv_cov)
        det = np.linalg.det(cov)
        det_list.append(det)

    # calcule les probabilit??s de chaque point des donn??es de test pour chaque classe
    # correspond ?? .predict dans la logique sklearn
    test_data1 = np.array(test_data1)
    t1, v1 = test_data1.shape
    test_data2 = np.array(test_data2)
    t2, v2 = test_data2.shape
    dens_prob1 = []
    dens_prob2 = []
    # calcule la valeur de la densit?? de probabilit?? pour chaque point de test

    for i in range(x):  # it??re sur toutes les classes
        # pour les points dans test_data1
        # TODO L2.E2.3 Compl??ter le calcul ici
        #mahalanobis1 = np.array([1 for j in range(t1)])
        temp1 = np.array([test_data1[j]-mean_list[i] for j in range(t1)])
        mahalanobis1 = np.array([(temp1[j]@inv_cov_list[i])@temp1[j].T for j in range(t1)])

        prob1 = 1 / np.sqrt(det_list[i] * (2 * np.pi) ** z) * np.exp(-mahalanobis1 / 2)
        dens_prob1.append(prob1)
        # pour les points dans test_data2
        #mahalanobis2 = np.array([1 for j in range(t2)])
        temp2 = np.array([test_data2[j]-mean_list[i] for j in range(t2)])
        mahalanobis2 = np.array([(temp2[j]@inv_cov_list[i])@temp2[j].T for j in range(t2)])
        prob2 = 1 / np.sqrt(det_list[i] * (2 * np.pi) ** z) * np.exp(-mahalanobis2 / 2)
        dens_prob2.append(prob2)

    return np.array(dens_prob1).T, np.array(dens_prob2).T  # reshape pour que les lignes soient les calculs pour 1 point original
