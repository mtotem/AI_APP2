
import time
import glob
import os
from functions import *
import helpers.analysis as an
from skimage import io as skiio
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.cluster import KMeans as km

class PPV:
    def __init__(self, n_nei, n_km):
        start = time.time()
        print('Starting PPV')

        image_folder = r"." + os.sep + "baseDeDonneesImages"
        _path = glob.glob(image_folder + os.sep + r"*.jpg")
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

        funclist = [avgBlue, avgRed, corner, nbedges]#, entropy, avgBlue, avgRed,
        # funclist=[avgBlue,avgRed, corner, nbedges, entropy] => 76 %
        # funclist = [avgBlue, avgRed, corner, nbedges] => 87.9 %

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
        extent = an.Extent_4d(ptList=data)

        ndonnees = 5000
        donneesTest = an.genDonneesTest_4d(ndonnees, extent)# _5d or _4d

        cluster_centers, cluster_labels = full_kmean(n_km, AllClass, class_labels, f'Repr??sentants des {n_nei}-moy', extent)
        full_ppv(n_nei, cluster_centers, cluster_labels, donneesTest, 'ppv test', extent, data, class_labels)
        print("hop")

def full_kmean(n_clusters, train_data, train_classes, title, extent):
    """
    Ex??cute l'algorithme des n_clusters-moyennes sur les donn??es de train_data ??tiquet??es dans train_classes
    Produit un graphique des repr??sentants de classes r??sultants
    Retourne les repr??sentants de classe obtenus et leur ??tiquette respective
    """
    cluster_centers, cluster_labels = kmean_alg(n_clusters, train_data)

    train_data = np.array(train_data)
    x, y, z = train_data.shape
    train_data = train_data.reshape(x * y, z)

    an.view_classification_results(train_data, cluster_centers, train_classes, cluster_labels, title, 'Donn??es d\'origine',
                                   f'Clustering de {n_clusters}-Means', extent)

    return cluster_centers, cluster_labels

def kmean_alg(n_clusters, data):
    """
    Calcule n_clusters repr??sentants de classe pour les classes contenues dans data (premi??re dimension)
    Retourne la suite des repr??sentants de classes et leur ??tiquette
    Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    """
    data = np.array(data)
    x, y, z = data.shape

    cluster_centers = []
    cluster_labels = np.zeros((n_clusters * x, 1))
    # calcule les repr??sentants pour chaque classe s??par??ment
    for i in range(x):
        # TODO L2.E3.3 compl??ter la logique pour utiliser la librairie ici
        kmeans_C = km(n_clusters=n_clusters,random_state=0)
        kmeans_C.fit(np.array(data[i]))
        cluster_centers.append(kmeans_C.cluster_centers_)
        cluster_labels[range(n_clusters * i, n_clusters * (i + 1))] = i  # assigne la classe en ordre ordinal croissant

    if n_clusters == 1:  # g??re les d??sagr??ments Python
        cluster_centers = np.array(cluster_centers)[:, 0]
    else:
        cluster_centers = np.array(cluster_centers)
        x, y, z = cluster_centers.shape
        cluster_centers = cluster_centers.reshape(x * y, z)
    return cluster_centers, cluster_labels

def full_ppv(n_neighbors, train_data, train_classes, datatest1, title, extent, datatest2=None, classestest2=None):
    """
    Classificateur PPV complet
    Utilise les donn??es de train_data ??tiquet??es dans train_classes pour cr??er un classificateur n_neighbors-PPV
    Trie les donn??es de test1 (non ??tiquet??es), datatest2 (optionnel, ??tiquet??es dans "classestest2"
    Calcule le taux d'erreur moyen pour test2 le cas ??ch??ant
    Produit un graphique des r??sultats pour test1 et test2 le cas ??ch??ant
    """
    predictions, predictions2 = ppv_classify(n_neighbors, train_data, train_classes.ravel(), datatest1, datatest2)
    predictions = predictions.reshape(len(datatest1), 1)

    error_class = 6  # optionnel, assignation d'une classe diff??rente ?? toutes les donn??es en erreur, aide pour la visualisation
    if np.asarray(datatest2).any():
        predictions2 = predictions2.reshape(len(datatest2), 1)
        # calcul des points en erreur ?? l'??chelle du syst??me

        error_indexes = calc_erreur_classification(classestest2, predictions2.reshape(classestest2.shape))
        predictions2[error_indexes] = error_class
        print(f'Taux de classification moyen sur l\'ensemble des classes, {title}: {100 * (1 - len(error_indexes) / len(classestest2))}%')

    an.view_classification_results(train_data, datatest1, train_classes, predictions, title, 'Repr??sentants de classe',
                                   f'Donn??es al??atoires class??es {n_neighbors}-PPV',
                                   extent, datatest2, predictions2 / error_class / 0.75,
                                   f'Pr??diction de {n_neighbors}-PPV, donn??es originales')

def ppv_classify(n_neighbors, train_data, classes, test1, test2=None):
    """
    Classifie test1 et test2 dans les classes contenues dans train_data et ??tiquet??es dans "classes"
        au moyen des n_neighbors plus proches voisins (distance euclidienne)
    Retourne les pr??dictions pour chaque point dans test1, test2
    Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    """
    # Creation classificateur
    # n_neighbors est le nombre k
    # metric est le type de distance entre les points. La liste est disponible ici:
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html#sklearn.neighbors.DistanceMetric
    # TODO L2.E3.1 Compl??ter la logique pour utiliser la librairie ici
    kNN = knn(n_neighbors, metric='minkowski')  # minkowski correspond ?? distance euclidienne lorsque le param??tre p = 2
    kNN.fit(train_data, classes)
    predictions_test1 = kNN.predict(test1)  # classifie les donn??es de test1
    predictions_test2 = kNN.predict(test2) if np.asarray(test2).any() else np.asarray([])  # classifie les donn??es de test2 si pr??sentes
    return predictions_test1, predictions_test2

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


