"""
Départ du laboratoire
Classification de 3 classes avec toutes les méthodes couvertes par l'APP
APP2 S8 GIA
TODO voir L1.E2, L2.E1, L2.E2, L2.E3, L3.E2
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt

from TroisClasses import TroisClasses

import helpers.analysis as an
import helpers.classifiers as classifiers



##########################################
def main():
    # Statistiques
    m1, cov1, valpr1, vectprop1 = an.calcModeleGaussien(TroisClasses.C1, '\nClasse 1')
    m2, cov2, valpr2, vectprop2 = an.calcModeleGaussien(TroisClasses.C2, '\nClasse 2')
    m3, cov3, valpr3, vectprop3 = an.calcModeleGaussien(TroisClasses.C3, '\nClasse 3')

    # Données d'origine et données décorrélées
    allClasses = [TroisClasses.C1, TroisClasses.C2, TroisClasses.C3]
    # TODO L2.E1.3
    coeffs = classifiers.get_borders(allClasses)
    if False:
        an.view_classes(allClasses, TroisClasses.extent,border_coeffs=coeffs)

        # TODO: move to static class?
        allDecorr = an.decorrelate(allClasses, vectprop1)
        m1d, cov1d, valpr1d, vectprop1d = an.calcModeleGaussien(allDecorr[0], '\nClasse 1d')
        m2d, cov2d, valpr2d, vectprop2d = an.calcModeleGaussien(allDecorr[1], '\nClasse 2d')
        m3d, cov3d, valpr3d, vectprop3d = an.calcModeleGaussien(allDecorr[2], '\nClasse 3d')
        # TODO L2.E1.3
        coeffs = classifiers.get_borders(allDecorr)

        an.view_classes(allDecorr, an.Extent(ptList=an.decorrelate(TroisClasses.extent.get_corners(), vectprop1)),border_coeffs=coeffs)

        # exemple d'une densité de probabilité arbitraire pour 1 classe
        an.creer_hist2D(TroisClasses.C1, 'C1',plot=True)

        # génération de données aléatoires
    ndonnees = 5000
    donneesTest = an.genDonneesTest(ndonnees, TroisClasses.extent)
    # Changer le flag dans les sections pertinentes pour chaque partie de laboratoire
    if True: # TODO L2.E2.2

        # classification
        # Bayes
        #                           (train_data, train_classes, donnee_test, title, extent, test_data, test_classes)
        classifiers.full_Bayes_risk(allClasses, TroisClasses.class_labels, donneesTest, 'Bayes risque #1', TroisClasses.extent, TroisClasses.data, TroisClasses.class_labels)

    if False: # TODO L2.E3
        # 1-PPV avec comme représentants de classes l'ensemble des points déjà classés
        #           full_ppv(n_neighbors, train_data, train_classes, datatest1, title, extent, datatest2=None, classestest2=None)
        classifiers.full_ppv(1, TroisClasses.data, TroisClasses.class_labels, donneesTest, '1-PPV avec données orig comme représentants', TroisClasses.extent)

        # 1-mean sur chacune des classes
        # suivi d'un 1-PPV avec ces nouveaux représentants de classes
        cluster_centers, cluster_labels = classifiers.full_kmean(1, allClasses, TroisClasses.class_labels, 'Représentants des 1-moy', TroisClasses.extent)
        classifiers.full_ppv(1, cluster_centers, cluster_labels, donneesTest, '1-PPV sur le 1-moy', TroisClasses.extent, TroisClasses.data, TroisClasses.class_labels)

    if False: # TODO L3.E2
        # nn puis visualisation des frontières
        n_hidden_layers = 3
        n_neurons = 8
        classifiers.full_nn(n_hidden_layers, n_neurons, TroisClasses.data, TroisClasses.class_labels, donneesTest,
                f'NN {n_hidden_layers} layer(s) caché(s), {n_neurons} neurones par couche', TroisClasses.extent, TroisClasses.data, TroisClasses.class_labels)


    plt.show()


#####################################
if __name__ == '__main__':
    main()
