import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.transform import swirl

class GloriaSimulate:
    """
    simulate different water spectra from different water body types and water types using GLORIA dataset
    Lehmann, Moritz K; Gurlin, Daniela; Pahlevan, Nima; Alikas, Krista; Anstee, Janet M; Balasubramanian, Sundarabalan V; Barbosa, Cláudio C F; Binding, Caren; Bracher, Astrid; Bresciani, Mariano; Burtner, Ashley; Cao, Zhigang; Conroy, Ted; Dekker, Arnold G; Di Vittorio, Courtney; Drayson, Nathan; Errera, Reagan M; Fernandez, Virginia; Ficek, Dariusz; Fichot, Cédric G; Gege, Peter; Giardino, Claudia; Gitelson, Anatoly A; Greb, Steven R; Henderson, Hayden; Higa, Hiroto; Irani Rahaghi, Abolfazl; Jamet, Cédric; Jiang, Dalin; Jordan, Thomas; Kangro, Kersti; Kravitz, Jeremy A; Kristoffersen, Arne S; Kudela, Raphael; Li, Lin; Ligi, Martin; Loisel, Hubert; Lohrenz, Steven; Ma, Ronghua; Maciel, Daniel A; Malthus, Tim J; Matsushita, Bunkei; Matthews, Mark; Minaudo, Camille; Mishra, Deepak R; Mishra, Sachidananda; Moore, Tim; Moses, Wesley J; Nguyen, Hà; Novo, Evlyn M L M; Novoa, Stéfani; Odermatt, Daniel; O'Donnell, David M; Olmanson, Leif G; Ondrusek, Michael; Oppelt, Natascha; Ouillon, Sylvain; Pereira Filho, Waterloo; Plattner, Stefan; Ruiz Verdú, Antonio; Salem, Salem I; Schalles, John F; Simis, Stefan G H; Siswanto, Eko; Smith, Brandon; Somlai-Schweiger, Ian; Soppa, Mariana A; Spyrakos, Evangelos; Tessin, Elinor; van der Woerd, Hendrik J; Vander Woude, Andrea J; Vandermeulen, Ryan A; Vantrepotte, Vincent; Wernand, Marcel Robert; Werther, Mortimer; Young, Kyana; Yue, Linwei (2022)
    : GLORIA - A global dataset of remote sensing reflectance and water quality from inland and coastal waters. PANGAEA, https://doi.org/10.1594/PANGAEA.948492
    :TODO
    """
    def __init__(self,fp):
        """
        :param fp (str): folder path to GLORIA dataset
        """
    
    def import_GLORIA(self):
        """
        import GLORIA datasets as pd dataframe
        """
    
    def image_distortion(self,im,rotation=0,strength=10,radius=120):
        """
        to simulate non-homogenous background spectra
        https://stackoverflow.com/questions/225548/resources-for-image-distortion-algorithms
        https://scikit-image.org/docs/stable/auto_examples/transform/plot_swirl.html
        """
        swirled = swirl(im, rotation=rotation, strength=strength, radius=radius)