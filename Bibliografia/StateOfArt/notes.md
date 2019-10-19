### Notes sobre l' State of the art- IMT segmentation

**Observació**: ningú parla de la classicació del tipus de placa!!

#### Reviews:

1-Loizou, C.P.: A review of ultrasound common carotid artery image and video segmentation techniques 
Comparació de molts mètodes diferents.
    * Bona  justificació de l'utilitat de classificar IMT.  (clinical signifianse amb referencies mèdiques)
    * Parla molt dels snake-like methods
    * No compara resultats neural networks però en parla
   
2-  Molinari, F., Zeng, G., Suri, J.S.: A state of the art review on IMT and wall segmentation techniques for carotid ultrasound.

#### Papers:

1- Molinari: Completely Automated Multiresolution Edge Snapper—A New Technique for an Accurate Carotid Ultrasound IMT Measurement.
    * Resol el problema en dos fases:
            1) automated CA (arteria carotida) recognition based on a combination of scale–space and statistical classification in a multiresolution framework 
            2) automated segmentation of lumen–intima (LI) and media–adventitia (MA) interfaces for the far (distal) wall and IMT measurement. 
    * Methods: ??? algun tipus de machine learning
    
2-Mechon-Lara RM: Automatic detection of the intima-media thickness in ultrasound images of the common carotid artery using **neural networks**. 
De pago. Seria interessant perque utilitza NN.
    
3- Dana E. Ilea: An Automatic 2D CAD Algorithm for the Segmentation of the IMT in Ultrasound Carotid Artery Images
    * Method: unsupervised Computer Aided Detection (CAD)

5 - Bastida-Jumilla: Segmentation of the Common Carotid Artery Walls Based on a Frequency Implementation of Active Contours
    * computer vision Technique: Active contours... 

7- Quin: An integrated method for atherosclerotic carotid plaque segmentation in ultrasound image
    * Algoritma consisteix en 3 fases! Podriem intentar aplicar-ho nosaltres!!
        1) Detectar la ROI
        2) Detectar regio candidata per la placa 
        3) Segmentació (4 metodes diferents)-
    * Aplica 4 machine learning methods per segmentar:
        - SVM with linear kernel, 
        - SVM with radial basis function kernel
        - AdaBoost
        - random forest
    * Best method:
     integrated the random forest and an auto-context model
    * Utilitza moltes mètriques diferents
    
?- Christine M Robertson: Carotid intima–media thickness and the prediction of vascular events.
    * Relació entre valor IMT i risc de diferents episodi cardiovascular
    * No fa experiments propis, reculls altres estudis que demostren la correlació entre un valor alt de ITM i episodis cardiovasculars.


