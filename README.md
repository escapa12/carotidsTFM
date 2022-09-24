## Deep Learning for the Detection and Characterization of the Carotid Artery in Ultrasound Imaging
	
In this thesis, we explore the effectiveness of Deep Learning techniques in attempting to automatize and improve the diagnosis of atheroma plaques. To achieve so we tackle the following problems: ultrasound image segmentation and plaque tissue classification.


### Datsets:
 * [REGICOR](https://regicor.cat/es/)
 * [NEFRONA](https://www.udetma.com/es/proyecto-nefrona/proyecto.html)


### Procedure:
 * Data Cleaning
 * Automatic GT generation
 * Data exploration
 * Comun Carotid Artery segmentation 
 * Tissue classification of atherosclerosi plaque 

### Implementations:
This project has been mostly implemented in python, only a part of the GT generation is implemented in MATLAB.
For the segmentation task we used a [keras fork](https://github.com/MarcBS/keras) running on Theano. The Fully Convolutional Network implemented to explore the carotid artery is based on the [semantic segmentation in Keras](https://github.com/beareme/keras_semantic_segmentation). 

For the classification of the plaque we used [keras](https://github.com/keras-team/keras) running tensorflow on the backend. 

### Delivered on 03-07-2018 by:
* Arnau Escapa
* Enric Sarlé
* Jonatan Piñol
