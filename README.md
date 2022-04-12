# Bilma
Bert In Latin aMericA

Bilma is a BERT implementation in tensorflow and trained on the Masked Language Model task under the https://sadit.github.io/regional-spanish-models-talk-2022/ datasets.

The regional models can be downloaded from http://geo.ingeotec.mx/~lgruiz/regional-models-bilma/

The accuracy of the models trained on the MLM task for different regions are:

![bilma-mlm-comp](https://user-images.githubusercontent.com/392873/163045798-89bd45c5-b654-4f16-b3e2-5cf404e12ddd.png)

We also fine tuned the models for emoticon prediction, the resulting accuracy is as follows:

![bilma-cls-comp](https://user-images.githubusercontent.com/392873/163046824-0109e00f-3a54-486e-b93e-fbe09fbc7588.png)

# Quick guide

You can see the demo notebooks for a quick guide on how to use the models.
