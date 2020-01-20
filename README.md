# Explaining Away Attacks Against Neural Networks

![image](/images/juxtaposition.png)

[This notebook](explain_away_attacks.ipynb) accompanies the blog post titled "Explaining Away Attacks Against Neural Networks" which can
be found [here](https://seansaito.github.io/2019/08/03/explain-adversarial/).


The steps in the notebook are as follows:
* Load a pre-trained InceptionV3 model from pytorch
* Generate adversarial examples using an ImageNet image
* Generate explanations for the model's prediction via SHAP
* Compare SHAP value explanations between the original image and the adversarial image

The notebook uses Python 3.6 and relies on the following dependencies:

```
matplotlib==3.1.1
numpy==1.17.0
Pillow==6.1.0
scipy==1.3.0
shap==0.29.3
torch==1.1.0.post2
torchvision==0.3.0
```
