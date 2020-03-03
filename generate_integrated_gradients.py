"""
Generate integrated gradients for clean images and their adversarial twins and construct a dataset
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import shap
import torch
import torch.nn.functional as F
import torchvision.models as models
from scipy.special import softmax
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from torchvision import datasets, transforms
from tqdm import tqdm

# Constants
path_to_imagenet_val = '/experiments/ImageNet/val_images/'
input_size = 224


def load_model(device):
    logger.info('Loading model')
    model = models.inception_v3(pretrained=True)
    model.eval()
    logger.info('Model in eval mode')
    model = model.to(device)
    return model


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad, targeted):
    # Targeted or not?
    direction = -1 if targeted else 1
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + direction * epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def get_data_loader(num_workers):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(path_to_imagenet_val, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=1, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    return data_loader


def attack(model, target_image, original_class, targeted, use_random_target,
           num_iterations, epsilon, device):
    """
    Main attack function

    Args:
        model: A Pytorch model
        target_image (torch.Tensor): The target image to produce
            the adverarial image from
        original_class (int): Index of the original class
        targeted (bool): Whether the attack is targeted or not
        num_iterations (int): Number of iterations to run BIM
        epsilon (float): Perturbation control parameter
        device (str): ID of the device to allocate model and Tensors

    Returns:
        If attack is successful:
            Tuple of (Adversarial image, original image, noise map,
            model confidence on adv image, model confidence on original image)
        else:
            None

    """
    # Convert relevant inputs into Pytorch Tensors
    target_image_tensor = target_image
    model = model.to(device)

    # Keep a clean copy
    clean_data = target_image_tensor.clone()
    target_image_tensor = target_image.to(device)
    target_image_tensor.requires_grad = True

    # Forward pass the data through the model
    logits = model(target_image_tensor)
    original_softmax = softmax(logits.data.cpu().numpy()).ravel()
    max_class = original_softmax.argmax()
    classes = np.argsort(original_softmax)
    second_most_confident_class = classes[-2]
    # logger.info('Max class: {}'.format(max_class))
    # logger.info('Second: {}'.format(second_most_confident_class))

    if max_class != original_class:
        return -1

    if targeted:
        if use_random_target:
            # Use a random class
            list_candidates = list(set(range(1000)) - {original_class})
            target_class = np.random.choice(list_candidates)
        else:
            # use second most confident class
            target_class = second_most_confident_class

        target_tensor = torch.tensor([target_class])
    else:
        target_tensor = torch.tensor([original_class])

    target_tensor = target_tensor.to(device)

    # Assign each tensor to the same device
    target_image_tensor = target_image_tensor.to(device, dtype=torch.float)

    # Run BIM attack
    adversarial_image = target_image_tensor
    for i in range(num_iterations):
        output = F.log_softmax(model(adversarial_image), dim=1)

        # Calculate the loss
        loss = F.nll_loss(output, target_tensor)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = target_image_tensor.grad.data

        # Call FGSM Attack
        adversarial_image = fgsm_attack(target_image_tensor, epsilon, data_grad, targeted=targeted)

    # Re-classify the perturbed image
    output = model(adversarial_image)

    # Get the probability
    max_prob = softmax(output.data.cpu().numpy()).max()

    # Check for success
    final_pred = output.max(1, keepdim=True)[1]

    if targeted:
        if final_pred.item() == target_tensor.item():
            # logger.info('Targeted attack is successful with probability {:.4f}'.format(max_prob))
            adv_ex = adversarial_image.squeeze().detach().cpu().numpy()
            clean_ex = clean_data.squeeze().detach().cpu().numpy()
            return adv_ex, clean_ex
    else:
        if final_pred.item() != target_tensor.item():
            # logger.info('Untargeted attack is successful')
            adv_ex = adversarial_image.squeeze().detach().cpu().numpy()
            clean_ex = clean_data.squeeze().detach().cpu().numpy()
            return adv_ex, clean_ex
    return None


def generate_shap_values(model, adv_images, clean_images, device, test_ratio=0.2):
    # Create the background data
    X, y = shap.datasets.imagenet50()
    X /= 255
    X = torch.tensor(X.swapaxes(-1, 1).swapaxes(2, 3)).float()
    X = X.to(device)

    num_images = len(adv_images)

    start = time.time()
    explainer = shap.GradientExplainer((model, model.Conv2d_4a_3x3),
                                       X, local_smoothing=0.9)
    adv_shap_values = []
    clean_shap_values = []
    for i in tqdm(range(num_images), total=num_images):
        adv_image = adv_images[i]
        clean_image = clean_images[i]
        adv_clean_images_pairs = np.stack([adv_image, clean_image])
        adv_clean_images_pairs = torch.tensor(adv_clean_images_pairs).float()
        adv_clean_images_pairs = adv_clean_images_pairs.to(device)
        shap_values, _ = explainer.shap_values(adv_clean_images_pairs,
                                               ranked_outputs=1, nsamples=50)
        # Sort the shap values and add to buffer
        adv_shap_values.append(sorted(shap_values[0][0].ravel()))
        clean_shap_values.append(sorted(shap_values[0][1].ravel()))

    adv_shap_values = np.stack(adv_shap_values)
    clean_shap_values = np.stack(clean_shap_values)
    train_test_split_idx = int((1 - test_ratio) * len(adv_images))
    adv_train = adv_shap_values[:train_test_split_idx]
    adv_test = adv_shap_values[train_test_split_idx:]
    clean_train = clean_shap_values[:train_test_split_idx]
    clean_test = clean_shap_values[train_test_split_idx:]
    logger.info(adv_train.shape)

    y_train = np.array([1] * len(adv_train) + [0] * len(clean_train))
    y_test = np.array([1] * len(adv_test) + [0] * len(clean_test))
    X_train = np.concatenate([adv_train, clean_train])
    X_test = np.concatenate([adv_test, clean_test])

    logger.info(X_train.shape)
    logger.info(X_test.shape)
    logger.info(y_train.shape)
    logger.info(y_test.shape)

    end = time.time()
    logger.info('Total SHAP value generation for {} images took {:.2f} seconds'.format(
        num_images * 2, end - start
    ))

    return X_train, X_test, y_train, y_test


def run_pipeline(num_workers, num_images, targeted, use_random_target, num_iterations, epsilon,
                 device):
    # Get the data_loader
    data_loader = get_data_loader(num_workers)

    # Get the model
    model = load_model(device=device)

    adv_images = []
    clean_images = []

    num_effective_samples = 0

    for idx, (target_image, label) in tqdm(enumerate(data_loader), total=num_images):
        if idx == num_images:
            break
        result = attack(model=model,
                        target_image=target_image,
                        original_class=label.item(),
                        targeted=targeted,
                        use_random_target=use_random_target,
                        num_iterations=num_iterations,
                        epsilon=epsilon,
                        device=device)
        if result == -1:
            # -1 means the model didn't predict correctly the first time
            continue
        else:
            num_effective_samples += 1
            if result:
                adv_image, clean_image = result
                adv_images.append(adv_image)
                clean_images.append(clean_image)

    logger.info('Attack success rate: {} ({}/{})'.format(len(adv_images) / num_effective_samples,
                                                         len(adv_images), num_effective_samples))
    adv_images, clean_images = np.stack(adv_images), np.stack(clean_images)
    logger.info('Adv images shape: {}'.format(adv_images.shape))

    data_loader = None
    del data_loader

    X_train, X_test, y_train, y_test = generate_shap_values(
        model=model,
        adv_images=adv_images,
        clean_images=clean_images,
        device=device
    )

    num_samples = len(X_train) + len(X_test)

    if targeted:
        if use_random_target:
            name = 'inception_v3_bim_targeted_random_{}.pkl'
        else:
            name = 'inception_v3_bim_targeted_next_confident_{}.pkl'
    else:
        name = 'inception_v3_bim_untargeted_{}.pkl'
    name = name.format(num_samples)

    joblib.dump({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }, filename=os.path.join('/tmp', name))

    eval_scikit_models(X_train, X_test, y_train, y_test)


def eval_scikit_models(X_train, X_test, y_train, y_test):
    lr = LogisticRegression()
    rf = RandomForestClassifier()

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    lr_score = lr.score(X_test, y_test)
    rf_score = rf.score(X_test, y_test)

    logger.info('Logistic Regression accuracy: {}'.format(lr_score))
    logger.info('RandomForest accuracy: {}'.format(rf_score))


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    logger = logging.getLogger(__name__)
    logger.info('Starting experiment')

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_images', required=False, default=100, type=int)
    parser.add_argument('--num_workers', required=False, default=1, type=int)
    parser.add_argument('--targeted', action='store_true')
    parser.add_argument('--use_random_target', action='store_true')
    parser.add_argument('--num_iterations', required=False, default=20, type=int)
    parser.add_argument('--epsilon', required=False, default=0.2, type=float)

    args = parser.parse_args()
    args = vars(args)

    logger.info('Arguments: {}'.format(args))

    num_images = int(args['num_images'])
    num_workers = int(args['num_workers'])
    targeted = bool(args['targeted'])
    use_random_target = bool(args['use_random_target'])
    num_iterations = int(args['num_iterations'])
    epsilon = float(args['epsilon'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('Device: {}'.format(device))

    if targeted:
        if use_random_target:
            logger.info('Using targeted attack with random targets')
        else:
            logger.info('Using targeted attack with second most confident class')
    else:
        logger.info('Using untargeted attacks')

    _ = run_pipeline(
        num_workers=num_workers,
        num_images=num_images,
        targeted=targeted,
        use_random_target=use_random_target,
        num_iterations=num_iterations,
        epsilon=epsilon,
        device=device
    )
