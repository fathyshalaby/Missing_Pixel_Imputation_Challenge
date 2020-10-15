import os
import numpy as np
import torch
import torch.utils.data
from datasets import getImages, ImageWidiSet
from utils import plot
from architectures import SimpleCNN
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pickle
import PIL
from matplotlib import pyplot as plt



def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device):
    """Function for evaluation of a model `model` on the data in `dataloader` on device `device`"""
    # Define a loss (mse loss)
    mse = torch.nn.MSELoss()
    # We will accumulate the mean loss in variable `loss`
    loss = torch.tensor(0., device=device)
    with torch.no_grad():  # We do not need gradients for evaluation
        # Loop over all samples in `dataloader`
        for data in tqdm(dataloader, desc="scoring", position=0):
            # Get a sample and move inputs and targets to device
            inputs, targets, mask = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)

            # mask = mask.to(dtype=torch.bool)

            # Get outputs for network
            outputs = model(inputs) * mask
            # predictions = [outputs[i, mask[i]] for i in range(len(outputs))]

            # Here we could clamp the outputs to the minimum and maximum values of inputs for better performance

            # Calculate mean mse loss over all samples in dataloader (accumulate mean losses in `loss`)
            # losses = torch.stack([mse(prediction, target.reshape((-1,))) for prediction, target in zip(predictions, targets)])
            # loss = losses.mean()
            loss = mse(outputs, targets)
    return loss


def main(results_path, network_config: dict, learningrate: int = 1e-3, weight_decay: float = 1e-5,
         n_updates: int = int(1e5), device: torch.device = torch.device("cuda:0")):
    """Main function that takes hyperparameters and performs training and evaluation of model"""
    # Prepare a path to plot to
    os.makedirs(results_path, exist_ok=True)
    plotpath = os.path.join(results_path, 'plots')
    os.makedirs(plotpath, exist_ok=True)
    # Load  dataset
    trainset = getImages(part='dataset_part_1/**')
    valset = getImages(part='dataset_part_4/**')
    #testset = getImages(part='dataset_part_2/**')
    # Create datasets and dataloaders with rotated targets without augmentation (for evaluation)
    trainingset_eval = ImageWidiSet(dataset=trainset)
    validationset = ImageWidiSet(dataset=valset)
    #testset = ImageWidiSet(dataset=testset)
    #trainloader = torch.utils.data.DataLoader(trainingset_eval, batch_size=1, shuffle=False, num_workers=22)
    valloader = torch.utils.data.DataLoader(validationset, batch_size=1, shuffle=False, num_workers=22)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=22)

    # Create datasets and dataloaders with rotated targets with augmentation (for training)
    trainingset_augmented = ImageWidiSet(dataset=trainset)
    trainloader_augmented = torch.utils.data.DataLoader(trainingset_augmented, batch_size=1, shuffle=True,
                                                        num_workers=22)

    # Define a tensorboard summary writer that writes to directory "results_path/tensorboard"
    writer = SummaryWriter(log_dir=os.path.join(results_path, 'tensorboard'))

    # Create Network
    net = SimpleCNN(**network_config)
    net.to(device)
    net.train()

    # Get mse loss function
    mse = torch.nn.MSELoss(reduction='sum')

    # Get adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learningrate, weight_decay=weight_decay)

    print_stats_at = 1e2  # print status to tensorboard every x updates
    plot_at = 2e3  # plot every x updates
    validate_at = len(trainingset_augmented)  # evaluate model on validation set and check for new best model every x updates
    update = 0  # current update counter
    best_validation_loss = np.inf  # best validation loss so far
    n_updates = len(trainingset_augmented)*3
    update_progess_bar = tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)  # progressbar

    # Save initial model as "best" model (will be overwritten later)
    torch.save(net, os.path.join(results_path, 'best_model.pt'))

    # Train until n_updates update have been reached
    while update < n_updates:
        for data in trainloader_augmented:
            # Get next samples in `trainloader_augmented`
            inputs, targets, mask = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            target_masks = mask.to(dtype=torch.bool)
            mask = mask.to(device)
            # Reset gradients
            optimizer.zero_grad()
            # Get outputs for network
            # print('mask', mask.shape)
            outputs = net(inputs) * mask
            # predictions = [outputs[i, target_masks[i]] for i in range(len(outputs))]
            # targetss = [targets[i, target_masks[i]] for i in range(len(targets))]
            # Calculate loss, do backward pass, and update weights
            # losses = torch.stack([mse(prediction, target.to(device))for prediction, target in zip(predictions, targetss)])
            loss = mse(outputs[target_masks], targets[target_masks])
            # loss = losses.mean()
            # loss = mse(predicted_image,target_image)
            loss.backward()
            optimizer.step()

            # Print current status and score
            if update % print_stats_at == 0 and update > 0:
                writer.add_scalar(tag="training/loss",
                                  scalar_value=loss.cpu().detach().numpy(),
                                  global_step=update)

            # Plot output
            if update % plot_at == 0:
                plot(inputs.detach().cpu().numpy(), targets.detach().cpu().numpy(),
                     outputs.detach().cpu().numpy() * 255,
                     plotpath, update)

            # Evaluate model on validation set
            if update % validate_at == 0 and update > 0:
                val_loss = evaluate_model(net, dataloader=valloader, device=device)
                writer.add_scalar(tag="validation/loss", scalar_value=val_loss.cpu(), global_step=update)
                # Add weights as arrays to tensorboard
                for i, param in enumerate(net.parameters()):
                    writer.add_histogram(tag=f'validation/param_{i}', values=param.cpu(),
                                         global_step=update)
                # Add gradients as arrays to tensorboard
                for i, param in enumerate(net.parameters()):
                    writer.add_histogram(tag=f'validation/gradients_{i}',
                                         values=param.grad.cpu(),
                                         global_step=update)
                # Save best model for early stopping
                if best_validation_loss > val_loss:
                    best_validation_loss = val_loss
                    print('new best model')
                    torch.save(net, os.path.join(results_path, 'best_model.pt'))

            update_progess_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            update_progess_bar.update()

            # Increment update counter, exit if maximum number of updates is reached
            update += 1
            if update >= n_updates:
                break

    update_progess_bar.close()
    torch.save(net, os.path.join(results_path, 'best_model.pt'))
    print('Finished Training!')

    # Load best model and compute score on test set
    print(f"Computing scores for best model")
    net = torch.load(os.path.join(results_path, 'best_model.pt'))
    #test_loss = evaluate_model(net, dataloader=testloader, device=device)
    val_loss = evaluate_model(net, dataloader=valloader, device=device)
    #train_loss = evaluate_model(net, dataloader=trainloader, device=device)

    print(f"Scores:")
    #print(f"test loss: {test_loss}")
    print(f"validation loss: {val_loss}")
    #print(f"training loss: {train_loss}")

    # Write result to file
    with open(os.path.join(results_path, 'results.txt'), 'w') as fh:
        print(f"Scores:", file=fh)
        #print(f"test loss: {test_loss}", file=fh)
        print(f"validation loss: {val_loss}", file=fh)
        #print(f"training loss: {train_loss}", file=fh)

    # Write predictions to file


def predict(results_path, network_config: dict, learningrate: int = 1e-3, weight_decay: float = 1e-5,
            n_updates: int = int(1e5), device: torch.device = torch.device("cuda:0")):
    pickle_file = open('example_testset.pkl', 'rb')
    plotpath = os.path.join(results_path, 'plots')

    test_list = pickle.load(pickle_file)
    net = SimpleCNN(**network_config)
    net = torch.load(os.path.join(results_path, 'best_model.pt'))
    net.eval()
    net.to(device)
    # pickle data
    images = test_list["images"]
    crop_sizes = test_list["crop_sizes"]
    crop_centers = test_list["crop_centers"]
    testset = ImageWidiSet(imagesRAW=[PIL.Image.fromarray(image) for image in images], centerList=crop_centers,
                           cropsizeList=crop_sizes, mode='test')
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    predicted_images = []
    # submission data
    plotpath = plotpath + '/test'
    i = 0
    for data in tqdm(testloader):
        X, Y, mask = data
        input_tensor = X.to(device)
        mask = mask.to(device)
        targetmask = mask.to(dtype=np.bool)
        # predicting the image
        outputs = net(input_tensor) * mask
        #print(outputs.max())
        outputs[:]*=255
        prediction = outputs.to(dtype=torch.uint8)
        # now the crop out the image.
        predicted_image = prediction[0][0].detach().cpu().numpy()
        x, y = crop_centers[i]
        dx, dy = crop_sizes[i]
        dx, dy = dx // 2, dy // 2

        predicted_image = predicted_image[x - dx:x + dx + 1, y - dy:y + dy + 1]
        #print(predicted_image)
        predicted_images.append(predicted_image)
        # Testing if it looks accurate.
        plot(input_tensor.detach().cpu().numpy(), input_tensor.detach().cpu().numpy(), outputs.detach().cpu().numpy(),
             plotpath, i)
        i += 1

    with open('submission_example105.pkl', 'wb') as sub:
        pickle.dump(predicted_images, sub)


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='path to config file', type=str)
    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file, 'r') as fh:
        config = json.load(fh)
    main(**config)
    predict(**config)
