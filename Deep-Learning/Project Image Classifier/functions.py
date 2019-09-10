import numpy as np
from collections import OrderedDict
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image


# Load Data
def load_data(data_directory):
    data_dir=data_directory
    train_dir=data_dir+'/train'
    valid_dir=data_dir+'/valid'
    test_dir=data_dir+'/test'
    
    # Define your transforms for the training, validation, and testing sets
    train_transf=transforms.Compose([transforms.RandomRotation(25),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    
    test_transf=transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    
    validation_transf=transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    
    # Load the datasets with ImageFolder
    trainData=datasets.ImageFolder(train_dir, transform=train_transf)
    testData=datasets.ImageFolder(test_dir, transform=test_transf)
    validData=datasets.ImageFolder(valid_dir, transform=validation_transf)

    # Using the image datasets and the trainforms, define the dataloaders
    train_data_loader=torch.utils.data.DataLoader(trainData, batch_size=128, shuffle=True)
    test_data_loader=torch.utils.data.DataLoader(testData, batch_size=128)
    valid_data_loader = torch.utils.data.DataLoader(validData, batch_size=128)
    
    return train_data_loader, test_data_loader, valid_data_loader, trainData, testData, validData





# Defining new Classifier 
def build_image_classifier(hidden_units, dropout, gpu, input_lr):
    
    model=models.alexnet(pretrained = True)
    
    for param in model.parameters(): 
        param.requires_grad=False

    classifier=nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(9216, hidden_units, bias=True)),
                              ('relu1', nn.ReLU()),
                              ('dropout1', nn.Dropout(dropout)),
                              ('fc2', nn.Linear(hidden_units, int(hidden_units/4), bias=True)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    # setting up new classifier 
    model.classifier=classifier
    
    # To set CUDA if it's enabled
    if torch.cuda.is_available() and gpu=='gpu':
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')

    # Add Cuda to the Model
    model.to(device)
    
    # Loss Function and optimizer
    loss_function=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.classifier.parameters(), lr=input_lr)

    return model, loss_function, optimizer





# Defining validation 
def validation(model, loss_function, valid_loader, device):
    
    validation_loss=0
    accuracy=0
    for inputs, labels in valid_loader:
        
        inputs, labels=inputs.to(device), labels.to(device)
        output=model.forward(inputs)
        validation_loss+=loss_function(output, labels).item()

        ps=torch.exp(output)
        equality=(labels.data == ps.max(dim=1)[1])
        accuracy+=equality.type(torch.FloatTensor).mean()
    
    return validation_loss, accuracy




# Train the model
def train(model, epochs, train_data_loader, test_data_loader, valid_data_loader, loss_function, optimizer, gpu, print_25=25):
    print("********************************** Initiating the Training process **********************************")
    print("")
    steps=0
    
    # To set CUDA if it's enabled
    if torch.cuda.is_available() and gpu == 'gpu':
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')

    # Add Cuda to the Model
    model.to(device)

    for epoch in range(epochs):
        running_loss=0
        model.train()
    
        for ii, (inputs, labels) in enumerate(train_data_loader):
            steps+=1
        
            inputs, labels=inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            outputs=model.forward(inputs)
            loss=loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss+=loss.item()
        
            if steps % print_25==0:
                with torch.no_grad():
                    validation_loss, val_accuracy=validation(model, loss_function, valid_data_loader, device)
            
                print("Epoch: {}/{} | ".format(epoch+1, epochs),
                      "Training Loss: {:.3f} | ".format(running_loss/print_25),
                      "Validation Loss: {:.3f} | ".format(validation_loss/len(test_data_loader)),
                      "Validation Accuracy: {:.2f}%".format(val_accuracy/len(test_data_loader)*100))
            
                running_loss=0
                model.train()
    
    print("************************************** Training Completed **************************************")
    return model, optimizer




# Saving the model
def save_model(model, trainData, file_path):
    '''
    This method will save the model with the given configuration to a specific path
    
    Arguments:
        model : model
        trainData : the trainData set with our transformations
        file_path : entire file path with the filename , to save our model
    '''
    
    # TODO: Save the checkpoint
    model.to("cpu") #no need to use cuda for saving/loading model.

    # TODO: Save the checkpoint 
    model.class_to_idx = trainData.class_to_idx 

    checkpoint = {'architecture': model.type,
                 'classifier': model.classifier,
                 'mapping': model.class_to_idx,
                 'state_dict': model.state_dict()}

    torch.save(checkpoint, file_path)




# Load the saved model    
def load_model(file_path):
    checkpoint=torch.load(file_path) 
    model=models.alexnet (pretrained = True) 
    
    model.type=checkpoint ['architecture']
    model.classifier=checkpoint ['classifier']
    model.load_state_dict (checkpoint ['state_dict'])
    model.class_to_idx=checkpoint ['mapping']
    
    for param in model.parameters(): 
        param.requires_grad=False 
    
    return model




# Process the image and returns the np array
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image_pil=Image.open(image)
   
    transform_img=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor=transform_img(image_pil)
    
    return image_tensor





# Display the image by plotting it
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax=plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image=image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    image=std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax




# Predicts the class (or classes) of the passed image
def predict(image_path, model_path, gpu, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model=load_model(model_path)
    
    # To set CUDA if it's enabled
    if torch.cuda.is_available() and gpu == 'gpu':
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')

    # Add Cuda to the Model
    model.to(device)

    model.eval()

    # Convert image from numpy to torch
    torch_image=torch.from_numpy(np.expand_dims(process_image(image_path),axis=0)).type(torch.FloatTensor).to(device)

    log_probs=model.forward(torch_image)
    linear_probs=torch.exp(log_probs)

    top_probs, top_labels=linear_probs.topk(topk)
    
    top_probs=np.array(top_probs)[0]
    top_labels=np.array(top_labels)[0]
    
    idx_to_class={val: key for key, val in model.class_to_idx.items()}
    
    top_classes=[idx_to_class [item] for item in top_labels]
    top_classes=np.array (top_classes)
    
    return top_probs, top_classes
