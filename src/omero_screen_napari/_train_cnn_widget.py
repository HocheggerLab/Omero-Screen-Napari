import numpy as np
import json
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
from magicgui import magicgui
from magicgui.widgets import Container, RadioButtons, Label, PushButton, LineEdit
from qtpy.QtWidgets import QFileDialog, QMessageBox

from omero_screen_napari.cnn_models import FocalLoss, ROIBasedDenseNetModel, ROIBasedEfficientNetB0Model, ROIBasedMobileNetV3Model, ROIBasedResNeXtModel, ROIBasedShuffleNetV2Model
from omero_screen_napari.gallery_userdata import UserData
from omero_screen_napari.gallery_userdata_singleton import userdata
from omero_screen_napari.omero_data import OmeroData
from omero_screen_napari.omero_data_singleton import omero_data
from omero_screen_napari.utils import omero_connect
from omero.gateway import FileAnnotationWrapper
from omero.sys import ParametersI
from omero.model import ProjectI, DatasetI
from omero.rtypes import rstring
from datetime import datetime
import time
import omero
from omero.model import FileAnnotationI, OriginalFileI
import os
import omero.rtypes as rtypes
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ROIDataset(Dataset):
    def __init__(self, rois, labels, transform=None):
        self.rois = rois
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        roi = self.rois[idx]
        label = self.labels[idx]
        if self.transform:
            roi = self.transform(roi)
        return roi, label

def train_cnn_widget(
    class_name: str | None = None,
    userdata: UserData | None = userdata,
) -> Container:
    widget = TrainCNNWidget(class_name, userdata, omero_data)
    return widget.container

class TrainCNNWidget:
    def __init__(self, class_name: str | None, user_data: UserData | None, omero_data: OmeroData, class_options: list[str] | None = None):
        
        self.user_data = user_data
        self.omero_data = omero_data
        self.model_options = ["DenseNet121", "ResNeXt50", "ShuffleNetV2", "EfficientNetB0", "MobileNetV3"]

        # Model selection
        self.select_model = RadioButtons(label="Select Model:", choices=self.model_options, value="DenseNet121")

        # Model name input
        self.model_name_input = LineEdit(label="Model Name:", value="My_Model")

        # Learning rate and epoch settings
        self.learning_rate_select = RadioButtons(label="Select Learning Rate:", choices=["0.001", "0.0001", "0.00001"], value="0.0001")
        self.epoch_select = RadioButtons(label="Select Epochs:", choices=["1", "5", "10", "20", "100"], value="10")
        
        # Folder selection, data loading, training, and test buttons
        self.select_folder_button = magicgui(self.open_folder_dialog, call_button="Select Dataset Folder")
        self.folder_name_label = Label(value="Selected Folder: None")
        self.load_data_button = PushButton(text="Load Data")
        self.load_data_button.clicked.connect(self.load_dataset)
        self.train_button = PushButton(text="Start Training")
        self.train_button.clicked.connect(self.start_training)
        self.test_button = PushButton(text="Start Testing")
        self.test_button.clicked.connect(self.start_testing)

        # Labels to show accuracies
        self.train_acc_label = Label(value="Training Accuracy: N/A")
        self.val_acc_label = Label(value="Validation Accuracy: N/A")
        self.test_accuracy_label = Label(value="Test Accuracy: N/A")

        # Loss function, optimizer, and batch size selection
        self.loss_function_select = RadioButtons(
            label="Select Loss Function:",
            choices=["CrossEntropyLoss", "FocalLoss"],
            value="CrossEntropyLoss"
        )

        self.optimizer_select = RadioButtons(
            label="Select Optimizer:",
            choices=["Adam", "SGD"],
            value="Adam"
        )

        self.batch_size_select = RadioButtons(
            label="Select Batch Size:",
            choices=["16", "32", "64", "128"],
            value="32"
        )

        # Weight decay selection
        self.weight_decay_select = RadioButtons(
            label="Select Weight Decay:",
            choices=["0.0", "0.0001", "0.001", "0.01"],
            value="0.0"
        )

        # Add this widget to the container
        self.container = Container(
            widgets=[
                self.select_model, self.model_name_input, self.learning_rate_select, self.epoch_select, 
                self.loss_function_select, self.optimizer_select, self.batch_size_select, 
                self.weight_decay_select, self.select_folder_button, self.folder_name_label, 
                self.load_data_button, self.train_button, self.test_button, 
                self.train_acc_label, self.val_acc_label, self.test_accuracy_label, 
            ]
        )
        self.dataset_folder = None
        self.class_options = None
        self.train_loader = None
        self.validation_loader = None
        self.test_loader = None
        self.device = torch.device("mps")
        self.image_channels = 2
        self.model = ROIBasedDenseNetModel(num_classes=7, num_channels=self.image_channels).to(self.device)

    def open_folder_dialog(self):
        folder_path = QFileDialog.getExistingDirectory(caption="Select Dataset Folder")
        if folder_path:
            self.dataset_folder = Path(folder_path)
            self.folder_name_label.value = f"Selected Folder: {self.dataset_folder.name}"

    def load_dataset(self):
        if not self.dataset_folder:
            QMessageBox.warning(None, "Warning", "Please select a dataset folder first.")
            return

        classifier_dir = self.dataset_folder
        npy_files = list(classifier_dir.glob("*.npy"))
        print(f'Found {len(npy_files)} .npy files:')
        
        meta_data_path = classifier_dir / "metadata.json"
        with meta_data_path.open("r") as json_file:
            metadata = json.load(json_file)
        
        self.class_options = metadata['class_options']
        if 'unassigned' in self.class_options:
            self.class_options.remove('unassigned')

        training_dicts = []
        i = 0
        for file in npy_files:
            print(str(i))
            training_files = np.load(file, allow_pickle=True)
            training_dicts.append(training_files[()])
            i = i + 1

        images, masks, targets = [], [], []
        for dict_ in training_dicts:
            for j in range(len(dict_['target'])):
                target = dict_['target'][j]
                if target in self.class_options:
                    images.append(dict_['data'][0][j])
                    masks.append(dict_['data'][1][j])
                    targets.append(target)

        # Encode labels to numerical values
        label_mapping = {label: idx for idx, label in enumerate(np.unique(targets))}
        numeric_labels = np.array([label_mapping[label] for label in targets])


        # Extract and combine all ROIs into a fixed-size array
        rois_array = self.extract_and_pad_rois(images, masks)

        print(f"Extracted ROIs array shape: {rois_array.shape}")

        # Split dataset
        train_images, test_images, train_rois, test_rois, train_labels, test_labels = train_test_split(
            images, rois_array, numeric_labels, test_size=0.2
        )
        train_images, val_images, train_rois, val_rois, train_labels, val_labels = train_test_split(
            train_images, train_rois, train_labels, test_size=0.1
        )

        # Save processed dataset
        np.savez(self.dataset_folder / 'cell_dataset_split_augmented.npz', 
                 train_images=train_images, train_rois=train_rois, train_labels=train_labels,
                 val_images=val_images, val_rois=val_rois, val_labels=val_labels,
                 test_images=test_images, test_rois=test_rois, test_labels=test_labels)
        print("Balanced and augmented dataset saved successfully")

        #self._upload_dataset_to_omero() # Commented out because dataset is too big and it takes too much time to upload to OMERO (11GB)

        # Create Dataloader
        self.create_data_loaders(train_images, train_rois, train_labels, val_images, val_rois, val_labels, test_images, test_rois, test_labels, int(self.batch_size_select.value))
        QMessageBox.information(None, "Success", "Dataset loaded and saved successfully.")

    @omero_connect
    def _upload_dataset_to_omero(self, conn=None):

        npy_file_path = Path(self.dataset_folder / 'cell_dataset_split_augmented.npz')
        dataset_name = self.dataset_folder.name + "_Dataset"
        dataset_id = self.get_dataset_ids_by_name(conn, dataset_name)
        
        # If dataset is not found, create a new dataset
        if not dataset_id:
            print(f"Dataset with the name '{dataset_name}' not found. Creating a new dataset.")
            new_dataset = DatasetI()
            new_dataset.name = rstring(dataset_name)
            new_dataset.description = rstring("Automatically created dataset for uploading npy files")
            conn.getUpdateService().saveObject(new_dataset)

            # Wait for a certain amount of time and retry to fetch the newly created dataset
            wait_time = 2  # Wait time in seconds
            max_retries = 10  # Max number of retries
            retries = 0

            while retries < max_retries:
                time.sleep(wait_time)
                dataset_id = self.get_dataset_ids_by_name(conn, dataset_name)
                if dataset_id:
                    dataset = conn.getObject("Dataset", dataset_id[0])
                    print("Dataset created and retrieved successfully.")
                    break
                retries += 1
                print(f"Retrying to fetch dataset... Attempt {retries}/{max_retries}")

            # Raise an error if the dataset could not be found within the given time
            if not dataset_id:
                raise Exception("Failed to retrieve the newly created dataset in OMERO after multiple attempts.")
        else:
            # If dataset already exists, retrieve the object using the ID
            dataset = conn.getObject("Dataset", dataset_id[0])

       # Check for existing File Annotations and delete any with the same name
        for ann in dataset.listAnnotations():
            if isinstance(ann, FileAnnotationWrapper):
                file_name = ann.getFile().getName()
                if file_name == npy_file_path.name:  # If file names match
                    print(f"Existing file found: {file_name}. Deleting it.")
                    conn.deleteObjects("Annotation", [ann.getId()], wait=True)
                    print(f"File {file_name} has been deleted.")

        # Upload the .npy file as a File Annotation
        file_ann = conn.createFileAnnfromLocalFile(
            str(npy_file_path),
            mimetype="application/octet-stream",  # MIME type for binary files
            ns="example.namespace",               # Namespace, customizable if desired
            desc="Uploaded .npy file"             # Description
        )

        # Add the File Annotation to the dataset
        dataset.linkAnnotation(file_ann)
        self._show_success_message("npy file successfully uploaded to omero")

    def get_dataset_ids_by_name(self, conn, dataset_name):
        query = "select obj from Dataset obj where obj.name = :name"
        params = ParametersI()
        params.addString("name", dataset_name)

        # Use findAllByQuery to get all datasets that match the given name
        datasets = conn.getQueryService().findAllByQuery(query, params)

        if datasets:
            dataset_ids = [dataset.getId().getValue() for dataset in datasets]
            return dataset_ids
        else:
            return None  # Return None if no matching datasets are found

    def _show_success_message(self, message: str):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText(message)
        msg_box.setWindowTitle("Success")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def extract_and_pad_rois(self, images, masks, target_size=(200, 200)):
        """
        Extracts ROIs from all images using the masks, pads them to a fixed size, and combines them in an array.

        Args:
            images (list or numpy array): 2-channel microscope images.
            masks (list or numpy array): Corresponding masks.
            target_size (tuple): Fixed ROI size (height, width).

        Returns:
            rois_array (numpy array): A numpy array containing the fixed-size ROIs.
        """
        rois_list = []

        for image, mask in zip(images, masks):
            # Extract the ROI using the mask
            binary_mask = (mask > 0).astype(np.uint8)
            roi = np.zeros_like(image)
            
            for channel in range(image.shape[-1]):
                roi[..., channel] = image[..., channel] * binary_mask

            # Pad the ROI to a fixed size
            h, w = roi.shape[:2]
            delta_w = target_size[1] - w
            delta_h = target_size[0] - h
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            padded_roi = cv2.copyMakeBorder(roi, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

            # Add the fixed-size ROI to the list
            rois_list.append(padded_roi)

        # Combine the ROIs into a numpy array
        rois_array = np.array(rois_list)
        return rois_array
    
    def data_augmentation(self, train_rois, train_labels):
        unique_elements, counts = np.unique(train_labels, return_counts=True)
        print("Initial class distribution:", dict(zip(unique_elements, counts)))

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        augmented_X = []
        augmented_y = []

        max_count = max(counts)

        for label in unique_elements:
            class_indices = np.where(train_labels == label)[0]
            class_X = train_rois[class_indices]
            class_y = train_labels[class_indices]
            
            current_count = len(class_X)
            target_count = max_count - current_count  # Calculate how many augmentation needed
            pbar = tqdm(total=target_count, desc=f"Augmenting for class {label}")
            
            generated_count = 0
            for batch in datagen.flow(class_X, class_y, batch_size=1, shuffle=False):
                augmented_X.append(batch[0][0])  # New image
                augmented_y.append(batch[1][0])  # Label
                generated_count += 1
                pbar.update(1)
                if generated_count >= target_count:
                    break
            
            pbar.close()
            
            # adding original data
            augmented_X.extend(class_X)
            augmented_y.extend(class_y)

        # Augmented dataset
        augmented_X = np.array(augmented_X)
        augmented_y = np.array(augmented_y)

        return augmented_X, augmented_y

    def create_data_loaders(self, train_images, train_rois, train_labels, val_images, val_rois, val_labels, test_images, test_rois, test_labels, b_size):
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.ConvertImageDtype(torch.float32),
            # transforms.Normalize([0.5, 0.5], [0.5, 0.5])
        ])

        train_dataset = ROIDataset(train_rois, train_labels, transform=data_transforms)
        self.train_loader = DataLoader(train_dataset, batch_size=b_size, shuffle=True)

        val_dataset = ROIDataset(val_rois, val_labels, transform=data_transforms)
        self.validation_loader = DataLoader(val_dataset, batch_size=b_size, shuffle=False)

        test_dataset = ROIDataset(test_rois, test_labels, transform=data_transforms)
        self.test_loader = DataLoader(test_dataset, batch_size=b_size, shuffle=False)

    def start_training(self):

        model_name = self.model_name_input.value
        self.model_save_path = Path(self.dataset_folder / f'{model_name}.pth')

        if not self.train_loader:
            QMessageBox.warning(None, "Warning", "Please load the data before starting training.")
            return

        if self.select_model.value == "DenseNet121":
            self.model = ROIBasedDenseNetModel(num_classes=len(self.class_options), num_channels=self.image_channels).to(self.device)
        elif self.select_model.value == "ResNeXt50":
            self.model = ROIBasedResNeXtModel(num_classes=len(self.class_options), num_channels=self.image_channels).to(self.device)
        elif self.select_model.value == "ShuffleNetV2":
            self.model = ROIBasedShuffleNetV2Model(num_classes=len(self.class_options), num_channels=self.image_channels).to(self.device)
        elif self.select_model.value == "EfficientNetB0":
            self.model = ROIBasedEfficientNetB0Model(num_classes=len(self.class_options), num_channels=self.image_channels).to(self.device)
        elif self.select_model.value == "MobileNetV3":
            self.model = ROIBasedMobileNetV3Model(num_classes=len(self.class_options), num_channels=self.image_channels).to(self.device)

         # Loss function selection
        if self.loss_function_select.value == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss()
        elif self.loss_function_select.value == "FocalLoss":
            criterion = FocalLoss(gamma=2)


        # Get learning rate and epoch values selected by the user
        learning_rate = float(self.learning_rate_select.value)
        weight_decay = float(self.weight_decay_select.value)

        if self.optimizer_select.value == "Adam":
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif self.optimizer_select.value == "SGD":
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        num_epochs = int(self.epoch_select.value)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            # Training loop
            with tqdm(self.train_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
                for rois, labels in tepoch:
                    rois, labels = rois.to(self.device), labels.to(torch.long).to(self.device)

                    optimizer.zero_grad()

                    # Forward pass
                    outputs = self.model(rois)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total_train += labels.size(0)
                    correct_train += (predicted == labels).sum().item()

                    tepoch.set_postfix(loss=loss.item())

                train_accuracy = 100 * correct_train / total_train
                train_loss = running_loss / len(self.train_loader)
                
                print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")


                # Validation
                self.model.eval()
                val_loss = 0.0
                correct_val = 0
                total_val = 0
                with torch.no_grad():
                    for rois, labels in self.validation_loader:
                        rois, labels = rois.to(self.device), labels.to(self.device)
                        outputs = self.model(rois)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total_val += labels.size(0)
                        correct_val += (predicted == labels).sum().item()

                val_loss /= len(self.validation_loader)
                val_accuracy = 100 * correct_val / total_val

                # Update
                self.train_acc_label.value = f"Training Accuracy: {train_accuracy:.4f}"
                self.val_acc_label.value = f"Validation Accuracy: {val_accuracy:.4f}"

        # Save the model after training is completed
        torch.save(self.model.state_dict(), self.model_save_path)
        print("Model saved successfully.")

        # Upload the model to OMERO
        self.upload_metadata_to_omero(self.model_name_input.value)
        self.upload_model_to_omero()
        
        QMessageBox.information(None, "Success", "Training completed and model uploaded successfully!")

    @omero_connect
    def upload_model_to_omero(self, conn=None):
        if not self.model_save_path.exists():
            raise Exception(f"Model file not found at {self.model_save_path}. Please ensure the model has been saved.")

        # Define the top-level project name
        top_level_project_name = "CNN_Models"

        # Generate the sub-dataset name based on the selected model and current date-time
        model_name = self.select_model.value  # Example: "DenseNet" or "ResNeXt"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        sub_dataset_name = self.model_name_input.value

        # Step 1: Check or create the top-level project
        project = conn.getObject("Project", attributes={"name": top_level_project_name})
        if project is None:
            print(f"Project with the name '{top_level_project_name}' not found. Creating a new project.")
            new_project = ProjectI()
            new_project.name = rstring(top_level_project_name)
            new_project.description = rstring("Top-level project for CNN models")
            project = conn.getUpdateService().saveAndReturnObject(new_project)

        # Step 2: Create the sub-dataset with the dynamic name
        print(f"Creating sub-dataset with name '{sub_dataset_name}'.")
        new_dataset = DatasetI()
        new_dataset.name = rstring(sub_dataset_name)
        new_dataset.description = rstring(f"Dataset for {model_name} model trained on {timestamp}")
        dataset = conn.getUpdateService().saveAndReturnObject(new_dataset)

        # Step 3: Link the sub-dataset to the project
        print(f"Linking dataset '{sub_dataset_name}' to project '{top_level_project_name}'.")
        link = omero.model.ProjectDatasetLinkI()
        link.setParent(omero.model.ProjectI(project.getId(), False))
        link.setChild(omero.model.DatasetI(dataset.getId(), False))
        conn.getUpdateService().saveObject(link)

        with open(self.model_save_path, 'rb') as model_file:
            file_size = os.path.getsize(self.model_save_path)

            # create OriginalFile
            omero_file = OriginalFileI()
            omero_file.setName(rstring(os.path.basename(self.model_save_path)))
            omero_file.setPath(rstring(os.path.dirname(self.model_save_path)))
            omero_file.setSize(rtypes.rlong(file_size))
            omero_file.setMimetype(rstring('application/octet-stream'))

            # Upload the model
            original_file = conn.getUpdateService().saveAndReturnObject(omero_file)

            file_ann = FileAnnotationI()
            file_ann.setFile(original_file)
            file_ann.setNs(rstring("omero.namespace.model"))

            # Save the file annotation object
            saved_file_ann = conn.getUpdateService().saveAndReturnObject(file_ann)

            # Ensure that the saved annotation is wrapped in the appropriate gateway object
            wrapped_annotation = conn.getObject("Annotation", saved_file_ann.id.val)

            # After the annotation is created, check and link it to the dataset
            if wrapped_annotation is not None:
                dataset = conn.getObject("Dataset", dataset.id)  # Refresh the dataset object using dataset.id
                dataset.linkAnnotation(wrapped_annotation)
                print("Model file successfully linked to the dataset.")
            else:
                print("Model file annotation could not be created.")

            self._show_success_message(f"Model successfully uploaded to Omero.")
            print(f"Model successfully uploaded to Omero: {self.model_save_path}")
    
    @omero_connect
    def upload_metadata_to_omero(self, model_name, conn=None):

        #model_name = self.select_model.value  # Example: "DenseNet" or "ResNeXt"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        sub_dataset_name = self.model_name_input.value

        # Get project and dataset
        project_name = "CNN_Models"
        project = conn.getObject("Project", attributes={"name": project_name})
        if not project:
            raise Exception(f"Project '{project_name}' does not exist.")
        
                # Step 1: Check or create the top-level project
        project = conn.getObject("Project", attributes={"name": project_name})
        if project is None:
            print(f"Project with the name '{project_name}' not found. Creating a new project.")
            new_project = ProjectI()
            new_project.name = rstring(project_name)
            new_project.description = rstring("Top-level project for CNN models")
            project = conn.getUpdateService().saveAndReturnObject(new_project)

        # Step 2: Create the sub-dataset with the dynamic name
        print(f"Creating sub-dataset with name '{sub_dataset_name}'.")
        new_dataset = DatasetI()
        new_dataset.name = rstring(sub_dataset_name)
        new_dataset.description = rstring(f"Dataset for {model_name} model trained on {timestamp}")
        dataset = conn.getUpdateService().saveAndReturnObject(new_dataset)

        # Step 3: Link the sub-dataset to the project
        print(f"Linking dataset '{sub_dataset_name}' to project '{project_name}'.")
        link = omero.model.ProjectDatasetLinkI()
        link.setParent(omero.model.ProjectI(project.getId(), False))
        link.setChild(omero.model.DatasetI(dataset.getId(), False))
        conn.getUpdateService().saveObject(link)


        dataset = None
        for dset in project.listChildren():
            if dset.getName() == model_name:
                dataset = dset
                break
        if not dataset:
            raise Exception(f"Dataset '{model_name}' does not exist under project '{project_name}'.")

        # Define metadata file path
        meta_data_path = self.dataset_folder / "metadata.json"
        if not meta_data_path.exists():
            raise Exception(f"Metadata file not found at {meta_data_path}.")

        # Upload the metadata file directly
        print(f"Uploading metadata file: {meta_data_path}")
        file_ann = conn.createFileAnnfromLocalFile(
            str(meta_data_path),  # Dosya yolu
            mimetype="application/json",  # MIME türü
            ns="omero.namespace.json",  # Namespace
            desc=f"Metadata file for model: {model_name}"  # Açıklama
        )

        if not file_ann:
            raise Exception("Failed to create FileAnnotation for metadata.")

        # Link the FileAnnotation to the dataset
        dataset.linkAnnotation(file_ann)
        print("Metadata file successfully linked to the dataset.")

        self._show_success_message("Metadata successfully uploaded to Omero.")



    def start_testing(self):
        if not self.test_loader:
            QMessageBox.warning(None, "Warning", "Please load the data before starting testing.")
            return

        self.model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        # Loss function selection
        if self.loss_function_select.value == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss()
        elif self.loss_function_select.value == "FocalLoss":
            criterion = FocalLoss(gamma=2)

        with torch.no_grad():
            for rois, labels in self.test_loader:
                rois, labels = rois.to(self.device), labels.to(self.device)
                outputs = self.model(rois)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_loss /= len(self.test_loader)
        test_accuracy = 100 * correct_test / total_test
        QMessageBox.information(None, "Testing Result", f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        self.test_accuracy_label.value = f"Test Accuracy: {test_accuracy:.2f}%"
