import json
import logging
from dataclasses import asdict
from pathlib import Path

from magicgui import magicgui
from magicgui.widgets import Container, Label
from qtpy.QtWidgets import QMessageBox

from omero_screen_napari.gallery_userdata import UserData
from omero_screen_napari.gallery_userdata_singleton import (
    userdata as global_user_data,
)
from omero_screen_napari.omero_data import OmeroData
from omero_screen_napari.omero_data_singleton import omero_data
from omero_screen_napari.utils import omero_connect

from omero.gateway import BlitzGateway
from omero.model import ProjectI, DatasetI, ProjectDatasetLinkI
import omero.rtypes as rtypes
from omero.model import FileAnnotationI, OriginalFileI, FilesetEntryI
import os

logger = logging.getLogger("omero-screen-napari")
logging.basicConfig(level=logging.DEBUG)



def setup_training_widget(
    class_options: list[str] | None = None,
    class_name: str | None = None,
    user_data: UserData | None = global_user_data,
) -> Container:
    widget = SetupTrainingWidget(class_options, class_name, user_data, omero_data)
    return widget.container


class ImageNavigator:
    def __init__(self, class_options: list | None):
        self.class_options = ["unassigned"] if class_options is None else class_options
        self.class_labels = Container(widgets=[
            Label(value=class_name) for class_name in self.class_options
        ])

    def add_class(self, class_name: str):
        if class_name and class_name not in self.class_options:
            self.class_options.append(class_name)
            self.refresh_class_labels()
            logger.debug(f"Class {class_name} added to choices.")

    def reset_class_options(self):
        self.class_options = ["unassigned"]
        self.refresh_class_labels()
        logger.debug("Class choices reset to default.")


    def refresh_class_labels(self):
        self.class_labels.clear()
        for class_name in self.class_options:
            self.class_labels.append(Label(value=class_name))


class SetupTrainingWidget:
    def __init__(
        self,
        class_options: list[str] | None,
        class_name: str | None,
        user_data: UserData | None,
        omero_data: OmeroData,
    ):
        self.image_navigator = ImageNavigator(class_options)
        self.user_data = user_data
        self.omero_data = omero_data
        self.class_name = class_name or "Classifier Name"
        self.meta_data_saver = MetaDataSaver(
            self.class_name,
            self.omero_data,
            self.user_data,
            self.image_navigator,
        )
        self.add_class = magicgui(
            call_button="Enter", text_input={"label": "Class name"}
        )(self.add_class)
        self.reset_class_options = magicgui(call_button="Reset class options")(
            self.reset_class_options
        )
        self.save_meta_data = magicgui(
            call_button="Save metadata",
            text_input={"label": "filename", "value": self.class_name},
        )(self.save_meta_data)

        self.container = self.create_container()


    def add_class(self, text_input: str):
        if text_input:
            self.image_navigator.add_class(text_input)
            self.add_class.text_input.value = ""

    def reset_class_options(self):
        self.image_navigator.reset_class_options()

    def save_meta_data(self, text_input: str):
        if new_classifier_name := text_input.strip():
            self.meta_data_saver._update_classifier_name(new_classifier_name)
        self.meta_data_saver.save_data()


    def create_container(self):
        return Container(
            widgets=[
                self.add_class,
                self.reset_class_options,
                self.image_navigator.class_labels,
                self.save_meta_data,
            ]
        )
class MetaDataSaver:
    def __init__(
        self,
        classifier_name: str,
        omero_data: OmeroData,
        user_data: UserData,
        image_navigator: ImageNavigator,
    ):
        self.classifier_name = classifier_name
        self.omero_data = omero_data
        self.user_data = user_data
        self.image_navigator = image_navigator
        self.home_dir = Path.home() / "omeroscreen_trainingdata"
        self.classifier_dir = self.home_dir / self.classifier_name
        self.meta_data_path = self._set_paths()
        self.metadata = self._create_metadata_dict()
        self._update_paths_and_metadata()

    def save_data(self):
        try:
            self._validate_classifier_name(self.classifier_name)
            if not self.classifier_dir.exists():
                self._create_and_save()
                return

            file_check = self._check_directory_contents()
            self._handle_saving_logic(file_check)
        except Exception as e:  # noqa: BLE001
            logger.error(e)
            self._show_error_message(str(e))
        
    @omero_connect
    def save_data_to_omero(self, conn=None):
        # Proje ve dataset bilgileri
        project_id = 3  # Mevcut proje ID'si
        dataset_name = self.classifier_name+"_Dataset"

        # Check if a dataset with the given name already exists in the project
        dataset = self.get_existing_dataset_in_project(conn, project_id, dataset_name)
        
        if dataset:
            print(f"Dataset already exists: {dataset_name}. Uploading file to the existing dataset.")
        else:
            # Create a new dataset if it doesn't already exist
            dataset = self.create_dataset_in_project(conn, project_id, dataset_name)
            print(f"New dataset created: {dataset_name}")

        # JSON dosyasını dataset'e yükleme
        json_file_path = self.meta_data_path
        self.upload_json_to_dataset(conn, dataset, json_file_path)

        # Check if the link between the project and dataset already exists
        project = conn.getObject("Project", project_id)
        existing_link = False

        for link in project.listChildren():
            if link.getId() == dataset.getId():
                existing_link = True
                break

        if not existing_link:
            link = ProjectDatasetLinkI()
            link.setParent(ProjectI(project_id, False))
            link.setChild(DatasetI(dataset.getId(), False))
            saved_link = conn.getUpdateService().saveObject(link)
            print("New link between project and dataset created.")
        else:
            print("The link between the project and dataset already exists. No new link was created.")

    # Function to check if a dataset with the specified name already exists in the project
    def get_existing_dataset_in_project(self, conn, project_id, dataset_name):
        project = conn.getObject("Project", project_id)
        if project is None:
            raise ValueError(f"No Project Found: {project_id}")

        # Look for the dataset with the specified name within the project
        for dataset in project.listChildren():
            if dataset.getName() == dataset_name:
                return dataset

        # Return None if no matching dataset is found
        return None
    

    # Proje ID'sini alarak dataset oluşturma
    def create_dataset_in_project(self, conn, project_id, dataset_name):
        project = conn.getObject("Project", project_id)
        if project is None:
            raise ValueError(f"No Project Found: {project_id}")
        
        # Yeni dataset oluştur
        dataset = DatasetI()
        dataset.setName(rtypes.rstring(dataset_name))  # omero.rtypes.rstring kullanımı
        dataset = conn.getUpdateService().saveAndReturnObject(dataset)
        
        return dataset

    # Dosyayı OMERO'ya yükleme fonksiyonu
    def upload_json_to_dataset(self, conn, dataset, json_file_path):

        # Check for existing annotations and delete if a file with the same name exists
        existing_annotations = list(dataset.listAnnotations(ns='omero.namespace.json'))
        for annotation in existing_annotations:
            linked_file = annotation.getFile()
            if linked_file.getName() == os.path.basename(json_file_path):
                print(f"Existing file {linked_file.getName()} found. Deleting it before uploading the new file.")
                conn.deleteObjects("Annotation", [annotation.getId()], deleteAnns=True)
                print(f"Existing file {linked_file.getName()} has been deleted.")

        with open(json_file_path, 'rb') as json_file:
            file_size = os.path.getsize(json_file_path)

            # OriginalFile oluştur
            omero_file = OriginalFileI()
            omero_file.setName(rtypes.rstring(os.path.basename(json_file_path)))
            omero_file.setPath(rtypes.rstring(os.path.dirname(json_file_path)))
            omero_file.setSize(rtypes.rlong(file_size))
            omero_file.setMimetype(rtypes.rstring('application/json'))

            # Dosyayı sunucuya yükleyin
            original_file = conn.getUpdateService().saveAndReturnObject(omero_file)
            
            # Dosyayı anotasyon olarak dataset'e ekleyin
            file_ann = FileAnnotationI()
            file_ann.setFile(original_file)
            file_ann.setNs(rtypes.rstring("omero.namespace.json"))

            # Save the file annotation object
            saved_file_ann = conn.getUpdateService().saveAndReturnObject(file_ann)

            # Ensure that the saved annotation is wrapped in the appropriate gateway object
            wrapped_annotation = conn.getObject("Annotation", saved_file_ann.id.val)

            # After the annotation is created, check and link it to the dataset
            if wrapped_annotation is not None:
                dataset = conn.getObject("Dataset", dataset.id)  # Refresh the dataset object using dataset.id
                dataset.linkAnnotation(wrapped_annotation)
                print("File successfully linked to the dataset.")
            else:
                print("File annotation could not be created.")
            
            # Dataset ile ilişkilendir
            # conn.getUpdateService().saveObject(dataset)
            # dataset.linkAnnotation(file_ann)
            # dataset = conn.getObject("Dataset", dataset.id.val)  # Dataset'i tekrar yükleyerek güncelle
            self._show_success_message(f"Metadata successfully saved to Omero.")
            print(f"JSON dosyası başarıyla dataset'e yüklendi: {json_file_path}")


    def _update_classifier_name(self, new_classifier_name: str):
        self.classifier_name = new_classifier_name
        self._update_paths_and_metadata()

    def _update_paths_and_metadata(self):
        self.classifier_dir = self.home_dir / self.classifier_name
        self.meta_data_path = self._set_paths()
        self.metadata = self._create_metadata_dict()

    def _create_and_save(self):
        self._create_directory(self.classifier_dir)
        self._save_metadata()

    def _check_directory_contents(self):
        logger.info(f"Directory {self.classifier_dir} already exists.")
        contents = list(self.classifier_dir.iterdir())
        return self._check_files(contents)

    def _handle_saving_logic(self, file_check):
        if file_check == "no_data":
            self._save_metadata()
        elif file_check == "metadata":
            if self._compare_metadata():
                if self._show_confirmation_dialog("Old metadata already present. Do you want to overwrite it anyway? Nothing will change"):
                    self._save_metadata()
            elif self._show_confirmation_dialog("Old metadata present but different. Do you want to change them?"):
                self._save_metadata()
        else:
            logger.error(f"Problem with file check in {self.classifier_dir}")
            raise ValueError(f"Problem with file check in {self.classifier_dir}")

    def _handle_metadata_present(self):
        if self._compare_metadata():
            self._save_training_data()
        elif self._show_confirmation_dialog(
            "Metadata has changed. Do you want to overwrite them and save {self.file_name}?"
        ):
            self.save_metadata()

    def _save_metadata(self):
        with self.meta_data_path.open("w") as json_file:
            json.dump(self.metadata, json_file)
        logger.info(f"Training Metadata saved to: {self.classifier_dir}")
        self._show_success_message(f"Metadata successfully saved to : {self.classifier_dir}.")
        self.save_data_to_omero()

    def _set_paths(self):
        return self.classifier_dir / "metadata.json"

    def _check_files(self, contents):
        has_metadata = any(file.name == "metadata.json" for file in contents)
        return "metadata" if has_metadata else "no_data"

    def _compare_metadata(self) -> bool:
        with self.meta_data_path.open("r") as json_file:
            existing_metadata = json.load(json_file)
        return existing_metadata == self.metadata

    def _validate_classifier_name(self, text_input: str):
        if not text_input.strip():
            logger.error("No classifier name provided.")
            raise ValueError("Failed to create directory: no classifier name provided.")

    def _create_directory(self, directory: Path):
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            raise ValueError(f"Failed to create directory {directory}: {e}") from e


    def _create_metadata_dict(self) -> dict:
        user_data_dict = asdict(self.user_data)
        user_data_dict.pop("well", None)
        return {
            "user_data": user_data_dict,
            "class_options": self.image_navigator.class_options,
        }

    def _show_error_message(self, message: str):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText(message)
        msg_box.setWindowTitle("Error")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def _show_success_message(self, message: str):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText(message)
        msg_box.setWindowTitle("Success")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def _show_confirmation_dialog(self, message: str) -> bool:
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText(message)
        msg_box.setWindowTitle('Warning')
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.No)
        reply = msg_box.exec_()
        return reply == QMessageBox.Yes
