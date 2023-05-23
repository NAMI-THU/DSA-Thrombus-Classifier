# Installation and usage of the DSA-Thrombus-Classifier
<https://github.com/NAMI-THU/DSA-Thrombus-Classifier>


## Setup

This section describes the steps that need to be taken in order to
install the application.

### Prerequisites

You will first have to install the following software. In most cases,
you can and should upgrade to the most current minor version in order to
profit from vulnerability patches and bugfixes. However, if you aim to
reproduce the results published in our paper, you should stick with the
same versions that are mentioned here in order to ensure to have the
same environment.

-   Python 3.10.5  
    <https://www.python.org/>

-   .NET 7.0 Desktop Runtime  
    <https://dotnet.microsoft.com/en-us/download/dotnet/7.0/runtime>

#### Optional steps for GPU support

If your computer is equipped with a NVIDIA graphics card and you wish to
utilize it in order to gain a significant speedup for the
classification, you will need to install CUDA along with your GPU
driver. The version we used in our paper was

-   NVIDIA GPU driver 526.98  
    <https://www.nvidia.com/Download/index.aspx>

-   NVIDIA CUDA Toolkit 11.7.1  
    <https://developer.nvidia.com/cuda-downloads>

If you further wish to train your own model, it is additionally
necessary to install cuDNN.

-   cuDNN library  
    <https://developer.nvidia.com/cudnn>

### Download of the application

In order to run the application, you will need two components. First,
the application itself along with the backend service. Second, you will
need a trained model to run the classification.

In order to get the application, choose from the following options.

-   If you like to simply use the application, download the latest
    application package from our Github page.  
    <https://github.com/NAMI-THU/DSA-Thrombus-Classifier/releases>

-   For reproducing the results of the paper, download the package found
    in the openscience framework or checkout the branch
    `2023-bvm-paper-reproducability`.  
    <https://osf.io/n8k4r/>

-   If you like to build the application yourself, checkout the github
    repository.  
    <https://github.com/NAMI-THU/DSA-Thrombus-Classifier>

In order to run the classification, download our pretrained models from
the openscience framework.

-   Trained models  
    <https://osf.io/n8k4r/>
	
-	Select
	`Demonstrator/Models.zip`

Extract the zip-archive file and place the directory in a place of your
choice. Please ensure to not change the internal structure of the
directory, as it has to contain a `lateral` and one `frontal`
subdirectory.

### Installation of the backend service

The application consists of two parts. One frontend module, which
provides an interface to the user. However, the classification itself is
done in a backend service, which runs with python. Before the first use,
you have to setup this backend service. For this:

1.  Navigate into the `Python-Server` directory

2.  Execute `setup.bat`

Depending on your hardware and internet connection, this might take a
while. All relevant packages are being downloaded and installed. After
all operations completed successfully, you are ready to go.

## Usage

### Selection of the model

To use the application, you first need to start the backend service and
afterwards launch the graphical user interface. For this simply run the `StartApplication.bat` in the root directory.
Alternatively, follow the following steps:

1.  Navigate into the `Python-Server` directory

2.  Execute `RunServer.bat`

3.  Navigate back and into the `UI` directory

4.  Execute `UI.exe`

Initially, you will need to select the folder with the pretrained
models. For this, select the `Select Model Folder`-Button and select the extracted folder that contains a
frontal and lateral subdirectory. The application will remember this
selection and automatically load the models at each start. However, if
you wish to change the model, you can always do so be reselecting the
directory.

### Loading images

Use the `Browse`-Buttons to select the
frontal- and lateral sequence view of your case. We currently support
nifti files, however, you can also load DICOM sequences and
automatically convert them with the respective button. Once both
sequences are selected, the application displays a preview image from
the middle of each sequence. The images are now automatically preloaded
in the backend.

### Classification

Once a model and both image sequences are selected, the round centered
classification button will be enabled. Use it to trigger the prediction
of the model. On the bottom, you will see the raw output of each model
(frontal and lateral). The mean of both are used to make a decision. If
the value is below the threshold, the case will be labeled as "no
thrombus detected", if it is above, it will be interpreted as detected
thrombus. Please be aware that this should not be used as medical
diagnosis, and only be interpreted as possible hint to trigger a medical
reassessment in case of predicted thrombi.

### Settings

In order to change the sensitivity of the system, it is possible to
change the threshold at which a classification is interpreted either
ways. For this, simply adjust the threshold by changing its value with
the respective slider. Expand the `metrics`-box to gain insight about the
expected behavior of the classifier.

The application supports the adjustment of additional settings. For
this, you need to edit the `appsettings.json` file in the UI directory.

|         **Option**          |             **Default behavior**              | **Description**                                                                                             |
|:---------------------------:|:---------------------------------------------:|:------------------------------------------------------------------------------------------------------------|
|       `AiServiceUrl`        |                      \-                       | The address of the backend service. Change this, if you run it on an external computer or a different port. |
|      `PlastimatchPath`      | 			`UI/plastimatch` 				  | The path to the plastimatch executable                                                                      |
| `ModelsEvaluationDirectory` | 			`UI/TestResults` 				  | The path to the results of the evaluation, which are used to estimate the metrics.                          |
|   `EnableEvaluationSetup`   |                      \-                       | If set to true, the application logs all predictions.                                                       |
|          `LogPath`          |               Working Directory               | The path where the logfiles should be created.                                                              |