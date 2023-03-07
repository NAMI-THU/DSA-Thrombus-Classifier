# DSA-Thrombus-Classifier
PoC demonstration tool to use CNNs to classify digital subtraction angiography images of the brain into thrombus-free and non-thrombus-free.
User application which uses [Mittmann et al.](https://pubmed.ncbi.nlm.nih.gov/35604489/)'s classifier.
We describe this application in our paper [TODO].

***This tool only represents a proof of concept and is no medical product!***

## Getting started
Before the first start, you need to setup the python service. For this, execute `Backend/setup.bat` . This might take a while.
In the meanwhile, you can download the pretrained model from our project page in the [OpenScienceFramework](https://osf.io/n8k4r/). Make sure to keep the structure of the folder.

To use the application, start the python backend using `Backend/RunServer.bat` . Afterwards, open the frontend application using `UI/Demonstrator.exe` .
You first need to select the model you want to use. For this, select the directory of your models, in which the two subdirectories, frontal and lateral, are contained. Once this is selected, the application will remember this at every start until you change it again. Afterwards, you are ready to go. For this, select the frontal and lateral image respectively. We support nifti as well as DICOM, however, for the ladder, you first need to convert the file using the conversion-button. Once both files are selected, the application displays a mid-sequence preview. You are then ready to run the classification using the respective robot-button. 

You might want to change the threshold that is used to determine whether a classification counts as thrombus-free or non-thrombus-free. By default, we recommend using a threshold of `0.57`, which is done by optimizing a trade-off between the different metrics. 

## Settings
Before the first use, you have to setup your `appsettings.json` file.
| Option      | Description |
| ----------- | ----------- |
| `AiServiceUrl` | The address and port where the python service serves at. When deploying the python service on the same machine, the default address is fine. |
| `PlastimatchPath` | The path to the [plastimatch](https://plastimatch.org/) executable. If no valid path is given (default), the application will look in the working directory for the executable. |
| `ModelsEvaluationDirectory` | The path to the test results which are used to calculate the metrics. By default, the application will look in the in the working directory for a folder named TestResults. |
| `EnableEvaluationSetup` | Use this option to log the outputs of the model. |
| `LogPath` | In case `EnableEvaluationSetup` is set to true, use this option to specify the folder where the results are written to. By default, the working directory will be used. |

***Important:***
***When shipping a new release, make sure that you do exclude the settings file! (Or tell the user how to migrate)
Otherwise, the configuration will be lost.***

## Building
You need to include [SimpleITKNative.dll and SimpleITKManaged.dll](https://github.com/SimpleITK/SimpleITK/releases) in the UI folder in order to be able to compile it.

## Attribution
Icon made by Freepik from www.flaticon.com
