# DSA-Thrombus-Classifier
PoC tool to use CNNs to classify digital subtraction angiography images of the brain into thrombus-free and non-thrombus-free.

# ThromboMapUI

A demonstrator for testing AI-based thrombus detection in DSA images.

## Include:
You need to include the SimpleITKNative.dll and SimpleITKManaged.dll in the ThrombomapUI/ThrombomapUI folder in order to be able to compile it.

## Update:
In order to update the submodule Python-Server, simply run
```
git submodule update --remote
```

Important:
When shipping a new version, make sure, that you do exclude the settings file! (Or tell the user how to migrate)
