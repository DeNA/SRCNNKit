# SRCNNKit
Implementation of Super Resolution (SR) with CoreML and Swift. You can use SR method in your app using SRCNNKit UIImageView extension.

For details, see the following presentaion:

https://www.tryswift.co/events/2018/tokyo/en/#coreml

**This project dosen't contain *.mlmodel yet. You should train your own model and import `SRCNN.mlmodel` to your project.**


## iOS

### Usage

```swift
import SRCNNKit

let imageView: UIImageView = ...
let image: UIImage = ...

imageView.setSRImage(image)
```

### Install
- Copy sources to your project.
- CocoaPods and Carthage will be supported soon.

### Requirements
- iOS11
- Xcode9.x

## Development SRCNNKit

### Setup
```shell
git submodule init
git submodule update
```

## Train Your own model

### Requirements
- Python 3.0+
- see `script/packages.txt`

### Convert Training Data

```shell
cd script
python3 convert.py <original train image dir> <train data dir>
python3 convert.py <original validation image dir> <validation data dir>
```

### Training
```shell
python3 train.py <tf log dir> <model output dir> <train data dir> <validation data dir>

```
### Plot Model Image
```shell
python plot.py <.h5 model path> <output dir>
```

### Convert Keras to CoreML Model
```shell
python3 coreml_convert.py <h5 mode path> <output dir>
```

### Validate CoreML Model
```shell
python3 coreml_predict.py <mlmodel path> <input patch image path> <output patch image path>
```

## Dependencies
https://github.com/hollance/CoreMLHelpers

## Licence
SRCNNKit is released under the MIT license. See [LICENSE](TBD) for details.

Copyright © 2018 DeNA Co., Ltd. All rights reserved.
