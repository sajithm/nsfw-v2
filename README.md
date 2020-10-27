# nsfw-v2
An NSFW detector serving responses over REST API developed using Keras (Tensorflow) and Flask in Python

## Introduction
A SFW/NSFW detector developed using Keras. It uses 3 sets of 2 convolution layers followed by a MaxPool. This is followed by a fully connected layer and SoftMax at the end.

The prediction is provided over REST API using Flask.

## Usage

```shell
curl -X POST -F image=@path_to_image.jpeg 'http://localhost:5000/predict'
```
Sample Response
```son
{  
   "is_safe":true,
   "predictions":{  
      "nsfw-nude":0.0003361131530255079,
      "nsfw-risque":0.2868056893348694,
      "nsfw-sex":0.008736947551369667,
      "nsfw-violence":0.06439296156167984,
      "sfw":0.6397283673286438
   },
   "success":true
}
```

## Dataset
The dataset is not included in the repository. ~It can be downloaded from [here](https://www.dropbox.com/s/opiqoh550jd1glb/dataset.zip?dl=0).~

The test/train images were resized to fit in 0.3MP (640x480 or less). Data is split into 5 categories: SFW, NSFW-Nude, NSFW-Sex, NSFW-Risque and NSFW-Violence. Each category has 5500 images - 5000 under training and 500 under testing sets.

Binary model is included in the repo as a ZIP archive. It is strongly recommended that you train the network with a larger datase.
