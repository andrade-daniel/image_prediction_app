# Image classification model with Deep learning

Scripts to train a classification model for prediction of the view of a car (from 'Frontal', 'Frontal right', 'Frontal left', 'Lateral', 'Rear right', 'Rear left', 'Rear') from images.
The resulting model is used in the app (check app folder).

Please note that the dataset used to train the model is not publicly available.

## Getting Started

You will essentially need three scripts:

- To preprocess the images from your dataset in the images folder (when running for the first time, let's say)

```
$ python process.py --input_enq_path=/images_folder --output_path=/output 
```

- To train the model

```
$ python train.py --n_epochs=100 --output_path=/output
```

- To predict, but also to test the model and retrieve metrics for validation

Test:

```
$ python predict.py --mode=test --output_path=/output
```

Prediction on one image:

```
$ python predict.py --mode=prediction --to_predict=one --input_path=/*.jpg --output_path=/output
```

Prediction on many images in one folder:

```
$ python predict.py --mode=prediction --to_predict=many --input_path=/images_folder --output_path=/output
```

Check for more arguments you can use.

### Extra

In the output folder, you can find a Jupyter notebook (stats_model_report.ipynb) that you can easily run to obtain the metrics and plots (option to use callbacks are included in train.py, though) to verify the performance of the model.
Run:
```
$ jupyter notebook
```