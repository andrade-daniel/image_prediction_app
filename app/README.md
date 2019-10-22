# An app for predicting a car view/frame

A simple app (I really recommend the streamlit package for quick-to-the-point light apps) where you can paste a image url of a car and see the prediction of the view (from 'Frontal', 'Frontal right', 'Frontal left', 'Lateral', 'Rear right', 'Rear left', 'Rear').


A screenshot of what you'll be able to see.

![alt text](https://raw.githubusercontent.com/andrade-daniel/image_prediction_app/master/app/image.png)

## Getting Started

After the necessary installations, you simply run:

```
$ streamlit run app.py
```

Try by pasting different url of images of cars (please note that the model shared was trained with only 100 epochs on pics that have similar perspectives/framing to the ones in the Carvana dataset).
