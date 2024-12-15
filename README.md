# ASRec: Adaptive Sequential Recommendation with Dynamic and Periodic Preferences Capturing
Official PyTorch Code base for "ASRec: Adaptive Sequential Recommendation with Dynamic and Periodic Preferences Capturing". The code is built on the [RecBole](https://github.com/RUCAIBox/RecBole) library, implemented by [@WLHao0313](https://github.com/WLHao0313).

* The model implementation is at `recbole/model/sequential_recommender/asrec.py`

## Abstract


<p align="center">
  <img src="figs/ASRec.PNG"/>
</p>



## Using the code:
The code is stable while using Python 3.7.12, PyTorch >= 1.13.1.
- Clone this repository:
```bash
git clone https://github.com/WLHao0313/ASRec
cd ASRec
```
- To install all the dependencies using conda:
```
conda create -n asrec python=3.7.12 -y
conda activate asrec
pip install -r requirements.txt
```

## Datasets
1) ml-100k - [Link](https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/MovieLens/ml-100k.zip)
2) ml-1M - [Link](https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/MovieLens/ml-1m.zip)
3) amazon-beauty - [Link](https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/Amazon_ratings/Amazon_Beauty.zip)
4) amazon-sports-outdoors - [Link](https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/Amazon_ratings/Amazon_Sports_and_Outdoors.zip)
5) yelp - [Link](https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/Yelp/yelp.zip)

Following is the statistics of the datasets we use:
<p align="center">
  <img src="figs/datasets.png"/>
</p>

## Training time
On ml-1m dataset:
```
python run_ASRec.py --dataset=ml-1m
```
For other datasets, simply replace "ml-1m" with the dataset name (e.g. ml-100k, amazon-beauty, amazon-sports-outdoors, yelp).

## Case Studies
The model ASRec successfully captures users' periodic and dynamic preference displays:
<p align="center">
  <img src="figs/case.png"/>
</p>
