# X-Ray-Pneumonia-Image-Classification

Yuhkai Lin & Peter Burton

Presentation: https://docs.google.com/presentation/d/1T57NZcnFOQudURk4JyrwX-yk_2hQXYutJAhIlV2DprA/edit#slide=id.p

## Background & Business Problem

Pneumonia is responsible for more childhood deaths than any other infectious disease worldwide, and early diagnosis and treatment are critical for short-term and long-term health outcomes for patients. One of the most common and cost effective means of diagnosing pneuomonia is through chest X-rays. 
We are given the task of using X-Ray data from pediatric patients to create a model that can assist medical professionals in diagnosing and evaluating X-rays. 
Correctly identifying and diagnosing Pneumonia and beginning treatment sooner can lead to faster recovery and better patient results. 
Providing a “second opinion” to medical professionals after initial diagnosis can help prevent patients from being misdiagnosed and untreated. 

## Dataset

Dataset from Mendeley: https://data.mendeley.com/datasets/rscbjbr9sj/2
![image](Images/Xraynormal.png)
![image](ImagesXraypne.png)

We are working from a dataset of  5,863 X-Ray images in two categories (Pneumonia/Normal). The Chest X-ray images were selected from pediatric patients ages one to five years old from Guangzhou Women and Children’s Medical Center. 
The dataset of images was then screened and graded by two expert physicians and then verified by a third party for accuracy. 
The dataset included a Train, Test, and Validation folder. Due to the low level of images in the validation folder, we created a train/test split from the Train data to create our own Train and Validation Images. 

## Modeling

We preprocessed images for modeling by using ImageDataGenerator to process images, dividing each image by 255 to adjust for greyscale, converting the image size to 64x64, and augmenting the data. 
We will use Convolutional Neural Network(CNN) algorithms to process and learn from thousands of images and predict from new X-rays if a patient has pneumonia 
Key metrics in our model:
•	Accuracy: Reliable predictions help medical professionals save time
•	Recall: Ensuring that we do not have any false negatives, since these could result in poor health outcomes
We will iterate through different models and choose a model that is able to give us the best results based around those parameters. 

## Dummy Classifier

Our first model is a dummy predictor model, that predicts based on the most frequent class(in this case pneumonia). The accuracy of this model is 74.3%. Since Dummy Classifier always chooses the most common class, it has a recall of 100%. 
![Dummy](Images/DummyConfusionMatrix.png)

## Baseline Model

Our next model was a baseline model using a Convolutional Neural Network(CNN). We created a basic CNN with only dense layers.  Compared to the dummy model, the baseline model was substantially better, with an accuracy of 89% and a recall of 90%. 
![AccuracyLine](Images/AccuracyOverEpochs1.png)
![RecallLine](Images/RecallOverEpochs1.png)

## Further CNN Models and VGG19


## Results

![AccuracyBar](Images/AccuracyBar.png)
![RecallBar](Images/RecallBar.png)

When we compared each of our models, we decided to select CNN Model #8, based on its extremely high recall score on both test and validation data. Prioritizing recall will ensure that our model misclassifies as few patients as possible that are positive for Pneumonia. In fact, out of the 390 Pnuemonia cases in our test set of unseen data, it correctly diagnosed 386, a False-Negative rate of just 1%. 
