# Deepfake detection using Deep Learning (ResNext and LSTM)

#### We have dockerised the [Django Application](https://github.com/nikhil150821/Deep_fack_Detection/tree/main/Django%20App) now you can spin up a container within seconds without worring about dependencies

S
## 1. Introduction
This projects aims in detection of video deepfakes using deep learning techniques like ResNext and LSTM. We have achived deepfake detection by using transfer learning where the pretrained ResNext CNN is used to obtain a feature vector, further the LSTM layer is trained using the features. For more details follow the [documentaion]

You can also watch [this Youtube video](https://www.youtube.com/watch?v=_q16aJTXVRE) to get a better intuition about the project.
You can watch [this playList](https://www.youtube.com/watch?v=quJ8Rv84oA0&list=PLNIj0dkfMA1FsD5xR4IEc8vdwr66_WExl) for step by step installation.


## 2. Directory Structure
For ease of understanding the project is structured in below format
```
Deepfake_detection_using_deep_learning
    |
    |--- Django Application
    |--- Model Creation
    |--- Documentaion
```
1. Django Application 
   - This directory consists of the django made application of our work. Where a user can upload the video and submit it to the model for prediction. The trained model performs the prediction and the result is displayed on the screen.
2. Model Creation
   - This directory consists of the step by step process of creating and training a deepfake detection model using our approach.
3. Documentation
   - This directory consists of all the documentation done during the project
   
## 3. System Architecture
<p align="center">
  <img src="https://github.com/nikhil150821/Deep_fack_Detection/blob/main/github_assets/System%20Architecture.png" />
</p>

## 5. Our Results

| Model Name | No of videos | No of Frames | Accuracy |
|------------|--------------|--------------|----------|
|model_84_acc_10_frames_final_data.pt |6000 |10 |84.21461|
|model_87_acc_20_frames_final_data.pt | 6000 |20 |87.79160|
|model_89_acc_40_frames_final_data.pt | 6000| 40 |89.34681|
|model_90_acc_60_frames_final_data.pt | 6000| 60 |90.59097 |
|model_91_acc_80_frames_final_data.pt | 6000 | 80 | 91.49818 |
|model_93_acc_100_frames_final_data.pt| 6000 | 100 | 93.58794|

## 6. Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<!-- <table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://www.linkedin.com/in/abhijitjadhav1998/"><img src="https://avatars.githubusercontent.com/u/38549908?v=4?s=100" width="100px;" alt="Abhijit Jadhav"/><br /><sub><b>Abhijit Jadhav</b></sub></a><br /><a href="#projectManagement-abhijitjadhav1998" title="Project Management">📆</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://vthonte.vercel.app/"><img src="https://avatars.githubusercontent.com/u/43621438?v=4?s=100" width="100px;" alt="Vishwesh Thonte"/><br /><sub><b>Vishwesh Thonte</b></sub></a><br /><a href="#maintenance-vthonte" title="Maintenance">🚧</a></td>
    </tr>
  </tbody>
</table> -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->


## 8. We welcome Open Source Contribution. 
### Below are the some changes that can be applied to the project. New Ideas will be appreciated.
- [ ] Deploying the applications in free cloud 
- [ ] Creating open source API for detection
- [ ] Batch processing of entire video instead of processing first 'x' frames.
- [ ] Optimizing the code for faster execution.
#### Completed 
- [X] Dockerizing the app
- [X] Enabling working of project on Non Cuda Computers. i.e on normal or AMD GPUs