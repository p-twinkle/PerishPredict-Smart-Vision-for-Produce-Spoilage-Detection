# PerishPredict Smart Vision for Produce Spoilage Detection
https://medium.com/@sarahgstephens/perishpredict-smart-vision-for-produce-spoilage-detection-2e34e5af344e

## Team Members

- Advaith Shankar ([advaithshankar](https://github.com/advaithshankar))
- Andrew White  ([andrewwhot](https://github.com/andrewwhot))
- Sarah Stephens ([sarahgstephens](https://github.com/sarahgstephens))
- Twinkle Panda ([p-twinkle](https://github.com/p-twinkle))
- Varsha Manju Jayakumar ([mj-varsha](https://github.com/mj-varsha))

## Problem Statement

Grocery retail faces significant challenges in managing food waste, particularly with perishable items like fresh produce. This project aims to address this issue by leveraging technology to identify spoiled or damaged produce before it reaches store shelves. By utilizing a pre-trained foundation model with image recognition capabilities, we can assess the quality of perishable products, ensuring only the best produce is offered to consumers, thereby enhancing profitability and customer experience.

## Dataset

Our project utilizes a dataset of approximately 30,000 images, including a wide variety of fruits and vegetables in both fresh and spoiled conditions. The dataset, sourced from Kaggle, contains labeled images for balanced and diverse training and evaluation.

### Key Features:
- **Size**: ~30,000 images
- **Categories**: Fresh and spoiled produce
- **Types**: Apples, bananas, tomatoes, cucumbers, and more
- **Labels**: Binary classification (fresh or spoilt)

### Preprocessing Steps:
- Resize and normalize images
- Split into training, testing, and hold-out validation sets

## Approaches

We plan to explore multiple approaches to detect subtle, early signs of spoilage:

1. **Deep CNNs**: Experiment with ResNet50, VGG-16, Inception, and EfficientNet
2. **Contrastive Learning**: Experiment with contrastive learning technique for a supervised problem
3. **Support Vector Machine (SVM)**: As an alternative to CNNs
4. **Open-Source VLMs and Multimodal LLMs**: Utilize APIs for vision-language models
5. **Fine-Tuning Open-Source VLMs**: Experiment with smaller parameter models like BLIP-2 and Llava-Next

## Applications

1. **Quality Control in Food Industry**: Automated systems for sorting fresh produce
2. **Reducing Food Waste**: Early detection of deteriorating produce
3. **Smart Refrigeration Systems**: Alerting users about nearing spoilage
4. **Disease Detection in Crops**: Early indicator of potential disease outbreaks

## Dataset Details

| Fresh / Spoilt | Produce      | No. of Images |
|----------------|--------------|---------------|
| Fresh          | Apple        | 3215          |
| Fresh          | Banana       | 3360          |
| Fresh          | Bitter Gourd | 327           |
| Fresh          | Capsicum     | 1269          |
| Fresh          | Cucumber     | 496           |
| Fresh          | Orange       | 1854          |
| Fresh          | Potato       | 806           |
| Fresh          | Tomato       | 2113          |
| Spoilt         | Apple        | 4236          |
| Spoilt         | Banana       | 3832          |
| Spoilt         | Bitter Gourd | 357           |
| Spoilt         | Capsicum     | 901           |
| Spoilt         | Cucumber     | 676           |
| Spoilt         | Orange       | 1998          |
| Spoilt         | Potato       | 1172          |
| Spoilt         | Tomato       | 2178          |
| Total          |              | 30357         |

Dataset Sources:
1. [Reddy, S. (2021). Fruits Fresh and Rotten for Classification. Kaggle.](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)
2. [Swoyam. (2023). Fresh and Stale Classification. Kaggle[1].](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification)

## References

1. Chin, A. (2022). How I Made AI to Detect Rotten Produce Using a CNN. Medium.
2. Habib. (2023). Binary Image Classification Using ResNet50. Medium.
3. GeeksforGeeks. (2021). Image Classification Using Support Vector Machine (SVM) in Python.
4. Luo, X., et al. (2024). Lift-On-Retrieval: Recycling Intermediate Outputs for Efficient Diffusion Inference. arXiv.
5. Olafenwa, J. (2021). Boost Your Image Classification Model with Pretrained VGG-16. Medium.
6. OpenCompass. Open-VLM Leaderboard. Hugging Face.
7. Qin, Y., et al. (2023). Understanding Diffusion Models: A Unified Perspective. arXiv.
8. TechLabs Hamburg. (2022). FruitShow: Detecting Whether a Fruit Is Still Edible or Rotten. Medium.
9. YoanFanClub. (2023). Prime Number Conspiracy, Explained. YouTube.
10. https://arxiv.org/html/2410.18200v1
11. https://lilianweng.github.io/posts/2021-05-31-contrastive/
12. https://www.yadavsaurabh.com/self-supervised-contrastive-learning-fundamentals/
13. https://learnopencv.com/contrastive-learning-simclr-and-byol-with-code-example/
14. https://blog.roboflow.com/contrastive-learning-machine-learning/
