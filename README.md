
-----

# Deep Learning and Reinforcement Learning Assignment

-----

## 📋 Student Information

| Field | Details |
|-------|---------|
| **Name** | K Venkata Vikram Reddy |
| **USN** | 1CD22AI027 |
| **Semester** | 7th |
| **Department** | Artificial Intelligence and Machine Learning (AIML) |
| **Subject** | Deep Learning and Reinforcement Learning |
| **Course Code** | BAI701 |
| **Academic Year** | 2025-2026 |

-----

## 📋 Table of Contents

1.  [AlexNet.py](https://www.google.com/search?q=%231-alexnetpy)
2.  [Classification.ipynb](https://www.google.com/search?q=%232-classificationipynb)
3.  [DeepReinforcementLearning.py](https://www.google.com/search?q=%233-deepreinforcementlearningpy)
4.  [LSTM.py](https://www.google.com/search?q=%234-lstmpy)
5.  [Rnn.py](https://www.google.com/search?q=%235-rnnpy)
6.  [TicTacToe.py](https://www.google.com/search?q=%236-tictactoepy)
7.  [Installation & Requirements](https://www.google.com/search?q=%23installation--requirements)
8.  [README.md](https://www.google.com/search?q=%23running-the-scripts)

-----

## 1\. AlexNet.py

### Original Implementation

  - Only defined the class structure.
  - No execution pipeline.
  - Relied on massive external datasets (ImageNet) which caused download issues.

**Original Output:**

<img width="791" height="694" alt="image" src="https://github.com/user-attachments/assets/d3544c04-ed4d-4f60-b66e-060609df7b98" />

### Improvements Made

#### **Architecture Enhancements**

  - ✅ **Operational Execution**: Added `model.compile` and `model.fit` loops to make the code fully executable.
  - ✅ **Hyperparameter Tuning**: Adjusted `Dropout` rate from 0.5 to **0.4** and reduced output classes from 1000 to **10** for rapid convergence during demos.

#### **Dataset & Training**

  - ✅ **Synthetic Data Generation**: Implemented NumPy-based dummy data generation. This eliminates the need to download 100GB+ datasets just to verify architecture functionality.
  - ✅ **Memory Optimization**: Adjusted batch sizes to run smoothly on standard lab hardware.

#### **Why These Changes?**

  - **Portability**: The code can now be run instantly on any machine without internet access or huge downloads.
  - **Demonstration**: Proves the network architecture works mathematically without the overhead of real data processing.

**Modified Output:**

<img width="794" height="767" alt="image" src="https://github.com/user-attachments/assets/4ef05d4a-4123-4453-85e2-87e41fa60401" />

-----

## 2\. Classification.ipynb

### Original Implementation

  - Basic Binary Classification (Cats vs Dogs).
  - Simple architecture without persistence.

### Improvements Made

#### **Architecture & Data Enhancements**

  - ✅ **Multi-Class Scaling**: Switched from binary classification to a complex **100-class Butterfly dataset**.
  - ✅ **Data Augmentation**: Implemented rotation, zooming, shearing, and flipping using `ImageDataGenerator` to prevent overfitting on the training data.
  - ✅ **Model Persistence**: Added `model.save()` functionality to persist the trained model (`model.h5`) and class indices (`.json`) for future inference.

#### **Visualization**

  - ✅ Prediction confidence scores.
  - ✅ Visual comparison of ground truth vs. predicted labels.

#### **Why These Changes?**

  - **Generalization**: Proves the model can handle complex, multi-class problems.
  - **Robustness**: Augmentation ensures the model learns features, not just memorizing pixels.

**Outputs:**

<img width="340" height="280" alt="image" src="https://github.com/user-attachments/assets/a397e213-b3be-4fb0-bd81-fd76b211717a" />
<img width="340" height="280" alt="image" src="https://github.com/user-attachments/assets/0da805fe-8944-4146-8702-71f7d410c083" />
<img width="340" height="280" alt="image" src="https://github.com/user-attachments/assets/e6a2be1b-ff24-4ce1-8ddd-2ea9da484fd6" />
<img width="340" height="280" alt="image" src="https://github.com/user-attachments/assets/4cfb1084-fb4e-41cc-aced-695189fc5edd" />


-----

## 3\. DeepReinforcementLearning.py

### Original Implementation

  - Fixed Graph structure.
  - Hardcoded goal (Node 10).
  - Static Q-Learning pathfinding.

**Original Output:**

<img width="340" height="280" alt="deepreinforcement learning" src="https://github.com/user-attachments/assets/cf2f0b2c-2de0-4b1d-8087-14433e982481" />

<img width="340" height="280" alt="deepreinforcement learning2" src="https://github.com/user-attachments/assets/c4a58ad0-e15f-49e1-af15-d223583f99b0" />

<img width="340" height="280" alt="deepreinforcement learning3" src="https://github.com/user-attachments/assets/a6735d1e-c7ce-401d-a426-8c8d7a3185a5" />

<img width="340" height="280" alt="deepreinforcement learning4" src="https://github.com/user-attachments/assets/262cc3fb-5693-414c-8ca4-ef7ca4ecb2ed" />


### Improvements Made

#### **Environment Engineering**

  - ✅ **Topology Modification**: Added a new "shortcut" edge between **Node 2 and Node 7**.
  - ✅ **Objective Change**: Changed the Goal Node from Node 10 to **Node 7**.

#### **Algorithm & Visualization**

  - ✅ **Dynamic Learning**: The Q-Learning agent successfully discovered the new shortcut `(2 -> 7)` instead of taking the long route.
  - ✅ **Enhanced Plotting**: Added descriptive titles and clearer node visualization to track the agent's logic.

#### **Why These Changes?**

  - **Proof of Learning**: By changing the map and the goal, we prove the agent is actively learning the environment rather than following a hardcoded path.
  - **Path Optimization**: Demonstrates the algorithm's ability to find the most efficient route.

**Modified Output:**

<img width="340" height="280" alt="modified Drl" src="https://github.com/user-attachments/assets/d4f18f2d-e70e-47e0-b80c-e6557a5e1fc0" />

<img width="340" height="280" alt="modified Drl2" src="https://github.com/user-attachments/assets/add277ff-d04e-405e-962b-0e9ac7098cad" />

<img width="558" height="122" alt="image" src="https://github.com/user-attachments/assets/cd42961f-32d2-4e24-abb6-ce0810398076" />


-----

## 4\. LSTM.py

### Original Implementation

  - Single LSTM layer (10 units).
  - Hardcoded Windows file path causing crashes.
  - Short look-back window (10 steps).

**Original Output:**

<img width="340" height="280" alt="op lstm" src="https://github.com/user-attachments/assets/088f08fa-b94b-4566-880f-80ad47c282d8" />
<img width="340" height="280" alt="op lstm2" src="https://github.com/user-attachments/assets/c958cbd1-f358-4913-b467-03a0213824a6" />
<img width="1211" height="425" alt="image" src="https://github.com/user-attachments/assets/5f78bb9e-dc85-43f4-a285-9dc1c9da8f8b" />

### Improvements Made

#### **Architecture & Data Enhancements**

  - ✅ **Synthetic Data**: Implemented Sine Wave generation. This ensures the code runs on any computer without needing specific CSV files in specific directories.
  - ✅ **Architecture Upgrade**: Increased LSTM units from 10 to **50**.
  - ✅ **Contextual Window**: Increased `time_stamp` from 10 to **12**.

#### **Why These Changes?**

  - **Seasonality**: Since the original problem was monthly airline passengers, a 12-step window better captures the yearly cycle.
  - **Robustness**: The Sine Wave generator proves the LSTM logic works on pure mathematical patterns.

**Modified Output:**

<img width="340" height="280" alt="modified lstm" src="https://github.com/user-attachments/assets/7cff73c4-0e33-46d3-a6fd-80d4a1417074" />
<img width="340" height="280" alt="modified lstm2" src="https://github.com/user-attachments/assets/e5bdde59-78b5-4a1a-8408-767ef39033b1" />
<img width="1063" height="420" alt="image" src="https://github.com/user-attachments/assets/3bc7b8b8-7a6c-4dce-a3af-83365077ec81" />

-----

## 5\. Rnn.py

### Original Implementation

  - SimpleRNN layer (vanishing gradient problem).
  - Short sequence length (6).
  - Basic text inputs ("Handsome boy...").

**Original Output:**

<img width="874" height="58" alt="image" src="https://github.com/user-attachments/assets/7f98d0be-e680-4032-8f05-f1cd6cd8ff42" />
### Improvements Made

#### **Architecture Enhancements**

  - ✅ **LSTM Upgrade**: Replaced `SimpleRNN` with **`LSTM`** to fix the vanishing gradient issue and retain long-term context.
  - ✅ **Technical Corpus**: Changed input text to "Artificial Intelligence involves machine learning..." to be more relevant to the course.

#### **Sequence Optimization**

  - ✅ **Extended Context**: Increased `seq_length` from 6 to **30**. This resolved ambiguity between repeated words (e.g., "machine *learning*" vs "deep *learning*").
  - ✅ **Intensive Training**: Increased epochs to **400** to ensure perfect memorization of the sentence.

#### **Why These Changes?**

  - **Coherence**: The LSTM with a longer window prevents the model from generating garbled or repetitive text.

**Modified Output:**

<img width="1809" height="74" alt="image" src="https://github.com/user-attachments/assets/3f961be4-682b-4d77-923f-630d506d861d" />

-----

## 6\. TicTacToe.py

### Original Implementation

  - 50,000 training rounds (slow).
  - Policy loading that crashed due to missing files.
  - Zero reward for draws.

**Original Output:**

<img width="321" height="914" alt="image" src="https://github.com/user-attachments/assets/9b201f90-394d-4e57-a2d6-4e5c205aef4e" />
<img width="321" height="1024" alt="image" src="https://github.com/user-attachments/assets/23c12278-0496-46d7-8db3-4339a32d9516" />


### Improvements Made

#### **Training Enhancements**

  - ✅ **Optimized Training**: Reduced rounds to **3,000** for faster execution while retaining strategy learning.
  - ✅ **Reward Logic**: Changed Tie/Draw reward from 0 to **0.2**.
  - ✅ **Self-Contained**: Disabled external file loading (`loadPolicy`) to prevent crashes.

#### **Why These Changes?**

  - **Strategic Play**: Rewarding ties teaches the agent to play defensively ("Not losing is better than losing"), which is optimal in Tic-Tac-Toe.
  - **Stability**: The script is now standalone and does not depend on pre-existing files.

**Modified Output:**

<img width="444" height="854" alt="image" src="https://github.com/user-attachments/assets/01eb4e66-3182-4251-a952-be2b94eb19cc" />
<img width="389" height="832" alt="image" src="https://github.com/user-attachments/assets/04a4535a-9335-4b96-aab0-b814102bf7c3" />
<img width="369" height="260" alt="image" src="https://github.com/user-attachments/assets/3f574ad6-dd31-4508-8205-d07bda5f1187" />


-----

## Installation & Requirements

### Required Libraries

```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn networkx
```

### System Requirements

  - **Python**: 3.8 or higher
  - **RAM**: 8GB minimum
  - **Storage**: \~2GB free space
  - **GPU**: Optional but recommended for faster training

-----

## Running the Scripts

### 1\. AlexNet

```bash
python AlexNet.py
```

*Generates synthetic data and trains for 1 epoch to verify architecture.*

### 2\. Classification (Jupyter Notebook)

Open `classification.ipynb` in Jupyter/Colab.
*Ensures dataset path is correct and runs augmentation pipeline.*

### 3\. Deep Reinforcement Learning

```bash
python DeepReinforcementLearning.py
```

*Displays graph visualization of the pathfinding agent.*

### 4\. LSTM

```bash
python LSTM.py
```

*Generates Sine wave data and plots prediction curves.*

### 5\. RNN

```bash
python Rnn.py
```

*Trains on text corpus and generates a sentence completion.*

### 6\. Tic Tac Toe

```bash
python TicTacToe.py
```

*Trains agent for 3000 rounds and starts a Human vs AI game.*

-----

## 📊 Results Summary

| Script | Status | Performance / Key Metric | Training Time (CPU) |
|--------|--------|---------------------|---------------------------|
| AlexNet.py | ✅ Modified | 10 Classes / Functional | \< 1 min |
| Classification.ipynb | ✅ Modified | 100 Classes / Augmentation | 10-15 min |
| DeepRL.py | ✅ Modified | Shortcut Found (2-\>7) | \< 1 min |
| LSTM.py | ✅ Modified | Sine Wave / 12-step lookback | 2-3 min |
| Rnn.py | ✅ Modified | Perfect Sentence Memorization | 3-5 min |
| TicTacToe.py | ✅ Modified | Defensive Play (Draw Reward) | 1-2 min |

-----

##  Learning Outcomes

Through this assignment, the following concepts were successfully implemented and understood:

### 1\. Convolutional Neural Networks (CNNs)

  - Architecture design (AlexNet)
  - Adaptation to multi-class problems (100 classes)
  - Importance of Data Augmentation in preventing overfitting

### 2\. Recurrent Neural Networks (RNNs)

  - Differences between SimpleRNN and LSTM (vanishing gradient)
  - Importance of Sequence Length in context retention
  - Time series prediction using LSTMs

### 3\. Reinforcement Learning

  - Q-learning fundamentals (Graph traversal)
  - Impact of Reward Shaping (TicTacToe Tie Reward)
  - Environment Design (Graph Topology modification)

-----

## 👨‍💻 Author Information

**Student:** K Venkata Vikram Reddy

**USN:** 1CD22AI027

**Department:** AIML, 7th Semester

**Course:** Deep Learning and Reinforcement Learning

**Submission Date:** December 2025

-----
