# DL-assignments
DeepReinforcementLearning Program

Changes:

Environment Modification: I altered the graph structure by adding a new edge between Node 2 and Node 7. This created a "shortcut" in the environment to test if the agent could discover new   optimal paths.

Objective Change: I changed the Goal Node from Node 10 to Node 7. This forces the Q-Learning algorithm to recalculate the entire reward matrix rather than relying on previous defaults.

Visualization: I added a descriptive title to the graph plot to clearly indicate the current goal and environment configuration during the output display.

Outputs:

Original  code outputs:





<img width="340" height="280" alt="deepreinforcement learning" src="https://github.com/user-attachments/assets/cf2f0b2c-2de0-4b1d-8087-14433e982481" />

<img width="340" height="280" alt="deepreinforcement learning2" src="https://github.com/user-attachments/assets/c4a58ad0-e15f-49e1-af15-d223583f99b0" />

<img width="340" height="280" alt="deepreinforcement learning3" src="https://github.com/user-attachments/assets/a6735d1e-c7ce-401d-a426-8c8d7a3185a5" />

<img width="340" height="280" alt="deepreinforcement learning4" src="https://github.com/user-attachments/assets/262cc3fb-5693-414c-8ca4-ef7ca4ecb2ed" />

<img width="532" height="648" alt="image" src="https://github.com/user-attachments/assets/6208f9dc-7330-41b5-9c4e-374a943bcc56" />



Modified code outputs:




<img width="340" height="280" alt="modified Drl" src="https://github.com/user-attachments/assets/d4f18f2d-e70e-47e0-b80c-e6557a5e1fc0" />

<img width="340" height="280" alt="modified Drl2" src="https://github.com/user-attachments/assets/add277ff-d04e-405e-962b-0e9ac7098cad" />

<img width="558" height="122" alt="image" src="https://github.com/user-attachments/assets/cd42961f-32d2-4e24-abb6-ce0810398076" />

RNN Program

Changes:
Architecture Upgrade: I replaced the basic SimpleRNN layer with an LSTM (Long Short-Term Memory) layer. SimpleRNN suffers from the vanishing gradient problem, whereas LSTM effectively retains information over longer sequences.

Sequence Optimization: I increased the seq_length from 6 to 30. This resolved ambiguity in the training text (specifically distinguishing between "machine learning" and "deep learning"), ensuring the generated output was coherent.

Training Intensity: I increased the training duration to 400 epochs to force the model to memorize the specific technical sentence about Artificial Intelligence perfectly.

Original code output:
<img width="874" height="58" alt="image" src="https://github.com/user-attachments/assets/7f98d0be-e680-4032-8f05-f1cd6cd8ff42" />

Modified code output:
<img width="1809" height="74" alt="image" src="https://github.com/user-attachments/assets/3f961be4-682b-4d77-923f-630d506d861d" />

AlexNet program

Changes:

Operational Execution: The original code only defined the class structure. I added the execution logic (model.compile and model.fit) to ensure the model actually runs and processes data.

Data Simulation: Instead of downloading the massive ImageNet dataset, I generated synthetic dummy data using NumPy. This demonstrates that the architecture functions correctly without requiring heavy external files.

Hyperparameter Tuning: I modified the Dropout rate from 0.5 to 0.4 and reduced the output layer from 1000 classes to 10 classes. This customized the network for a lighter, faster classification task.

Original Code Output:

<img width="791" height="694" alt="image" src="https://github.com/user-attachments/assets/d3544c04-ed4d-4f60-b66e-060609df7b98" />

Modified Code Output:

<img width="794" height="767" alt="image" src="https://github.com/user-attachments/assets/4ef05d4a-4123-4453-85e2-87e41fa60401" />

Classification

Changes:
Dataset Substitution: I replaced the binary "Cats vs Dogs" dataset with a complex 100-class dataset (Butterflies). 

Data Augmentation: I implemented specific data augmentation strategies (Rotation, Zoom, Shearing) using ImageDataGenerator. This improves the model's generalization capabilities and prevents overfitting on the training data.

Model Persistence: I added functionality to serialize and save the trained model (model.h5) and class indices (JSON). This allows the model to be reloaded later for predictions without retraining.

Ouputs:


<img width="340" height="280" alt="image" src="https://github.com/user-attachments/assets/a397e213-b3be-4fb0-bd81-fd76b211717a" />
<img width="340" height="280" alt="image" src="https://github.com/user-attachments/assets/0da805fe-8944-4146-8702-71f7d410c083" />
<img width="606" height="651" alt="image" src="https://github.com/user-attachments/assets/e6a2be1b-ff24-4ce1-8ddd-2ea9da484fd6" />
<img width="613" height="642" alt="image" src="https://github.com/user-attachments/assets/4cfb1084-fb4e-41cc-aced-695189fc5edd" />



LSTM Program

Changes:
Contextual Window: I increased the time_stamp (look-back period) from 10 to 12. Since the dataset represents monthly passengers, a 12-step window allows the model to capture yearly seasonality effectively.

Model Capacity: I upgraded the architecture from LSTM(10) to LSTM(50). Increasing the number of units improved the model's ability to learn complex patterns, resulting in a prediction curve that closely matches the actual data.

Original Code Outputs:

<img width="340" height="280" alt="op lstm" src="https://github.com/user-attachments/assets/088f08fa-b94b-4566-880f-80ad47c282d8" />
<img width="340" height="280" alt="op lstm2" src="https://github.com/user-attachments/assets/c958cbd1-f358-4913-b467-03a0213824a6" />
<img width="1211" height="425" alt="image" src="https://github.com/user-attachments/assets/5f78bb9e-dc85-43f4-a285-9dc1c9da8f8b" />


Modified code outputs:

<img width="340" height="280" alt="modified lstm" src="https://github.com/user-attachments/assets/7cff73c4-0e33-46d3-a6fd-80d4a1417074" />
<img width="340" height="280" alt="modified lstm2" src="https://github.com/user-attachments/assets/e5bdde59-78b5-4a1a-8408-767ef39033b1" />
<img width="1063" height="420" alt="image" src="https://github.com/user-attachments/assets/3bc7b8b8-7a6c-4dce-a3af-83365077ec81" />


TicTacToe Program
Changes:
Training Efficiency: I reduced the training iterations from 50,000 to 3,000. This makes the code executable in a reasonable time frame for demonstration purposes while still allowing the agent to learn basic strategies.

Reward Logic Modification: I adjusted the reward function for a Tie (Draw) from 0 to 0.2. This reinforces defensive play, teaching the agent that achieving a draw is a more favorable outcome than losing.

Self-Contained Execution: I disabled the external loadPolicy function. This prevents FileNotFound errors and ensures the agent trains a fresh model every time the script is run.

Original Code Outputs:



<img width="321" height="914" alt="image" src="https://github.com/user-attachments/assets/9b201f90-394d-4e57-a2d6-4e5c205aef4e" />
<img width="321" height="1024" alt="image" src="https://github.com/user-attachments/assets/23c12278-0496-46d7-8db3-4339a32d9516" />

Modified Code Outputs:


<img width="444" height="854" alt="image" src="https://github.com/user-attachments/assets/01eb4e66-3182-4251-a952-be2b94eb19cc" />
<img width="389" height="832" alt="image" src="https://github.com/user-attachments/assets/04a4535a-9335-4b96-aab0-b814102bf7c3" />
<img width="369" height="260" alt="image" src="https://github.com/user-attachments/assets/3f574ad6-dd31-4508-8205-d07bda5f1187" />


                                                           THE END
