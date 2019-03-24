# Neural Network Model
## Shopee National Data Science Challenge 2019 (Advanced Category)
### Team: WhalerTheFirst

Code will be made public in https://github.com/yulonglong/nn_ndsc_shopee after the challenge ends.

**Environment:**  
This code is only tested using the following configuration:
- Ubuntu 16.04.6 LTS  
- Python 3.5.2 (default, GCC 5.4.0)
- CUDA 10.1 with Nvidia Pascal GPUs
- Python pip packages (`pip3 install -r requirements.txt`):
	- torch (PyTorch)  
	- h5py  
	- numpy  
	- scikit-learn  
	- scipy  
	- pydot  
	- opencv-python  

**Model details:**  
One model is trained for each category (fashion, beauty, mobile) and each sub-category (brand, colour, etc), and frame this problem as a multiclass classification problem. We trained 21 separate models for each of the 21 sub-categories in this dataset.

NLP model sequence:
- Lookup table layer (Embedding) to convert the words in the title to word embeddings (trainable)
- 1D CNN with window-size 3 to capture trigram representation
- Bidirectional LSTM to capture contextual information
- For each layer above, there is a residual component where the output is sum with the input
- Attention mechanism to pool the vectors as a form of weighted average
- Linear Highway layers to for the model to learn more complex functions

Image model sequence:
- Resize the image into 640x640 pixels, with 3 RGB channels
- CNN Unit consists of a CNN with kernel size 3, with batch normalization, and ReLU activation function
- CNN block consists of either two or three CNN units
- Stack 4 CNN blocks, doubling the number of filters/channel for each block (e.g., 3, 32, 64, 128, 256), with Max Pooling after each CNN block
- Average Pooling after the last CNN block
- Flatten the image into a vector, and apply Linear Highway layers
- (Tried another variation of CNN with residual where the output is sum with the input)

Combine both NLP and Image output vectors by concatenation, and then apply another set of Linear Highway layers to add more complexity and trainable parameters to the combined representation of text and image.

Apply softmax and obtain top 2 classes with the highest probability.

**Train the model:**  
Assuming all the packages are installed and data is stored in `./data` directory: 
- `./train_run.sh <dataset_category> <gpu_num> <gpu_name> <subcategory_start_index> <subcategory_end_index>`
- For example, to train on beauty dataset for all sub-categories (0,1,2,3,4) with GPU0, a GTX1080:
	- `./train_run.sh beauty 0 GTX1080 0 5`
- Hence, to train on all categories and subcategories on GPU 0, 1, and 2:
	- `./train_run.sh beauty 0 GTX1080 0 5`
	- `./train_run.sh fashion 1 GTX1080 0 5`
	- `./train_run.sh mobile 2 GTX1080 0 11`
- The output will be in a new `output` folder in the same directory (e.g., `./data/`) 
