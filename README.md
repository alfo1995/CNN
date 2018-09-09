# CNN - Convolution Neural Network


The goal of this project is to build up a Convolutional Neural Network (CNN) which analyzes images composed by characters and numbers.

This CNN has to be able to recognize:

* which character the image is
* which is the font of the image
* whether the character is bold or not, italic or not

At the beginning we have 94 characters and 28 font and before applying the *augmentation* the number of images is given by $94 \cdot 28 = 2632$.


The *augmentations* we decide to apply to our dataset are the following:

* 3 types of rotation
    * random.uniform(-60,60)
    * random.uniform(-65,25)
    * random.uniform(-60,30)
  
* Salt and Pepper noise with probability equal to $0.02$
* Zoom
    * we create a function which takes in input an array image and a zoom factor. We fix the factor to        $1.5$
* A random translation with the **transform.rescale** function of skimage package

**_How do we proceed?_**

We take in input the original image and we apply the augmentation functions above. 
Each image obtained is passed through the augmentation functions (rotation, zoom) once again and so on.

The result is $89k$ images.


# Description of the final model

In the final model we've devided our network as follows:

**_Net1_:**

From the input, the first network starts with a  **_convolution_** whose **filters_size** is equal to $32$ is implemented. It has followed by another **_convolution_** with the same **filters_size**.
The size of the kernel is respective $5$ and $3$.

These two convolutions are followed by a **Max_pooling** with **pool_size** equal to $2$ and then a **drop_out** with $0.25$ probability to switch neurons off.

**_Net structure_**: **Font**

The net structure behind the **font** output is composed by a **_convolution_** (with **filters_size** equal to $32$ and **kernel_size** equal to $3$).
A **flatten** follows the **_convolution_**.

Now, **_Dense_** (with **relu** as activation functions and $32$ as number of neurons) and **drop_out** are alternate before the output.



**_Net2_:**

Here, we have a **_convolution_** whose number of **filters** is equal to $16$ and **kernel_size** $3$.
Then, a **Max_pooling** with **pool_size** equal to $2$ and a **drop_out** with $0.25$.


**_Net structure_**: **Char**

In this net, we have a **_convolution_** (with **filters_size** equal to $32$ and **kernel_size** equal to $3$) followed by a **Flatten**.
As we did in the previous net,**_Dense_** (with **relu** as activation functions and $32$ as number of neurons) and **drop_out** are alternate before the output.


**_Net3_:**:

The configuration of the net is as the same as the *net1*.

The difference is about the parameters. We have $32$ **_filters_** and **_kernel_size_** equal to $5$ for the first **_convolution_** and $3$ for the second one.


**_Net structure_**: **Bold** and **Italic**

Even here the structure is as the same as the **Char** structure.
What changes is the number of filters of the **_convolution_** ($32$) and the number of **_Dense_** ($16$).


# Performance

The performance obtained on the test set is:

- Char = 0.008
- Font = 0.148
- Bold = 0.572
- Italics = 0.611

**_Partial accuracy_ = 0.35**

 
# Comparison

One of the first model which got us back a "good" score had the following simply structure:

- **_Convolution_** with $64$ **Filters** and **Kernel_size** equal to $3$
- **_Max_pooling_**
- **_Drop_out_**
- **_Flatten_**

With this structure we achieved a score equal to $0.19$.
