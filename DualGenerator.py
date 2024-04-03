import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

#### Data parameter
mb_size = 256
X_dim = 784
z_dim = 784
TrianNum = 10000
para_dim = 10

h1_dim = 128
h2_dim = 256
h3_dim = 512
h4_dim = 256
h5_dim = 128

n_round = 6
n_test  = 6
n_disc  = 3
n_gene  = 10

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)     #MNIST

def plot(samples):
    fig = plt.figure(figsize=(3, 3))
    gs = gridspec.GridSpec(3, 3)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    return fig
    
def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.5, shape=shape)
    return tf.Variable(initial)


##### Network sturcture 
# Train Data compose of [0:784]            [784:794]           
# Train Data compose of [28*28 pixels data][feature data ]
# Train Traget is 28*28 data **[784 pixels Target]** [10 feature data ]
XDisc = tf.placeholder(tf.float32, shape=[None, X_dim])             #Real Data for Generator 784 array
ZDisc = tf.placeholder(tf.float32, shape=[None, z_dim+para_dim])    #Input Data for Discriminator 784 array

# Left network ############
LeftW0 = weight_variable([z_dim+para_dim, h1_dim])
LeftB0 = bias_variable([h1_dim])
LeftW1 = weight_variable([h1_dim, h2_dim])
LeftB1 = bias_variable([h2_dim])
LeftW2 = weight_variable([h2_dim, h3_dim])
LeftB2 = bias_variable([h3_dim])
LeftW3 = weight_variable([h3_dim, h4_dim])
LeftB3 = bias_variable([h4_dim])
LeftW4 = weight_variable([h4_dim, h5_dim])
LeftB4 = bias_variable([h5_dim])
LeftW5 = weight_variable([h5_dim, z_dim])
LeftB5 = bias_variable([z_dim])

LeftNN = [LeftW0,LeftB0,LeftW1,LeftB1,LeftW2,LeftB2,LeftW3,LeftB3,LeftW4,LeftB4,LeftW5,LeftB5]
initail_LeftNN = tf.variables_initializer(LeftNN)

def Left(ZDisc): # Generator network
    OutGene0    = tf.nn.leaky_relu(tf.matmul(ZDisc,     LeftW0) + LeftB0)
    OutGene1    = tf.nn.leaky_relu(tf.matmul(OutGene0,  LeftW1) + LeftB1)
    OutGene2    = tf.nn.leaky_relu(tf.matmul(OutGene1,  LeftW2) + LeftB2)
    OutGene3    = tf.nn.leaky_relu(tf.matmul(OutGene2,  LeftW3) + LeftB3)
    OutGene4    = tf.nn.leaky_relu(tf.matmul(OutGene3,  LeftW4) + LeftB4)
    OutGene5    = tf.nn.leaky_relu(tf.matmul(OutGene4,  LeftW5) + LeftB5)
    return tf.nn.sigmoid(OutGene5)
    
# Right network ############
RightW0 = weight_variable([z_dim+para_dim, h1_dim])
RightB0 = bias_variable([h1_dim])
RightW1 = weight_variable([h1_dim, h2_dim])
RightB1 = bias_variable([h2_dim])
RightW2 = weight_variable([h2_dim, h3_dim])
RightB2 = bias_variable([h3_dim])
RightW3 = weight_variable([h3_dim, h4_dim])
RightB3 = bias_variable([h4_dim])
RightW4 = weight_variable([h4_dim, h5_dim])
RightB4 = bias_variable([h5_dim])
RightW5 = weight_variable([h5_dim, z_dim])
RightB5 = bias_variable([z_dim])

RightNN = [RightW0,RightB0,RightW1,RightB1,RightW2,RightB2,RightW3,RightB3,RightW4,RightB4,RightW5,RightB5]
initail_RightNN = tf.variables_initializer(RightNN)

def Right(ZDisc): # Generator network
    OutGene0    = tf.nn.leaky_relu(tf.matmul(ZDisc,     RightW0) + RightB0)
    OutGene1    = tf.nn.leaky_relu(tf.matmul(OutGene0,  RightW1) + RightB1)
    OutGene2    = tf.nn.leaky_relu(tf.matmul(OutGene1,  RightW2) + RightB2)
    OutGene3    = tf.nn.leaky_relu(tf.matmul(OutGene2,  RightW3) + RightB3)
    OutGene4    = tf.nn.leaky_relu(tf.matmul(OutGene3,  RightW4) + RightB4)
    OutGene5    = tf.nn.leaky_relu(tf.matmul(OutGene4,  RightW5) + RightB5)
    return tf.nn.sigmoid(OutGene5)

##### Generator Network Define  
Right       = Right(ZDisc)                                                      #Right Generator output
Left        = Left(ZDisc)                                                       #Left Generator output

##### loss function of right discriminator (MSE)   
GRight  = ( tf.pow(Right-XDisc,2))
GRight  = tf.reduce_mean( GRight )
Right_solver        = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5,beta2=0.9,epsilon=1e-08).minimize(GRight, var_list=RightNN)

##### loss function of left  discriminator (MSE)   
Gleft   = (tf.pow(Left-XDisc,2))
Gleft     = tf.reduce_mean( Gleft ) 
Left_solver        = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5,beta2=0.9,epsilon=1e-08).minimize(Gleft, var_list=LeftNN)

#### Tensorflow variable initial ##
init = tf.global_variables_initializer()
if not os.path.exists('./dual_Gen/'):
    os.makedirs('./dual_Gen')
sess = tf.Session()
sess.run(init)

count = 0
for i  in range (TrianNum):
    if i%100 == 0:                                                              #for print
        print(i)
        X_mb, feature = mnist.train.next_batch(9)                               #load mnist data
        zdata   = np.random.rand( 9, z_dim)                                     #random array

        vector = np.random.randint(2)                                           
        if vector == 0:
            zdataF  = np.append(zdata,feature,axis = 1)                         # array + feature
            for _ in range(n_test):
                zdata   = sess.run(Left,feed_dict={ ZDisc : zdataF} )           # left starts generating
                zdataF  = np.append(zdata,feature,axis = 1)                     # array + feature
                zdata   = sess.run(Right,feed_dict={ ZDisc : zdataF })          # right starts generating
                zdataF  = np.append(zdata,feature,axis = 1)                     # array + feature

            sampleL       = sess.run(Left,feed_dict={ ZDisc : zdataF } )
            sampleR       = sess.run(Right,feed_dict={  ZDisc : zdataF})      
                
        if vector == 1:                                                         # mirror
            zdataF  = np.append(zdata,feature,axis = 1)
            for _ in range(n_test):
                zdata   = sess.run(Right,feed_dict={ ZDisc : zdataF })
                zdataF  = np.append(zdata,feature,axis = 1)
                zdata   = sess.run(Left,feed_dict={ ZDisc : zdataF} )
                zdataF  = np.append(zdata,feature,axis = 1)
               
            sampleR       = sess.run(Right,feed_dict={  ZDisc : zdataF})   
            sampleL       = sess.run(Left,feed_dict={ ZDisc : zdataF } )
            
        fig = plot(sampleL[0:9])
        plt.savefig('dual_Gen/{}Left.png'.format(str(count).zfill(3)), bbox_inches='tight')
        plt.close(fig)
                    
        fig = plot(sampleR[0:9])
        plt.savefig('dual_Gen/{}Right.png'.format(str(count).zfill(3)), bbox_inches='tight')
        count += 1
        plt.close(fig)

    X_mb, feature = mnist.train.next_batch(2*mb_size)                           # load mnist data for train
    X_mbR, X_mbL = X_mb[0:mb_size],X_mb[mb_size:]                               # divide mnist data for train
    featureR,featureL = feature[0:mb_size], feature[mb_size:]                   # divide feature for train
    zdata   = np.random.rand( mb_size, z_dim)                                   # random array
    #X_mbR = X_mbL
    if i%2==0:
        zdataL  = np.append(zdata,featureL,axis = 1)                            # array + feature
        for _ in range(n_round):
            for i in range(n_gene):
                sess.run(Left_solver,feed_dict={ XDisc: X_mbL,ZDisc : zdataL } )# left starts training
            zdata   = sess.run(Left,feed_dict={ ZDisc : zdataL } )              # left starts generating
            zdataR  = np.append(zdata,featureR,axis = 1)                        # array + feature
            
            for i in range(n_gene):
                sess.run(Right_solver,feed_dict={ XDisc: X_mbR,ZDisc : zdataR })# right starts training
            zdata   = sess.run(Right,feed_dict={ ZDisc : zdataR } )             # right starts generating
            zdataL  = np.append(zdata,featureL,axis = 1)                        # array + feature

    if i%2==1:                                                                  # mirror
        zdataR  = np.append(zdata,featureR,axis = 1)

        for _ in range(n_round):
            for i in range(n_gene):
                sess.run(Right_solver,feed_dict={ XDisc: X_mbR,ZDisc : zdataR } )
            zdata   = sess.run(Right,feed_dict={ ZDisc : zdataR } )
            zdataL  = np.append(zdata,featureL,axis = 1)

            for i in range(n_gene):
                sess.run(Left_solver,feed_dict={ XDisc: X_mbL,ZDisc : zdataL } )
            zdata   = sess.run(Left,feed_dict={ ZDisc : zdataL } )
            zdataR  = np.append(zdata,featureR,axis = 1)


    