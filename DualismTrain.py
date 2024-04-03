import numpy as np
import tensorflow.compat.v1 as tf
import os
import datetime
import pandas as pd
#import funlib.funlib1 as fl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

print('tf_ver',tf.__version__)	 ###tensorflow ver 1.15.0

###### Title and folder check
TitleofPic = 'GAN with dualism'

Datalogflag=1
if not os.path.exists('./dualism/'):
    os.makedirs('./dualism/') 
    
Path = './dualism/'

'''      
if not os.path.isfile('./dualismlog.csv'):
    Datalogflag=0
    Begin = 0

if  os.path.isfile('./NewBalanceDataLog.csv'):    
    DataLog = pd.read_csv('./NewBalanceDataLog.csv',encoding = 'big5')
    DataLog = DataLog.values
    Begin = len(DataLog)

    
print('netnum = '+str(Begin))
'''   
###### data composition
# Train Data compose of [1:16]       [16:251]           [251:312]           [312:318]
# Train Data compose of [15misn data][data open to now ][first 5 days Kline][price,price change & target this min]
PreDataTrain    = pd.read_csv('./PreDataTrain.csv')
PreDataTrain    = PreDataTrain.values
       
# Test & Verificationm
# Data compose of [0:1]      [1:16]       [16:251]           [251:312]           [312:318]
# Data compose of [oepn flag][15misn data][data open to now ][first 5 days Kline][price,price change & target this min]
PreDataTest     = pd.read_csv('./PreDataTest0.csv')
PreDataTest     = PreDataTest.values        
PreDataValid    = pd.read_csv('./PreDataTest1.csv')
PreDataValid    = PreDataValid.values       

# Train Price Data compose of [0:15]                [15:30]           
# Train Price Data compose of [previous 15mins data][following 45 mins data data/3mins ]
# Train Traget is middle 30 mins data [15mins data] **[30mins Target]** [15mins data ]
PricDataTrain   = pd.read_csv('./PricDataTrain.csv')
PricDataTrain   = PricDataTrain.values      

#### Data parameter
NumTrain,PreTrain,NextTrain,LastTrain = 30,15,10,5  #Total,previous 15mins data,30mins Target data/3mins,Last 15mins data  
z_dim = 10                                          #30mins Target data/3mins->10 input data for GAN equal to output                                              
mb_size,half_mb = 512,256                           #batch of data
n_round,n_disc,n_gene,n_test = 6,3,2,6              #train num/round,/round (discriminator),/round (Generator),/round (test & Verificationm) 
TrianNum = 1000                                     #Total round TrianNum, stable round 
Numoffeature = 317                                  #len of Train Data
RowofTest   = len(PreDataTest)
RowofValid  = len(PreDataValid)
Data_Test   = PreDataTest[:,1:Numoffeature-5]       #remove unuse para 
Data_Valid  = PreDataValid[:,1:Numoffeature-5]      #remove unuse para 
PreDataTrain = PreDataTrain[:,0:Numoffeature-6]     #remove unuse para 

### Function Define

def weight_variable(shape):
    weight = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(weight)

def bias_variable(shape):
    bias = tf.constant(0.5, shape=shape)
    return tf.Variable(bias)

def Pic_Save(title,data,savepath,xl,yl,days):
    raw = len(data)
    x = np.arange(0,raw/days,1/days)
    x = x[0:raw]
    plt.figure() 
    plt.title(title) 
    plt.plot(x,data)
    plt.savefig(savepath)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.close()    

##### Network sturcture   
XDisc = tf.placeholder(tf.float32, shape=[None,1*NumTrain])           # train data for Discrominator [NumTrain] 
YDisc = tf.placeholder(tf.float32, shape=[None,1*PreTrain])             # post data
lDisc = tf.placeholder(tf.float32, shape=[None,1*LastTrain])            # last data
ZGene = tf.placeholder(tf.float32, shape=[None,Numoffeature-6+z_dim])   # feature + train data for generator

# Left network ############
LeftW0 = weight_variable([Numoffeature-6+z_dim,320])
LeftB0 = bias_variable([320])
LeftW1 = weight_variable([320,512])
LeftB1 = bias_variable([512])
LeftW2 = weight_variable([ 512, 512])
LeftB2 = bias_variable([512])
LeftW3 = weight_variable([512, 400])
LeftB3 = bias_variable([400])
LeftW4 = weight_variable([400, 256])
LeftB4 = bias_variable([256])
LeftW5 = weight_variable([256, 128])
LeftB5 = bias_variable([ 128])
LeftW6 = weight_variable([ 128, 10])
LeftB6 = bias_variable([10])

LeftNN = [LeftW0,LeftB0,LeftW1,LeftB1,LeftW2,LeftB2,LeftW3,LeftB3,LeftW4,LeftB4,LeftW5,LeftB5,LeftW6,LeftB6]
#initail_LeftNN = tf.variables_initializer(LeftNN)

def Left(ZGene): 
    OutGene0    = tf.nn.leaky_relu(tf.matmul(ZGene,     LeftW0) + LeftB0)
    OutGene1    = tf.nn.leaky_relu(tf.matmul(OutGene0,  LeftW1) + LeftB1)
    OutGene2    = tf.nn.leaky_relu(tf.matmul(OutGene1,  LeftW2) + LeftB2)
    OutGene3    = tf.nn.leaky_relu(tf.matmul(OutGene2,  LeftW3) + LeftB3)
    OutGene4    = tf.nn.leaky_relu(tf.matmul(OutGene3,  LeftW4) + LeftB4)
    OutGene5    = tf.nn.leaky_relu(tf.matmul(OutGene4,  LeftW5) + LeftB5)
    return tf.nn.sigmoid(tf.matmul(OutGene5, LeftW6) + LeftB6)
    
# Right network ############
RightW0 = weight_variable([Numoffeature-6+z_dim,320])
RightB0 = bias_variable([320])
RightW1 = weight_variable([320,512])
RightB1 = bias_variable([512])
RightW2 = weight_variable([ 512, 512])
RightB2 = bias_variable([512])
RightW3 = weight_variable([512, 400])
RightB3 = bias_variable([400])
RightW4 = weight_variable([400, 256])
RightB4 = bias_variable([256])
RightW5 = weight_variable([256, 128])
RightB5 = bias_variable([ 128])
RightW6 = weight_variable([ 128, 10])
RightB6 = bias_variable([10])

RightNN = [RightW0,RightB0,RightW1,RightB1,RightW2,RightB2,RightW3,RightB3,RightW4,RightB4,RightW5,RightB5,RightW6,RightB6]
#initail_RightNN = tf.variables_initializer(RightNN)

def Right(ZGene): 
    OutGene0    = tf.nn.leaky_relu(tf.matmul(ZGene,     RightW0) + RightB0)
    OutGene1    = tf.nn.leaky_relu(tf.matmul(OutGene0,  RightW1) + RightB1)
    OutGene2    = tf.nn.leaky_relu(tf.matmul(OutGene1,  RightW2) + RightB2)
    OutGene3    = tf.nn.leaky_relu(tf.matmul(OutGene2,  RightW3) + RightB3)
    OutGene4    = tf.nn.leaky_relu(tf.matmul(OutGene3,  RightW4) + RightB4)
    OutGene5    = tf.nn.leaky_relu(tf.matmul(OutGene4,  RightW5) + RightB5)
    return tf.nn.sigmoid(tf.matmul(OutGene5, RightW6) + RightB6)
    
#Right Discriminator 
WRDisc0   = weight_variable([1*NumTrain, 256]) 
BRDisc0   = bias_variable([256]) 
WRDisc1   = weight_variable([ 256, 512])
BRDisc1   = bias_variable([512])
WRDisc2   = weight_variable([ 512, 512])
BRDisc2   = bias_variable([512])
WRDisc3   = weight_variable([512, 256])
BRDisc3   = bias_variable([256])
WRDisc4   = weight_variable([256, 128])
BRDisc4   = bias_variable([128])
WRDisc5   = weight_variable([128, 64])
BRDisc5   = bias_variable([64])
WRDisc6   = weight_variable([64, 1])
BRDisc6   = bias_variable([1])

RDiscNN = [WRDisc0,BRDisc0,WRDisc1,BRDisc1,WRDisc2,BRDisc2,WRDisc3,BRDisc3,WRDisc4,BRDisc4,WRDisc5,BRDisc5,WRDisc6,BRDisc6]
#initail_RDiscNN = tf.variables_initializer(RDiscNN)

def RDiscNet(XDisc): 
    OutDisc = tf.nn.leaky_relu(tf.matmul(XDisc, WRDisc0) + BRDisc0)
    OutDisc0 = tf.nn.leaky_relu(tf.matmul(OutDisc, WRDisc1) + BRDisc1)
    OutDisc1 = tf.nn.leaky_relu(tf.matmul(OutDisc0, WRDisc2) + BRDisc2)
    OutDisc2 = tf.nn.leaky_relu(tf.matmul(OutDisc1, WRDisc3) + BRDisc3)
    OutDisc3 = tf.nn.leaky_relu(tf.matmul(OutDisc2, WRDisc4) + BRDisc4)
    OutDisc4 = tf.nn.leaky_relu(tf.matmul(OutDisc3, WRDisc5) + BRDisc5)
    return tf.nn.sigmoid(tf.matmul(OutDisc4, WRDisc6) + BRDisc6)

#Left Discriminator  
WLDisc0   = weight_variable([1*NumTrain, 256]) 
BLDisc0   = bias_variable([256]) 
WLDisc1   = weight_variable([ 256, 512])
BLDisc1   = bias_variable([512])
WLDisc2   = weight_variable([ 512, 512])
BLDisc2   = bias_variable([512])
WLDisc3   = weight_variable([512, 256])
BLDisc3   = bias_variable([256])
WLDisc4   = weight_variable([256, 128])
BLDisc4   = bias_variable([128])
WLDisc5   = weight_variable([128, 64])
BLDisc5   = bias_variable([64])
WLDisc6   = weight_variable([64, 1])
BLDisc6   = bias_variable([1])

LDiscNN = [WLDisc0,BLDisc0,WLDisc1,BLDisc1,WLDisc2,BLDisc2,WLDisc3,BLDisc3,WLDisc4,BLDisc4,WLDisc5,BLDisc5,WLDisc6,BLDisc6]
#initail_LDiscNN = tf.variables_initializer(LDiscNN)

def LDiscNet(XDisc): 
    OutDisc = tf.nn.leaky_relu(tf.matmul(XDisc, WLDisc0) + BLDisc0)
    OutDisc0 = tf.nn.leaky_relu(tf.matmul(OutDisc, WLDisc1) + BLDisc1)
    OutDisc1 = tf.nn.leaky_relu(tf.matmul(OutDisc0, WLDisc2) + BLDisc2)
    OutDisc2 = tf.nn.leaky_relu(tf.matmul(OutDisc1, WLDisc3) + BLDisc3)
    OutDisc3 = tf.nn.leaky_relu(tf.matmul(OutDisc2, WLDisc4) + BLDisc4)
    OutDisc4 = tf.nn.leaky_relu(tf.matmul(OutDisc3, WLDisc5) + BLDisc5)
    return tf.nn.sigmoid(tf.matmul(OutDisc4, WLDisc6) + BLDisc6)

##### Generator Network Define
Right0      = Right(ZGene)                          ## mid data      
Right1      = tf.concat([YDisc,Right0], axis=1)     ## pre data + mid data 
Right       = tf.concat([Right1,lDisc], axis=1)     ## pre data + mid data + last data 

Left0       = Left(ZGene)                           
Left1       = tf.concat([YDisc,Left0], axis=1)
Left        = tf.concat([Left1,lDisc], axis=1)

##### Discriminator Network Define
RDTrue      = RDiscNet(XDisc)                       ## 1 of discriminator                 
RDFake      = RDiscNet(Right)                       ## Train Target of discriminator 
RDFalse     = RDiscNet(Left)                        ## 0 of discriminator 

LDTrue      = LDiscNet(XDisc)                       
LDFake      = LDiscNet(Left)
LDFalse     = LDiscNet(Right)

avgRDTrue    = tf.reduce_mean(RDTrue)
avgRDFake    = tf.reduce_mean(RDFake)
avgRDFalse   = tf.reduce_mean(RDFalse)

avgLDTrue    = tf.reduce_mean(LDTrue)
avgLDFake    = tf.reduce_mean(LDFake)
avgLDFalse   = tf.reduce_mean(LDFalse)

##### loss function of right discriminator (cross entropy)
RDCETrue    = -( 1*tf.log(RDTrue) + 1*tf.log(1-RDFalse) ) 
RDCEGene    = -( LDFake*tf.log(1-RDFake) + (1-LDFake)*tf.log(RDFake))
RDCE        = tf.reduce_mean( RDCETrue ) + tf.reduce_mean( RDCEGene ) 
RD_solver   = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5,beta2=0.9,epsilon=1e-08).minimize(tf.reduce_sum(RDCE),var_list = RDiscNN)

##### loss function of left discriminator (cross entropy)
LDCETrue    = -( 1*tf.log(LDTrue) + 1*tf.log(1-LDFalse)  )
LDCEGene    = -( RDFake*tf.log(1-LDFake) + (1-RDFake)*tf.log(LDFake))
LDCE        = tf.reduce_mean( LDCETrue ) + tf.reduce_mean( LDCEGene ) 
LD_solver   = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5,beta2=0.9,epsilon=1e-08).minimize(tf.reduce_sum(LDCE),var_list = LDiscNN)

##### loss function of right generator (cross entropy)    
GRight      = -1*tf.log(RDFake) #- 1*tf.log(1-LDFalse)
GRight      = tf.reduce_mean(GRight) 
Right_solver= tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5,beta2=0.9,epsilon=1e-08).minimize(GRight, var_list=RightNN)

##### loss function of left generator (cross entropy)     
Gleft       = -1*tf.log(LDFake) #- 1*tf.log(1-RDFalse)
Gleft       = tf.reduce_mean(Gleft) 
Left_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5,beta2=0.9,epsilon=1e-08).minimize(Gleft, var_list=LeftNN)

### Tensorflow variable initial 
init        = tf.global_variables_initializer()
saver_right = tf.train.Saver(RightNN,max_to_keep=31)
saver_left  = tf.train.Saver(LeftNN,max_to_keep=31)
sess = tf.Session()
sess.run(init)

TestWR,TestCu,ValidWR,ValidCu = np.zeros(TrianNum),np.zeros(TrianNum),np.zeros(TrianNum), np.zeros(TrianNum) # Data for record WR : winrate / Cu : cumsum
partial = half_mb  ##real data : generator data = half_mb : mb_size-half_mb 

for i  in range (TrianNum):
    
    # load batch data
    #Train,PriceTrain    = np.zeros((mb_size,len(PreDataTrain[0]))),np.zeros((mb_size,len(PricDataTrain[0])))    
    rand                = int(np.round( np.random.rand()*(len(PreDataTrain)-mb_size)))                           
    Train,PriceTrain    = PreDataTrain[rand:rand+mb_size,:],PricDataTrain[rand:rand+mb_size,:]  # parameter , price data
    
    # generate random data
    zdata       = np.random.normal(0,1,size=[mb_size,z_dim])        #random array
    sample      = np.append(Train,zdata,axis=1)                     #para + random array
    
    if i%2==0:
        for _ in range(n_round):
            
            for _ in range(n_gene):
                sess.run(Left_solver,feed_dict={ ZGene : sample,YDisc : PriceTrain[:,0:1*PreTrain],lDisc :PriceTrain[:,1*(PreTrain+NextTrain):1*(PreTrain+NextTrain+LastTrain)] } )     # left generator Train
                
            zdata   = sess.run(Left0,feed_dict={ ZGene : sample } )     # left generator output
            sample  = np.append(Train,zdata,axis=1)                     # parameter + left output
            LData   = sess.run(Left,feed_dict={ ZGene : sample,YDisc : PriceTrain[:,0:1*PreTrain],lDisc :PriceTrain[:,1*(PreTrain+NextTrain):1*(PreTrain+NextTrain+LastTrain)] } )      # LData = Left(para + pre output) 
            LData   = np.append(PriceTrain[:partial],LData[partial:],axis=0)    ###給Discrominator包含兩組generator互丟 
            
            for _ in range(n_disc):
                #sess.run(RD_solver,feed_dict={XDisc: PriceTrain, ZGene: sample, YDisc : PriceTrain[:,0:1*PreTrain],lDisc :PriceTrain[:,1*(PreTrain+NextTrain):1*(PreTrain+NextTrain+LastTrain)] }) # Right discriminator train 
                sess.run(RD_solver,feed_dict={XDisc: LData, ZGene: sample, YDisc : PriceTrain[:,0:1*PreTrain],lDisc :PriceTrain[:,1*(PreTrain+NextTrain):1*(PreTrain+NextTrain+LastTrain)] }) #mix real data with gen data 
                    
            for _ in range(n_gene):
                sess.run(Right_solver,feed_dict={ ZGene : sample,YDisc : PriceTrain[:,0:1*PreTrain],lDisc :PriceTrain[:,1*(PreTrain+NextTrain):1*(PreTrain+NextTrain+LastTrain)] } )    # right generator Train
                
            zdata   = sess.run(Right0,feed_dict={ ZGene : sample } )    # right generator output
            sample  = np.append(Train,zdata,axis=1) # parameter + right output
            RData   = sess.run(Right,feed_dict={ ZGene : sample,YDisc : PriceTrain[:,0:1*PreTrain],lDisc :PriceTrain[:,1*(PreTrain+NextTrain):1*(PreTrain+NextTrain+LastTrain)] } )     # RData = right(para + pre output)
            RData   = np.append(PriceTrain[:partial],RData[partial:],axis=0)    ###給Discrominator包含兩組generator互丟 
        
            for _ in range(n_disc):
                #sess.run(LD_solver,feed_dict={XDisc: PriceTrain, ZGene: sample, YDisc : PriceTrain[:,0:1*PreTrain],lDisc :PriceTrain[:,1*(PreTrain+NextTrain):1*(PreTrain+NextTrain+LastTrain)] }) # Left discriminator train
                sess.run(LD_solver,feed_dict={XDisc: RData, ZGene: sample, YDisc : PriceTrain[:,0:1*PreTrain],lDisc :PriceTrain[:,1*(PreTrain+NextTrain):1*(PreTrain+NextTrain+LastTrain)] }) #mix real data with gen data 
            
    if i%2==1:
        for _ in range(n_round):
            
            for _ in range(n_gene):
                sess.run(Right_solver,feed_dict={ ZGene : sample,YDisc : PriceTrain[:,0:1*PreTrain],lDisc :PriceTrain[:,1*(PreTrain+NextTrain):1*(PreTrain+NextTrain+LastTrain)] } )    # right generator Train
                
            zdata   = sess.run(Right0,feed_dict={ ZGene : sample } )    # right generator output
            sample  = np.append(Train,zdata,axis=1) # parameter + right output
            RData   = sess.run(Right,feed_dict={ ZGene : sample,YDisc : PriceTrain[:,0:1*PreTrain],lDisc :PriceTrain[:,1*(PreTrain+NextTrain):1*(PreTrain+NextTrain+LastTrain)] } )     # RData = right(para + pre output)
            RData   = np.append(PriceTrain[:partial],RData[partial:],axis=0)
            
            for _ in range(n_disc):
                #sess.run(LD_solver,feed_dict={XDisc: PriceTrain, ZGene: sample, YDisc : PriceTrain[:,0:1*PreTrain],lDisc :PriceTrain[:,1*(PreTrain+NextTrain):1*(PreTrain+NextTrain+LastTrain)] }) # Left discriminator train
                sess.run(LD_solver,feed_dict={XDisc: RData, ZGene: sample, YDisc : PriceTrain[:,0:1*PreTrain],lDisc :PriceTrain[:,1*(PreTrain+NextTrain):1*(PreTrain+NextTrain+LastTrain)] }) #mix real data with gen data 
            
            for _ in range(n_gene):
                sess.run(Left_solver,feed_dict={ ZGene : sample,YDisc : PriceTrain[:,0:1*PreTrain],lDisc :PriceTrain[:,1*(PreTrain+NextTrain):1*(PreTrain+NextTrain+LastTrain)] } )
                
            zdata   = sess.run(Left0,feed_dict={ ZGene : sample } )     # left generator output
            sample  = np.append(Train,zdata,axis=1) # parameter + left output
            LData   = sess.run(Left,feed_dict={ ZGene : sample,YDisc : PriceTrain[:,0:1*PreTrain],lDisc :PriceTrain[:,1*(PreTrain+NextTrain):1*(PreTrain+NextTrain+LastTrain)] } )      # LData = Left(para + pre output)
            LData   = np.append(PriceTrain[:partial],LData[partial:],axis=0)
        
            for _ in range(n_disc):
                #sess.run([RD_solver],feed_dict={XDisc: PriceTrain, ZGene: sample, YDisc : PriceTrain[:,0:1*PreTrain],lDisc :PriceTrain[:,1*(PreTrain+NextTrain):1*(PreTrain+NextTrain+LastTrain)] }) # Right discriminator train
                sess.run(RD_solver,feed_dict={XDisc: LData, ZGene: sample, YDisc : PriceTrain[:,0:1*PreTrain],lDisc :PriceTrain[:,1*(PreTrain+NextTrain):1*(PreTrain+NextTrain+LastTrain)] }) #mix real data with gen data 
    
    #### Test Data for generator
    sample      = np.random.normal(0,1,size=[RowofTest,z_dim])      # random array
    sample      = np.append(Data_Test,sample,axis=1)                # para + random array
    
    vector = np.random.randint(2)
    if vector == 0:
        for _ in range(n_test):
            zdata   = sess.run(Left0,feed_dict={ ZGene : sample} )  # left starts generating
            sample  = np.append(Data_Test,zdata,axis=1)             # para + LData
            zdata   = sess.run(Right0,feed_dict={ ZGene : sample }) # right starts generating
            sample  = np.append(Data_Test,zdata,axis=1)             # para + RData
            
        DataL       = sess.run(Left0,feed_dict={ ZGene : sample } ) # left result
        sample      = np.append(Data_Test,zdata,axis=1)
        DataR       = sess.run(Right0,feed_dict={  ZGene : sample}) # right result     
            
    if vector == 1:
        for _ in range(n_test):
            zdata   = sess.run(Right0,feed_dict={ ZGene : sample} ) # right starts generating
            sample  = np.append(Data_Test,zdata,axis=1)             # para + RData
            zdata   = sess.run(Left0,feed_dict={ ZGene : sample })  # left starts generating
            sample  = np.append(Data_Test,zdata,axis=1)             # para + LData
            
        DataR       = sess.run(Right0,feed_dict={  ZGene : sample}) # right result  
        sample      = np.append(Data_Test,zdata,axis=1)
        DataL       = sess.run(Left0,feed_dict={ ZGene : sample } ) # Left result

    #### Test Data log      
    Data            = (DataR+DataL)/2                                               # average of two net
    Data            = np.reshape(Data,(RowofTest,10))               
    DataDiff        = np.reshape(Data[:,9]-Data[:,0],(RowofTest))                   # output flag 
    DataSig         = np.piecewise(DataDiff, [DataDiff < 0, DataDiff > 0], [-1, 1]) # output flag to binary
    DataProfit      = DataSig*PreDataTest[:,-2]                                     # flag * pricechange per min
    TrueSig         = PreDataTest[:,-1]                                             # the flag lebal by data
    index           = np.arange(0,RowofTest)
    TestWR[i]       = len(index[DataSig == TrueSig])/RowofTest*100                  # win rate of flag in output and true
    TCumData        = np.cumsum(DataProfit)                                         
    TestCu[i]       = TCumData[-1]                                                  # cumsum data of result
   
    #### Valid Data for generator     
    sample          = np.random.normal(0,1,size=[RowofValid,z_dim]) # random array
    sample          = np.append(Data_Valid,sample,axis=1)           # para + random array
    
    vector = np.random.randint(2)
    if vector == 0:
        for _ in range(n_test):
            zdata   = sess.run(Left0,feed_dict={ ZGene : sample} )  # left starts generating
            sample  = np.append(Data_Valid,zdata,axis=1)            # para + LData
            zdata   = sess.run(Right0,feed_dict={ ZGene : sample }) # right starts generating
            sample  = np.append(Data_Valid,zdata,axis=1)            # para + RData
            
        DataL       = sess.run(Left0,feed_dict={ ZGene : sample } ) # left result
        sample      = np.append(Data_Valid,zdata,axis=1)
        DataR       = sess.run(Right0,feed_dict={  ZGene : sample}) # right result    
            
    if vector == 1:
        for _ in range(n_test):
            zdata   = sess.run(Right0,feed_dict={ ZGene : sample} ) # right starts generating
            sample  = np.append(Data_Valid,zdata,axis=1)            # para + RData
            zdata   = sess.run(Left0,feed_dict={ ZGene : sample })  # left starts generating
            sample  = np.append(Data_Valid,zdata,axis=1)            # para + LData
            
        DataR       = sess.run(Right0,feed_dict={  ZGene : sample}) # right result 
        sample      = np.append(Data_Valid,zdata,axis=1)
        DataL       = sess.run(Left0,feed_dict={ ZGene : sample } ) # Left result
    
    ###### valid Data log 
    Data            = (DataR+DataL)/2                                               # average of two net
    Data            = np.reshape(Data,(RowofValid,10))
    DataDiff        = np.reshape(Data[:,9]-Data[:,0],(RowofValid))                  # output flag 
    DataSig         = np.piecewise(DataDiff, [DataDiff < 0, DataDiff > 0], [-1, 1]) # output flag to binary
    DataProfit      = DataSig*PreDataValid[:,-2]                                    # flag * pricechange per min
    TrueSig         = PreDataValid[:,-1]                                            # the flag lebal by data
    index           = np.arange(0,RowofValid)
    ValidWR[i]      = len(index[DataSig == TrueSig])/RowofValid*100                 # win rate of flag in output and true
    VCumData        = np.cumsum(DataProfit) 
    ValidCu[i]      = VCumData[-1]                                                  # cumsum data of result
    
    #judgfun         = TWR[i]*TCum[i]  ##用Test選結果
    
    ## Print 
    if i%100==50:
        
        # load batch data
        #Train,PriceTrain    = np.zeros((mb_size,len(PreDataTrain[0]))),np.zeros((mb_size,len(PricDataTrain[0])))    
        rand                = int(np.round( np.random.rand()*(len(PreDataTrain)-mb_size)))                           
        Train,PriceTrain    = PreDataTrain[rand:rand+mb_size,:],PricDataTrain[rand:rand+mb_size,:]  # parameter , price data
        
        # generate random data
        zdata       = np.random.normal(0,1,size=[mb_size,z_dim])        #random array
        sample      = np.append(Train,zdata,axis=1)                     #para + random array
        
        for _ in range(n_test):
            zdata   = sess.run(Left0,feed_dict={ ZGene : sample} )      # left starts generating
            sample  = np.append(Train,zdata,axis=1) # para + LData
            zdata   = sess.run(Right0,feed_dict={ ZGene : sample} )     # right starts generating
            sample  = np.append(Train,zdata,axis=1) # para + RData
        
        # output average of Discriminator
        RDT,LDT      = sess.run([avgRDTrue,avgLDTrue],feed_dict={ XDisc: PriceTrain, ZGene: sample,YDisc :  PriceTrain[:,0:1*PreTrain],lDisc :PriceTrain[:,1*(PreTrain+NextTrain):1*(PreTrain+NextTrain+LastTrain)]})
        RDF,LDF      = sess.run([avgRDFake,avgLDFake],feed_dict={ XDisc: PriceTrain, ZGene: sample,YDisc :  PriceTrain[:,0:1*PreTrain],lDisc :PriceTrain[:,1*(PreTrain+NextTrain):1*(PreTrain+NextTrain+LastTrain)]})
        FRD,FLD      = sess.run([avgRDFalse,avgLDFalse],feed_dict={ XDisc: PriceTrain, ZGene: sample,YDisc :  PriceTrain[:,0:1*PreTrain],lDisc :PriceTrain[:,1*(PreTrain+NextTrain):1*(PreTrain+NextTrain+LastTrain)]})
        
        print(str(i),' RDT= ',RDT,' RDF= ' ,RDF,' FARD= ' ,FRD,' // LDT=',LDT,' LDF= ',LDF,' FALD= ',FLD) 
        
        # print generated data and true data
        DataL   = sess.run(Left ,feed_dict={  ZGene : sample, YDisc : PriceTrain[:,0:1*PreTrain],lDisc :PriceTrain[:,1*(PreTrain+NextTrain):1*(PreTrain+NextTrain+LastTrain)]})
        DataR   = sess.run(Right,feed_dict={  ZGene : sample, YDisc : PriceTrain[:,0:1*PreTrain],lDisc :PriceTrain[:,1*(PreTrain+NextTrain):1*(PreTrain+NextTrain+LastTrain)]})
        TrueSample              = PriceTrain[0,0:1*NumTrain]
        Sample         = (DataR[0,:]+DataL[0,:])/2
        TrueSample,Sample  = np.reshape(TrueSample,(NumTrain,1)), np.reshape(Sample,(NumTrain,1))
        
        plt.figure()  
        plt.title('Right Net Sample')
        plt.xlim((0, NumTrain))
        plt.ylim((0, 1))
        plt.plot(Sample[:,0],color='y',label='Open')
        plt.xlabel('min')
        plt.ylabel('Normal Value')
        plt.savefig(Path + str(i)+'GAN_Sample')
        plt.close()
        
        plt.figure()  
        plt.title('True Sample')
        plt.xlim((0, NumTrain))
        plt.ylim((0, 1))
        plt.plot(TrueSample[:,0],color='y',label='Open')
        plt.xlabel('min')
        plt.ylabel('Normal Value')
        plt.savefig(Path + str(i)+'TrueData')
        plt.close()
        # print generated data and true data
        Pic_Save('TestWR = ' +str(TestWR[i]),TCumData,Path+'TestWR'+str(i),'Day','Point',250)
        Pic_Save('ValidWR = '+str(ValidWR[i]),VCumData,Path+'ValidWR'+str(i),'Day','Point',250)

#print winrate and cumsum for all 
fig     = plt.figure()
ax      = fig.add_subplot(111)  
ax1     = plt.subplot(311)
plt.plot(ValidCu)
ax2     = plt.subplot(312)
plt.plot(ValidWR)
ax3     = plt.subplot(313)
plt.plot(TestCu)
ax1.title.set_text('Valid : Cumsum')
ax2.title.set_text('Valid : WinRate')
ax3.title.set_text('Test : Cumsum')
ax.set_xlabel('Train Num')
ax.set_ylabel('Value')
plt.subplots_adjust(hspace=0.6)
plt.savefig(Path+TitleofPic+'CumsumTest')
plt.close()

fig     = plt.figure()
ax      = fig.add_subplot(111)  
ax1     = plt.subplot(311)
plt.plot(ValidCu)
ax2     = plt.subplot(312)
plt.plot(ValidWR)
ax3     = plt.subplot(313)
plt.plot(TestWR)
ax1.title.set_text('Valid : Cumsum')
ax2.title.set_text('Valid : WinRate')
ax3.title.set_text('Test : WinRate')
ax.set_xlabel('Train Num')
ax.set_ylabel('Value')
plt.subplots_adjust(hspace=0.6)
plt.savefig(Path+TitleofPic+'WinRateTest')
plt.close()
sess.close()



