import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 

rho1 = 0.0
rho2 = 0.5
rho3 = 0.05
rho4 = 0.005
rho4_1 = 0.009
rho4_2 = 0.001
rho5 = 0.0005
rho6 = 0.00005
rho7 = 0.000005


time_1_ANIL_lower = 133.47 #cuda 3
time_2_ANIL_lower = 131.55 #cuda 1
time_3_ANIL_lower = 133.32 #cuda 1
time_4_ANIL_lower = 132.89 #cuda 3

data_nsam_t1_1shot_t1 = np.load('save_results/results_ANIL_miniimagenet_5way_1shot_t1.npy')
data_sam_t1_1shot_rho3_t1 = np.load('save_results/results_BiSAM_0.05_miniimagenet_5way_1shot_lower_t1.npy')
data_sam_t1_1shot_rho4_t1 = np.load('save_results/results_BiSAM_0.005_miniimagenet_5way_1shot_lower_t1.npy')
data_sam_t1_1shot_rho5_t1 = np.load('save_results/results_BiSAM_0.0005_miniimagenet_5way_1shot_lower_t1.npy')
data_sam_t1_1shot_rho6_t1 = np.load('save_results/results_BiSAM_5e-05_miniimagenet_5way_1shot_lower_t1.npy')

data_nsam_t1_1shot_t2 = np.load('save_results/results_ANIL_miniimagenet_5way_1shot_t2.npy')
data_sam_t1_1shot_rho3_t2 = np.load('save_results/results_BiSAM_0.05_miniimagenet_5way_1shot_lower_t4.npy')
data_sam_t1_1shot_rho4_t2 = np.load('save_results/results_BiSAM_0.005_miniimagenet_5way_1shot_lower_t2.npy')
data_sam_t1_1shot_rho5_t2 = np.load('save_results/results_BiSAM_0.0005_miniimagenet_5way_1shot_lower_t2.npy')
data_sam_t1_1shot_rho6_t2 = np.load('save_results/results_BiSAM_5e-05_miniimagenet_5way_1shot_lower_t2.npy')

data_nsam_t1_1shot_t3 = np.load('save_results/results_ANIL_miniimagenet_5way_1shot_t3.npy')
data_sam_t1_1shot_rho3_t3 = np.load('save_results/results_BiSAM_0.05_miniimagenet_5way_1shot_lower_t3.npy')
data_sam_t1_1shot_rho4_t3 = np.load('save_results/results_BiSAM_0.005_miniimagenet_5way_1shot_lower_t3.npy')
data_sam_t1_1shot_rho5_t3 = np.load('save_results/results_BiSAM_0.0005_miniimagenet_5way_1shot_lower_t3.npy')
data_sam_t1_1shot_rho6_t3 = np.load('save_results/results_BiSAM_5e-05_miniimagenet_5way_1shot_lower_t3.npy')



test_acc_nsam_t1_1shot_t1  = data_nsam_t1_1shot_t1[:,0]
test_loss_nsam_t1_1shot_t1 = data_nsam_t1_1shot_t1[:,1]

test_acc_sam_t1_1shot_rho3_t1 = data_sam_t1_1shot_rho3_t1[:,0]
test_loss_sam_t1_1shot_rho3_t1 = data_sam_t1_1shot_rho3_t1[:,1]

test_acc_sam_t1_1shot_rho4_t1  = data_sam_t1_1shot_rho4_t1[:,0]
test_loss_sam_t1_1shot_rho4_t1 = data_sam_t1_1shot_rho4_t1[:,1]

test_acc_sam_t1_1shot_rho5_t1  = data_sam_t1_1shot_rho5_t1[:,0]
test_loss_sam_t1_1shot_rho5_t1 = data_sam_t1_1shot_rho5_t1[:,1]

test_acc_sam_t1_1shot_rho6_t1  = data_sam_t1_1shot_rho6_t1[:,0]
test_loss_sam_t1_1shot_rho6_t1 = data_sam_t1_1shot_rho6_t1[:,1]




test_acc_nsam_t1_1shot_t2  = data_nsam_t1_1shot_t2[:,0]
test_loss_nsam_t1_1shot_t2 = data_nsam_t1_1shot_t2[:,1]

test_acc_sam_t1_1shot_rho3_t2 = data_sam_t1_1shot_rho3_t2[:,0]
test_loss_sam_t1_1shot_rho3_t2 = data_sam_t1_1shot_rho3_t2[:,1]

test_acc_sam_t1_1shot_rho4_t2  = data_sam_t1_1shot_rho4_t2[:,0]
test_loss_sam_t1_1shot_rho4_t2 = data_sam_t1_1shot_rho4_t2[:,1]

test_acc_sam_t1_1shot_rho5_t2  = data_sam_t1_1shot_rho5_t2[:,0]
test_loss_sam_t1_1shot_rho5_t2 = data_sam_t1_1shot_rho5_t2[:,1]

test_acc_sam_t1_1shot_rho6_t2  = data_sam_t1_1shot_rho6_t2[:,0]
test_loss_sam_t1_1shot_rho6_t2 = data_sam_t1_1shot_rho6_t2[:,1]




test_acc_nsam_t1_1shot_t3  = data_nsam_t1_1shot_t3[:,0]
test_loss_nsam_t1_1shot_t3 = data_nsam_t1_1shot_t3[:,1]

test_acc_sam_t1_1shot_rho3_t3 = data_sam_t1_1shot_rho3_t3[:,0]
test_loss_sam_t1_1shot_rho3_t3 = data_sam_t1_1shot_rho3_t3[:,1]

test_acc_sam_t1_1shot_rho4_t3  = data_sam_t1_1shot_rho4_t3[:,0]
test_loss_sam_t1_1shot_rho4_t3 = data_sam_t1_1shot_rho4_t3[:,1]

test_acc_sam_t1_1shot_rho5_t3  = data_sam_t1_1shot_rho5_t3[:,0]
test_loss_sam_t1_1shot_rho5_t3 = data_sam_t1_1shot_rho5_t3[:,1]

test_acc_sam_t1_1shot_rho6_t3  = data_sam_t1_1shot_rho6_t3[:,0]
test_loss_sam_t1_1shot_rho6_t3 = data_sam_t1_1shot_rho6_t3[:,1]



test_acc_nsam_t1_1shot_avg = (test_acc_nsam_t1_1shot_t1+test_acc_nsam_t1_1shot_t2+test_acc_nsam_t1_1shot_t3)/3
test_loss_nsam_t1_1shot_avg = (test_loss_nsam_t1_1shot_t1+test_loss_nsam_t1_1shot_t2+test_loss_nsam_t1_1shot_t3)/3

test_acc_sam_t1_1shot_rho3_avg = (test_acc_sam_t1_1shot_rho3_t1+test_acc_sam_t1_1shot_rho3_t2+test_acc_sam_t1_1shot_rho3_t3)/3
test_loss_sam_t1_1shot_rho3_avg = (test_loss_sam_t1_1shot_rho3_t1+test_loss_sam_t1_1shot_rho3_t2+test_loss_sam_t1_1shot_rho3_t3)/3

test_acc_sam_t1_1shot_rho4_avg = (test_acc_sam_t1_1shot_rho4_t1+test_acc_sam_t1_1shot_rho4_t2+test_acc_sam_t1_1shot_rho4_t3)/3
test_loss_sam_t1_1shot_rho4_avg = (test_loss_sam_t1_1shot_rho4_t1+test_loss_sam_t1_1shot_rho4_t2+test_loss_sam_t1_1shot_rho4_t3)/3

test_acc_sam_t1_1shot_rho5_avg = (test_acc_sam_t1_1shot_rho5_t1+test_acc_sam_t1_1shot_rho5_t2+test_acc_sam_t1_1shot_rho5_t3)/3
test_loss_sam_t1_1shot_rho5_avg = (test_loss_sam_t1_1shot_rho5_t1+test_loss_sam_t1_1shot_rho5_t2+test_loss_sam_t1_1shot_rho5_t3)/3

test_acc_sam_t1_1shot_rho6_avg = (test_acc_sam_t1_1shot_rho6_t1+test_acc_sam_t1_1shot_rho6_t2+test_acc_sam_t1_1shot_rho6_t3)/3
test_loss_sam_t1_1shot_rho6_avg = (test_loss_sam_t1_1shot_rho6_t1+test_loss_sam_t1_1shot_rho6_t2+test_loss_sam_t1_1shot_rho6_t3)/3



print('max accuracy (trial 1) ANIL                       (5-way 1-shot) (in first 1000 epochs):  {} (in {} epochs)'.format(np.max(test_acc_nsam_t1_1shot_t1[0:1000]), np.argmax(test_acc_nsam_t1_1shot_t1[0:1000])))
print('max accuracy (trial 1) ANIL+SAM_lower rho=0.05    (5-way 1-shot) (in first 1000 epochs):  {} (in {} epochs)'.format(np.max(test_acc_sam_t1_1shot_rho3_t1[0:1000]), np.argmax(test_acc_sam_t1_1shot_rho3_t1[0:1000])))
print('max accuracy (trial 1) ANIL+SAM_lower rho=0.005   (5-way 1-shot) (in first 1000 epochs):  {} (in {} epochs)'.format(np.max(test_acc_sam_t1_1shot_rho4_t1[0:1000]), np.argmax(test_acc_sam_t1_1shot_rho4_t1[0:1000])))
print('max accuracy (trial 1) ANIL+SAM_lower rho=0.0005  (5-way 1-shot) (in first 1000 epochs):  {} (in {} epochs)'.format(np.max(test_acc_sam_t1_1shot_rho5_t1[0:1000]), np.argmax(test_acc_sam_t1_1shot_rho5_t1[0:1000])))
print('max accuracy (trial 1) ANIL+SAM_lower rho=0.00005 (5-way 1-shot) (in first 1000 epochs):  {} (in {} epochs)\n'.format(np.max(test_acc_sam_t1_1shot_rho6_t1[0:1000]), np.argmax(test_acc_sam_t1_1shot_rho6_t1[0:1000])))

print('max accuracy (trial 2) ANIL                       (5-way 1-shot) (in first 1000 epochs): \t {} (in {} epochs)'.format(np.max(test_acc_nsam_t1_1shot_t2[0:1000]), np.argmax(test_acc_nsam_t1_1shot_t2[0:1000])))
print('max accuracy (trial 2) ANIL+SAM_lower rho=0.05    (5-way 1-shot) (in first 1000 epochs): \t {} (in {} epochs)'.format(np.max(test_acc_sam_t1_1shot_rho3_t2[0:1000]), np.argmax(test_acc_sam_t1_1shot_rho3_t2[0:1000])))
print('max accuracy (trial 2) ANIL+SAM_lower rho=0.005   (5-way 1-shot) (in first 1000 epochs): \t {} (in {} epochs)'.format(np.max(test_acc_sam_t1_1shot_rho4_t2[0:1000]), np.argmax(test_acc_sam_t1_1shot_rho4_t2[0:1000])))
print('max accuracy (trial 2) ANIL+SAM_lower rho=0.0005  (5-way 1-shot) (in first 1000 epochs): \t {} (in {} epochs)'.format(np.max(test_acc_sam_t1_1shot_rho5_t2[0:1000]), np.argmax(test_acc_sam_t1_1shot_rho5_t2[0:1000])))
print('max accuracy (trial 2) ANIL+SAM_lower rho=0.00005 (5-way 1-shot) (in first 1000 epochs): \t {} (in {} epochs)\n'.format(np.max(test_acc_sam_t1_1shot_rho6_t2[0:1000]), np.argmax(test_acc_sam_t1_1shot_rho6_t2[0:1000])))

print('max accuracy (trial 3) ANIL                       (5-way 1-shot) (in first 1000 epochs): {} (in {} epochs)'.format(np.max(test_acc_nsam_t1_1shot_t3[0:1000]), np.argmax(test_acc_nsam_t1_1shot_t3[0:1000])))
print('max accuracy (trial 3) ANIL+SAM_lower rho=0.05    (5-way 1-shot) (in first 1000 epochs): {} (in {} epochs)'.format(np.max(test_acc_sam_t1_1shot_rho3_t3[0:1000]), np.argmax(test_acc_sam_t1_1shot_rho3_t3[0:1000])))
print('max accuracy (trial 3) ANIL+SAM_lower rho=0.005   (5-way 1-shot) (in first 1000 epochs): {} (in {} epochs)'.format(np.max(test_acc_sam_t1_1shot_rho4_t3[0:1000]), np.argmax(test_acc_sam_t1_1shot_rho4_t3[0:1000])))
print('max accuracy (trial 3) ANIL+SAM_lower rho=0.0005  (5-way 1-shot) (in first 1000 epochs): {} (in {} epochs)'.format(np.max(test_acc_sam_t1_1shot_rho5_t3[0:1000]), np.argmax(test_acc_sam_t1_1shot_rho5_t3[0:1000])))
print('max accuracy (trial 3) ANIL+SAM_lower rho=0.00005 (5-way 1-shot) (in first 1000 epochs): {} (in {} epochs)\n'.format(np.max(test_acc_sam_t1_1shot_rho6_t3[0:1000]), np.argmax(test_acc_sam_t1_1shot_rho6_t3[0:1000])))

print('max avg accuracy (3 trials) ANIL                       (5-way 1-shot) (in first 1000 epochs): {} (in {} epochs)'.format(np.max(test_acc_nsam_t1_1shot_avg[0:1000]), np.argmax(test_acc_nsam_t1_1shot_avg[0:1000])))
print('max avg accuracy (3 trials) ANIL+SAM_lower rho=0.05    (5-way 1-shot) (in first 1000 epochs): {} (in {} epochs)'.format(np.max(test_acc_sam_t1_1shot_rho3_avg[0:1000]), np.argmax(test_acc_sam_t1_1shot_rho3_avg[0:1000])))
print('max avg accuracy (3 trials) ANIL+SAM_lower rho=0.005   (5-way 1-shot) (in first 1000 epochs): {} (in {} epochs)'.format(np.max(test_acc_sam_t1_1shot_rho4_avg[0:1000]), np.argmax(test_acc_sam_t1_1shot_rho4_avg[0:1000])))
print('max avg accuracy (3 trials) ANIL+SAM_lower rho=0.0005  (5-way 1-shot) (in first 1000 epochs): {} (in {} epochs)'.format(np.max(test_acc_sam_t1_1shot_rho5_avg[0:1000]), np.argmax(test_acc_sam_t1_1shot_rho5_avg[0:1000])))
print('max avg accuracy (3 trials) ANIL+SAM_lower rho=0.00005 (5-way 1-shot) (in first 1000 epochs): {} (in {} epochs)\n'.format(np.max(test_acc_sam_t1_1shot_rho6_avg[0:1000]), np.argmax(test_acc_sam_t1_1shot_rho6_avg[0:1000])))


plt.figure("test loss")
plt.title("test loss vs epochs - miniimagenet (5-way 1-shot)")
plt.grid()
plt.xlabel('epoch')
plt.ylabel("test loss")
plt.plot(test_loss_nsam_t1_1shot_avg)
#plt.plot(test_loss_sam_t1_1shot_rho1)
plt.plot(test_loss_sam_t1_1shot_rho3_avg)
plt.plot(test_loss_sam_t1_1shot_rho4_avg)
plt.plot(test_loss_sam_t1_1shot_rho5_avg)
plt.plot(test_loss_sam_t1_1shot_rho6_avg)
#plt.plot(test_loss_sam_t1_1shot_rho7)
#plt.plot(test_loss_nsam_t1_5shot)
plt.legend(['ANIL',
			#'ANIL 5-way 5-shot',
			#'ANIL+SAM (rho={})'.format(rho1),
			'ANIL+SAM (lower) (rho={})'.format(rho3),
			'ANIL+SAM (lower) (rho={})'.format(rho4),
			'ANIL+SAM (lower) (rho={})'.format(rho5),
			'ANIL+SAM (lower) (rho={})'.format(rho6),
			#'ANIL+SAM (rho={})'.format(rho7),
			])
plt.savefig("./save_results/test_loss_miniimagenet_avg.png", dpi=600)
plt.show()
#plt.savefig("stepsize3.png", dpi=600)


plt.figure("test accuracy")
plt.title("test accuracy vs epochs - miniimagenet (5-way 1-shot)")
plt.grid()
plt.xlabel('epoch')
plt.ylabel("test accuracy")
plt.plot(test_acc_nsam_t1_1shot_avg)
#plt.plot(test_acc_sam_t1_1shot_rho1)
plt.plot(test_acc_sam_t1_1shot_rho3_avg)
plt.plot(test_acc_sam_t1_1shot_rho4_avg)
plt.plot(test_acc_sam_t1_1shot_rho5_avg)
plt.plot(test_acc_sam_t1_1shot_rho6_avg)
#plt.plot(test_acc_sam_t1_1shot_rho7)
# plt.plot(test_acc_nsam_t1_5shot)
#plt.plot(test_acc_sam_t1_rho1[0:100])
#plt.yscale("log")
plt.legend(['ANIL',
			#'ANIL 5-way 5-shot',
			#'ANIL+SAM (rho={})'.format(rho1),
			'ANIL+SAM (lower) (rho={})'.format(rho3),
			'ANIL+SAM (lower) (rho={})'.format(rho4),
			'ANIL+SAM (lower) (rho={})'.format(rho5),
			'ANIL+SAM (lower) (rho={})'.format(rho6),
			#'ANIL+SAM (rho={})'.format(rho7),
			])
plt.savefig("./save_results/test_acc_miniimagenet_avg.png", dpi=600)
plt.show()


