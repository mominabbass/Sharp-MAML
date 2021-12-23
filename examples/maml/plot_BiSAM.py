import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 

rho1 = 0.0
rho2 = 0.5
rho3 = 0.05
rho4 = 0.005

data_nsam_t1 = np.load('save_results/results_MAML.npy')
data_sam_t1_rho1 = np.load('save_results/results_BiSAM_0.0.npy')
data_sam_t1_rho2 = np.load('save_results/results_BiSAM_0.5.npy')
data_sam_t1_rho3 = np.load('save_results/results_BiSAM_0.05.npy')
data_sam_t1_rho3_adap = np.load('save_results/results_BiSAM_0.05_adap.npy')
data_sam_t1_rho4 = np.load('save_results/results_BiSAM_0.005.npy')



# data_nsam_t2 = np.load('save_results/minibatch_bs_10_vbs_10_olrmu_3e-06_0.0_ilrmu_0.3_0.0_eta_0.5_T_10_K_10dec15/results_trial2.npy')
# data_sam_t2 = np.load('save_results/minibatch_bs_10_vbs_10_olrmu_3e-06_0.0_ilrmu_0.3_0.0_eta_0.5_T_10_K_10dec15/results_dec15_trial2.npy')

# data_nsam_t3 = np.load('save_results/minibatch_bs_10_vbs_10_olrmu_3e-06_0.0_ilrmu_0.3_0.0_eta_0.5_T_10_K_10dec15/results_trial3.npy')
# data_sam_t3 = np.load('save_results/minibatch_bs_10_vbs_10_olrmu_3e-06_0.0_ilrmu_0.3_0.0_eta_0.5_T_10_K_10dec15/results_dec15_trial3.npy')

# data_nsam_t4 = np.load('save_results/minibatch_bs_10_vbs_10_olrmu_3e-06_0.0_ilrmu_0.3_0.0_eta_0.5_T_10_K_10dec15/results_trial4.npy')
# data_sam_t4 = np.load('save_results/minibatch_bs_10_vbs_10_olrmu_3e-06_0.0_ilrmu_0.3_0.0_eta_0.5_T_10_K_10dec15/results_dec15_trial4.npy')

test_acc_nsam_t1  = data_nsam_t1[:,0]
test_loss_nsam_t1 = data_nsam_t1[:,1]

#time_nsam_t1 = data_nsam_t1[:,2]

# test_loss_nsam_t2 = data_nsam_t2[:,0]
# test_acc_nsam_t2  = data_nsam_t2[:,1]
# time_nsam_t2 = data_nsam_t2[:,2]

# test_loss_nsam_t3 = data_nsam_t3[:,0]
# test_acc_nsam_t3  = data_nsam_t3[:,1]
# time_nsam_t3 = data_nsam_t3[:,2]

# test_loss_nsam_t4 = data_nsam_t4[:,0]
# test_acc_nsam_t4  = data_nsam_t4[:,1]
# time_nsam_t4 = data_nsam_t4[:,2]


# test_loss_nsam = (test_loss_nsam_t1+test_loss_nsam_t2+test_loss_nsam_t3+test_loss_nsam_t4)/4
# test_acc_nsam = (test_acc_nsam_t1+test_acc_nsam_t2+test_acc_nsam_t3+test_acc_nsam_t4)/4


test_acc_sam_t1_rho1  = data_sam_t1_rho1[:,0]
test_loss_sam_t1_rho1 = data_sam_t1_rho1[:,1]


test_acc_sam_t1_rho2  = data_sam_t1_rho2[:,0]
test_loss_sam_t1_rho2 = data_sam_t1_rho2[:,1]

test_acc_sam_t1_rho3  = data_sam_t1_rho3[:,0]
test_loss_sam_t1_rho3 = data_sam_t1_rho3[:,1]
test_acc_sam_t1_rho3_adap  = data_sam_t1_rho3_adap[:,0]
test_loss_sam_t1_rho3_adap = data_sam_t1_rho3_adap[:,1]

test_acc_sam_t1_rho4  = data_sam_t1_rho4[:,0]
test_loss_sam_t1_rho4 = data_sam_t1_rho4[:,1]

#time_sam_t1 = data_sam_t1[:,2]

# test_loss_sam_t2 = data_sam_t2[:,0]
# test_acc_sam_t2  = data_sam_t2[:,1]
# time_sam_t2 = data_sam_t2[:,2]

# test_loss_sam_t3 = data_sam_t3[:,0]
# test_acc_sam_t3  = data_sam_t3[:,1]
# time_sam_t3 = data_sam_t3[:,2]

# test_loss_sam_t4 = data_sam_t4[:,0]
# test_acc_sam_t4  = data_sam_t4[:,1]
# time_sam_t4 = data_sam_t4[:,2]


# test_loss_sam = (test_loss_sam_t1+test_loss_sam_t2+test_loss_sam_t3+test_loss_sam_t4)/4
# test_acc_sam = (test_acc_sam_t1+test_acc_sam_t2+test_acc_sam_t3+test_acc_sam_t4)/4


plt.figure("test loss")
plt.title("test loss vs epochs - Omniglot (5-way 5-shot)")
plt.grid()
plt.xlabel('epoch')
plt.ylabel("test loss")
plt.plot(test_loss_nsam_t1)
plt.plot(test_loss_sam_t1_rho1)
#plt.plot(test_loss_sam_t1_rho2)
#plt.plot(test_loss_sam_t1_rho3)
#plt.plot(test_loss_sam_t1_rho4)
#plt.plot(test_loss_sam_t1_rho3_adap)
#plt.yscale("log")
plt.legend(['MAML', 'MAML+SAM (rho={})'.format(rho1), 
			'MAML+SAM (rho={})'.format(rho2), 
			'MAML+SAM (rho={})'.format(rho3), 
			'MAML+SAM (rho={})'.format(rho4), 
			'MAML+SAM (rho={}, adap)'.format(rho3)])
plt.savefig("./save_results/test_loss.png", dpi=600)
plt.show()
#plt.savefig("stepsize3.png", dpi=600)

plt.figure("test accuracy")
plt.title("test accuracy vs epochs - Omniglot (5-way 5-shot)")
plt.grid()
plt.xlabel('epoch')
plt.ylabel("test accuracy")
plt.plot(test_acc_nsam_t1)
plt.plot(test_acc_sam_t1_rho1)
#plt.plot(test_acc_sam_t1_rho2)
#plt.plot(test_acc_sam_t1_rho3)
#plt.plot(test_acc_sam_t1_rho4)
#plt.plot(test_acc_sam_t1_rho3_adap)
#plt.yscale("log")
plt.legend(['MAML', 'MAML+SAM (rho={})'.format(rho1), 
			'MAML+SAM (rho={})'.format(rho2), 
			'MAML+SAM (rho={})'.format(rho3), 
			'MAML+SAM (rho={})'.format(rho4), 
			'MAML+SAM (rho={}, adap)'.format(rho3)])
plt.savefig("./save_results/test_acc.png", dpi=600)
plt.show()


