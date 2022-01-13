import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 

def get_sub(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub_s = "ₐ₈cDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    res = x.maketrans(''.join(normal), ''.join(sub_s))
    return x.translate(res)

rho1 = 0.0
rho2 = 0.5
rho3 = 0.05
rho4 = 0.005
rho5 = 0.0005
rho6 = 0.00005

#all times obtained on cuda:1
total_time_maml 		   = 1615.58			#1591.33 on cuda:3
total_time_lower_sam_maml  = 1585.92
total_time_upper_sam_maml  = 1604.79
total_time_both_sam_maml   = 1586.20

# print('total_time_maml: ', total_time_maml)
# print('total_time_lower_sam_maml: ', total_time_lower_sam_maml)
# print('total_time_upper_sam_maml: ', total_time_upper_sam_maml)
# print('total_time_both_sam_maml: ',  total_time_both_sam_maml)

maml_acc_best = 55.00
maml_acc_avg = 47.44

# maml_sam_upper_acc_best_5e2 = 95.27
# maml_sam_upper_acc_avg_5e2  = 94.32

# maml_sam_upper_acc_best_5e3 = 96.35
# maml_sam_upper_acc_avg_5e3  = 95.62

# maml_sam_upper_acc_best_5e4 = 96.56
# maml_sam_upper_acc_avg_5e4  = 95.82

# maml_sam_upper_acc_best_5e5 = 96.50
# maml_sam_upper_acc_avg_5e5  = 97.12


maml_sam_lower_acc_best_5e2 = 57.33
maml_sam_lower_acc_avg_5e2  = 46.67

maml_sam_lower_acc_best_5e3 = 52.33
maml_sam_lower_acc_avg_5e3  = 45.22

maml_sam_lower_acc_best_5e4 = 54.33
maml_sam_lower_acc_avg_5e4  = 46.44

maml_sam_lower_acc_best_5e5 = 53.33
maml_sam_lower_acc_avg_5e5  = 45.89


# maml_sam_both_acc_best_5e2 = 95.29
# maml_sam_both_acc_avg_5e2  = 94.58

# maml_sam_both_acc_best_5e3 = 96.29
# maml_sam_both_acc_avg_5e3  = 95.85

# maml_sam_both_acc_best_5e4 = 96.37
# maml_sam_both_acc_avg_5e4  = 96.06

# maml_sam_both_acc_best_5e5 = 96.67
# maml_sam_both_acc_avg_5e5  = 95.74



# best_upper_acc = [maml_sam_upper_acc_best_5e2, maml_sam_upper_acc_best_5e3, maml_sam_upper_acc_best_5e4, maml_sam_upper_acc_best_5e5]
# avg_upper_acc = [maml_sam_upper_acc_avg_5e2, maml_sam_upper_acc_avg_5e3, maml_sam_upper_acc_avg_5e4, maml_sam_upper_acc_avg_5e5]

best_lower_acc = [maml_sam_lower_acc_best_5e2, maml_sam_lower_acc_best_5e3, maml_sam_lower_acc_best_5e4, maml_sam_lower_acc_best_5e5]
avg_lower_acc = [maml_sam_lower_acc_avg_5e2, maml_sam_lower_acc_avg_5e3, maml_sam_lower_acc_avg_5e4, maml_sam_lower_acc_avg_5e5]

# best_both_acc = [maml_sam_both_acc_best_5e2, maml_sam_both_acc_best_5e3, maml_sam_both_acc_best_5e4, maml_sam_both_acc_best_5e5]
# avg_both_acc = [maml_sam_both_acc_avg_5e2, maml_sam_both_acc_avg_5e3, maml_sam_both_acc_avg_5e4, maml_sam_both_acc_avg_5e5]




# x_axis = [1e-5, 1e-4, 1e-3, 1e-2]

# print('best lower maml orig: ', maml_acc_best) 
# print('avg lower maml morig:  ', maml_acc_avg) 

print('best lower: ', best_lower_acc) 
print('avg lower:  ', avg_lower_acc) 
# print('\nbest upper: ', best_upper_acc) 
# print('avg upper:  ', avg_upper_acc) 
# print('\nbest both: ', best_both_acc) 
# print('avg both:  ', avg_both_acc) 


# plt.figure("test loss")
# plt.title("test loss vs epochs - omniglot (20-way 1-shot)")
# plt.grid()
# plt.xlabel('perturbation $\epsilon$')
# plt.ylabel("5-way accuracy")
# plt.plot(best_lower_acc, linestyle="--",  marker='+')
# plt.plot(best_upper_acc, linestyle="--",  marker='+')
# plt.plot(best_both_acc, linestyle="--",  marker='+')
# plt.legend([#'MAML',
# 			'$SAM+MAML_{lower}$',
# 			'$SAM+MAML_{upper}$',
# 			'$SAM+MAML_{both}$',
# 			])
# #plt.savefig("./save_results_min/test_loss_miniimagenet_avg.png", dpi=600)
# plt.show()
# #plt.savefig("stepsize3.png", dpi=600)



# plt.figure("test accuracy")
# plt.title("test accuracy vs epochs - miniimagenet (5-way 1-shot)")
# plt.grid()
# plt.xlabel('epoch')
# plt.ylabel("test accuracy")
# plt.plot(test_acc_nsam_t1_1shot_avg)
# #plt.plot(test_acc_sam_t1_1shot_rho1)
# plt.plot(test_acc_sam_t1_1shot_rho3_avg)
# plt.plot(test_acc_sam_t1_1shot_rho4_avg)
# plt.plot(test_acc_sam_t1_1shot_rho5_avg)
# plt.plot(test_acc_sam_t1_1shot_rho6_avg)
# #plt.plot(test_acc_sam_t1_1shot_rho7)
# # plt.plot(test_acc_nsam_t1_5shot)
# #plt.plot(test_acc_sam_t1_rho1[0:100])
# #plt.yscale("log")
# plt.legend(['MAML',
# 			# 'MAML 5-way 5-shot',
# 			#'MAML+SAM (rho={})'.format(rho1),
# 			'MAML+SAM (lower) (rho={})'.format(rho3),
# 			'MAML+SAM (lower) (rho={})'.format(rho4),
# 			'MAML+SAM (lower) (rho={})'.format(rho5),
# 			'MAML+SAM (lower) (rho={})'.format(rho6),
# 			#'MAML+SAM (rho={})'.format(rho7),
# 			])
# plt.savefig("./save_results_min/test_acc_miniimagenet_avg.png", dpi=600)
# plt.show()


