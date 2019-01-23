import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

files= os.listdir("test_out")
p_files=[]
best_scores=[]
pass_epochs=[]
pars = []
layerss = []
mem_sizes=[]
update_everys=[]
learn_everys=[]
learning_rates =[]
eps_decays =[]
double_qnets=[]
delayers=[]
for f in files:
    if f[-2:]==".p":
        p_files.append(f)
        score_list, mean_score_list, pars = pickle.load(open("test_out/"+f, "rb"))
        solved_epoch=10000000
        solved=False
        best_score = 0
        for epoch in range(len(mean_score_list)):
            if mean_score_list[epoch] >13.0:
                if not solved:
                    solved_epoch=100*epoch
                    solved=True
            best_score=max(best_score, mean_score_list[epoch])

        best_scores.append(best_score)
        pass_epochs.append(solved_epoch)
        layerss.append(pars['layers_sel'])
        mem_sizes.append(pars['mem_size_sel'])
        update_everys.append(pars['update_every_sel'])
        learn_everys.append(pars['learn_every_sel'])
        learning_rates.append(pars['learning_rate_sel'])
        eps_decays.append(pars['eps_decay_sel'])
        double_qnets.append(pars['double_qnet_sel'])
        delayers.append(pars['delayer_sel'])

res_dict={"file_name": p_files, "best_score": best_scores, "pass_epoch":pass_epochs, "layer":layerss,
          "mem_size":mem_sizes, "update_every":update_everys, "learn_every":learn_everys,
          "learning_rate":learning_rates, "eps_decay": eps_decays, "double_qnet": double_qnets, "delayer":delayers}

res_df=pd.DataFrame(res_dict)

res_df=res_df.sort_values("best_score", ascending=False)
res_df.to_csv("optim_results.csv")


f= res_df.file_name[63]
score_list, mean_score_list, pars = pickle.load(open("test_out/" + f, "rb"))
plt.figure()
plt.plot(score_list)
x = range(len(mean_score_list))
plt.plot(np.array(x)*100.0, mean_score_list)
plt.plot([0, 800], [13, 13])
plt.ylabel('Score')
plt.xlabel('Episode #')

# colors= ["ro", "go", "rs", "gs" ]
#
#
# plt.figure() #mem_size: 5000+
# plt.plot(res_df["mem_size"].astype(float), res_df["best_score"], "ro")
# plt.figure() #learn every 4 or 16?
# plt.plot(res_df["learn_every"].astype(float), res_df["best_score"], "ro")
# plt.figure()
# plt.plot(res_df["double_qnet"].astype(float), res_df["best_score"], "ro")
# plt.figure()
# plt.plot(res_df["delayer"].astype(float), res_df["best_score"], "ro")
# plt.figure()
# plt.plot(res_df["eps_decay"].astype(float), res_df["best_score"], "ro")
#
# best_res_df=res_df[res_df["mem_size"]>=5000]
# best_res_df=best_res_df[res_df["eps_decay"]==0.99]
# best_res_df=best_res_df[res_df["learn_every"]==4]
# best_res_df=best_res_df[res_df["update_every"]<10]
# best_res_df=best_res_df[res_df["delayer"]==True]
# best_res_df=best_res_df[res_df["double_qnet"]==True]
#
# plt.figure() # update every:2
# plt.plot(best_res_df["update_every"].astype(float), best_res_df["best_score"], "ro")
#
# plt.figure() #learning rate 0.005
# plt.plot(best_res_df["learning_rate"].astype(float), best_res_df["best_score"], "ro")