"""
Show trained DQnet Agents in action on a gym or BananaCollection environment
"""

from banana_env import BananaEnv
from q_agent import QAgent
from time import sleep

top_opt_networks_ckp=[
"top_configs/qnet_Banana_local_test_64.ckp",
"top_configs/qnet_Banana_local_test_7.ckp",
"top_configs/qnet_Banana_local_test_53.ckp",
"top_configs/qnet_Banana_local_test_34.ckp",
"top_configs/qnet_Banana_local_test_28.ckp"]


bad_opt_networks_ckp="bad_config/qnet_Banana_local_test_87.ckp"
seeded_test_64_ckp={"local":"top_configs/qnet_banana_local_episode_740.ckp",
                    "target":"top_configs/qnet_banana_target_episode_740.ckp",
                    "delayer": "top_configs/qnet_banana_delayer_episode_740.ckp"}

sel_network = 1 #select one of the top scoring checkponts. The network will have different parameters
load_from_seeded_64: bool = False # ignore previus selection and select the re-trained (seeded) network for test_64 configuration. Here the environment was solved (>13.5) after 740 episodes
load_bad_network: bool = False # inglore the previous selection and load the worst network resulting form optimization.

env = BananaEnv()

agent = QAgent(action_space=env.get_action_space_size(), state_space=env.get_state_space_size())
if load_bad_network:
    agent.load_checkpoint(bad_opt_networks_ckp)
elif load_from_seeded_64: #target and delayer weights are not actually needed here, they would be needed to resume training as it was left.
    agent.load_checkpoint(local_checkpoint=seeded_test_64_ckp["local"],
                          target_checkpoint=seeded_test_64_ckp["target"],
                          delayer_checkpoint=seeded_test_64_ckp["delayer"])
else:
    agent.load_checkpoint(top_opt_networks_ckp[sel_network])

env.reset()
done=False
for i in range(5):
    print("Episode {:d}\n score: ".format(i), end=" " )
    done = False
    env.reset()
    score = 0
    while not done:
        exp = agent.act(env, 0)
        score = score + exp.reward
        if abs(exp.reward)>0.01:
            print(str(int(score)), sep=' ', end=' ', flush=True)
        done =exp.done
        sleep(0.02)
    print("\nfinal score:" + str(score)+"\n")
    sleep(1)
