from banana_env import BananaEnv
from q_agent import QAgent
from time import sleep

top_networks_ckp=[
"top_configs/qnet_Banana_local_test_64.ckp",
"top_configs/qnet_Banana_local_test_7.ckp",
"top_configs/qnet_Banana_local_test_53.ckp",
"top_configs/qnet_Banana_local_test_34.ckp",
"top_configs/qnet_Banana_local_test_28.ckp"]

sel_network = 1

env = BananaEnv()
agent = QAgent(action_space=env.get_action_space_size(), state_space=env.get_state_space_size())
agent.load_checkpoint(top_networks_ckp[sel_network])
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
