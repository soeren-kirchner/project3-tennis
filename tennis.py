
import numpy as np
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment


from multi_agent import MultiAgent

env = UnityEnvironment(file_name="Tennis.app")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
print("Number of agents:", num_agents)

action_size = brain.vector_action_space_size
print("Size of each action:", action_size)

states = env_info.vector_observations
state_size = states.shape[1]
print("There are {} agents. Each observes a state with length: {}".format(states.shape[0], state_size))
print("The state for the first agent looks like:", states[0])


def gather(env_info):
    rewards = env_info.rewards
    next_states = env_info.vector_observations
    dones = env_info.local_done
    return rewards, next_states, dones 


def print_status_and_save(multi_agent, episode, scores, total_scores, save=False):
    total_scores = np.array(total_scores).T
    max = total_scores[0][-1]
    avg100 = total_scores[1][-1]
    agents_values = ", ".join([f'{value: .2f}' for value in scores])
    if save:
        multi_agent.save('max')
    savedMark = "[S]" if save else ""
    print(f"Ep {episode:4}\tMaxAvg100: {avg100:.2f}\tMax: {max:.2f} [{agents_values}]\t -- {multi_agent.agents[0].noise} {savedMark}")  


def plot_scores(scores_array, labels, save_as=None):
    fig = plt.figure(figsize=(12,7))

    for scores, label in zip(scores_array, labels):
        scores = np.asarray(scores)
        if scores.ndim > 1:
            transposed = scores.T
            plt.plot(np.arange(1, len(scores)+1), transposed[0], color='silver', label=label)
            plt.plot(np.arange(1, len(scores)+1), transposed[1], label=f'{label} - Avg100')
        else:
            plt.plot(np.arange(1, len(scores)+1), scores, label=label)

    plt.axhline(y=.5, color='green', linestyle='dashed')
    for y in [1.0, 2.0, 2.7]:
        plt.axhline(y=y, color='lightgray', linestyle='dashed')

    plt.legend(loc='lower right')

    plt.ylabel('Score')
    plt.xlabel('Episode #')

    if save_as:
        plt.savefig(save_as)

    plt.show()


def train(multi_agent, n_episodes=3000, max_t=1000):

    print(multi_agent)

    total_scores = []
    maxAvg100 = 0

    for episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]       # reset the environment    
        states = env_info.vector_observations                   # get the current state (for each agent)
        scores = np.zeros(num_agents)                           # initialize the score (for each agent)
        multi_agent.reset()
        
        for _ in range(max_t):
            actions = multi_agent.act(states)
            env_info = env.step(actions)[brain_name]            # send all actions to the environment
            rewards, next_states, dones = gather(env_info)
            multi_agent.step(states, actions, rewards, next_states, dones) # send actions to the agent
            scores += env_info.rewards                          # update the score (for each agent)
            states = next_states                                # roll over states to next time step
            
            if np.any(dones):                                   # exit loop if episode finished
                break
        
        avg100 = np.mean(np.asarray(total_scores)[-100:]) if len(total_scores) > 0 else 0
        total_scores.append([max(scores), avg100])
        print_status_and_save(multi_agent, episode, scores, total_scores, avg100 > maxAvg100)
        maxAvg100 = max(maxAvg100, avg100)

    multi_agent.save("last")
    return total_scores


def play(multi_agent, n_episodes=1, max_t=1000):

    env_info = env.reset(train_mode=False)[brain_name] 

    for _ in range(1, n_episodes+1):
        states = env_info.vector_observations                   # get the current state (for each agent)
        multi_agent.reset()
        
        for _ in range(max_t):
            actions = multi_agent.act(states, False)
            env_info = env.step(actions)[brain_name]            # send all actions to the environment
            _, next_states, _ = gather(env_info)
            states = next_states  


if __name__ == "__main__":
    multi_agent = MultiAgent(num_agents, state_size, action_size, noise_type="normal")
    multi_agent.load(suffix="max")

    play(multi_agent, n_episodes=100)
    print(len(multi_agent.memory))

    env.close()