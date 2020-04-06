import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from net import es
import numpy as np
import math
import os


def do_rollouts(model, random_seed, return_queue, env, negative_tag, max_episode_length):
    episode_return = 0
    episode_step = 0
    obs = env.reset()
    episode_return = 0
    episode_step = 0
    for step in range(max_episode_length):
        obs = torch.FloatTensor(np.expand_dims(obs, 0))
        output = model.forward(obs)
        prob = F.softmax(output, -1)
        action = prob.max(1)[1].detach().item()
        next_obs, reward, done, info = env.step(action)
        episode_return += reward
        episode_step += 1
        if done:
            break
        obs = next_obs
    return_queue.put([random_seed, episode_return, episode_step, negative_tag])


def perturb_model(model, random_seed, env, sigma):
    new_model = es(env.observation_space.shape[0], env.action_space.n)
    anti_model = es(env.observation_space.shape[0], env.action_space.n)
    new_model.load_state_dict(model.state_dict())
    anti_model.load_state_dict(model.state_dict())
    np.random.seed(random_seed)
    for (key, value), (anti_key, anti_value) in zip(new_model.get_params(), anti_model.get_params()):
        eps = np.random.normal(0, 1, value.size())
        value += torch.FloatTensor(sigma * eps)
        anti_value -= torch.FloatTensor(sigma * eps)
    return new_model, anti_model


def update_gradient(synced_model, returns, random_seeds, neg_list, episode_num, step_num, chkpt_dir, unperturb_result, model_num, learning_rate, sigma, lr_decay):
    def fitness_shape(returns):
        sort_returns = sorted(returns)[:: -1]
        returns_length = len(returns)
        shaped_returns = []
        denominator = sum([max(0, math.log(returns_length / 2 + 1, 2) - math.log(sort_returns.index(r) + 1, 2)) for r in returns])
        for r in returns:
            numerator = max(0, math.log(returns_length / 2 + 1, 2) - math.log(sort_returns.index(r) + 1, 2))
            shaped_returns.append(numerator / denominator + 1 / returns_length)
        return shaped_returns

    def unperturbed_rank(returns, unperturbed_results):
        nth_place = 1
        for r in returns:
            if r > unperturb_result:
                nth_place += 1
        return nth_place

    batch_size = len(returns)
    shaped_returns = fitness_shape(returns)
    rank = unperturbed_rank(returns, unperturb_result)

    for i in range(model_num):
        np.random.seed(random_seeds[i])
        negative_multiplier = -1 if neg_list[i] else 1
        reward = shaped_returns[i]
        for key, value in synced_model.get_params():
            eps = np.random.normal(0, 1, value.size())
            grad_step = learning_rate / (model_num * sigma) * reward * negative_multiplier * eps
            value += torch.FloatTensor(grad_step)
        learning_rate = learning_rate * lr_decay

    torch.save(synced_model.state_dict(), os.path.join(chkpt_dir, 'latest.pkl'))
    return synced_model


def render_env(model, env):
    while True:
        obs = env.reset()
        this_episode_return = 0
        while True:
            prob = F.softmax(model.forward(torch.FloatTensor(np.expand_dims(obs, 0))))
            action = prob.max(1)[1].detach().item()
            next_obs, reward, done, _ = env.step(action)
            env.render()
            this_episode_return += reward
            obs = next_obs
            if done:
                print('reward: {:.1f}'.format(this_episode_return))
                break


def generate_seeds_models(synced_model, env, sigma):
    np.random.seed()
    random_seed = np.random.randint(2 ** 30)
    new_model, anti_model = perturb_model(synced_model, random_seed, env, sigma)
    return random_seed, new_model, anti_model


def train(synced_model, env, chkpt_dir, max_gradient_updates, model_num, sigma, max_episode_length, learning_rate, lr_decay, variable_ep_len):
    episode_num = 0
    total_step_num = 0
    for epoch in range(max_gradient_updates):
        processes = []
        return_queue = mp.Queue()
        all_seeds = []
        all_models = []
        for j in range(int(model_num / 2)):
            random_seed, new_model, anti_model = generate_seeds_models(synced_model, env, sigma)
            all_seeds.append(random_seed)
            all_seeds.append(random_seed)
            all_models.extend([new_model, anti_model])

        is_negative = True
        while all_models:
            perturbed_model = all_models.pop()
            seed = all_seeds.pop()
            p = mp.Process(
                target=do_rollouts,
                args=(
                    perturbed_model,
                    seed,
                    return_queue,
                    env,
                    is_negative,
                    max_episode_length
                )
            )
            p.start()
            processes.append(p)
            is_negative = not is_negative

        p = mp.Process(
            target=do_rollouts,
            args=(
                synced_model,
                'dummy_seed',
                return_queue,
                env,
                'dummy_neg',
                max_episode_length
            )
        )
        p.start()
        processes.append(p)
        for p in processes:
            p.join()

        raw_results = [return_queue.get() for p in processes]
        random_seeds, all_returns, all_steps, negative_tag = zip(* raw_results)
        random_seeds = list(random_seeds)
        all_returns = list(all_returns)
        all_steps = list(all_steps)
        negative_tag = list(negative_tag)
        unperturbed_index = random_seeds.index('dummy_seed')
        random_seeds.pop(unperturbed_index)
        unperturbed_return = all_returns.pop(unperturbed_index)
        all_steps.pop(unperturbed_index)
        negative_tag.pop(unperturbed_index)

        total_step_num += sum(all_steps)
        episode_num += len(all_returns)
        synced_model = update_gradient(
            synced_model,
            all_returns,
            random_seeds,
            negative_tag,
            episode_num,
            total_step_num,
            chkpt_dir,
            unperturbed_return,
            model_num,
            learning_rate,
            sigma,
            lr_decay
        )

        print('epoch: {}  unperturbed_return: {}'.format(epoch, unperturbed_return))

        if variable_ep_len:
            max_episode_length = int(2 * sum(all_steps) / len(all_steps))
