import os
import random
import pickle
import argparse
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Import custom evaluators
from advGenerate.environments.dl_environment_eval import DL_IDS_Evaluator
from advGenerate.environments.vib_environment_eval import VIB_IDS_Evaluator
from advGenerate.environments.transformer_environment_eval import Transformer_IDS_Evaluator
from advGenerate.environments.cnn_environment_eval import CNN_IDS_Evaluator
from advGenerate.environments.pandora_seq_environment_eval import PandoraSeq_IDS_Evaluator


# 1. Amoeba State Encoder
class StateEncoder(nn.Module):
    def __init__(self, device, state_num=2, num_layers=2, hidden_size=256):
        super(StateEncoder, self).__init__()
        self.num_layers = num_layers
        self.device = device
        self.hidden_size = hidden_size
        self.encoder = nn.GRU(
            input_size=state_num,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(
            self.device
        )

    def step(self, X, h=None):
        batch_size, seq_len, _ = X.size()
        if h is None:
            h = self.init_hidden(batch_size)
        encoded, h = self.encoder(X, h)
        return encoded.squeeze(dim=1), h


# 2. IDS Discriminator Wrapper
class CustomIDSDCRWrapper(nn.Module):
    """Wraps dynamically loaded IDS evaluators to act as Amoeba's discriminator."""
    def __init__(self, evaluator):
        super().__init__()
        self.evaluator = evaluator

    def forward(self, generated_flow):
        flow_np = generated_flow.detach().cpu().numpy()[0]
        sizes = [abs(f[0]) for f in flow_np]

        timestamps = [abs(f[1]) for f in flow_np]
        ipds = [timestamps[0]] + [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]

        _, _, pred_class = self.evaluator.predict(ipds, sizes, max_seq_len=128)

        score = 1.0 if pred_class != 0 else 0.0
        return torch.tensor([score], dtype=torch.float32)

# 3. Amoeba Gym Environment
class AmoebaCustomEnv(gym.Env):
    def __init__(self, discriminator, state_encoder, data, device):
        super().__init__()
        self.discriminator = discriminator
        self.state_encoder = state_encoder
        self.data = data
        self.device = device

        self.data_penalty = 1.0
        self.time_penalty = 2.0
        self.truncate_penalty = 0.05

        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([0.5, 0.1]), shape=(2,))
        self.observation_space = spaces.Box(-1, 1, shape=(256 * 2 + 1 + 2,))

        self.reset_virtual_flow_states()

    def reset_virtual_flow_states(self):
        self.generated_pkts = []
        self.original_pkts = []
        self.last_gen_hidden_states = None
        self.last_original_hidden_state = None
        self.truncate_num = 0

    def get_sample(self):
        while True:
            item = random.choice(self.data)
            if isinstance(item, tuple) and len(item) == 2:
                ipds, sizes = list(item[0]), list(item[1])
            elif isinstance(item, dict):
                ipds, sizes = list(item.get("ipds", [])), list(item.get("sizes", []))
            else:
                continue

            if len(ipds) > 5:
                timestamps = np.cumsum(ipds)
                flow = np.column_stack((sizes, timestamps))
                return torch.tensor(flow, dtype=torch.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pkt_counter = self.step_c = 0
        self.reset_virtual_flow_states()
        self.original_flow = self.get_sample()

        init_ob = self.original_flow[self.pkt_counter].to(self.device)
        self.original_pkts.append(init_ob)
        self.remaining_bytes = init_ob[0].item()
        self.last_t = init_ob[1].item()

        return self.get_encoded_states(), {}

    def get_encoded_states(self):
        with torch.no_grad():
            enc_orig, self.last_original_hidden_state = self.state_encoder.step(
                self.original_pkts[-1].view(1, 1, 2), self.last_original_hidden_state)

            if len(self.generated_pkts) == 0:
                enc_adv = torch.zeros_like(enc_orig, device=self.device)
            else:
                enc_adv, self.last_gen_hidden_states = self.state_encoder.step(
                    self.generated_pkts[-1].view(1, 1, 2), self.last_gen_hidden_states)

        enc_orig = enc_orig.detach().cpu().numpy().reshape((-1,))
        enc_adv = enc_adv.detach().cpu().numpy().reshape((-1,))
        next_pkt = self.original_pkts[-1].detach().cpu().numpy()
        binary_signal = np.ones((1,)) if self.truncate_num != 0 else np.zeros((1,))

        return np.concatenate([enc_orig, enc_adv, binary_signal, next_pkt], axis=0)

    def step(self, action):
        self.step_c += 1
        terminated = False

        pkt_size = self.remaining_bytes * (1.0 + action[0])
        t = self.last_t + action[1]

        added_delay = t - self.last_t
        pkt_diff = abs(pkt_size - self.remaining_bytes)

        action_tensor = torch.tensor([pkt_size, t], device=self.device, dtype=torch.float32)
        self.generated_pkts.append(action_tensor)

        # Reconstruct complete flow by appending remaining unperturbed original packets
        if self.pkt_counter + 1 < len(self.original_flow):
            rem_flow = self.original_flow[self.pkt_counter + 1:].clone()
            orig_last_t = self.original_flow[self.pkt_counter][1].item()
            time_shift = t - orig_last_t
            rem_flow[:, 1] += time_shift
            full_flow = torch.cat([torch.stack(self.generated_pkts, dim=0), rem_flow.to(self.device)], dim=0)
        else:
            full_flow = torch.stack(self.generated_pkts, dim=0)

        with torch.no_grad():
            generated_flow = full_flow.view(1, -1, 2)
            clf_result = self.discriminator(generated_flow).item()

        reward = -clf_result - (self.data_penalty * pkt_diff * 0.001) - (self.time_penalty * added_delay)

        next_pkt = torch.zeros(2)
        self.pkt_counter += 1
        if self.pkt_counter < len(self.original_flow) and self.step_c < 60:
            next_pkt = self.original_flow[self.pkt_counter]
            self.remaining_bytes = next_pkt[0].item()
            self.last_t = next_pkt[1].item()
        else:
            terminated = True

        next_pkt = next_pkt.to(self.device)
        self.original_pkts.append(next_pkt)

        return self.get_encoded_states(), reward, terminated, False, {}

# 4. Helper Evaluator Loader
def get_evaluator(defense: str, world_type: str, dataset: str, attack: str, feedback_mode: str = "hard_labeled"):
    num_classes = 3 if world_type == "open_world" else 2

    if defense == "dl":
        model_path = f"models/{world_type}/{dataset}/dl/{attack}_ids.pth"
        if not os.path.exists(model_path): return None
        return DL_IDS_Evaluator(models_dir=model_path, mode=feedback_mode, threshold=0.3)

    elif defense == "vib":
        model_path = f"models/{world_type}/{dataset}/vib/{attack}_vib.pth"
        if not os.path.exists(model_path): return None
        return VIB_IDS_Evaluator(models_dir=model_path, mode=feedback_mode, threshold=0.5, num_classes=num_classes)

    elif defense == "transformer":
        model_path = f"models/{world_type}/{dataset}/transformer/{attack}_ids.pth"
        if not os.path.exists(model_path): return None
        return Transformer_IDS_Evaluator(models_dir=model_path, mode=feedback_mode, threshold=0.5, num_classes=num_classes)

    elif defense == "cnn":
        model_path = f"models/{world_type}/{dataset}/cnn/{attack}_ids.pth"
        if not os.path.exists(model_path): return None
        return CNN_IDS_Evaluator(models_dir=model_path, mode=feedback_mode, threshold=0.5)

    elif defense in ["pandora", "pandora_seq"]:
        suffix = "_ids_open_world.pth" if world_type == "open_world" else "_ids.pth"
        model_path = f"models/{world_type}/{dataset}/pandora_seq/{attack}{suffix}"
        if not os.path.exists(model_path):
            model_path_base = f"models/{world_type}/{dataset}/pandora_seq/{attack}"
            if os.path.exists(model_path_base):
                model_path = model_path_base
            else:
                return None
        return PandoraSeq_IDS_Evaluator(models_dir=model_path, mode=feedback_mode, threshold=0.5)

    return None

# 5. Evaluation Execution
def run_amoeba_eval(data, evaluator, model, state_encoder, device, max_steps):
    ter_count = 0
    acr_count = 0
    mal_count = 0
    total_valid = 0
    total_all_steps = 0

    state_encoder.eval()

    with torch.no_grad():
        for item in tqdm(data, desc="Attacking Flows (Amoeba)", leave=False):
            if isinstance(item, tuple) and len(item) == 2:
                ipds, sizes = list(item[0]), list(item[1])
            elif isinstance(item, dict):
                ipds, sizes = list(item.get("ipds", [])), list(item.get("sizes", []))
            else:
                continue

            if len(ipds) <= 1:
                continue

            _, _, initial_class = evaluator.predict(ipds, sizes, max_seq_len=128)
            if initial_class == 0:
                continue

            total_valid += 1
            timestamps = np.cumsum(ipds).tolist()
            remaining_bytes = sizes[0]
            last_t = timestamps[0]

            generated_pkts = [[sizes[0], timestamps[0]]]
            pkt_counter = 0

            last_orig_h = None
            last_gen_h = None

            orig_pkt_tensor = torch.tensor([[sizes[0], timestamps[0]]], dtype=torch.float32, device=device).view(1, 1, 2)
            enc_orig, last_orig_h = state_encoder.step(orig_pkt_tensor, last_orig_h)

            steps_taken = 0
            last_pred_class = initial_class

            for step in range(1, max_steps + 1):
                if len(generated_pkts) == 1:
                    enc_adv = torch.zeros_like(enc_orig, device=device)
                else:
                    adv_pkt_tensor = torch.tensor([generated_pkts[-1]], dtype=torch.float32, device=device).view(1, 1, 2)
                    enc_adv, last_gen_h = state_encoder.step(adv_pkt_tensor, last_gen_h)

                enc_orig_np = enc_orig.detach().cpu().numpy().reshape((-1,))
                enc_adv_np = enc_adv.detach().cpu().numpy().reshape((-1,))
                next_pkt_np = np.array([remaining_bytes, last_t], dtype=np.float32)
                binary_signal = np.zeros((1,), dtype=np.float32)

                obs = np.concatenate([enc_orig_np, enc_adv_np, binary_signal, next_pkt_np], axis=0)

                action, _ = model.predict(obs, deterministic=True)

                pkt_size = remaining_bytes * (1.0 + float(action[0]))
                t = last_t + float(action[1])

                generated_pkts.append([pkt_size, t])

                # Extract modified prefix
                eval_sizes = [abs(p[0]) for p in generated_pkts]
                eval_ts = [abs(p[1]) for p in generated_pkts]

                # Append remaining original flow packets to evaluate full flow candidate
                if pkt_counter + 1 < len(sizes):
                    rem_sizes = sizes[pkt_counter + 1:]
                    last_gen_t = eval_ts[-1]
                    rem_ipds = [
                        timestamps[i] - timestamps[i - 1]
                        for i in range(pkt_counter + 1, len(timestamps))
                    ]
                    rem_ts = []
                    curr_t = last_gen_t
                    for delta in rem_ipds:
                        curr_t += delta
                        rem_ts.append(curr_t)

                    full_sizes = eval_sizes + rem_sizes
                    full_ts = eval_ts + rem_ts
                else:
                    full_sizes = eval_sizes
                    full_ts = eval_ts

                full_ipds = [full_ts[0]] + [
                    full_ts[i] - full_ts[i - 1] for i in range(1, len(full_ts))
                ]

                _, _, last_pred_class = evaluator.predict(full_ipds, full_sizes, max_seq_len=128)

                steps_taken = step

                pkt_counter += 1
                if pkt_counter < len(sizes):
                    remaining_bytes = sizes[pkt_counter]
                    last_t = timestamps[pkt_counter]
                    next_orig_tensor = torch.tensor([[remaining_bytes, last_t]], dtype=torch.float32, device=device).view(1, 1, 2)
                    enc_orig, last_orig_h = state_encoder.step(next_orig_tensor, last_orig_h)

                if last_pred_class == 0 or last_pred_class == 2:
                    break

            total_all_steps += steps_taken

            if last_pred_class == 0:
                ter_count += 1
            elif last_pred_class == 2:
                acr_count += 1
            else:
                mal_count += 1

            if total_valid >= 3000:
                break

    ter = (ter_count / total_valid * 100) if total_valid > 0 else 0.0
    acr = (acr_count / total_valid * 100) if total_valid > 0 else 0.0
    mal_rate = (mal_count / total_valid * 100) if total_valid > 0 else 0.0
    avg_steps = (total_all_steps / total_valid) if total_valid > 0 else 0.0

    return ter, acr, mal_rate, avg_steps

# 6. Main Pipeline
def main():
    parser = argparse.ArgumentParser(description="Amoeba Baseline Evaluation & Training Pipeline")
    parser.add_argument("--world_type", type=str, default="closed_world")
    parser.add_argument("--datasets", "--dataset", nargs="+", default=["cic2023"], dest="datasets")
    parser.add_argument("--defenses", "--defense", nargs="+", default=["dl"], dest="defenses")
    parser.add_argument("--attacks", "--attack", nargs="+", default=["dos"], dest="attacks")
    parser.add_argument("--feedback_mode", type=str, default="hard_labeled")
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--train_timesteps", type=int, default=10000)
    parser.add_argument("--out_md", type=str, default="amoeba_evaluation_results.md")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.out_md, "w") as f:
        f.write(f"# Amoeba Baseline Results ({args.world_type.upper()})\n\n")

        for dataset in args.datasets:
            f.write(f"## {dataset}\n\n")
            f.flush()

            for defense in args.defenses:
                f.write(f"### {dataset} | {defense.upper()}\n\n")
                f.flush()

                for attack in args.attacks:
                    dataset_pkl = f"data/{dataset}/{attack}.pkl"

                    if not os.path.exists(dataset_pkl):
                        print(f"[!] Dataset pickle not found: {dataset_pkl}. Skipping...")
                        continue

                    evaluator = get_evaluator(defense, args.world_type, dataset, attack, args.feedback_mode)
                    if evaluator is None:
                        print(f"[!] IDS model for {defense}/{args.world_type}/{dataset}/{attack} not found. Skipping...")
                        continue

                    with open(dataset_pkl, "rb") as pkl_file:
                        raw_data = pickle.load(pkl_file)

                    model_dir = f"models/amoeba_trained/{args.world_type}/{dataset}/{defense}/{attack}"
                    model_path = os.path.join(model_dir, "ppo_agent.zip")
                    enc_path = os.path.join(model_dir, "state_encoder.pth")

                    state_encoder = StateEncoder(device, state_num=2, num_layers=2, hidden_size=256).to(device)

                    # Auto-Train PPO model if not already saved
                    if not (os.path.exists(model_path) and os.path.exists(enc_path)):
                        print(f"[*] [Amoeba] No pre-trained model found for {dataset}/{defense}/{attack}. Training PPO agent...")
                        os.makedirs(model_dir, exist_ok=True)
                        ids_wrapper = CustomIDSDCRWrapper(evaluator).to(device)

                        env_kwargs = dict(discriminator=ids_wrapper, state_encoder=state_encoder, data=raw_data, device=device)
                        vec_env = make_vec_env(AmoebaCustomEnv, n_envs=4, env_kwargs=env_kwargs)

                        policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=[256, 64, 32])
                        model = PPO("MlpPolicy", vec_env, verbose=0, learning_rate=5e-4, policy_kwargs=policy_kwargs, n_steps=60)
                        model.learn(total_timesteps=args.train_timesteps)

                        model.save(os.path.join(model_dir, "ppo_agent"))
                        torch.save(state_encoder.state_dict(), enc_path)
                        print(f"[+] Training complete! Model saved to {model_dir}")
                    else:
                        print(f"[*] [Amoeba] Loading pre-trained model from {model_dir}")
                        model = PPO.load(model_path, device=device)
                        state_encoder.load_state_dict(torch.load(enc_path, map_location=device))

                    print(f"[*] [Amoeba] Running Evaluation: Dataset={dataset} | Defense={defense} | Attack={attack}")
                    ter, acr, mal_rate, avg_steps = run_amoeba_eval(
                        raw_data, evaluator, model, state_encoder, device, args.max_steps
                    )

                    f.write(f"#### {dataset} | {defense.upper()} | {attack}\n\n")
                    f.write(f"- **TER**: `{ter:.2f}%`  \n")
                    f.write(f"- **ACR**: `{acr:.2f}%`  \n")
                    f.write(f"- **Remains Malicious**: `{mal_rate:.2f}%`  \n")
                    f.write(f"- **Avg Steps**: `{avg_steps:.2f}`  \n\n")
                    f.flush()

    print(f"\n[★] Finished Amoeba Evaluation! Output written to: {args.out_md}")

if __name__ == "__main__":
    main()
