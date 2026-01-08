from verl import DataProto
import torch
import numpy as np

class ProcessRewardManager:
    
    def __init__(self, tokenizer, num_examine, normalize_by_length) -> None:
        self.tokenizer = tokenizer

    def __call__(self, data: DataProto, return_dict: bool = False):
         # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

            turn_reward = data_item.non_tensor_batch['rewards']
            reward_tensor[i, valid_response_length - 1] = torch.tensor(turn_reward, dtype=torch.float32, device=prompt_ids.device)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {},
            }
        else:
            return reward_tensor