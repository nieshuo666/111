import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .AlphaEdit_hparams import AlphaEditHyperParams
# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}


def apply_AlphaEdit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    cache_template: Optional[str] = None,
    cache_c = None,
    P = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests[:10]:
        print(
            f"MEMIT request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    # weights = {}
    # for layer in hparams.layers:
    #     param_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
    #     try:
    #         weight_param = nethook.get_parameter(model, param_name)
    #         print("++++++++++++++")
    #         print(id(weight_param))
    #         print("++++++++++++++")
    #         weights[param_name] = weight_param
    #     except LookupError as e:
    #         print(f"无法找到参数 {param_name}，错误信息：{e}")
    
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # print("pre_weight: {}",weights)
    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    # print("+++++++++++++++++++++++++++++++")
    # print(context_templates)
    # print("+++++++++++++++++++++++++++++++")
    # # 减输入
    # context_templates[1] = context_templates[1][:2]
    # print("-------------------")
    # print(context_templates)
    # print("-------------------")
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        # Retrieve k/v pair if already stored in cache
        # cache_fname = (
        #     Path(
        #         str(cache_template).format(
        #             z_layer, hparams.clamp_norm_factor, request["case_id"]
        #         )
        #     )
        #     if cache_template is not None
        #     else None
        # )
        data_loaded = False
        # if (
        #     cache_fname is not None  # Require cache template
        #     and cache_fname.exists()  # Cache file must exist
        # ):
        #     try:
        #         data = np.load(cache_fname)
        #         z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
        #         data_loaded = True
        #     except Exception as e:
        #         print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )

            z_list.append(cur_z)

            # if cache_fname is not None:
            #     cache_fname.parent.mkdir(exist_ok=True, parents=True)
            #     np.savez(
            #         cache_fname,
            #         **{
            #             "v_star": cur_z.detach().cpu().numpy(),
            #         },
            #     )
            #     print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1)

    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")
        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T

        # print(zs.device)
        # print(cur_zs.device)
        cur_zs=cur_zs.to(zs.device)


        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers  ΔAlphaEdit = RK₁ᵀP (KₚKₚᵀP + K₁K₁ᵀP + I)⁻¹
        # print(P[i,:,:].cuda().device)
        # print(layer_ks.device)
        # print(cache_c[i,:,:].cuda().device)
        # print(resid.device)
        p = P[i,:,:].cuda().to(zs.device)
        layer_ks=layer_ks.to(zs.device)
        c = cache_c[i,:,:].cuda().to(zs.device)
        ii=torch.eye(layer_ks.shape[0], dtype=torch.float,device="cuda").to(zs.device)
        resid =resid.to(zs.device)
        upd_matrix = torch.linalg.solve(
                p @ (layer_ks @ layer_ks.T + c) + hparams.L2*ii, p @ layer_ks @ resid.T
                # P[i,:,:].cuda() @ (layer_ks @ layer_ks.T + cache_c[i,:,:].cuda()) + hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float,device="cuda"), P[i,:,:].cuda() @ layer_ks @ resid.T
        )
        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        norm_before = torch.norm(weights[weight_name].flatten())
        print("norm_before:{}".format(norm_before))
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))
        # print(weights[weight_name].device)
        # print(upd_matrix.device)
        upd_matrix=upd_matrix.to(zs.device)
        weights[weight_name]=weights[weight_name].to(zs.device)
        print(upd_matrix)
        # print("名字是：{}",format(weight_name))
        with torch.no_grad():
            weights[weight_name][...] += upd_matrix
        for param_name, param in model.named_parameters():
            if weight_name == param_name:
                with torch.no_grad():
                    upd_matrix=upd_matrix.to(param.device)
                    param+=upd_matrix
                    print("77777")
                    print(param)
        # Clear GPU memory
        #del U,S,cov
        norm_after = torch.norm(weights[weight_name].flatten())
        print("norm_after: {}",format(norm_after))
        print("layers:{}",format(layer))
        # print(weights[weight_name])
        for x in [layer_ks, cur_zs, targets, upd_matrix]:
            x.cpu()
            del x
        # 更新作为之前的编辑Kp
    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        cache_c[i,:,:] += layer_ks.cpu() @ layer_ks.cpu().T
    # print("post_weight: {}",weights)
    # for param_name, param in model.named_parameters():
    #     if "model.layers.4.mlp.down_proj.weight" == param_name:
    #         with torch.no_grad():
    #         # 简单示例：给权重参数每个元素加0.1
    #             print(param)
    # for param_name, param in model.named_parameters():
    #     if "model.layers.5.mlp.down_proj.weight" == param_name:
    #         with torch.no_grad():
    #         # 简单示例：给权重参数每个元素加0.1
    #             print(param)
    print(f"Deltas successfully computed for {list(weights.keys())}")
    return model, cache_c


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """
    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)
    print(key)
    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")
    # file_path = "/data/ns/AlphaEdit-main/data/stats/llama3-8b-instruct/wikipedia_stats/{}.npz".format(layer_name)
    return (
        torch.inverse(COV_CACHE[key].to("cuda")) if inv else COV_CACHE[key].to("cuda")
    )
    # npz_file = np.load(file_path)
    # # Assuming the covariance matrix is stored under the key 'array' in the .npz file
    # cov_matrix_np = npz_file['mom2.mom2']  # Extract the array from the npz file
    # cov_matrix_torch = torch.from_numpy(cov_matrix_np).float()  # Convert to PyTorch tensor
    # if inv:
    #     # Compute the inverse of the covariance matrix
    #     cov_matrix_torch = torch.inverse(cov_matrix_torch)
    # print("---------")
    # print(cov_matrix_torch)
    # print("---------")
    # return cov_matrix_torch.to("cuda")  # Move the tensor to CUDA


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
