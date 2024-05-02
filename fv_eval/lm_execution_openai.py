import pandas as pd
from tqdm import tqdm

import openai
from fv_eval import evaluation, prompts_svagen_design2sva, data, prompts_svagen_nl2sva


def lm_generate_sva_openai(
    model_name: str = "gpt-4",
    temp: float = 0.0,
    system_prompt: str = "",
    user_prompt: str = "",
    reference_solution: str = "",
    do_preprocessing: bool = True,
    metric_name: str = "bleu",
    max_tokens: int = 100,
):  
    assert 'gpt' in model_name
    client = openai.OpenAI()
    try:
        lm_response = client.chat.completions.create(
            model = model_name,
             messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens = max_tokens,
            temperature = temp
        )
        response_text = lm_response.choices[0].message
        if not metric_name:
           return {"lm_code_response": response_text, metric_name: 0.0}
        else:
            return evaluation.evaluate_sva_similarity(
                lm_response=response_text,
                reference_solution=reference_solution,
                do_preprocessing=do_preprocessing,
                metric_name="bleu",
            )
    except:
        return {"lm_code_response": "", metric_name: 0.0}


def run_svagen(
    save_dir: str,
    dataset_path: str = "data/svagen_3.csv",
    model_name_list: list[str] = ["gpt-4"],
    temp: float = 0.0,
    do_preprocessing: bool = True,
    metric_name: str = "bleu",
):
    num_icl_examples = dataset_path.split(".csv")[0].split("_")[-1]
    svagen_df = pd.read_csv(dataset_path, sep=",")

    print(f'Running SVAGen benchmark on {dataset_path}')
    for model_name in model_name_list:
        model_results = []
        for idx, row in tqdm(
            svagen_df.iterrows(),
            total=svagen_df.shape[0],
            desc=f"Running for {model_name}",
        ):
            system_prompt = prompts_svagen_nl2sva.SVAGEN_HEADER
            question_prompt = prompts_svagen_nl2sva.SVAGEN_QUESTION_PREAMBLE
            question_prompt += row.prompt
            question_prompt += "\n" + prompts_svagen_nl2sva.SVAGEN_QUESTION_POSTAMBLE
            user_prompt = prompts_svagen_nl2sva.ICL_EXAMPLE
            user_prompt += "\n\n" + prompts_svagen_nl2sva.SVAGEN_TB_PREAMBLE
            user_prompt += "\n" + row.testbench_context
            user_prompt += "\n\n" + question_prompt
            response_dict = {
                "dut_name": row.dut_name,
                "task_id": row.task_id,
                "question_prompt":  question_prompt,
                "full_prompt":  user_prompt,
                "question_prompt":  question_prompt,
                "ref_solution": row.ref_solution,
            }
            response_dict.update(
                lm_generate_sva_adlrchat(
                    model_name=model_name,
                    temp=temp,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    reference_solution=row.ref_solution,
                    do_preprocessing=do_preprocessing,
                    metric_name=metric_name,
                )
            )
            model_results.append(response_dict)
        model_results_df = pd.DataFrame(model_results)
        model_results_df[f"avg_{metric_name}"] = model_results_df[metric_name].mean()
        model_results_df.to_csv(
            save_dir + f"/{model_name}_svagen_{num_icl_examples}.csv"
        )


def run_svagen_from_dut(
    save_dir: str,
    dataset_path: str = "data_agr/data",
    model_name_list: list[str] = ["gpt-4"],
    temp: float = 0.0,
    do_preprocessing: bool = True,
    metric_name: str = "bleu",
):
    dut_rtl_texts = data.read_sv_testbenches(
        testbench_dir=dataset_path
    )
    print(f'Running SVAGen benchmark on {dataset_path}')
    for model_name in model_name_list:
        model_results = []
        for dut_name in tqdm(dut_rtl_texts.keys()):
            system_prompt = prompts_svagen_design2sva.SVAGEN_HEADER
            user_prompt = prompts_svagen_design2sva.SVAGEN_DUT_PREAMBLE
            user_prompt += dut_rtl_texts[dut_name]
            user_prompt += "\n\n" + prompts_svagen_design2sva.SVAGEN_QUESTION_PREAMBLE
            response_dict = {
                "dut_name": dut_name,
                "question_prompt":  user_prompt,
            }
            response_dict.update(
                lm_generate_sva_adlrchat(
                    model_name=model_name,
                    temp=temp,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    do_preprocessing=do_preprocessing,
                    metric_name="",
                    max_tokens=1500
                )
            )
            model_results.append(response_dict)
            model_results_df = pd.DataFrame(model_results)
            model_results_df.to_csv(
                save_dir + f"/{model_name}_svagen.csv"
            )