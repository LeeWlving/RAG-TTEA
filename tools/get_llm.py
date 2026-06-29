import os, yaml


def get_llm(model_name):
    if model_name == "deepseek-chat":
        from ._llm_engines import OpenAIEngine
        return OpenAIEngine(
            model_name="deepseek-chat",
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
    if model_name == "qwen-plus":
        from ._llm_engines import OpenAIEngine
        return OpenAIEngine(
            model_name="qwen-plus",
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
            base_url=os.environ.get(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
        )

    # load model configs from yaml
    with open(os.environ["KEYS_PATH"], "r") as f:
        keys_cfg = yaml.safe_load(f)['llm']
    
    if model_name == "gpt4o-mini":
        from ._llm_engines import AzureOpenAIEngine
        model_kwargs = keys_cfg['azure']['gpt4o-mini']
        llm = AzureOpenAIEngine(model_name=model_kwargs['model_name'],
                                api_key=model_kwargs['api_key'],
                                azure_endpoint=model_kwargs['azure_endpoint'],
                                api_version=model_kwargs['api_version'])
    elif model_name == "gpt4o":
        from ._llm_engines import AzureOpenAIEngine
        model_kwargs = keys_cfg['azure']['gpt4o']
        llm = AzureOpenAIEngine(model_name=model_kwargs['model_name'],
                                api_key=model_kwargs['api_key'],
                                azure_endpoint=model_kwargs['azure_endpoint'],
                                api_version=model_kwargs['api_version'])
    elif model_name == "gpt4o-mini-openai":
        from ._llm_engines import OpenAIEngine
        model_kwargs = keys_cfg['openai']['gpt4o-mini']
        llm = OpenAIEngine(model_name=model_kwargs['model_name'],
                           api_key=model_kwargs['api_key'])
    elif model_name in keys_cfg.get('deepseek', {}):
        from ._llm_engines import OpenAIEngine
        model_kwargs = keys_cfg['deepseek'][model_name]
        llm = OpenAIEngine(model_name=model_kwargs.get('model_name', model_name),
                           api_key=model_kwargs['api_key'],
                           base_url=model_kwargs.get('base_url', 'https://api.deepseek.com'))
        
    elif model_name == "claude-4.5-sonnet":
        from ._llm_engines import AnthropicVertexEngine
        model_kwargs = keys_cfg['gcp']['claude-4-5-sonnet']
        llm = AnthropicVertexEngine(model_name=model_kwargs['model_name'],
                                    project_id=model_kwargs['project_id'],
                                    region=model_kwargs['region'])
    # elif model_name == "gemini-3-flash":
    #     model_kwargs = keys_cfg['gcp']['gemini-3-flash']
    #     llm = GeminiEngine(model_name=model_kwargs['model_name'], 
    #                        project_id=model_kwargs['project_id'], 
    #                        region=model_kwargs['region'])
        

    elif model_name == "llama3-8B-I":
        from ._llama_engines import LlamaEngine
        llm = LlamaEngine(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", device="cuda:0")
    elif model_name == "qwen2.5-7B-I":
        from ._llama_engines import LlamaEngine
        llm = LlamaEngine(model_name="Qwen/Qwen2.5-7B-Instruct", device="cuda:0")
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    
    return llm 





if __name__ == "__main__":
    import os
    import sys
    pwd = "./Extraction-AD-Pipeline"
    os.environ["PYTHONPATH"] = pwd + ":" + os.environ.get("PYTHONPATH", "")
    os.environ["KEYS_PATH"] = pwd + "/keys.yaml"
    sys.path.append(pwd)
    
    # Example usage gpt4o
    engine = get_llm("gpt4o-mini-openai")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    response = engine(messages)
    print(response)
    
