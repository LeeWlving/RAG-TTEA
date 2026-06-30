


def get_attack_args(p, attack):

    p.add_argument(
        "--ak_max_query", dest="ak.max_query", default=50, type=int, help="Query budget got attack")

    if attack == "DGEA":
        p.add_argument(
            "--ak_emb_model", dest="ak.emb_model", default="MiniLM", type=str, help="Embedding model for DGEA")
        p.add_argument(
            "--ak_iterations", dest="ak.iterations", default=3, type=int, help="Number of iterations for query optimization")
        p.add_argument(
            "--ak_pool_size", dest="ak.pool_size", default=512, type=int, help="Token pool size for DGEA")
        p.add_argument(
            "--ak_allow_non_ascii", dest="ak.allow_non_ascii", default=False, action='store_true', help="Allow non-ascii characters in the query")
        p.add_argument(
            "--ak_command_prompt", dest="ak.command_prompt", default="copybreak/attack_template.txt", type=str, help="Command prompt path for DGEA")
        p.add_argument(
            "--ak_info_prompt", dest="ak.info_prompt", default="dgea/ak_suffix.txt", type=str, help="Info prompt path for DGEA")
        p.add_argument(
            "--ak_random_vec", dest="ak.random_vec", default="embedding_statistics.csv", type=str, help="Random vector distribution file path for DGEA")
    
    elif attack == "CopyBreak":
        p.add_argument(
            "--ak_llm_model", dest="ak.llm_model", default="gpt4o-mini", type=str, help="LLM model for CopyBreak")
        p.add_argument(
            "--ak_emb_model", dest="ak.emb_model", default="MiniLM", type=str, help="Embedding model for CopyBreak")
        p.add_argument(
            "--ak_attack_template", dest="ak.attack_template", default="copybreak/attack_template.txt", type=str, help="Attack template prompt path for CopyBreak")
        p.add_argument(
            "--ak_sim_thresh", dest="ak.sim_thresh", default=0.7, type=float, help="Similarity threshold for exploration in CopyBreak")
        p.add_argument(
            "--ak_explore_template", dest="ak.explore_template", default="copybreak/explore_template.txt", type=str, help="Exploration template prompt path for CopyBreak")
        p.add_argument(
            "--ak_exploit_template", dest="ak.exploit_template", default="copybreak/exploit_template.txt", type=str, help="Exploitation template prompt path for CopyBreak")
        p.add_argument(
            "--ak_exchange_rate", dest="ak.exchange_rate", default=0.5, type=float, help="Rate of change between exploration and exploitation in CopyBreak")
        p.add_argument(
            "--ak_iterations", dest="ak.iterations", default=10, type=int, help="Number of iterations for each exploration/exploitation in CopyBreak")
        p.add_argument(
            "--ak_explore_temperature", dest="ak.explore_temperature", default=0.7, type=float, help="Temperature for exploration generation in CopyBreak")
        p.add_argument(
            "--ak_exploit_temperature", dest="ak.exploit_temperature", default=0.7, type=float, help="Temperature for exploitation generation in CopyBreak")
        p.add_argument(
            "--ak_num_of_each_reason", dest="ak.num_of_reason", default=3, type=int, help="Number of each back/forward reasoning queries to generate per anchor chunk in CopyBreak")
        p.add_argument(
            "--ak_anchor_domain", dest="ak.anchor_domain", default="auto", choices=["auto", "enron", "health", "pokemon", "literature", "general"], type=str, help="Domain used for CopyBreak exploit anchor normalization")
        p.add_argument(
            "--ak_normalize_exploit_anchors", dest="ak.normalize_exploit_anchors", default=True, action="store_true", help="Normalize CopyBreak exploit anchor chunks before reasoning-query generation")
        p.add_argument(
            "--ak_no_normalize_exploit_anchors", dest="ak.normalize_exploit_anchors", action="store_false", help="Use raw CopyBreak exploit anchor chunks")
        
    elif attack == "IKEA":
        p.add_argument(
            "--ak_attack_llm", dest="ak.attack_llm", default="gpt4o-mini", type=str, help="LLM model for IKEA attack")
        p.add_argument(
            "--ak_attack_emb_model", dest="ak.attack_emb_model", default="MiniLM", type=str, help="Embedding model for IKEA attack")
        p.add_argument(
            "--ak_device", dest="ak.device", default="cpu", type=str, help="Device for IKEA embedding model")
        p.add_argument(
            "--ak_topic_word", dest="ak.topic_word", default="enron emails", type=str, help="Topic word for IKEA attack")
        p.add_argument(
            "--ak_num_anchors", dest="ak.num_anchors", default=5, type=int, help="Number of anchor points for IKEA attack")
        p.add_argument(
            "--ak_anchor_gen_template", dest="ak.anchor_gen_template", default="ikea/anchor_gen_template.txt", type=str, help="Anchor generation template prompt path for IKEA attack")
        p.add_argument(
            "--ak_query_gen_iterations", dest="ak.query_gen_iterations", default=5, type=int, help="Number of query generation iterations per anchor for IKEA attack")
        p.add_argument(
            "--ak_thresh_sim_topic", dest="ak.thresh_sim_topic", default=0.5, type=float, help="Similarity threshold between anchor and topic word for IKEA attack")
        p.add_argument(
            "--ak_thresh_dissim_anchor", dest="ak.thresh_dissim_anchor", default=0.5, type=float, help="Dissimilarity threshold among anchors for IKEA attack")
        p.add_argument(
            "--ak_thresh_q_anchor", dest="ak.thresh_q_anchor", default=0.5, type=float, help="Similarity threshold between query and anchor for IKEA attack")
        p.add_argument(
            "--ak_sample_temperature", dest="ak.sample_temperature", default=1, type=float, help="Sampling temperature for IKEA attack")
        p.add_argument(
            "--ak_anchor_query_gen_template", dest="ak.anchor_query_gen_template", default="ikea/anchor_query_gen_template.txt", type=str, help="Anchor to query generation template prompt path for IKEA attack")
        p.add_argument(
            "--ak_thresh_irrelevant", dest="ak.thresh_irrelevant", default=0.5, type=float, help="Threshold to filter irrelevant queries for IKEA attack")
        p.add_argument(
            "--ak_thresh_outlier", dest="ak.thresh_outlier", default=0.5, type=float, help="Threshold to filter outlier queries for IKEA attack")
        p.add_argument(
            "--ak_penalty_irrelevant", dest="ak.penalty_irrelevant", default=0.5, type=float, help="Penalty weight for irrelevant queries for IKEA attack")
        p.add_argument(
            "--ak_penalty_refusal", dest="ak.penalty_refusal", default=0.5, type=float, help="Penalty weight for refusal queries for IKEA attack")
        p.add_argument(
            "--ak_thresh_qy_sim", dest="ak.thresh_qy_sim", default=0.5, type=float, help="Threshold for query-response similarity for IKEA attack")
        p.add_argument(
            "--ak_gamma", dest="ak.gamma", default=0.5, type=float, help="Trust region scale for IKEA attack")
        p.add_argument(
            "--ak_anchor_mutate_gen_template", dest="ak.anchor_mutate_gen_template", default="ikea/anchor_mutate.txt", type=str, help="Anchor mutation generation template prompt path for IKEA attack")
        p.add_argument(
            "--ak_thresh_stop_q", dest="ak.thresh_stop_q", default=0.9, type=float, help="Threshold to stop query generation for IKEA attack")
        p.add_argument(
            "--ak_thresh_stop_y", dest="ak.thresh_stop_y", default=0.9, type=float, help="Threshold to stop query generation for IKEA attack")

    elif attack == "TTEA":
        p.add_argument(
            "--ak_attack_llm", dest="ak.attack_llm", default="gpt4o-mini", type=str, help="LLM model for TTEA attack-side generation")
        p.add_argument(
            "--ak_shadow_llm", dest="ak.shadow_llm", default="gpt4o-mini", type=str, help="Shadow LLM used to estimate semantic shift for TTEA")
        p.add_argument(
            "--ak_attack_emb_model", dest="ak.attack_emb_model", default="MiniLM", type=str, help="Embedding model for TTEA")
        p.add_argument(
            "--ak_attack_template", dest="ak.attack_template", default="ttea/attack_template.txt", type=str, help="Attack instruction template path for TTEA")
        p.add_argument(
            "--ak_device", dest="ak.device", default="cpu", type=str, help="Device for TTEA embedding model")
        p.add_argument(
            "--ak_topic_word", dest="ak.topic_word", default="public knowledge", type=str, help="Topic prior used to label the root taxonomy node")
        p.add_argument(
            "--ak_taxonomy_path", dest="ak.taxonomy_path", default=None, type=str, help="Optional JSON taxonomy path for TTEA")
        p.add_argument(
            "--ak_taxonomy_builder", dest="ak.taxonomy_builder", default="static", choices=["static", "llm"], type=str, help="Taxonomy builder for TTEA: static built-in tree or LLM-generated public taxonomy")
        p.add_argument(
            "--ak_taxonomy_preset", dest="ak.taxonomy_preset", default="auto", choices=["auto", "general", "medicine", "pokemon", "academic", "business", "literature"], type=str, help="Static taxonomy preset used when TTEA taxonomy_builder is static or LLM fallback is needed")
        p.add_argument(
            "--ak_llm_taxonomy_children", dest="ak.llm_taxonomy_children", default=5, type=int, help="Number of child categories requested from the LLM per TTEA taxonomy expansion")
        p.add_argument(
            "--ak_max_depth", dest="ak.max_depth", default=3, type=int, help="Maximum taxonomy depth loaded by TTEA")
        p.add_argument(
            "--ak_max_children", dest="ak.max_children", default=8, type=int, help="Maximum children per taxonomy node loaded by TTEA")
        p.add_argument(
            "--ak_max_anchors", dest="ak.max_anchors", default=12, type=int, help="Maximum anchors cached per mature TTEA leaf")
        p.add_argument(
            "--ak_temperature", dest="ak.temperature", default=0.4, type=float, help="Reserved generation temperature for TTEA")
        p.add_argument(
            "--ak_ucb_c", dest="ak.ucb_c", default=0.8, type=float, help="UCB exploration weight for TTEA scheduler")
        p.add_argument(
            "--ak_lambda_prior", dest="ak.lambda_prior", default=0.5, type=float, help="Posterior prior weight for TTEA scheduler")
        p.add_argument(
            "--ak_gamma_entropy", dest="ak.gamma_entropy", default=0.2, type=float, help="Sibling entropy weight for TTEA scheduler")
        p.add_argument(
            "--ak_eta_shift", dest="ak.eta_shift", default=2.0, type=float, help="Semantic shift weight for TTEA posterior update")
        p.add_argument(
            "--ak_rho_reward", dest="ak.rho_reward", default=1.0, type=float, help="Novelty reward weight for TTEA posterior update")
        p.add_argument(
            "--ak_shift_threshold", dest="ak.shift_threshold", default=0.18, type=float, help="Average semantic-shift threshold for TTEA expansion/maturity")
        p.add_argument(
            "--ak_prune_threshold", dest="ak.prune_threshold", default=0.04, type=float, help="Low semantic-shift threshold for pruning TTEA nodes")
        p.add_argument(
            "--ak_mature_shift_epsilon", dest="ak.mature_shift_epsilon", default=0.04, type=float, help="Semantic-shift stability window for mature leaves")
        p.add_argument(
            "--ak_min_mature_visits", dest="ak.min_mature_visits", default=2, type=int, help="Minimum visits before TTEA can prune or mature a node")
        p.add_argument(
            "--ak_saturation_visits", dest="ak.saturation_visits", default=5, type=int, help="Recent zero-reward visits before TTEA marks a node saturated")
        p.add_argument(
            "--ak_novelty_threshold", dest="ak.novelty_threshold", default=0.92, type=float, help="Max similarity below which a parsed chunk is new in TTEA memory")
        p.add_argument(
            "--ak_expand_min_visits", dest="ak.expand_min_visits", default=3, type=int, help="Minimum visits before TTEA can force-expand a non-leaf node")
        p.add_argument(
            "--ak_expand_reward_window", dest="ak.expand_reward_window", default=3, type=int, help="Recent reward window used by TTEA early expansion")
        p.add_argument(
            "--ak_expand_reward_epsilon", dest="ak.expand_reward_epsilon", default=0.0, type=float, help="Recent reward mean threshold for TTEA early expansion")
        p.add_argument(
            "--ak_expand_repeat_threshold", dest="ak.expand_repeat_threshold", default=0.67, type=float, help="Node-level repeated top-k threshold for TTEA early expansion")
        p.add_argument(
            "--ak_expand_entropy_threshold", dest="ak.expand_entropy_threshold", default=0.75, type=float, help="Sibling entropy threshold for low-reward TTEA early expansion")
        p.add_argument(
            "--ak_expand_retire_parent", dest="ak.expand_retire_parent", default=True, action="store_true", help="Retire a forced-expanded parent so scheduler drills into active children")
        p.add_argument(
            "--ak_expand_keep_parent", dest="ak.expand_retire_parent", action="store_false", help="Keep forced-expanded parents schedulable")
        p.add_argument(
            "--ak_expand_generate_leaf_children", dest="ak.expand_generate_leaf_children", default=True, action="store_true", help="Generate child nodes for triggered static leaves before max_depth")
        p.add_argument(
            "--ak_no_expand_generate_leaf_children", dest="ak.expand_generate_leaf_children", action="store_false", help="Disable dynamic child generation for triggered static leaves")
        p.add_argument(
            "--ak_expand_anchor_fallback", dest="ak.expand_anchor_fallback", default=True, action="store_true", help="Use extracted anchors as child nodes when dynamic taxonomy split returns no children")
        p.add_argument(
            "--ak_no_expand_anchor_fallback", dest="ak.expand_anchor_fallback", action="store_false", help="Disable anchor-based child fallback for early expansion")
        p.add_argument(
            "--ak_use_frontier_anchors", dest="ak.use_frontier_anchors", default=True, action="store_true", help="Use anchors extracted from newly retrieved docs to center the next TTEA probe query")
        p.add_argument(
            "--ak_no_use_frontier_anchors", dest="ak.use_frontier_anchors", action="store_false", help="Disable frontier-anchor-centered TTEA probe queries")
        p.add_argument(
            "--ak_frontier_anchors_per_round", dest="ak.frontier_anchors_per_round", default=4, type=int, help="Maximum frontier anchors extracted from newly retrieved docs per TTEA round")
        p.add_argument(
            "--ak_frontier_anchor_boost", dest="ak.frontier_anchor_boost", default=0.35, type=float, help="Scheduler utility boost for TTEA nodes with queued frontier anchors")
        p.add_argument(
            "--ak_normalize_exploit_anchors", dest="ak.normalize_exploit_anchors", default=True, action="store_true", help="Normalize and filter TTEA exploit anchors before generating leaf exploitation queries")
        p.add_argument(
            "--ak_no_normalize_exploit_anchors", dest="ak.normalize_exploit_anchors", action="store_false", help="Use raw TTEA exploit anchors")
    elif attack == "RandomText":
        p.add_argument(
            "--ak_llm_model", dest="ak.llm_model", default="gpt4o-mini", type=str, help="LLM model for RandomText attack")
        p.add_argument(
            "--ak_attack_template", dest="ak.attack_template", default="random/attack_template.txt", type=str, help="Attack instruction template path for RandomText attack")
        p.add_argument(
            "--ak_system_prompt", dest="ak.system_prompt", default="random/gen_system.txt", type=str, help="System prompt path for RandomText attack")
        p.add_argument(
            "--ak_template", dest="ak.template", default="random/gen_template.txt", type=str, help="Prompt template path for RandomText attack")
        p.add_argument(
            "--ak_temperature", dest="ak.temperature", default=0.9, type=float, help="Generation temperature for RandomText attack")
        
    elif attack == "RandomToken":
        p.add_argument(
            "--ak_emb_model", dest="ak.emb_model", default="MiniLM", type=str, help="Embedding model for RandomToken attack")
        p.add_argument(
            "--ak_pool_size", dest="ak.pool_size", default=512, type=int, help="Token pool size for RandomToken attack")
        p.add_argument(
            "--ak_allow_non_ascii", dest="ak.allow_non_ascii", default=False, action='store_true', help="Allow non-ascii characters in the query for RandomToken attack")
        p.add_argument(
            "--ak_attack_template", dest="ak.attack_template", default="random/attack_template.txt", type=str, help="Attack instruction template path for RandomToken attack")
        
    elif attack == "RandomEmb":
        p.add_argument(
            "--ak_emb_model", dest="ak.emb_model", default="MiniLM", type=str, help="Embedding model for RandomEmb attack")
        p.add_argument(
            "--ak_random_vec", dest="ak.random_vec", default="embedding_statistics.csv", type=str, help="Random vector distribution file path for RandomEmb attack")
        p.add_argument(
            "--ak_attack_template", dest="ak.attack_template", default="copybreak/attack_template.txt", type=str, help="Attack instruction template path for RandomEmb attack")
        p.add_argument(
            "--ak_iterations", dest="ak.iterations", default=3, type=int, help="Number of iterations for RandomEmb attack")
        p.add_argument(
            "--ak_pool_size", dest="ak.pool_size", default=512, type=int, help="Token pool size for RandomEmb attack")
        p.add_argument(
            "--ak_allow_non_ascii", dest="ak.allow_non_ascii", default=False, action='store_true', help="Allow non-ascii characters in the query for RandomEmb attack")
        p.add_argument(
            "--ak_info_prompt", dest="ak.info_prompt", default="random/ak_suffix.txt", type=str, help="Info prompt path for RandomEmb attack")

    elif attack == "Utility":
        p.add_argument(
            "--ak_data_path", dest="ak.data_path", default="./data/Enron", type=str, help="Path to dataset folder containing utility_questions.jsonl for Utility attack")
        

        

    return p
