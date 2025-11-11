import asyncio
import json
import re
from typing import Union
from collections import Counter, defaultdict
import warnings
import json_repair
import pandas as pd
import io

from .utils import (
    list_of_list_to_csv,
    truncate_list_by_token_size,
    split_string_by_multi_markers,
    logger,
    locate_json_string_body_from_string,
    process_combine_contexts,
    clean_str,
    edge_vote_path,
    encode_string_by_tiktoken,
    decode_tokens_by_tiktoken,
    is_float_regex,
    pack_user_ass_to_openai_messages,
    compute_mdhash_id,
    calculate_similarity,
    cal_path_score_list,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS


def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])

    edge_keywords = clean_str(record_attributes[4])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entitiy_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]

    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )

    # description = await _handle_entity_relation_summary(
    #     entity_name, description, global_config
    # )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_keywords.extend(
            split_string_by_multi_markers(already_edge["keywords"], [GRAPH_FIELD_SEP])
        )

    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    keywords = GRAPH_FIELD_SEP.join(
        sorted(set([dp["keywords"] for dp in edges_data] + already_keywords))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    # description = await _handle_entity_relation_summary(
    #     (src_id, tgt_id), description, global_config
    # )
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
    )

    return edge_data


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    entity_name_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())
    # if global_config['RAGmode'] == 'minirag':
    #     # entity_extract_prompt = PROMPTS["entity_extraction_noDes"]
    #     entity_extract_prompt = PROMPTS["entity_extraction"]
    # else:
    entity_extract_prompt = PROMPTS["entity_extraction"]

    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]

    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        final_result = await use_llm_func(hint_prompt)

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    print()  # clear the progress bar
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    all_relationships_data = await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if not len(all_relationships_data):
        logger.warning(
            "Didn't extract any relationships, maybe your LLM is not working"
        )
        return None

    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)
    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + " " + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    if entity_name_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="Ename-"): {
                "content": dp["entity_name"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_name_vdb.upsert(data_for_vdb)

    if relationships_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "content": dp["keywords"]
                + " " + dp["src_id"]
                + " " + dp["tgt_id"]
                + " " + dp["description"],
            }
            for dp in all_relationships_data
        }

        await relationships_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst


async def local_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    context = None
    use_model_func = global_config["llm_model_func"]

    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query)
    result = await use_model_func(kw_prompt)
    json_text = locate_json_string_body_from_string(result)

    try:
        keywords_data = json.loads(json_text)
        keywords = keywords_data.get("low_level_keywords", [])
        keywords = ", ".join(keywords)
    except json.JSONDecodeError:
        try:
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[1].split("}")[0] + "}"

            keywords_data = json.loads(result)
            keywords = keywords_data.get("low_level_keywords", [])
            keywords = ", ".join(keywords)
        # Handle parsing error
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"]
    if keywords:
        context = await _build_local_query_context(
            keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )

    print("\n\n CONTEXT DONE \n\n")
    print()
    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    return response


async def _build_local_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await entities_vdb.query(query, top_k=query_param.top_k)

    if not len(results):
        return None
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]  # what is this text_chunks_db doing.  dont remember it in airvx.  check the diagram.
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )
    logger.info(
        f"Local query uses {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list = [
        ["id", "source", "target", "description", "keywords", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])

    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )

    # Add null check for node data
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v  # Add source_id check
    }

    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            if this_edges:  # Add check for None edges
                for e in this_edges:
                    if (
                        e[1] in all_one_hop_text_units_lookup
                        and c_id in all_one_hop_text_units_lookup[e[1]]
                    ):
                        relation_counts += 1

            chunk_data = await text_chunks_db.get_by_id(c_id)
            if chunk_data is not None and "content" in chunk_data:  # Add content check
                all_text_units_lookup[c_id] = {
                    "data": chunk_data,
                    "order": index,
                    "relation_counts": relation_counts,
                }

    # Filter out None values and ensure data has content
    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not all_text_units:
        logger.warning("No valid text units found")
        return []

    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )

    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units = [t["data"] for t in all_text_units]
    return all_text_units


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = set()
    for this_edges in all_related_edges:
        all_edges.update([tuple(sorted(e)) for e in this_edges])
    all_edges = list(all_edges)
    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )
    return all_edges_data


async def global_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    context = None
    use_model_func = global_config["llm_model_func"]

    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query)
    result = await use_model_func(kw_prompt)
    json_text = locate_json_string_body_from_string(result)

    try:
        keywords_data = json.loads(json_text)
        keywords = keywords_data.get("high_level_keywords", [])
        keywords = ", ".join(keywords)
    except json.JSONDecodeError:
        try:
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[1].split("}")[0] + "}"

            keywords_data = json.loads(result)
            keywords = keywords_data.get("high_level_keywords", [])
            keywords = ", ".join(keywords)

        except json.JSONDecodeError as e:
            # Handle parsing error
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"]
    if keywords:
        context = await _build_global_query_context(
            keywords,
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            text_chunks_db,
            query_param,
        )

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    return response


async def _build_global_query_context(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await relationships_vdb.query(keywords, top_k=query_param.top_k)

    if not len(results):
        return None

    edge_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in results]
    )

    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")
    edge_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"]) for r in results]
    )
    edge_datas = [
        {"src_id": k["src_id"], "tgt_id": k["tgt_id"], "rank": d, **v}
        for k, v, d in zip(results, edge_datas, edge_degree)
        if v is not None
    ]
    edge_datas = sorted(
        edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )

    use_entities = await _find_most_related_entities_from_relationships(
        edge_datas, query_param, knowledge_graph_inst
    )
    use_text_units = await _find_related_text_unit_from_relationships(
        edge_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    logger.info(
        f"Global query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} text units"
    )
    relations_section_list = [
        ["id", "source", "target", "description", "keywords", "weight", "rank"]
    ]
    for i, e in enumerate(edge_datas):
        relations_section_list.append(
            [
                i,
                e["src_id"],
                e["tgt_id"],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(use_entities):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    entity_names = set()
    for e in edge_datas:
        entity_names.add(e["src_id"])
        entity_names.add(e["tgt_id"])

    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(entity_name) for entity_name in entity_names]
    )

    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(entity_name) for entity_name in entity_names]
    )
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(entity_names, node_datas, node_degrees)
    ]

    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )

    return node_datas


async def _find_related_text_unit_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]

    all_text_units_lookup = {}

    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            if c_id not in all_text_units_lookup:
                all_text_units_lookup[c_id] = {
                    "data": await text_chunks_db.get_by_id(c_id),
                    "order": index,
                }

    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    all_text_units: list[TextChunkSchema] = [t["data"] for t in all_text_units]

    return all_text_units


async def hybrid_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    low_level_context = None
    high_level_context = None
    use_model_func = global_config["llm_model_func"]

    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query)

    result = await use_model_func(kw_prompt)
    json_text = locate_json_string_body_from_string(result)
    try:
        keywords_data = json.loads(json_text)
        hl_keywords = keywords_data.get("high_level_keywords", [])
        ll_keywords = keywords_data.get("low_level_keywords", [])
        hl_keywords = ", ".join(hl_keywords)
        ll_keywords = ", ".join(ll_keywords)
    except json.JSONDecodeError:
        try:
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[1].split("}")[0] + "}"
            keywords_data = json.loads(result)
            hl_keywords = keywords_data.get("high_level_keywords", [])
            ll_keywords = keywords_data.get("low_level_keywords", [])
            hl_keywords = ", ".join(hl_keywords)
            ll_keywords = ", ".join(ll_keywords)
        # Handle parsing error
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"]
    if ll_keywords:
        low_level_context = await _build_local_query_context(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )

    if hl_keywords:
        high_level_context = await _build_global_query_context(
            hl_keywords,
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            text_chunks_db,
            query_param,
        )

    context = combine_contexts(high_level_context, low_level_context)

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )
    return response


def combine_contexts(high_level_context, low_level_context):
    # Function to extract entities, relationships, and sources from context strings

    def extract_sections(context):
        entities_match = re.search(
            r"-----Entities-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )
        relationships_match = re.search(
            r"-----Relationships-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )
        sources_match = re.search(
            r"-----Sources-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )

        entities = entities_match.group(1) if entities_match else ""
        relationships = relationships_match.group(1) if relationships_match else ""
        sources = sources_match.group(1) if sources_match else ""

        return entities, relationships, sources

    # Extract sections from both contexts

    if high_level_context is None:
        warnings.warn(
            "High Level context is None. Return empty High entity/relationship/source"
        )
        hl_entities, hl_relationships, hl_sources = "", "", ""
    else:
        hl_entities, hl_relationships, hl_sources = extract_sections(high_level_context)

    if low_level_context is None:
        warnings.warn(
            "Low Level context is None. Return empty Low entity/relationship/source"
        )
        ll_entities, ll_relationships, ll_sources = "", "", ""
    else:
        ll_entities, ll_relationships, ll_sources = extract_sections(low_level_context)

    # Combine and deduplicate the entities

    combined_entities = process_combine_contexts(hl_entities, ll_entities)
    combined_entities = chunking_by_token_size(combined_entities, max_token_size=2000)
    # Combine and deduplicate the relationships
    combined_relationships = process_combine_contexts(
        hl_relationships, ll_relationships
    )
    combined_relationships = chunking_by_token_size(
        combined_relationships, max_token_size=2000
    )
    # Combine and deduplicate the sources
    combined_sources = process_combine_contexts(hl_sources, ll_sources)
    combined_sources = chunking_by_token_size(combined_sources, max_token_size=2000)
    # Format the combined context
    return f"""
-----Entities-----
```csv
{combined_entities}
```
-----Relationships-----
```csv
{combined_relationships}
```
-----Sources-----
```csv
{combined_sources}
```
"""


async def naive_query(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    use_model_func = global_config["llm_model_func"]
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return PROMPTS["fail_response"]
    chunks_ids = [r["id"] for r in results]

    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    maybe_trun_chunks = truncate_list_by_token_size(
        chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    section = "--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])
    if query_param.only_need_context:
        return section
    sys_prompt_temp = PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=section, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )

    if len(response) > len(sys_prompt):
        response = (
            response[len(sys_prompt) :]
            .replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    return response


async def path2chunk(
    scored_edged_reasoning_path, knowledge_graph_inst, pairs_append, query, max_chunks=5
):
    already_node = {}
    for k, v in scored_edged_reasoning_path.items():
        node_chunk_id = None

        for pathtuple, scorelist in v["Path"].items():
            if pathtuple in pairs_append:
                use_edge = pairs_append[pathtuple]
                edge_datas = []
                edge_datas = await asyncio.gather(
                    *[knowledge_graph_inst.get_edge(r[0], r[1]) for r in use_edge]
                )
                text_units = [
                    split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
                    for dp in edge_datas  # chunk ID
                ][0]

            else:
                use_edge = []
                text_units = []

            node_datas = await asyncio.gather(
                *[knowledge_graph_inst.get_node(pathtuple[0])]
            )
            for dp in node_datas:
                text_units_node = split_string_by_multi_markers(
                    dp["source_id"], [GRAPH_FIELD_SEP]
                )
                text_units = text_units + text_units_node

            node_datas = await asyncio.gather(
                *[knowledge_graph_inst.get_node(ents) for ents in pathtuple[1:]]
            )
            if query is not None:
                for dp in node_datas:
                    text_units_node = split_string_by_multi_markers(
                        dp["source_id"], [GRAPH_FIELD_SEP]
                    )
                    descriptionlist_node = split_string_by_multi_markers(
                        dp["description"], [GRAPH_FIELD_SEP]
                    )
                    if descriptionlist_node[0] not in already_node.keys():
                        already_node[descriptionlist_node[0]] = None

                        if len(text_units_node) == len(descriptionlist_node):
                            if len(text_units_node) > 5:
                                max_ids = int(max(5, len(text_units_node) / 2))
                                should_consider_idx = calculate_similarity(
                                    descriptionlist_node, query, k=max_ids
                                )
                                text_units_node = [
                                    text_units_node[i] for i in should_consider_idx
                                ]
                                already_node[descriptionlist_node[0]] = text_units_node
                    else:
                        text_units_node = already_node[descriptionlist_node[0]]
                    if text_units_node is not None:
                        text_units = text_units + text_units_node

            count_dict = Counter(text_units)
            total_score = scorelist[0] + scorelist[1] + 1
            for key, value in count_dict.items():
                count_dict[key] = value * total_score
            if node_chunk_id is None:
                node_chunk_id = count_dict
            else:
                node_chunk_id = node_chunk_id + count_dict
        v["Path"] = []
        if node_chunk_id is None:
            node_datas = await asyncio.gather(*[knowledge_graph_inst.get_node(k)])
            for dp in node_datas:
                text_units_node = split_string_by_multi_markers(
                    dp["source_id"], [GRAPH_FIELD_SEP]
                )
                count_dict = Counter(text_units_node)

            for id in count_dict.most_common(max_chunks):
                v["Path"].append(id[0])
            # v['Path'] = count_dict.most_common(max_chunks)#[]
        else:
            for id in count_dict.most_common(max_chunks):
                v["Path"].append(id[0])
            # v['Path'] = node_chunk_id.most_common(max_chunks)
    return scored_edged_reasoning_path


# import time
# import asyncio
# from collections import Counter, OrderedDict
# from typing import Dict, Any, List, Tuple

# async def path2chunk(
#     scored_edged_reasoning_path: Dict[str, Any],
#     knowledge_graph_inst,
#     pairs_append: Dict[Tuple[str, ...], List[Tuple[str, str]]],
#     query: str,
#     max_chunks: int = 5,
# ):
#     """
#     Optimized path->chunk mapping:
#       - Batch fetch edges and nodes per scored node (k)
#       - Cap nodes per k via a frequency heuristic
#       - Use a small in-process LRU cache for nodes
#       - Print plain timings for edge fetch, node fetch, processing
#     Returns modified scored_edged_reasoning_path with v['Path'] replaced by list of chunk ids.
#     """

#     # Create a small module-local LRU cache on the function object (persistent across calls)
#     if not hasattr(path2chunk, "GLOBAL_NODE_CACHE"):
#         path2chunk.GLOBAL_NODE_CACHE = OrderedDict()
#     GLOBAL_NODE_CACHE: OrderedDict = path2chunk.GLOBAL_NODE_CACHE
#     GLOBAL_NODE_CACHE_MAX = 20000  # tune as needed for memory budget

#     overall_start = time.time()
#     print(">>> path2chunk start")

#     # Tunables (adjust to your environment)
#     MAX_NODES_PER_K = 1024     # max unique nodes to fetch per scored node 'k'
#     NODE_BATCH_SIZE = 128      # batch size for asyncio.gather when fetching nodes
#     EDGE_BATCH_SIZE = 256      # batch size for edges if many
#     # Iterate scoring nodes sequentially (to avoid overwhelming backend); can be parallelized with a semaphore
#     for k, v in scored_edged_reasoning_path.items():
#         k_start = time.time()

#         # v["Path"] expected to be mapping: pathtuple -> scorelist
#         path_items: List[Tuple[Tuple[str, ...], List[float]]] = list(v.get("Path", {}).items()) if v.get("Path") else []

#         # 1) Determine edges to fetch and node frequencies (for prioritization)
#         edges_to_fetch: List[Tuple[str, str]] = []
#         node_freq_counter = Counter()
#         for pathtuple, _ in path_items:
#             # count nodes for frequency heuristic
#             for n in pathtuple:
#                 node_freq_counter[n] += 1
#             # collect edge pairs if present in pairs_append
#             if pathtuple in pairs_append:
#                 edges = pairs_append[pathtuple]
#                 # edges is expected to be list of (src, tgt) pairs
#                 for pair in edges:
#                     edges_to_fetch.append(pair)

#         # 2) Batch fetch edges (if any)
#         edge_fetch_start = time.time()
#         edge_map: Dict[Tuple[str, str], Dict] = {}
#         if edges_to_fetch:
#             # Deduplicate edges while preserving order
#             seen = set()
#             unique_edges = []
#             for p in edges_to_fetch:
#                 tp = tuple(p)
#                 if tp not in seen:
#                     seen.add(tp)
#                     unique_edges.append(tp)
#             # Fetch in batches
#             for i in range(0, len(unique_edges), EDGE_BATCH_SIZE):
#                 batch = unique_edges[i : i + EDGE_BATCH_SIZE]
#                 batch_results = await asyncio.gather(
#                     *[knowledge_graph_inst.get_edge(a, b) for (a, b) in batch],
#                     return_exceptions=False,
#                 )
#                 for pair, ed in zip(batch, batch_results):
#                     edge_map[pair] = ed  # ed may be None or dict
#         edge_fetch_time = time.time() - edge_fetch_start
#         print(f'[{k}] Edge fetch time: {edge_fetch_time:.3f}s (unique_edges: {len(edge_map)})')

#         # 3) Choose nodes to fetch using frequency heuristic and global cache
#         node_fetch_start = time.time()
#         # If no nodes in path_items, we'll later fallback to fetching node 'k' itself
#         node_list_to_consider = [n for n, _ in node_freq_counter.most_common(MAX_NODES_PER_K)]

#         # Partition into cached and missing
#         cached_nodes = {}
#         missing_nodes = []
#         for n in node_list_to_consider:
#             if n in GLOBAL_NODE_CACHE:
#                 cached_nodes[n] = GLOBAL_NODE_CACHE[n]
#                 # refresh LRU position
#                 GLOBAL_NODE_CACHE.move_to_end(n)
#             else:
#                 missing_nodes.append(n)

#         # Batch-fetch missing nodes
#         node_results_map = dict(cached_nodes)  # name -> payload
#         if missing_nodes:
#             for i in range(0, len(missing_nodes), NODE_BATCH_SIZE):
#                 batch = missing_nodes[i : i + NODE_BATCH_SIZE]
#                 batch_results = await asyncio.gather(
#                     *[knowledge_graph_inst.get_node(name) for name in batch],
#                     return_exceptions=False,
#                 )
#                 for name, res in zip(batch, batch_results):
#                     node_results_map[name] = res
#                     # update global cache LRU
#                     GLOBAL_NODE_CACHE[name] = res
#                     GLOBAL_NODE_CACHE.move_to_end(name)
#                     if len(GLOBAL_NODE_CACHE) > GLOBAL_NODE_CACHE_MAX:
#                         GLOBAL_NODE_CACHE.popitem(last=False)

#         node_fetch_time = time.time() - node_fetch_start
#         print(f'[{k}] Node fetch time: {node_fetch_time:.3f}s (fetched: {len(node_results_map) - len(cached_nodes)}, cached: {len(cached_nodes)})')

#         # 4) For each pathtuple build text_units using fetched edge_map and node_results_map
#         process_start = time.time()
#         node_chunk_counter = Counter()
#         # Per-k small cache for description-based selection (same as earlier logic)
#         already_node_selection: Dict[str, List[str] or None] = {}

#         for pathtuple, scorelist in path_items:
#             text_units: List[str] = []

#             # collect edge-derived text units
#             if pathtuple in pairs_append:
#                 for (a, b) in pairs_append[pathtuple]:
#                     ed = edge_map.get((a, b))
#                     if ed:
#                         src = ed.get("source_id", "") or ""
#                         parts = split_string_by_multi_markers(src, [GRAPH_FIELD_SEP])
#                         if parts:
#                             text_units.extend(parts)

#             # add head node's source_id parts
#             head = pathtuple[0] if len(pathtuple) > 0 else None
#             if head:
#                 dp_head = node_results_map.get(head)
#                 if dp_head:
#                     parts = split_string_by_multi_markers(dp_head.get("source_id", "") or "", [GRAPH_FIELD_SEP])
#                     if parts:
#                         text_units.extend(parts)

#             # add tail nodes (with description-based selection if query is present)
#             for tail in pathtuple[1:]:
#                 dp = node_results_map.get(tail)
#                 if not dp:
#                     continue
#                 descriptionlist_node = split_string_by_multi_markers(dp.get("description", "") or "", [GRAPH_FIELD_SEP])
#                 text_units_node = split_string_by_multi_markers(dp.get("source_id", "") or "", [GRAPH_FIELD_SEP])

#                 selected_text_units_node = text_units_node
#                 if query is not None and descriptionlist_node:
#                     desc_key = descriptionlist_node[0]
#                     if desc_key not in already_node_selection:
#                         already_node_selection[desc_key] = None
#                         # only apply similarity-based selection when lengths match and list is long
#                         if len(text_units_node) == len(descriptionlist_node) and len(text_units_node) > 5:
#                             max_ids = int(max(5, len(text_units_node) / 2))
#                             try:
#                                 should_consider_idx = calculate_similarity(descriptionlist_node, query, k=max_ids)
#                                 selected_text_units_node = [text_units_node[i] for i in should_consider_idx]
#                                 already_node_selection[desc_key] = selected_text_units_node
#                             except Exception:
#                                 # If similarity fails, fall back to full list
#                                 selected_text_units_node = text_units_node
#                                 already_node_selection[desc_key] = selected_text_units_node
#                     else:
#                         selected_text_units_node = already_node_selection[desc_key] or []

#                 if selected_text_units_node:
#                     text_units.extend(selected_text_units_node)

#             # Aggregate counts scaled by score
#             if text_units:
#                 count_dict = Counter(text_units)
#                 s0 = scorelist[0] if len(scorelist) > 0 else 0
#                 s1 = scorelist[1] if len(scorelist) > 1 else 0
#                 total_score = s0 + s1 + 1
#                 for key_, val_ in count_dict.items():
#                     node_chunk_counter[key_] += val_ * total_score

#         # 5) Fallback: if nothing gathered from paths, use node 'k' own source_id
#         if not node_chunk_counter:
#             dp_k = node_results_map.get(k)
#             if dp_k is None:
#                 # fetch single node k (not in our selected list) - do this individually
#                 dp_k = await knowledge_graph_inst.get_node(k)
#                 # update global cache a bit
#                 if dp_k is not None:
#                     GLOBAL_NODE_CACHE[k] = dp_k
#                     GLOBAL_NODE_CACHE.move_to_end(k)
#                     if len(GLOBAL_NODE_CACHE) > GLOBAL_NODE_CACHE_MAX:
#                         GLOBAL_NODE_CACHE.popitem(last=False)
#             if dp_k:
#                 text_units_node = split_string_by_multi_markers(dp_k.get("source_id", "") or "", [GRAPH_FIELD_SEP])
#                 node_chunk_counter.update(text_units_node)

#         # 6) Select top chunk ids
#         top_ids = [item[0] for item in node_chunk_counter.most_common(max_chunks)]
#         v["Path"] = top_ids

#         process_time = time.time() - process_start
#         total_k_time = time.time() - k_start
#         print(f'[{k}] Processing time: {process_time:.3f}s, total_k_time: {total_k_time:.3f}s, selected_chunks: {len(top_ids)}')

#     total_time = time.time() - overall_start
#     print(f">>> path2chunk done, total time: {total_time:.3f}s")
#     return scored_edged_reasoning_path

def scorednode2chunk(input_dict, values_dict):
    for key, value_list in input_dict.items():
        input_dict[key] = [
            values_dict.get(val, None) for val in value_list if val in values_dict
        ]
        input_dict[key] = [val for val in input_dict[key] if val is not None]


def kwd2chunk(ent_from_query_dict, chunks_ids, chunk_nums):
    final_chunk = Counter()
    final_chunk_id = []
    for key, list_of_dicts in ent_from_query_dict.items():
        total_id_scores = Counter()
        id_scores_list = []
        id_scores = {}
        for d in list_of_dicts:
            if d == list_of_dicts[0]:
                score = d["Score"] * 2
            else:
                score = d["Score"]
            path = d["Path"]

            for id in path:
                if id == path[0] and id in chunks_ids:
                    score = score * 10
                if id in id_scores:
                    id_scores[id] += score
                else:
                    id_scores[id] = score
        id_scores_list.append(id_scores)

        for scores in id_scores_list:
            total_id_scores.update(scores)
        final_chunk = final_chunk + total_id_scores  # .most_common(3)

    for i in final_chunk.most_common(chunk_nums):
        final_chunk_id.append(i[0])
    return final_chunk_id


async def _build_mini_query_context(
    ent_from_query,
    type_keywords,
    originalquery,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    entity_name_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    embedder,
    query_param: QueryParam,
):
    imp_ents = []
    nodes_from_query_list = []
    ent_from_query_dict = {}

    for ent in ent_from_query:
        ent_from_query_dict[ent] = []
        results_node = await entity_name_vdb.query(ent, top_k=query_param.top_k)

        nodes_from_query_list.append(results_node)
        ent_from_query_dict[ent] = [e["entity_name"] for e in results_node]

    candidate_reasoning_path = {}

    for results_node_list in nodes_from_query_list:
        candidate_reasoning_path_new = {
            key["entity_name"]: {"Score": key["distance"], "Path": []}
            for key in results_node_list
        }

        candidate_reasoning_path = {
            **candidate_reasoning_path,
            **candidate_reasoning_path_new,
        }
    for key in candidate_reasoning_path.keys():
        candidate_reasoning_path[key][
            "Path"
        ] = await knowledge_graph_inst.get_neighbors_within_k_hops(key, 2)
        imp_ents.append(key)

    short_path_entries = {
        name: entry
        for name, entry in candidate_reasoning_path.items()
        if len(entry["Path"]) < 1
    }
    sorted_short_path_entries = sorted(
        short_path_entries.items(), key=lambda x: x[1]["Score"], reverse=True
    )
    save_p = max(1, int(len(sorted_short_path_entries) * 0.2))
    top_short_path_entries = sorted_short_path_entries[:save_p]
    top_short_path_dict = {name: entry for name, entry in top_short_path_entries}
    long_path_entries = {
        name: entry
        for name, entry in candidate_reasoning_path.items()
        if len(entry["Path"]) >= 1
    }
    candidate_reasoning_path = {**long_path_entries, **top_short_path_dict}
    node_datas_from_type = await knowledge_graph_inst.get_node_from_types(
        type_keywords
    )  # entity_type, description,...

    maybe_answer_list = [n["entity_name"] for n in node_datas_from_type]
    imp_ents = imp_ents + maybe_answer_list
    scored_reasoning_path = cal_path_score_list(
        candidate_reasoning_path, maybe_answer_list
    )

    results_edge = await relationships_vdb.query(
        originalquery, top_k=len(ent_from_query) * query_param.top_k
    )
    goodedge = []
    badedge = []
    for item in results_edge:
        if item["src_id"] in imp_ents or item["tgt_id"] in imp_ents:
            goodedge.append(item)
        else:
            badedge.append(item)
    scored_edged_reasoning_path, pairs_append = edge_vote_path(
        scored_reasoning_path, goodedge
    )
    scored_edged_reasoning_path = await path2chunk(
        scored_edged_reasoning_path,
        knowledge_graph_inst,
        pairs_append,
        originalquery,
        max_chunks=3,
    )

    entites_section_list = []
    node_datas = await asyncio.gather(
        *[
            knowledge_graph_inst.get_node(entity_name)
            for entity_name in scored_edged_reasoning_path.keys()
        ]
    )
    node_datas = [
        {**n, "entity_name": k, "Score": scored_edged_reasoning_path[k]["Score"]}
        for k, n in zip(scored_edged_reasoning_path.keys(), node_datas)
    ]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                n["entity_name"],
                n["Score"],
                n.get("description", "UNKNOWN"),
            ]
        )
    entites_section_list = sorted(
        entites_section_list, key=lambda x: x[1], reverse=True
    )
    entites_section_list = truncate_list_by_token_size(
        entites_section_list,
        key=lambda x: x[2],
        max_token_size=query_param.max_token_for_node_context,
    )

    entites_section_list.insert(0, ["entity", "score", "description"])
    entities_context = list_of_list_to_csv(entites_section_list)

    scorednode2chunk(ent_from_query_dict, scored_edged_reasoning_path)

    results = await chunks_vdb.query(originalquery, top_k=int(query_param.top_k / 2))
    chunks_ids = [r["id"] for r in results]
    final_chunk_id = kwd2chunk(
        ent_from_query_dict, chunks_ids, chunk_nums=int(query_param.top_k / 2)
    )

    if not len(results_node):
        return None

    if not len(results_edge):
        return None

    use_text_units = await asyncio.gather(
        *[text_chunks_db.get_by_id(id) for id in final_chunk_id]
    )
    text_units_section_list = [["id", "content"]]

    for i, t in enumerate(use_text_units):
        if t is not None:
            text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Sources-----
```csv
{text_units_context}
```
""", entities_context, text_units_context

# import asyncio
# import time

# async def _build_mini_query_context(
#     ent_from_query,
#     type_keywords,
#     originalquery,
#     knowledge_graph_inst: BaseGraphStorage,
#     entities_vdb: BaseVectorStorage,
#     entity_name_vdb: BaseVectorStorage,
#     relationships_vdb: BaseVectorStorage,
#     chunks_vdb: BaseVectorStorage,
#     text_chunks_db: BaseKVStorage[TextChunkSchema],
#     embedder,
#     query_param: QueryParam,
# ):
#     start_total = time.time()
#     print(">>> Starting _build_mini_query_context")

#     # 1️⃣ Entity name lookups
#     t0 = time.time()
#     nodes_from_query_list = []
#     ent_from_query_dict = {}

#     for ent in ent_from_query:
#         results_node = await entity_name_vdb.query(ent, top_k=query_param.top_k)
#         nodes_from_query_list.append(results_node)
#         ent_from_query_dict[ent] = [e["entity_name"] for e in results_node]
#     print(f"[TIME] Entity name lookups: {time.time() - t0:.3f}s")

#     # 2️⃣ Build candidate reasoning path
#     t0 = time.time()
#     candidate_reasoning_path = {}
#     for results_node_list in nodes_from_query_list:
#         for key in results_node_list:
#             candidate_reasoning_path[key["entity_name"]] = {
#                 "Score": key["distance"],
#                 "Path": [],
#             }
#     print(f"[TIME] Build candidate reasoning path: {time.time() - t0:.3f}s")

#     # 3️⃣ Get neighbors (within 2 hops)
#     t0 = time.time()
#     imp_ents = []
#     for key in candidate_reasoning_path.keys():
#         candidate_reasoning_path[key]["Path"] = await knowledge_graph_inst.get_neighbors_within_k_hops(key, 2)
#         imp_ents.append(key)
#     print(f"[TIME] Neighbor retrieval: {time.time() - t0:.3f}s")

#     # 4️⃣ Filter short and long paths
#     t0 = time.time()
#     short_path_entries = {name: entry for name, entry in candidate_reasoning_path.items() if len(entry["Path"]) < 1}
#     sorted_short_path_entries = sorted(short_path_entries.items(), key=lambda x: x[1]["Score"], reverse=True)
#     save_p = max(1, int(len(sorted_short_path_entries) * 0.2))
#     top_short_path_entries = sorted_short_path_entries[:save_p]
#     top_short_path_dict = {name: entry for name, entry in top_short_path_entries}
#     long_path_entries = {name: entry for name, entry in candidate_reasoning_path.items() if len(entry["Path"]) >= 1}
#     candidate_reasoning_path = {**long_path_entries, **top_short_path_dict}
#     print(f"[TIME] Short/long path filter: {time.time() - t0:.3f}s")

#     # 5️⃣ Get node data by type
#     t0 = time.time()
#     node_datas_from_type = await knowledge_graph_inst.get_node_from_types(type_keywords)
#     print(f"[TIME] Get node data from types: {time.time() - t0:.3f}s")

#     # 6️⃣ Calculate path scores
#     t0 = time.time()
#     maybe_answer_list = [n["entity_name"] for n in node_datas_from_type]
#     imp_ents = imp_ents + maybe_answer_list
#     scored_reasoning_path = cal_path_score_list(candidate_reasoning_path, maybe_answer_list)
#     print(f"[TIME] Calculate path scores: {time.time() - t0:.3f}s")

#     # 7️⃣ Query relationships
#     t0 = time.time()
#     results_edge = await relationships_vdb.query(originalquery, top_k=len(ent_from_query) * query_param.top_k)
#     print(f"[TIME] Relationship query: {time.time() - t0:.3f}s")

#     # 8️⃣ Filter good/bad edges
#     t0 = time.time()
#     goodedge, badedge = [], []
#     for item in results_edge:
#         if item["src_id"] in imp_ents or item["tgt_id"] in imp_ents:
#             goodedge.append(item)
#         else:
#             badedge.append(item)
#     print(f"[TIME] Edge filtering: {time.time() - t0:.3f}s")

#     # 9️⃣ Edge vote and chunk mapping
#     t0 = time.time()
#     scored_edged_reasoning_path, pairs_append = edge_vote_path(scored_reasoning_path, goodedge)
#     scored_edged_reasoning_path = await path2chunk(
#         scored_edged_reasoning_path, knowledge_graph_inst, pairs_append, originalquery, max_chunks=3
#     )
#     print(f"[TIME] Edge vote + path2chunk: {time.time() - t0:.3f}s")

#     # 🔟 Retrieve nodes
#     t0 = time.time()
#     node_datas = await asyncio.gather(
#         *[knowledge_graph_inst.get_node(entity_name) for entity_name in scored_edged_reasoning_path.keys()]
#     )
#     print(f"[TIME] Node retrieval: {time.time() - t0:.3f}s")

#     # 11️⃣ Build entities section
#     t0 = time.time()
#     entites_section_list = []
#     for k, n in zip(scored_edged_reasoning_path.keys(), node_datas):
#         entites_section_list.append([n.get("entity_name", k), scored_edged_reasoning_path[k]["Score"], n.get("description", "UNKNOWN")])
#     entites_section_list = sorted(entites_section_list, key=lambda x: x[1], reverse=True)
#     entites_section_list = truncate_list_by_token_size(
#         entites_section_list, key=lambda x: x[2], max_token_size=query_param.max_token_for_node_context
#     )
#     entites_section_list.insert(0, ["entity", "score", "description"])
#     entities_context = list_of_list_to_csv(entites_section_list)
#     print(f"[TIME] Build entity context: {time.time() - t0:.3f}s")

#     # 12️⃣ Query chunks
#     t0 = time.time()
#     results = await chunks_vdb.query(originalquery, top_k=int(query_param.top_k / 2))
#     print(f"[TIME] Chunk query: {time.time() - t0:.3f}s")

#     # 13️⃣ Get text units
#     t0 = time.time()
#     chunks_ids = [r["id"] for r in results]
#     final_chunk_id = kwd2chunk(ent_from_query_dict, chunks_ids, chunk_nums=int(query_param.top_k / 2))
#     use_text_units = await asyncio.gather(*[text_chunks_db.get_by_id(id) for id in final_chunk_id])
#     print(f"[TIME] Text unit retrieval: {time.time() - t0:.3f}s")

#     # 14️⃣ Build text context
#     t0 = time.time()
#     text_units_section_list = [["id", "content"]]
#     for i, t in enumerate(use_text_units):
#         if t is not None:
#             text_units_section_list.append([i, t["content"]])
#     text_units_context = list_of_list_to_csv(text_units_section_list)
#     print(f"[TIME] Build text context: {time.time() - t0:.3f}s")

#     print(f"✅ TOTAL EXECUTION TIME: {time.time() - start_total:.3f}s")

#     return f"""
# -----Entities-----
# ```csv
# {entities_context}
# -----Sources-----

# {text_units_context}


# """, entities_context, text_units_context


async def minirag_query(  # MiniRAG
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    entity_name_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    embedder,
    query_param: QueryParam,
    global_config: dict,
) -> str:
    use_model_func = global_config["llm_model_func"]
    kw_prompt_temp = PROMPTS["minirag_query2kwd"]
    TYPE_POOL, TYPE_POOL_w_CASE = await knowledge_graph_inst.get_types()
    kw_prompt = kw_prompt_temp.format(query=query, TYPE_POOL=TYPE_POOL)
    result = await use_model_func(kw_prompt)

    try:
        keywords_data = json_repair.loads(result)

        type_keywords = keywords_data.get("answer_type_keywords", [])
        entities_from_query = keywords_data.get("entities_from_query", [])[:5]

    except json.JSONDecodeError:
        try:
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[1].split("}")[0] + "}"
            keywords_data = json_repair.loads(result)
            type_keywords = keywords_data.get("answer_type_keywords", [])
            entities_from_query = keywords_data.get("entities_from_query", [])[:5]

        # Handle parsing error
        except Exception as e:
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"]

    context, entities_context, text_units_context = await _build_mini_query_context(
        entities_from_query,
        type_keywords,
        query,
        knowledge_graph_inst,
        entities_vdb,
        entity_name_vdb,
        relationships_vdb,
        chunks_vdb,
        text_chunks_db,
        embedder,
        query_param,
    )

    # print("entities_context: ", entities_context)
    # print("text_units_context: ", text_units_context)
    # print(kk)
    

    # with open("/u/a/s/asinghal28/private/NLP/Advanced-NLP/hw2/MiniRAG_self/logs/entities_context.csv", "w", newline="", encoding="utf-8") as f:
    #     f.write(entities_context)

    # with open("/u/a/s/asinghal28/private/NLP/Advanced-NLP/hw2/MiniRAG_self/logs/text_units_context.csv", "w", newline="", encoding="utf-8") as f:
    #     f.write(text_units_context)

    df = pd.read_csv(io.StringIO(text_units_context))
    context_json_data = df.to_json(orient="records")



    if query_param.only_need_context:
        # print(type(context))
        return context
    if context is None:
        return PROMPTS["fail_response"]

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )

    return query, context_json_data,  sys_prompt, response
