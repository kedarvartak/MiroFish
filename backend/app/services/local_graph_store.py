"""
本地图谱存储与检索服务
用于在未配置 ZEP_API_KEY 时，提供可替代的图谱能力。
"""

import json
import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger

logger = get_logger('mirofish.local_graph_store')


class LocalGraphStore:
    """基于 JSON 文件的本地图谱存储"""

    def __init__(self):
        self.base_dir = os.path.join(Config.UPLOAD_FOLDER, 'local_graphs')
        os.makedirs(self.base_dir, exist_ok=True)

    def _graph_path(self, graph_id: str) -> str:
        return os.path.join(self.base_dir, f'{graph_id}.json')

    def _now(self) -> str:
        return datetime.now().isoformat()

    def _load_graph(self, graph_id: str) -> Dict[str, Any]:
        path = self._graph_path(graph_id)
        if not os.path.exists(path):
            raise ValueError(f"图谱不存在: {graph_id}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_graph(self, graph: Dict[str, Any]):
        graph_id = graph.get('graph_id')
        if not graph_id:
            raise ValueError("graph_id 不能为空")
        path = self._graph_path(graph_id)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)

    def create_graph(self, name: str, description: str = "") -> str:
        graph_id = f"mirofish_{uuid.uuid4().hex[:16]}"
        graph = {
            "graph_id": graph_id,
            "name": name,
            "description": description,
            "created_at": self._now(),
            "ontology": {"entity_types": [], "edge_types": []},
            "episodes": [],
            "nodes": [],
            "edges": [],
        }
        self._save_graph(graph)
        return graph_id

    def delete_graph(self, graph_id: str):
        path = self._graph_path(graph_id)
        if os.path.exists(path):
            os.remove(path)

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        graph = self._load_graph(graph_id)
        graph["ontology"] = {
            "entity_types": ontology.get("entity_types", []),
            "edge_types": ontology.get("edge_types", []),
        }
        self._save_graph(graph)

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        graph = self._load_graph(graph_id)
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        return {
            "graph_id": graph_id,
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

    def get_all_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        return self._load_graph(graph_id).get("nodes", [])

    def get_all_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        return self._load_graph(graph_id).get("edges", [])

    def get_node_by_uuid(self, node_uuid: str) -> Optional[Dict[str, Any]]:
        for filename in os.listdir(self.base_dir):
            if not filename.endswith('.json'):
                continue
            path = os.path.join(self.base_dir, filename)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    graph = json.load(f)
                for node in graph.get("nodes", []):
                    if node.get("uuid") == node_uuid:
                        return node
            except Exception:
                continue
        return None

    def add_text(self, graph_id: str, text: str) -> str:
        graph = self._load_graph(graph_id)
        episode_id = f"ep_{uuid.uuid4().hex[:16]}"
        graph.setdefault("episodes", []).append({
            "uuid": episode_id,
            "type": "text",
            "data": text,
            "processed": True,
            "created_at": self._now(),
        })

        entities, relations = self._extract_kg(graph, text)
        node_uuid_map = self._upsert_nodes(graph, entities)
        self._upsert_edges(graph, relations, node_uuid_map)
        self._save_graph(graph)
        return episode_id

    def _extract_kg(self, graph: Dict[str, Any], text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        ontology = graph.get("ontology", {})
        entity_types = [x.get("name", "Entity") for x in ontology.get("entity_types", []) if x.get("name")]
        edge_types = [x.get("name", "related_to") for x in ontology.get("edge_types", []) if x.get("name")]

        try:
            llm = LLMClient()
            prompt = (
                "请从文本中抽取知识图谱信息并仅返回 JSON 对象。\n"
                "输出格式: "
                "{\"entities\":[{\"name\":\"\",\"type\":\"\",\"summary\":\"\",\"attributes\":{}}],"
                "\"relations\":[{\"source\":\"\",\"target\":\"\",\"name\":\"\",\"fact\":\"\",\"attributes\":{}}]}\n"
                f"可用实体类型: {entity_types if entity_types else ['Entity']}\n"
                f"可用关系类型: {edge_types if edge_types else ['related_to']}\n"
                "要求: 不要臆造，最多抽取10个实体和20条关系。\n"
                f"文本:\n{text[:4000]}"
            )
            result = llm.chat_json(
                messages=[
                    {"role": "system", "content": "你是知识图谱抽取器，只返回合法JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=2000,
            )
            entities = result.get("entities", []) if isinstance(result, dict) else []
            relations = result.get("relations", []) if isinstance(result, dict) else []
            if entities:
                return entities[:10], relations[:20]
        except Exception as e:
            logger.debug(f"LLM抽取失败，降级规则抽取: {e}")

        tokens = re.findall(r'[\u4e00-\u9fff]{2,8}|[A-Za-z][A-Za-z0-9_]{1,20}', text)
        dedup = []
        seen = set()
        for token in tokens:
            key = token.lower()
            if key in seen:
                continue
            seen.add(key)
            dedup.append(token)
            if len(dedup) >= 10:
                break

        default_type = entity_types[0] if entity_types else "Entity"
        entities = [{"name": name, "type": default_type, "summary": "", "attributes": {}} for name in dedup]
        rel_name = edge_types[0] if edge_types else "related_to"
        relations = []
        for i in range(len(dedup) - 1):
            relations.append({
                "source": dedup[i],
                "target": dedup[i + 1],
                "name": rel_name,
                "fact": f"{dedup[i]} 与 {dedup[i + 1]} 存在关联",
                "attributes": {},
            })
        return entities, relations

    def _normalize_entity(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        name = str(entity.get("name", "")).strip()
        entity_type = str(entity.get("type", "Entity")).strip() or "Entity"
        summary = str(entity.get("summary", "")).strip()
        attrs = entity.get("attributes", {})
        if not isinstance(attrs, dict):
            attrs = {}
        return {
            "name": name,
            "type": entity_type,
            "summary": summary,
            "attributes": attrs,
        }

    def _upsert_nodes(self, graph: Dict[str, Any], entities: List[Dict[str, Any]]) -> Dict[str, str]:
        nodes = graph.setdefault("nodes", [])
        index: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for node in nodes:
            key = (str(node.get("name", "")).strip().lower(), self._primary_label(node))
            index[key] = node

        node_uuid_map: Dict[str, str] = {}
        for raw in entities:
            normalized = self._normalize_entity(raw)
            if not normalized["name"]:
                continue
            key = (normalized["name"].lower(), normalized["type"])
            existing = index.get(key)
            if existing:
                if normalized["summary"] and not existing.get("summary"):
                    existing["summary"] = normalized["summary"]
                if normalized["attributes"]:
                    existing.setdefault("attributes", {}).update(normalized["attributes"])
                node_uuid = existing["uuid"]
            else:
                node_uuid = f"node_{uuid.uuid4().hex[:16]}"
                node = {
                    "uuid": node_uuid,
                    "name": normalized["name"],
                    "labels": ["Entity", normalized["type"]],
                    "summary": normalized["summary"],
                    "attributes": normalized["attributes"],
                    "created_at": self._now(),
                }
                nodes.append(node)
                index[key] = node
            node_uuid_map[normalized["name"]] = node_uuid
        return node_uuid_map

    def _primary_label(self, node: Dict[str, Any]) -> str:
        labels = node.get("labels", [])
        for label in labels:
            if label not in ["Entity", "Node"]:
                return label
        return "Entity"

    def _upsert_edges(self, graph: Dict[str, Any], relations: List[Dict[str, Any]], node_uuid_map: Dict[str, str]):
        edges = graph.setdefault("edges", [])
        existing = set()
        for edge in edges:
            key = (
                edge.get("source_node_uuid", ""),
                edge.get("target_node_uuid", ""),
                edge.get("name", ""),
                edge.get("fact", ""),
            )
            existing.add(key)

        for rel in relations:
            source_name = str(rel.get("source", "")).strip()
            target_name = str(rel.get("target", "")).strip()
            if not source_name or not target_name:
                continue
            source_uuid = node_uuid_map.get(source_name)
            target_uuid = node_uuid_map.get(target_name)
            if not source_uuid or not target_uuid:
                continue
            name = str(rel.get("name", "related_to")).strip() or "related_to"
            fact = str(rel.get("fact", "")).strip() or f"{source_name} 与 {target_name} 相关"
            attrs = rel.get("attributes", {})
            if not isinstance(attrs, dict):
                attrs = {}
            key = (source_uuid, target_uuid, name, fact)
            if key in existing:
                continue
            edges.append({
                "uuid": f"edge_{uuid.uuid4().hex[:16]}",
                "name": name,
                "fact": fact,
                "source_node_uuid": source_uuid,
                "target_node_uuid": target_uuid,
                "source_node_name": source_name,
                "target_node_name": target_name,
                "attributes": attrs,
                "created_at": self._now(),
                "valid_at": None,
                "invalid_at": None,
                "expired_at": None,
                "episodes": [],
            })
            existing.add(key)
