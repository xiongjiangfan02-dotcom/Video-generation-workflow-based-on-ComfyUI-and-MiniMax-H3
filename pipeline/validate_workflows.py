"""Check saved UI graphs against their matching executable API graphs."""

import json
from pathlib import Path

ROOT = Path("E:/AI/ComfyUI")


def main():
    for path in sorted((ROOT / "workflows").glob("Drama_*.json")):
        graph = json.loads(path.read_text(encoding="utf-8"))
        api = json.loads((ROOT / "pipeline" / (path.stem + ".api.json")).read_text(encoding="utf-8"))
        nodes = {node["id"]: node for node in graph["nodes"]}
        assert len(nodes) == len(graph["nodes"]) == len(api)
        assert graph["last_node_id"] == max(nodes)
        for link_id, source, output_slot, target, input_slot, kind in graph["links"]:
            output = nodes[source]["outputs"][output_slot]
            incoming = nodes[target]["inputs"][input_slot]
            assert output["type"] == kind
            assert link_id in output["links"] and incoming["link"] == link_id
            assert api[str(target)]["inputs"][incoming["name"]] == [str(source), output_slot]
        for node in nodes.values():
            for incoming in node["inputs"]:
                expected = api[str(node["id"])]["inputs"].get(incoming["name"])
                assert (incoming["link"] is not None) == isinstance(expected, list)
        print(f"VALID {path.name}: {len(nodes)} nodes, {len(graph['links'])} links")


if __name__ == "__main__":
    main()
