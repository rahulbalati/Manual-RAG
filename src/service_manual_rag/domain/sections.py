"""Section tree utilities."""

from service_manual_rag.domain.models import Section


def flatten_sections(sections: list[Section]) -> list[Section]:
    result: list[Section] = []
    for section in sections:
        result.append(section)
        result.extend(flatten_sections(section.children))
    return result


def heading_path(
    target: Section,
    roots: list[Section],
) -> list[str]:
    path: list[str] = []

    def dfs(node: Section, current: list[str]) -> bool:
        current = [*current, node.title]
        if node.section_id == target.section_id:
            path.extend(current)
            return True
        for child in node.children:
            if dfs(child, current):
                return True
        return False

    for root in roots:
        if dfs(root, []):
            break

    return path
