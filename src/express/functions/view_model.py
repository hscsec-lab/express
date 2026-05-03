from tqdm import tqdm
from textual.app import App, ComposeResult
from textual.widgets import Tree, Header, DataTable, Footer, TabbedContent, TabPane, Static

from express.base.model import Model
from express.math.SVDAnalyzer import SVDAnalyzer


class ModelTreeViewer(Static):
    """
    Model tree viewer with recursive parameter summation for parent nodes.
    """

    def __init__(self, model_name, data):
        super().__init__()
        self.model_name = model_name
        self.data = data
        self.tree_data = {}
        self._preprocess_tree()

    def _preprocess_tree(self):
        """
        Build a nested dictionary and calculate cumulative parameter counts.
        """
        for key, info in self.data.items():
            parts = key.split('.')
            current = self.tree_data
            for part in parts:
                current = current.setdefault(part, {"_sum": 0, "_info": None})
                current["_sum"] += info['params']
            current["_info"] = info

    def compose(self) -> ComposeResult:
        tree = Tree(self.model_name)
        self._render_node(tree.root, self.tree_data)
        yield tree

    def _render_node(self, tree_node, data_dict):
        """
        Recursively render tree nodes with formatted parameter sums.
        """
        for name, content in data_dict.items():
            if name == "_sum" or name == "_info":
                continue

            count_str = format_params(content["_sum"])
            info = content["_info"]

            if info:
                fp, er = info.get('fp', ""), info.get('er', 0.0)
                label = f"{name} [blue]({count_str})[/] [dim](ER: {er:.2f})[/] [yellow]SVs: {fp}[/]"
            else:
                label = f"{name} [blue]({count_str})[/]"

            new_node = tree_node.add(label, expand=False)
            self._render_node(new_node, content)

class ModelTableViewer(Static):
    """
    Model table viewer with sortable columns for parameter analysis.
    """

    def __init__(self, data):
        super().__init__()
        self.data = data

    def compose(self) -> ComposeResult:
        table = DataTable()
        table.zebra_stripes = True
        table.cursor_type = "row"
        table.add_column("Param Name", key="name")
        table.add_column("Count", key="count")
        table.add_column("ER(Effective Rank)", key="er")
        table.add_column("Top SVs(Singular Values)", key="svs")

        for name, info in self.data.items():
            svs = ", ".join([f"{x:.2f}" for x in info['fp']]) if isinstance(info['fp'], list) else str(info['fp'])
            count_str = format_params(info['params'])
            table.add_row(name, count_str, f"{info['er']:.4f}", svs, label=name)
        yield table

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        """
        Handle sorting logic when a header is clicked.
        """
        table = self.query_one(DataTable)
        if event.column_index == 1:
            table.sort(event.column_key, key=self._parse_count, reverse=True)
        elif event.column_index == 2:
            table.sort(event.column_key, key=float, reverse=True)
        else:
            table.sort(event.column_key)

    def _parse_count(self, value: str) -> float:
        """
        Convert human-readable count back to float for sorting.
        """
        mapping = {'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}
        if value[-1] in mapping:
            return float(value[:-1]) * mapping[value[-1]]
        return float(value)


class UnifiedInspector(App):
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, model_name, data):
        super().__init__()
        self.model_name = model_name
        self.data = data

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Tree Hierarchy"):
                yield ModelTreeViewer(self.model_name, self.data)
            with TabPane("Flat Table"):
                yield ModelTableViewer(self.data)
        yield Footer()


def format_params(num: int) -> str:
    """
    Format parameter count into human-readable string (K, M, B, T).
    """
    if num < 1e3:
        return str(num)
    for unit, val in [('T', 1e12), ('B', 1e9), ('M', 1e6), ('K', 1e3)]:
        if num >= val:
            return f"{num / val:.2f}{unit}"
    return str(num)


def pre_analyze_model(state_dict, calc_fp=False, calc_er=False):
    """
    Revised to handle meta devices and ensure all layers are counted.
    """
    results = {}
    for k, v in tqdm(state_dict.items(), desc="Analyzing Layers"):
        # 获取基础信息（即使在 meta device 上，numel 和 shape 也是可读的）
        num_params = v.numel()
        shape = list(v.shape)

        # 初始化结果
        results[k] = {
            "params": num_params,
            "shape": shape,
            "fp": '',
            "er": 0.0
        }

        # 只有不在 meta 设备上且需要计算时才进行 SVD
        # 如果在 meta 设备上，SVD 无法运行，直接跳过计算逻辑但保留条目
        if not v.is_meta:
            if calc_fp:
                results[k]["fp"] = SVDAnalyzer.get_svd_fingerprint(v)
            if calc_er:
                results[k]["er"] = SVDAnalyzer.get_effective_rank(v)
        else:
            # 可选：在 FP 处标注该层被 offload 了
            results[k]["fp"] = "[Offloaded]"

    return results

def view_model(model: Model, calc_fp=False, calc_er=False):
    state_dict = model.model.state_dict()
    data = pre_analyze_model(state_dict, calc_fp, calc_er)
    UnifiedInspector(model.path.__str__(), data).run()
