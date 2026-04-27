from textual.containers import VerticalScroll
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from textual.app import App, ComposeResult
from textual.widgets import Tree, Header, DataTable, Footer, TabbedContent, TabPane, Static

from express.math.SVDAnalyzer import SVDAnalyzer

class ModelTreeViewer(Static):
    """
    Model tree viewer.
    """
    def __init__(self, model_name, data):
        super().__init__()
        self.model_name = model_name
        self.data = data

    def compose(self) -> ComposeResult:
        tree = Tree(self.model_name)
        for key, info in self.data.items():
            self._build_branch(tree.root, key.split('.'), info)
        yield tree

    def _build_branch(self, node, parts, info):
        for i, part in enumerate(parts):
            is_leaf = (i == len(parts) - 1)
            found = next((c for c in node.children if str(c.label).split(' ')[0] == part), None)
            if found:
                node = found
            else:
                label = f"{part} [dim]({info['er']:.2f})[/]" if is_leaf else part
                node = node.add(label, expand=False)

class ModelTableViewer(Static):
    """
    Model table viewer.
    """
    def __init__(self, data):
        super().__init__()
        self.data = data

    def compose(self) -> ComposeResult:
        table = DataTable()
        table.zebra_stripes = True
        table.add_columns("Param Name", "ER", "Top SVs")
        for name, info in self.data.items():
            svs = ", ".join([f"{x:.2f}" for x in info['sv']])
            table.add_row(name, f"{info['er']:.2f}", svs)
        yield table

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

def pre_analyze_model(state_dict):
    """单线程顺序计算，物理内存/显存友好型"""
    results = {}
    # 直接迭代，无需包装 Future
    for k, v in tqdm(state_dict.items(), desc="Analyzing Layers"):
        # 如果后续要用 GPU，这里可以加上 v = v.cuda()
        results[k] = {
            "shape": list(v.shape),
            "fp": SVDAnalyzer.get_svd_fingerprint(v),
            "er": SVDAnalyzer.get_effective_rank(v)
        }
    return results

def view_model(model_path: str):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    state_dict = model.state_dict()
    data = pre_analyze_model(state_dict)
    UnifiedInspector(model_path, data).run()

if __name__ == '__main__':
    model_path = '/models/HIVE0.5-6B-1115-sft'
    view_model(model_path)
