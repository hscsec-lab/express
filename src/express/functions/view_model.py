from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedModel
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
                if is_leaf:
                    fp = info.get('fp', "")
                    er = info.get('er', 0.0)
                    label = f"{part} [dim](ER: {er:.2f})[/] [yellow]SVs: {fp}[/]"
                else:
                    label = part
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
        table.cursor_type = "row"  # 设置光标为整行选中，体验更好
        # 添加列并获取列键（Column Key），方便排序
        table.add_column("Param Name", key="name")
        table.add_column("ER(Effective Rank)", key="er")
        table.add_column("Top SVs(Singular Values)", key="svs")

        for name, info in self.data.items():
            if isinstance(info['fp'], list):
                svs = ", ".join([f"{x:.2f}" for x in info['fp']])
            else:
                svs = str(info['fp'])
            # 填入数据
            table.add_row(name, info['er'], svs, label=name)
        yield table

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        """
        当点击表头时触发排序
        """
        table = self.query_one(DataTable)
        # 根据点击的列进行排序
        # ER 列（索引1）按数值排，其他按字符串排
        if event.column_index == 1:
            table.sort(event.column_key, key=float, reverse=True)
        else:
            table.sort(event.column_key)
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

def view_model(model: PreTrainedModel):
    state_dict = model.state_dict()
    data = pre_analyze_model(state_dict)
    UnifiedInspector(model_path, data).run()

if __name__ == '__main__':
    model_path = '/models/HIVE0.5-6B-1115-sft'
    view_model(AutoModelForCausalLM.from_pretrained(model_path))
